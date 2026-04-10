import numpy as np
import heapq
import matplotlib.pyplot as plt
import os
import imageio
import glob
def fmm_algorithm(node_list, source_nodes):
    """
        From the source nodes, fmm_algoritm runs the FMM
    """
    
    # Internal function to avoid multiple push and get a very long list of values
    def heap_push(node, dist):
        if dist < best_dist_in_heap.get(node.idx, float('inf')):
            best_dist_in_heap[node.idx] = dist
            heapq.heappush(trial_nodes, (dist, node.idx, node))

    print('FMM running ...')
    print()

    # ---- Step 1 --> Initialization ----
    trial_nodes = []
    best_dist_in_heap = {}

    # Mark source nodes as ALIVE before anything else
    for node in source_nodes:
        assert node.dist < float('inf'), f"Source node {node.node_tag} has dist=inf!"
        node.state = 'ALIVE'
    
    for node in node_list:
        if node.state == 'TRIAL':
            heap_push(node, node.dist)
    
    
    # ---- Step 2 --> Principal Loop - FMM ---- 
    while trial_nodes:
        # Point 1 -> Extract the lower distance node
        d_min, _, current_node = heapq.heappop(trial_nodes)

        # Security check: a node could be inserted many times in the list with different distances
        if current_node.state == 'ALIVE':
            continue
        
        # Check if the extracted distance is still the lowest one -> discard futiles duplicates
        if d_min > best_dist_in_heap.get(current_node.idx, float('inf')):
            continue   

        # Point 2 -> set the minimum distance node as ALIVE 
        current_node.state = 'ALIVE'

        # Point 3 -> Recompute all node distance (if not ALIVE) and put all of them as TRIAL
        for neighbor in current_node.neighbors:
            
            # Recompute the distance of the neighbour nodes only if not ALIVE
            if neighbor.state != 'ALIVE':
                old_dist = neighbor.dist
                new_dist = eikonal_sol(neighbor)

                # Update the trial distance only if it is lower than the previously computed (always true if the node was FAR)
                # see pag 4
                if new_dist < old_dist:
                    neighbor.dist = new_dist
                    neighbor.state = 'TRIAL' # Set the node status as TRIAL, in case it was FAR
                    heap_push(neighbor, new_dist)



def eikonal_sol(node, F=1.0):
    """"
    Compute Locally the approximate solution of the Eikonal Equation
    For acute triangles only, for the moment
    """

    # First - Distance initialization
    dist = float('inf')

    # For compute the distance D from all adjacents triangles in 
    for tri in node.adjacent_triangles:
        
        others = [n for n in tri.nodes if n.node_tag != node.node_tag]
        node_a, node_b = others[0], others[1]

        # CASE 1 - Both other nodes are ALIVE
        if node_a.state == 'ALIVE' and node_b.state == 'ALIVE':

            # Assuming T(A) < T(B)
            if node_a.dist > node_b.dist:
                # If it not the case swap the nodes
                node_a, node_b = node_b, node_a

            # From definitions
            u = node_b.dist - node_a.dist
            # Computations of the lengths of the sides of the triangles
            #! Is it more efficient to compute the distance each time or compute them once and store them?
            # For now I'll just compute them each time
            a = node.distance_to_other_node(node_b)    # BC
            b = node.distance_to_other_node(node_a)    # AC
            c = node_a.distance_to_other_node(node_b)  # AB

            # Computation of the angle theta using the cosine theorem
            #          a^2 = b^2 + c^2 - 2bc cos_theta 
            CA = node_a.coords - node.coords
            CB = node_b.coords - node.coords
            cos_theta = np.dot(CA, CB) / (b * a)
            sin_theta = np.sqrt(1 - cos_theta**2)   
            #! Possible error if cos_theta > 1 due to numerical errors, gemini suggest to use max(0, 1-cos_theta^2), i do not think it is useful

            # Quadratic equation coefficient
            A = a**2 + b**2 -2*a*b*cos_theta
            B = 2*b*u*(a*cos_theta - b)
            C = b**2 * (u**2 - F**2 * a**2 * sin_theta**2)

            Delta  = B**2 - 4 * A * C
            t = float('inf') # if there is no solution, the time is infinite - Do not update it
           
            # Solve only if Delta >= 0 
            if Delta >= 0: 
                t_sol = (-B + np.sqrt(Delta)) / (2 * A)

                #! Check that t_sol can be very small
                if t_sol > 1e-12 and u < t_sol:
                    cond = b * (t_sol - u) / t_sol
                    if a * cos_theta < cond < a / cos_theta and u < t_sol:
                       return  t_sol + node_a.dist

            if t == float('inf'):
                t = min(b * F + node_a.dist, a * F + node_b.dist)

            dist = min(dist, t)

        # CASE 2 --> Degenerate cases
        # in these cases we need to compute the 1d distance
        elif node_a.state == 'ALIVE':
            dist_1d = node_a.dist + node.distance_to_other_node(node_a) * F  # T(A) + b*F
            dist = min(dist, dist_1d)

        elif node_b.state == 'ALIVE':
            dist_1d = node_b.dist + node.distance_to_other_node(node_b) * F # T(B) + c*F
            dist = min(dist, dist_1d)
       
       #! Can we delete these two elif (and consequently the if at the start) and put them alltogether without checking if the two nodes are alive?
    return dist



#########################################
##### FMM with circular wavefornt #######
#########################################

def fmm_algorithm_circ(node_list, source_nodes):
    """
        From the source nodes, fmm_algoritm runs the FMM
    """
    frame_id = 0
    # Internal function to avoid multiple push and get a very long list of values
    def heap_push(node, dist):
        if dist < best_dist_in_heap.get(node.idx, float('inf')):
            best_dist_in_heap[node.idx] = dist
            heapq.heappush(trial_nodes, (dist, node.idx, node))

    print('FMM running ...')
    print()

    # ---- Step 1 --> Initialization ----
    trial_nodes = []
    best_dist_in_heap = {}

    node_virtual_source = {} # Global dictionary: node.idx --> corner NODE object
                             # When the wavefront reaches a node via a 1D fallback (i.e. the geodesic
                             # exits the triangle through a vertex), that vertex becomes a new virtual
                             # source. This dictionary stores, for each such node, the corner NODE
                             # from which it was reached, so that future propagations can use it
                             # as the new local origin instead of trying to back-trace to the
                             # original source S.

    # Mark source node as ALIVE before anything else
    for node in source_nodes:
        assert node.dist < float('inf'), f"Source node {node.node_tag} has dist=inf!"
        node.state = 'ALIVE'
        # In case there is no virtual source saved in the dictionary, it means that 
        # the virtual source is the original one --> save it to simplify the code
        node.virtual_source = node
        node_virtual_source[node.idx] = node

    for node in node_list:
        if node.state == 'TRIAL':
            node.virtual_source = source_nodes[0]
            node_virtual_source[node.idx] = node.virtual_source
            heap_push(node, node.dist)
        # if node.state == 'ALIVE' and node is not source_nodes:
        #     node.virtual_source = source_nodes[0]
        #     node_virtual_source[node.idx] = node.virtual_source

            

    
    # ---- Step 2 --> Principal Loop - FMM ---- 
    while trial_nodes:
        # Point 1 -> Extract the lower distance node
        d_min, _, current_node = heapq.heappop(trial_nodes)

        # Security check: a node could be inserted many times in the list with different distances
        if current_node.state == 'ALIVE':
            continue
        
        # Check if the extracted distance is still the lowest one -> discard futiles duplicates
        if d_min > best_dist_in_heap.get(current_node.idx, float('inf')):
            continue   

        # Point 2 -> set the minimum distance node as ALIVE 
        current_node.state = 'ALIVE'


        # PROBLEM: In the last version, the update of the virtual source/edge_curvature
        #          was done in the eikonal solver, when the nodes were all in the state 
        #          TRIAL. However this would have caused a loop fallback since it will not
        #          be updated after it has been reached for the first time, propagating  
        #          an error, since the first reach is not usually the best one.
        # SOLUTION: Update the curvature or virtual source only when the node has become 
        #           ALIVE, i.e. when the minimum distance has been computed. 
        if hasattr(current_node, 'virtual_source'):
            if current_node.virtual_source is not None:
                node_virtual_source[current_node.idx] = current_node.virtual_source
                # if current_node.IsLastUpdateOned:
                #     node_virtual_source[current_node.idx] = current_node.Source1d
                # else:
                #     node_virtual_source[current_node.idx] = current_node.virtual_source



        # Point 3 -> Recompute all node distance (if not ALIVE) and put all of them as TRIAL
        for neighbor in current_node.neighbors:
            
            # Recompute the distance of the neighbour nodes only if not ALIVE
            if neighbor.state != 'ALIVE':
                old_dist = neighbor.dist

                # Now the function eikonal_sol_circ does not alter the global state, update
                # only when the node becomes ALIVE.
                new_dist, node_vs, is_1d_fallback, isOnTheSameLine, t_1d_list, node_1d_list = eikonal_sol_circ(neighbor, node_virtual_source)
                new_node_vs = node_vs

                
                
                
                # Update the trial distance only if it is lower than the previously computed 
                # (always true if the node was FAR)
                if new_dist < old_dist:
                    neighbor.dist = new_dist
                    if is_1d_fallback and not isOnTheSameLine:
                        neighbor.IsLastUpdateOned = True
                        neighbor.Source1d = node_vs
                        neighbor.virtual_source = node_vs
                    else:
                        neighbor.IsLastUpdateOned = False
                        neighbor.virtual_source = node_vs
                    neighbor.state = 'TRIAL'
                    heap_push(neighbor, new_dist)
        #Save_Front_Frame(node_list, frame_id)
        #Save_Debug_Frame(node_list, current_node, node_virtual_source, frame_id, folder = "debug_frames")
        #Save_Debug_Frame_3D(node_list, current_node, node_virtual_source, frame_id, folder="debug_frames_3D")
        frame_id += 1
    #Make_GIF()
    #Make_GIF(folder="debug_frames", gif_name="front_source.gif")
    #Make_GIF(folder="debug_frames_3D", gif_name="cyl_front_source.gif")


def eikonal_sol_circ(node, node_virtual_source):
    samelinetol = 1e-1
    dist = float('inf')  # Distance Initialization
    best_node_vs = None
    isOnTheSameLine = False
    is_1d_fallback = False
    #is_1d_fallback = False
    t_1d_list = []
    node_list = []
    for tri in node.adjacent_triangles: # Compute all the distance from all adjactens triangles
        others = [n for n in tri.nodes if n.node_tag != node.node_tag]
        node_a, node_b = others[0], others[1]
        
        normal = tri.compute_outward_normal()
        tA_1d = node_a.dist + node.distance_to_other_node(node_a)
        tB_1d = node_b.dist + node.distance_to_other_node(node_b)
        # t_1d_list.append(tA_1d)
        # t_1d_list.append(tB_1d)
        # node_list.append(node_a)
        # node_list.append(node_b)
    
        # ------------------------------------------
        # CASE 1: Both nodes are ALIVE
        # ------------------------------------------
        if node_a.state == 'ALIVE' and node_b.state == 'ALIVE':
            
            if node_a.dist > node_b.dist: # d(A) < d(B) as a convention
                node_a, node_b = node_b, node_a
            tA_1d = node_a.dist + node.distance_to_other_node(node_a)
            tB_1d = node_b.dist + node.distance_to_other_node(node_b)
            d_A_raw = node_a.dist
            d_B_raw = node_b.dist

            # Get the registered virtual sources (if present) for the vertices A and B
            corner_a = node_virtual_source.get(node_a.idx)
            corner_b = node_virtual_source.get(node_b.idx)
            if corner_b is None or corner_a is None:
                print("corner a or b is None")
            # else:
            #     print("Is not None")
            best_local_dist = dist 
            # Evaluate which is the minimum of the two
            if tA_1d <= tB_1d and tA_1d < best_local_dist:
                best_local_dist = tA_1d
                #isOnTheSameLine = abs(np.linalg.norm(np.cross((node.coords - node_a.coords), (node.coords - node_a.virtual_source.coords) - np.dot((node.coords - node_a.virtual_source.coords), normal)*normal))) < 1e-3
                
                v1 = node.coords - node_a.coords
                v2 = (node.coords - node_a.virtual_source.coords)
                v2 = v2 - np.dot(v2, normal) * normal
                
                n1 = np.linalg.norm(v1)
                n2 = np.linalg.norm(v2)
                
                if n1 > 1e-14 and n2 > 1e-14:
                    isOnTheSameLine = np.linalg.norm(np.cross(v1/n1, v2/n2)) < samelinetol
                else:
                    isOnTheSameLine = True

                is_1d_fallback = True
                # Update correctly the source
                best_local_vs =  node_a if isOnTheSameLine == False else corner_a

            elif tB_1d < tA_1d and tB_1d < best_local_dist:
                best_local_dist = tB_1d
                #isOnTheSameLine = abs(np.linalg.norm(np.cross((node.coords - node_b.coords), (node.coords - node_b.virtual_source.coords) - np.dot((node.coords - node_b.virtual_source.coords), normal)*normal))) < 1e-3
                v1 = node.coords - node_b.coords
                v2 = (node.coords - node_b.virtual_source.coords)
                v2 = v2 - np.dot(v2, normal) * normal
                
                n1 = np.linalg.norm(v1)
                n2 = np.linalg.norm(v2)
                
                if n1 > 1e-14 and n2 > 1e-14:
                    isOnTheSameLine = np.linalg.norm(np.cross(v1/n1, v2/n2)) < samelinetol
                else:
                    isOnTheSameLine = True
                best_local_vs  =  node_b if isOnTheSameLine == False else corner_b
                is_1d_fallback = True


            

            #*if corner_a.coords[0] != 0.2: print(corner_a.coords)

            # Attempt 2D Eikonal from corner_b (only if different from corner_a to save computations)
            if corner_a is not corner_b:#Todo here?
                #print("not the same corner")
                #TODO======================================================|
                #TODO======================================================V
                #*d(A) < d(B) so corner_b is closer to C
                #*Get closest corner
                # d_to_corner_b = np.linalg.norm(corner_b.coords - node.coords)
                # d_to_corner_a = np.linalg.norm(corner_a.coords - node.coords)
                # if d_to_corner_a < d_to_corner_b:
                #     close_corner = corner_a
                #     other_corner = corner_b
                #     mynode = node_b #* Left node on my drawing
                #     mynode2 = node_a
                # else:
                #     close_corner = corner_b
                #     other_corner = corner_a
                #     mynode = node_a
                #     mynode2 = node_b
                #print(other_corner.coords)
                if corner_b.dist < corner_a.dist:
                    close_corner = corner_a
                    other_corner = corner_b
                    mynode = node_b
                    mynode2 = node_a 
                else:
                    close_corner = corner_b
                    other_corner = corner_a
                    mynode = node_a
                    mynode2 = node_b

                x_hat = (close_corner.coords - other_corner.coords) / np.linalg.norm(close_corner.coords - other_corner.coords)
                x_hat = x_hat - np.dot(x_hat, normal) * normal
                x_hat = x_hat / np.linalg.norm(x_hat)

                y_hat = np.cross(normal, x_hat)#node_a.coords - np.dot(x_hat, node_a.coords - other_corner.coords) * x_hat
                y_hat = y_hat / np.linalg.norm(y_hat)
                #*Check if C is either C+ or C-
                if np.dot((node.coords - other_corner.coords) - np.dot((node.coords - other_corner.coords), normal) * normal, y_hat) >= 0 and np.dot((mynode.coords - other_corner.coords) - np.dot((mynode.coords - other_corner.coords), normal) * normal, y_hat)>0:
                    isCplus = True
                else:
                    isCplus = False
                # v1 = (node.coords - other_corner.coords) - np.dot((node.coords - other_corner.coords), normal) * normal
                # v2 = x_hat
                # n1 = np.linalg.norm(v1)
                # n2 = np.linalg.norm(v2)
                # if n1 > 1e-14 and n2 > 1e-14:
                #     isOnTheSameLine = np.linalg.norm(np.cross(v1/n1, v2/n2)) < samelinetol
                # else:
                #     isOnTheSameLine = True

                #isOnTheSameLine = abs(np.linalg.norm(np.cross((node.coords - other_corner.coords) - np.dot((node.coords - other_corner.coords), normal) * normal, x_hat))) < 1e-3
                #if isOnTheSameLine: print(node.coords)
                # if isOnTheSameLine:
                #     # Force 1D update → pas de nouvelle source
                #     if tA_1d <= tB_1d:
                #         best_local_dist = tA_1d
                #         best_local_vs = corner_a   # garde la même source
                #     else:
                #         best_local_dist = tB_1d
                #         best_local_vs = corner_b

                #     if best_local_dist < dist:
                #         dist = best_local_dist
                #         best_node_vs = best_local_vs

                #     continue  # skip complètement le 2D
                # t_2d_a = compute_2d_eikonal(node_a, node_b, node, node_a.dist, node_b.dist, corner_a)
                # t_2d_b = compute_2d_eikonal(node_a, node_b, node, node_a.dist, node_b.dist, corner_b)
                # if t_2d_a < t_2d_b:
                #     if t_2d_a < best_local_dist:
                #         best_local_dist = t_2d_a
                #         best_local_vs = corner_a
                # else:
                #     if t_2d_b < best_local_dist:
                #         best_local_dist = t_2d_b
                #         best_local_vs = corner_b

                if isCplus:
                    #t_2d_b = compute_2d_eikonal(mynode, mynode2, node, mynode.dist, other_corner.dist + np.linalg.norm(other_corner.coords - mynode2.coords), other_corner)
                    t_2d_b = compute_2d_eikonal(mynode, mynode2, node, mynode.dist, mynode2.dist, other_corner)
                    # if t_2d_b == float('inf'):
                    #     t_2d_b = compute_2d_eikonal(mynode2, mynode, node, mynode2.dist, mynode.dist, close_corner)

                    #t_2d_b = compute_2d_eikonal(node_a, node_b, node, d_A_raw-corner_a.dist, d_B_raw, corner_a)
                    if t_2d_b < best_local_dist:
                        best_local_dist = t_2d_b
                        best_local_vs = other_corner
                        is_1d_fallback = False
                else:
                    #t_2d_b = compute_2d_eikonal(mynode, mynode2, node, close_corner.dist +  np.linalg.norm(close_corner.coords - mynode.coords), mynode2.dist, close_corner)
                    t_2d_b = compute_2d_eikonal(mynode, mynode2, node, mynode.dist, mynode2.dist, close_corner)
                    # if t_2d_b == float('inf'):
                    #     t_2d_b = compute_2d_eikonal(mynode2, mynode, node, mynode2.dist, mynode.dist, other_corner)
                        
                    #t_2d_b = compute_2d_eikonal(node_a, node_b, node, d_A_raw-corner_a.dist, d_B_raw-corner_b.dist, corner_b)
                    if t_2d_b < best_local_dist:
                        best_local_dist = t_2d_b
                        best_local_vs = close_corner
                        is_1d_fallback = False
                #print(y_hat)
                #TODO======================================================^
                #TODO======================================================|
                # t_2d_b = compute_2d_eikonal(node_a, node_b, node, d_A_raw, d_B_raw, corner_b)
                
            else:#*same source
                #print("Same source")
                # 2D circular update ---> try with corner_a as a source
                t_2d_a = compute_2d_eikonal(node_a, node_b, node, d_A_raw, d_B_raw, corner_a)
                if t_2d_a < best_local_dist:
                    best_local_dist = t_2d_a
                    best_local_vs = corner_a
                    is_1d_fallback = False
                    

                # t_2d_b = compute_2d_eikonal(node_a, node_b, node, d_A_raw, d_B_raw, corner_b)
                # if t_2d_b < best_local_dist:
                #     best_local_dist = t_2d_b
                #     best_local_vs = corner_b
                #     is_1d_fallback = False
                #     isOnTheSameLine = True
                

            # Update of the distance only if it is lower than the previously computed
            if best_local_dist < dist:
                dist = best_local_dist
                best_node_vs = best_local_vs

        # ------------------------------------------
        # CASE 2: Degenerate (Only one node ALIVE)
        # ------------------------------------------
        # 1D fallback
        elif node_a.state == 'ALIVE':
            tA_1d = node_a.dist + node.distance_to_other_node(node_a)
            if tA_1d < dist:
                dist = tA_1d
                is_1d_fallback = True
                
                v1 = node.coords - node_a.coords
                v2 = (node.coords - node_a.virtual_source.coords)
                v2 = v2 - np.dot(v2, normal) * normal
                
                n1 = np.linalg.norm(v1)
                n2 = np.linalg.norm(v2)
                
                if n1 > 1e-14 and n2 > 1e-14:
                    isOnTheSameLine = np.linalg.norm(np.cross(v1/n1, v2/n2)) < samelinetol
                else:
                    isOnTheSameLine = True
                    
                best_node_vs = node_virtual_source.get(node_a.idx) if isOnTheSameLine else node_a  
                # if node.virtual_source == node_virtual_source.get(node_a.idx):
                #     best_local_vs = node_virtual_source.get(node_a.idx)  
 
        elif node_b.state == 'ALIVE':
            tB_1d = node_b.dist + node.distance_to_other_node(node_b)
            if tA_1d < dist:
                dist = tB_1d
                is_1d_fallback = True
                
                v1 = node.coords - node_b.coords
                v2 = (node.coords - node_b.virtual_source.coords)
                v2 = v2 - np.dot(v2, normal) * normal
                
                n1 = np.linalg.norm(v1)
                n2 = np.linalg.norm(v2)
                
                if n1 > 1e-14 and n2 > 1e-14:
                    isOnTheSameLine = np.linalg.norm(np.cross(v1/n1, v2/n2)) < samelinetol
                    
                else:
                    isOnTheSameLine = True
                    
                best_node_vs = node_virtual_source.get(node_b.idx) if isOnTheSameLine else node_b 
                # if node.virtual_source == node_virtual_source.get(node_b.idx):
                #     best_local_vs = node_virtual_source.get(node_b.idx)  

    # if node.coords[0] == -0.5 and node.coords[2] == 1: print(dist) 
    return dist, best_node_vs, is_1d_fallback, isOnTheSameLine, t_1d_list, node_list


# ========== Helper Functions ============
def compute_2d_eikonal(node_a, node_b, node_c, d_A_raw, d_B_raw, S_prime):
    """
    Computes the 2D Eikonal solution from a specific virtual source S_prime.
    Returns float('inf') if the upwind condition fails or geometry is degenerate.
    """
    # Extract the distances SS'
    if S_prime is None:
        S_prime_dist = 0.0
        print("None")
        
    else:
        S_prime_dist = S_prime.dist
        
    d_A = d_A_raw - S_prime_dist 
    d_B = d_B_raw - S_prime_dist

    # ====== STEP 1 ---> Obtain all necessary information about the element AB
    AB = node_b.coords - node_a.coords
    AC = node_c.coords - node_a.coords
    normal = np.cross(AB, AC)
    
    # Avoid all degenerate cases: three collinear points...
    # if np.linalg.norm(normal) / (np.linalg.norm(AB) * np.linalg.norm(AC)) < 1e-14:
    #     return float('inf') 

    normal = normal / np.linalg.norm(normal)
    L = np.linalg.norm(AB)
    # if L < 1e-16:
    #     return float('inf')   

    # Create a local frame centered in A with x // to AB and poiniting towards B
    x_hat = AB / L
    y_hat = np.cross(normal, x_hat)
    y_hat = y_hat / np.linalg.norm(y_hat)

    # ====== STEP 2 ---> C coordinates in the local frame
    x_C = np.dot(AC, x_hat)
    y_C = np.dot(AC, y_hat)

    # ====== STEP 3 ---> Virtual Source S coordinates in the local frame
    x_S = (d_A**2 - d_B**2 + L**2) / (2*L)
    sq_y_S = d_A**2 - x_S**2

    # Avoid Floating Point Errors
    if sq_y_S < 0.0:
        sq_y_S = 0.0

    if S_prime is not None:
        sign_y_S = -np.sign(y_C)
        if sign_y_S == 0:
            sign_y_S = 1.0
    else:
        #! If S_prime is None (original source), we assume the source is on the 
        # opposite side of the vertex C as an initial fallback
        sign_y_S = -np.sign(y_C)
        if sign_y_S == 0:
            sign_y_S = 1.0 # degenerate case: C exactly on AB
    
    # The position of the source S can be finally obtained
    y_S = sign_y_S * np.sqrt(sq_y_S)

    # ====== STEP 4 ---> UPWIND CONDITION (Shadow Zone Detection) ---
    # The line from S(x_S, y_S) to C(x_C, y_C) must pass through the edge AB.
    # Intersect the ray with the local x-axis (y=0).
    if abs(y_C - y_S) > 1e-12:
        x_int = x_S - y_S * (x_C - x_S) / (y_C - y_S)
    else:
        x_int = -1.0 # Force fail if line is parallel

    # If the ray falls outside [0, L], the wave is bending around a corner.
    if not (-L/100 <= x_int <= L + L/100):
        return float('inf')

    # ====== STEP 5 ---> Determine the distane of C from the source S
    t_local = np.sqrt( (x_C-x_S)**2 + (y_C-y_S)**2  )
    
    return t_local + S_prime_dist

"""
def edge_key(node_1, node_2):

    Function needed to confront the same edge in the different iterations of the algorithm.

    In fact the edge AB can be re-updated different times and can happen that dB becomes 
    smaller than dA, in that case the local frame of reference would flip. 

    In that case the curvature sign would not be consistent anymore with the reference
    return (min(node_1.idx, node_2.idx), max(node_1.idx, node_2.idx))


def store_virtual_source(node_1, node_2, S, edge_curvature):
    It saves the 3D position of the virtual source S for the edge (node_1, node_2).
    It does not overwrite it if already present

    #! Understand if when we will need to reinitialize the virtual source, we will need to change this code

    key = edge_key(node_1, node_2)

    if key in edge_curvature:
        return
    
    #Update the position of the virtual source S for the edge
    edge_curvature[key] = S.copy()

"""


def Save_Front_Frame(node_list, frame_id, folder="frames"):

    if not os.path.exists(folder):
        os.makedirs(folder)

    x_alive, y_alive = [], []
    x_trial, y_trial = [], []
    x_far, y_far = [], []

    for n in node_list:

        if n.state == 'ALIVE':
            x_alive.append(n.coords[0])
            y_alive.append(n.coords[1])

        elif n.state == 'TRIAL':
            x_trial.append(n.coords[0])
            y_trial.append(n.coords[1])

        else:
            x_far.append(n.coords[0])
            y_far.append(n.coords[1])

    plt.figure(figsize=(6,6))

    plt.scatter(x_far, y_far, c="lightgray", s=5)
    plt.scatter(x_trial, y_trial, c="orange", s=10)
    plt.scatter(x_alive, y_alive, c="blue", s=10)
    xmin = 0
    xmax = 1
    ymin = 0
    ymax = xmax
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)

    plt.gca().set_aspect("equal")

    filename = f"{folder}/frame_{frame_id:05d}.png"
    plt.savefig(filename, dpi=150)
    plt.close()

def Make_GIF(folder="frames", gif_name="front.gif"):

    files = sorted(glob.glob(folder + "/*.png"))

    images = []
    for f in files:
        images.append(imageio.imread(f))

    imageio.mimsave(gif_name, images, duration=1)

def Save_Debug_Frame(node_list, current_node, node_virtual_source, frame_id, folder="debug_frames"):

    if not os.path.exists(folder):
        os.makedirs(folder)

    # --- Tous les nodes en gris ---
    x_all = [n.coords[0] for n in node_list]
    y_all = [n.coords[1] for n in node_list]

    plt.figure(figsize=(6,6))
    plt.scatter(x_all, y_all, c="lightgray", s=5, label = "TRIAL")

    x_alive = [n.coords[0] for n in node_list if n.state == "ALIVE"]
    y_alive = [n.coords[1] for n in node_list if n.state == "ALIVE"]
    plt.scatter(x_alive, y_alive,
                c="blue", s=8, alpha=0.6, label="ALIVE")

    # --- Node courant (ALIVE) ---
    plt.scatter(current_node.coords[0], current_node.coords[1],
                c="gold", s=40, label="Current ALIVE")

    # --- Source associée ---
    source = node_virtual_source.get(current_node.idx)

    if source is not None:
        plt.scatter(source.coords[0], source.coords[1],
                    c="red", s=40, label="Virtual Source")

    # --- Mise en forme ---
    plt.gca().set_aspect("equal")

    # Fix les limites (important)
    xmin = 0
    xmax = 1
    ymin = 0
    ymax = xmax
    plt.xlim(xmin-0.1, xmax+0.1)
    plt.ylim(ymin-0.1, ymax+0.1)
    x_line = np.linspace(0.2, 0.8, 100)
    y_line = (5/3) * (x_line - 0.2)

    plt.plot(x_line, y_line, 'r--', linewidth=2)

    plt.legend()

    filename = f"{folder}/frame_{frame_id:05d}.png"
    plt.savefig(filename, dpi=150)
    plt.close()

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def Save_Debug_Frame_3D(node_list, current_node, node_virtual_source, frame_id, folder="debug_frames_3D"):
    
    if not os.path.exists(folder):
        os.makedirs(folder)

    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection='3d')

    # --- tous les nodes ---
    x_all = [n.coords[0] for n in node_list]
    y_all = [n.coords[1] for n in node_list]
    z_all = [n.coords[2] for n in node_list]
    ax.scatter(x_all, y_all, z_all, c="lightgray", s=5)

    # --- nodes ALIVE ---
    alive_nodes = [n for n in node_list if n.state == "ALIVE"]  # adapte si besoin
    if alive_nodes:
        x_alive = [n.coords[0] for n in alive_nodes]
        y_alive = [n.coords[1] for n in alive_nodes]
        z_alive = [n.coords[2] for n in alive_nodes]
        ax.scatter(x_alive, y_alive, z_alive, c="blue", s=10)

    # --- node courant ---
    ax.scatter(current_node.coords[0],
               current_node.coords[1],
               current_node.coords[2],
               c="gold", s=60, label="Current ALIVE")

    # --- source ---
    source = node_virtual_source.get(current_node.idx)
    if source is not None:
        ax.scatter(source.coords[0],
                   source.coords[1],
                   source.coords[2],
                   c="red", s=60, label="Virtual Source")

    # --- limites ---
    ax.set_xlim(min(x_all), max(x_all))
    ax.set_ylim(min(y_all), max(y_all))
    ax.set_zlim(min(z_all), max(z_all))

    ax.set_box_aspect([
        max(x_all)-min(x_all),
        max(y_all)-min(y_all),
        max(z_all)-min(z_all)
    ])

    ax.view_init(elev=110, azim=60)

    ax.legend()

    filename = f"{folder}/frame_{frame_id:05d}.png"
    plt.savefig(filename, dpi=150)
    plt.close()