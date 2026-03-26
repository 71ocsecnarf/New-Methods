import numpy as np
import heapq

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
            cos_theta = np.clip((b**2 - c**2 + a**2) / (2*b*c), -1.0, 1.0)
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
                t_sol = (-B + np.sqrt(Delta)) / (2 *A)

                #! Check that t_sol can be very small
                if t_sol > 1e-12 and u < t_sol:
                    cond = b * (t_sol - u) / t_sol
                    if 0 < cond < c:
                        t = t_sol + node_a.dist

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
    edge_curvature = {}      # Global dictionary: edge (idx_A, idx_B) --> virtual source position S (3D coords)
                             # Tracks the position of the virtual source on each edge of the mesh.

    node_virtual_source = {} # Global dictionary: node.idx --> corner NODE object
                             # When the wavefront reaches a node via a 1D fallback (i.e. the geodesic
                             # exits the triangle through a vertex), that vertex becomes a new virtual
                             # source. This dictionary stores, for each such node, the corner NODE
                             # from which it was reached, so that future propagations can use it
                             # as the new local origin instead of trying to back-trace to the
                             # original source S.

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


        # PROBLEM: In the last version, the update of the virtual source/edge_curvature
        #          was done in the eikonal solver, when the nodes were all in the state 
        #          TRIAL. However this would have caused a loop fallback since it will not
        #          be updated after it has been reached for the first time, propagating  
        #          an error, since the first reach is not usually the best one.
        # SOLUTION: Update the curvature or virtual source only when the node has become 
        #           ALIVE, i.e. when the minimum distance has been computed. 
        if hasattr(current_node, 'edge_updates') and current_node.edge_updates:
            for k, S in current_node.edge_updates:
                if k not in edge_curvature:
                    edge_curvature[k] = S.copy()

        if hasattr(current_node, 'virtual_source'):
            if current_node.virtual_source is not None:
                node_virtual_source[current_node.idx] = current_node.virtual_source
            else:
                node_virtual_source.pop(current_node.idx, None)



        # Point 3 -> Recompute all node distance (if not ALIVE) and put all of them as TRIAL
        for neighbor in current_node.neighbors:
            
            # Recompute the distance of the neighbour nodes only if not ALIVE
            if neighbor.state != 'ALIVE':
                old_dist = neighbor.dist

                # Now the function eikonal_sol_circ does not alter the global state, update
                # only when the node becomes ALIVE.
                new_dist, edge_upd, node_vs = eikonal_sol_circ(neighbor, edge_curvature, node_virtual_source)

                # Update the trial distance only if it is lower than the previously computed 
                # (always true if the node was FAR)
                if new_dist < old_dist:
                    neighbor.dist = new_dist
                    # Save the date of the edge update and the virtual source temporaneally 
                    # to not update them globally
                    neighbor.edge_updates = edge_upd
                    neighbor.virtual_source = node_vs
                    neighbor.state = 'TRIAL' # Set the node status as TRIAL, in case it was FAR
                    heap_push(neighbor, new_dist)




def eikonal_sol_circ(node, edge_curvature, node_virtual_source):
    
    dist = float('inf')  # Distance Initialization
    best_edge_updates = []
    best_node_vs = None

    for tri in node.adjacent_triangles: # Compute all the distance from all adjactens triangles
        others = [n for n in tri.nodes if n.node_tag != node.node_tag]
        node_a, node_b = others[0], others[1]

        # ------------------------------------------
        # CASE 1: Both nodes are ALIVE
        # ------------------------------------------
        if node_a.state == 'ALIVE' and node_b.state == 'ALIVE':
            if node_a.dist > node_b.dist: # d(A) < d(B) as a convention
                node_a, node_b = node_b, node_a

            d_A_raw = node_a.dist
            d_B_raw = node_b.dist

            # Get the registered virtual sources (if present) for the vertices A and B
            corner_a = node_virtual_source.get(node_a.idx)
            corner_b = node_virtual_source.get(node_b.idx)

            # --- CONFLICT CHECK: A and B have different virtual sources ---
            #  ---> They belong to different wavefronts, 2D circular wave is invalid.
            #! Problem: I do not know if forcing it to the 1D case is correct

            if corner_a is not corner_b:  # two different virtual sources --> force the 1D fallback 
                tA = d_A_raw + node.distance_to_other_node(node_a)
                tB = d_B_raw + node.distance_to_other_node(node_b)
                if tA <= tB:
                    t_1d = tA
                    # Update correctly the source
                    new_source = corner_a if corner_a is not None else node_a
                else:
                    t_1d = tB
                    new_source = corner_b if corner_b is not None else node_b
                # Update the distance if it is the lowest
                if t_1d < dist:
                    dist = t_1d
                    #! I do not know if this method for the assignement of the 
                    #! source is too slow 
                    best_edge_updates = []
                    best_node_vs = new_source
                continue

            # --- SAME SOURCE ---
            if corner_a is not None:
                S_prime = corner_a
                d_A = d_A_raw - S_prime.dist 
                d_B = d_B_raw - S_prime.dist
            else:
                S_prime = None
                d_A = d_A_raw
                d_B = d_B_raw

            # ====== STEP 1 ---> Obtain all necessary information about the element AB
            AB = node_b.coords - node_a.coords
            AC = node.coords - node_a.coords
            normal = np.cross(AB, AC)
            
            # Avoid all degenerate cases: trhee collinear points...
            if np.linalg.norm(normal) < 1e-14:
                continue 

            normal = normal / np.linalg.norm(normal)
            L = np.linalg.norm(AB)
            if L < 1e-14:
                continue   
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
                # In fact, per construction, sq_y_S >= 0 always

            key_AB = edge_key(node_a, node_b)
            # Store in each edge of triangle the position of the virtual source S

            if key_AB in edge_curvature:
                # It means that the virtual source position is already known
                # --> Read the dictionary

                # Re-project the position of the source in the local frame
                S_dict = edge_curvature[key_AB]
                S_local = S_dict - node_a.coords
                y_S_test = np.dot(S_local, y_hat)
                sign_y_S = np.sign(y_S_test) 

                if sign_y_S == 0:
                    sign_y_S = 1.0  # degenerate case: C exactly on AB 
            else:
                # If it is the first time we cross this edge, we impose that the position 
                # of the source S is on the opposite side of the vertex C
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
            if not (-1e-10 <= x_int <= L + 1e-10):
                tA = d_A_raw + node.distance_to_other_node(node_a)
                tB = d_B_raw + node.distance_to_other_node(node_b)

                if tA <= tB:
                    t_1d = tA
                    new_source = corner_a if corner_a is not None else node_a
                else:
                    t_1d = tB
                    new_source = corner_b if corner_b is not None else node_b

                if t_1d < dist:
                    dist = t_1d
                    best_edge_updates = []
                    best_node_vs = new_source
                continue


            # ====== STEP 5 ---> Determine the distane of C from the source S
            t_local = np.sqrt( (x_C-x_S)**2 + (y_C-y_S)**2  )
            # If a virtual source has been used, take that into consideration
            if S_prime is not None: 
                t_total = t_local + S_prime.dist
            else:
                t_total = t_local

            if t_total < dist:
                dist = t_total

                # Prepare the virtual source to return
                best_node_vs = S_prime
                
                # Compute the global position of the virtual source S
                S_global = node_a.coords + x_S * x_hat + y_S * y_hat

                    # ====== STEP 6 ---> Update only locally and leave the update to the FMM algorithm
                best_edge_updates = [
                    (edge_key(node_a, node), S_global),
                    (edge_key(node_b, node), S_global)
                    ]


        # ------------------------------------------
        # CASE 2: Degenerate (Only one node ALIVE)
        # ------------------------------------------
        # 1D fallback
        elif node_a.state == 'ALIVE':
            t_1d = node_a.dist + node.distance_to_other_node(node_a)
            if t_1d < dist:
                dist = t_1d
                best_edge_updates = []
                best_node_vs = node_virtual_source.get(node_a.idx)
 
        elif node_b.state == 'ALIVE':
            t_1d = node_b.dist + node.distance_to_other_node(node_b)
            if t_1d < dist:
                dist = t_1d
                best_edge_updates = []
                best_node_vs = node_virtual_source.get(node_b.idx)
    
    return dist, best_edge_updates, best_node_vs



def edge_key(node_1, node_2):
    """
    Function needed to confront the same edge in the different iterations of the algorithm.

    In fact the edge AB can be re-updated different times and can happen that dB becomes 
    smaller than dA, in that case the local frame of reference would flip. 

    In that case the curvature sign would not be consistent anymore with the reference
    """
    return (min(node_1.idx, node_2.idx), max(node_1.idx, node_2.idx))

"""
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

