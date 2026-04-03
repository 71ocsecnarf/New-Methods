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
            cos_theta = np.clip((b**2 - c**2 + a**2) / (2*b*c), -1.0, 1.0)
            #cos_theta = np.clip((b**2 + c**2 - a**2) / (2*b*c), -1.0, 1.0)
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
                if t_sol > 1e-12 * (node_a.dist + 1.0) and u <= t_sol + 1e-12*(abs(u)+1.0):
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

        """
        elif node_b.state == 'ALIVE':
            dist_1d = node_b.dist + node.distance_to_other_node(node_b) * F # T(B) + c*F
            dist = min(dist, dist_1d)
        """
        
       
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
    edge_curvature = {}


    # Mark source nodes as ALIVE before anything else
    for node in source_nodes:
        assert node.dist < float('inf'), f"Source node {node.node_tag} has dist=inf!"
        node.state = 'ALIVE'
        # In case there is no virtual source saved in the dictionary, it means that 
        # the virtual source is the original one --> save it to simplify the code
        node.virtual_source = node

    for node in node_list:
        if node.state == 'TRIAL':
            heap_push(node, node.dist)
            node.virtual_source = source_nodes[0]

    
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

        #! DA fare prima
        # Update the edge_curvature dictionary now that the node is ALIVE
        if hasattr(current_node, 'opt_S') and current_node.opt_S is not None:
            store_virtual_source(current_node.opt_A, current_node, current_node.opt_S, edge_curvature)
            store_virtual_source(current_node.opt_B, current_node, current_node.opt_S, edge_curvature)


        # Point 3 -> Recompute all node distance (if not ALIVE) and put all of them as TRIAL
        for neighbor in current_node.neighbors:
            
            # Recompute the distance of the neighbour nodes only if not ALIVE
            if neighbor.state != 'ALIVE':
                old_dist = neighbor.dist

                new_dist, node_vs, opt_A, opt_B, opt_S = eikonal_sol_circ(neighbor, edge_curvature)

                # Update the trial distance only if it is lower than the previously computed 
                # (always true 
                #  the node was FAR)
                if new_dist < old_dist:
                    neighbor.dist = new_dist
                    # Save the date of the edge update and the virtual source temporaneally 
                    # to not update them globally
                    neighbor.virtual_source = node_vs

                    # Save the optimal configuration for edge_curvature update when ALIVE
                    neighbor.opt_A = opt_A
                    neighbor.opt_B = opt_B
                    neighbor.opt_S = opt_S

                    neighbor.state = 'TRIAL' # Set the node status as TRIAL, in case it was FAR
                    heap_push(neighbor, new_dist)
            
        #Save_Debug_Frame(node_list, current_node, current_node.virtual_source, frame_id)
        #frame_id += 1
    
    #Make_GIF(folder="debug_frames", gif_name="front_source.gif")
                




def eikonal_sol_circ(node, edge_curvature):

    # Initialization
    dist = float('inf')  
    best_node_vs = None
    best_A = None
    best_B = None
    best_S_coords = None

    # Inspect all adjacent triangles
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
            corner_a = node_a.virtual_source
            corner_b = node_b.virtual_source

            # Compute the 1D update to confront them with the 2D updated distances
            tA_1d = d_A_raw + node.distance_to_other_node(node_a)
            tB_1d = d_B_raw + node.distance_to_other_node(node_b)


            #!!!!!!!!! ARE WE SURE
            # Evaluate which is the minimum of the two
            if tA_1d <= tB_1d:
                best_local_dist = tA_1d
                # Update correctly the source

                #best_local_vs = corner_a if corner_a is not None else node_a
                best_local_vs = node_a
                best_local_S = node_a.coords

                #! I am not sure
                #best_local_vs   = node_a.virtual_source  # inherit, not reset
                #best_local_S    = node_a.virtual_source.coords  # keep propagating it
                
            else:
                best_local_dist = tB_1d
                #best_local_vs = corner_b if corner_b is not None else node_b
                best_local_vs = node_b
                best_local_S = node_b.coords
                #best_local_vs   = node_b.virtual_source  # inherit, not reset
                #best_local_S    = node_b.virtual_source.coords


            # 2D circular update ---> try with corner_a as a source
            t_2d_a, S_coords_a = compute_2d_eikonal(node_a, node_b, node, d_A_raw, d_B_raw, corner_a, edge_curvature)
            #if t_2d_a <= best_local_dist + 1e-10:
            if t_2d_a <= best_local_dist:
                best_local_dist = t_2d_a
                best_local_vs = corner_a
                best_local_S = S_coords_a

            # Attempt 2D Eikonal from corner_b (only if different from corner_a to save computations)
            if corner_a is not corner_b:
                t_2d_b, S_coords_b = compute_2d_eikonal(node_a, node_b, node, d_A_raw, d_B_raw, corner_b, edge_curvature)
                #if t_2d_b <= best_local_dist + 1e-10:
                if t_2d_b <= best_local_dist:
                    best_local_dist = t_2d_b
                    best_local_vs = corner_b
                    best_local_S = S_coords_b
            
            # Update of the distance only if it is lower than the previously computed
            if best_local_dist < dist:
                
        
                ##IMPORTANT CHECK: THE FRONT MUST ALWAYS MOVE FORWARD, THE COMPUTED DISTANCE MUST ALWAYS 
                # HIGHER THAN THE DISTANCES FROM THE OTHER TWO VERTICES A AND B
                #    ---> SOMETHING LIKE AN UPWIND CONDITION
                

                
                if best_local_dist < node_a.dist-1e-10 or best_local_dist < node_b.dist-1e-10:
                    # in this case force a 1D fallback and the node a becomes the new virtual source for C

                    if tA_1d <= tB_1d:
                        dist = tA_1d
                        #best_node_vs = node_a.virtual_source
                        best_node_vs = node_a
                        best_A = node_a
                        best_B = node_b
                        best_S_coords = node_a.coords
                    else:
                        dist = tB_1d
                        #best_node_vs = node_b.virtual_source
                        best_node_vs = node_b
                        best_A = node_a
                        best_B = node_b
                        best_S_coords = node_a.coords
                else:
                    # Accept the 2D circular result
                    dist = best_local_dist
                    best_node_vs = best_local_vs
                    best_A, best_B, best_S_coords = node_a, node_b, best_local_S

                """
                # IMPROVED VERSION: remove the over-aggressive upwind check
                # Trust the intersection check already done inside compute_2d_eikonal
                if best_local_dist < dist:
                    # Only reject if clearly non-causal (use a relative tolerance)
                    tol = 1e-8 * (node_a.dist + node_b.dist + 1.0)
                    if best_local_dist < min(node_a.dist, node_b.dist) - tol:
                         # Genuine non-causal: 1D fallback
                        dist_via_a = node_a.dist + node.distance_to_other_node(node_a)
                        dist_via_b = node_b.dist + node.distance_to_other_node(node_b)
                        if dist_via_a <= dist_via_b:
                            dist = dist_via_a
                            #best_node_vs = node_a.virtual_source
                            best_node_vs = node_a
                            best_A, best_B, best_S_coords = None, None, None
                        else:
                            dist = dist_via_b
                            #best_node_vs = node_b.virtual_source
                            best_node_vs = node_b
                            best_A, best_B, best_S_coords = None, None, None
                else:
                    # Accept the 2D circular result
                    dist = best_local_dist
                    best_node_vs = best_local_vs
                    best_A, best_B, best_S_coords = node_a, node_b, best_local_S
                """
                


        # ------------------------------------------
        # CASE 2: Degenerate (Only one node ALIVE)
        # ------------------------------------------
        # 1D fallback
        elif node_a.state == 'ALIVE':
            t_1d = node_a.dist + node.distance_to_other_node(node_a)
            if t_1d < dist:
                dist = t_1d
                best_node_vs = node_a.virtual_source
                #best_node_vs = node_a
                best_S_coords = node_a.virtual_source.coords
                best_A = node_a
                best_B = node_b

        #else: 
        #    print('Eikonal_solver_failed')

        """
        elif node_b.state == 'ALIVE':
            t_1d = node_b.dist + node.distance_to_other_node(node_b)
            if t_1d < dist:
                dist = t_1d
                best_node_vs = node_b.virtual_source
                #best_node_vs = node_b
                best_S_coords = node_b.virtual_source.coords
                best_A = node_a
                best_B = node_b
        
        """
        
    
    return dist, best_node_vs, best_A, best_B, best_S_coords


# ========== Helper Functions ============
def compute_2d_eikonal(node_a, node_b, node_c, d_A_raw, d_B_raw, S_prime, edge_curvature):
    """
    Computes the 2D Eikonal solution from a specific virtual source S_prime.
    Returns float('inf') if the upwind condition fails or geometry is degenerate.
    """

    # Extract the distances SS'
    if S_prime is None:
        S_prime_dist = 0.0
    else:
        S_prime_dist = S_prime.dist
        
    d_A = d_A_raw - S_prime_dist 
    d_B = d_B_raw - S_prime_dist

    # ====== STEP 1 ---> Obtain all necessary information about the element AB
    AB = node_b.coords - node_a.coords
    AC = node_c.coords - node_a.coords
    normal = np.cross(AB, AC)
    
    # Avoid all degenerate cases: three collinear points...
    if np.linalg.norm(normal) < 1e-14:
        return float('inf'), None

    normal = normal / np.linalg.norm(normal)
    L = np.linalg.norm(AB)
    if L < 1e-14:
        return float('inf'), None   

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

    #! Change
    if sq_y_S < -1e-12:
        return float('inf'), None # Reject Unfeasible Solutions
    elif sq_y_S < 0.0:
        sq_y_S = 0.0 # Avoid floating point errorsù

    # To retrieve the sign of the curvature we need to know if the segment AB in local frame 
    # as the same convention used globally: (idx_min --> idx_max)
    key_AB = edge_key(node_a, node_b)

    if key_AB in edge_curvature:
       # It means that the virtual source position is already known
        # --> Read the dictionary
        
        # Re-project the position of the source in the local frame
        S_dict = edge_curvature[key_AB]
        S_local = S_dict - node_a.coords
        y_S_test = np.dot(S_local, y_hat)
        sign_y_S = np.sign(y_S_test)

        """
        if sign_y_S == 0:
            sign_y_S = 1.0   # degenerate case: C exactly on AB
        """


    else:
        # If it is the first time we cross this edge, we impose that the position 
        # of the source S is on the opposite side of the vertex C
        
        sign_y_S = -np.sign(y_C)
        if sign_y_S == 0:
            sign_y_S = 1.0   # degenerate case: C exactly on AB

    
    # The position of the source S can be finally obtained
    y_S = sign_y_S * np.sqrt(sq_y_S)

    # Reconstruct S position in the global 2D coordinates for future edges
    S_global = node_a.coords + x_S*x_hat + y_S*y_hat

        
    
    # ====== STEP 4 ---> UPWIND CONDITION (Shadow Zone Detection) ---
    # The line from S(x_S, y_S) to C(x_C, y_C) must pass through the edge AB.
    # Intersect the ray with the local x-axis (y=0).
    if abs(y_C - y_S) > 1e-10:
        x_int = x_S - y_S * (x_C - x_S) / (y_C - y_S)
    else:
        x_int = -1.0 # Force fail if line is parallel

    # If the ray falls outside [0, L], the wave is bending around a corner.
    if not (-1e-12 <= x_int <= L + 1e-12):
        return float('inf'), None
    
       
    # ====== STEP 5 ---> Determine the distane of C from the source S
    t_local = np.sqrt( (x_C-x_S)**2 + (y_C-y_S)**2  )
    
    return t_local + S_prime_dist, S_global


def edge_key(node_1, node_2):
    """
     Function needed to confront the same edge in the different iterations of the algorithm.

    In fact the edge AB can be re-updated different times and can happen that dB becomes 
    smaller than dA, in that case the local frame of reference would flip. 

    In that case the curvature sign would not be consistent anymore with the reference
    """
    return (min(node_1.idx, node_2.idx), max(node_1.idx, node_2.idx))


def store_virtual_source(node_1, node_2, S, edge_curvature):
    """
    It saves the 3D position of the virtual source S for the edge (node_1, node_2).
    It does not overwrite it if already present
    """
    
    key = edge_key(node_1, node_2)

    """
    if key in edge_curvature:
        return

    """
    
    #Update the position of the virtual source S for the edge
    edge_curvature[key] = S.copy()



def update_dir(node_a, node_c):
    """
    Check if the update of the distance comes from inside of the triangle or outside
     i.e. re-initialization of the curvature 
    """

    update_consistent = 1

    return update_consistent








# ===================================================
# =============== Debugging functions ===============
# ===================================================

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
    plt.scatter(x_all, y_all, c="lightgray", s=5)

    # --- Node courant (ALIVE) ---
    plt.scatter(current_node.coords[0], current_node.coords[1],
                c="blue", s=40, label="Current ALIVE")

    # --- Source associée ---
    source = node_virtual_source

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