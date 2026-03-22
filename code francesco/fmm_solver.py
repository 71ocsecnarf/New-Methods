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
    edge_curvature = {}  # Global dictionary containing all edges and the curvature of the front on it
                         # (indx_A indx_B)  ---> k (curvature)

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
                new_dist = eikonal_sol_circ(neighbor, edge_curvature)

                # Update the trial distance only if it is lower than the previously computed (always true if the node was FAR)
                # see pag 4
                if new_dist < old_dist:
                    neighbor.dist = new_dist
                    neighbor.state = 'TRIAL' # Set the node status as TRIAL, in case it was FAR
                    heap_push(neighbor, new_dist)




def eikonal_sol_circ(node, edge_curvature, F=1.0):
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

            # We need to have T(A) < T(B)
            if node_a.dist > node_b.dist:
                # If it not the case swap the nodes
                node_a, node_b = node_b, node_a

            d_A = node_a.dist
            d_B = node_b.dist

            # STEP 1 ---> obtain all the necessary info about the element AB
            AB = node_b.coords - node_a.coords
            AC = node.coords - node_a.coords
            normal = np.cross(AC, AB)
            normal = normal / np.linalg.norm(normal)#*unit normal vector

            L = np.linalg.norm(AB) # compute the distance AB
            # Local frame centered in A with x parallel to AB
            x_hat = AB / L
            y_hat = -np.cross(normal, x_hat)
            y_hat = y_hat / np.linalg.norm(y_hat)


            # STEP 3 ---> C Coordinates on the local frame
            
            x_C = np.dot(AC,x_hat)
            y_C = np.dot(AC,y_hat)


            # STEP 4 ---> Virtual Source S coordinates
            x_S = (d_A**2 - d_B**2 + L**2) / (2*L)
            sq_y_S = d_A**2 - x_S**2

            # Check - if the value under the square root is under lower than zero: fall back to the 1D case
            if sq_y_S < 0.0:
            # fallback 1D
                t = min(d_A + node.distance_to_other_node(node_a),
                        d_B + node.distance_to_other_node(node_b))
                dist = min(dist, t)
                continue

            key_AB = edge_key(node_a, node_b)
            # The sign is in the reference frame of the edge (min_idx → max_idx).
            # If node_a.idx > node_b.idx it means that the local frame 
            # is inverted wrt the local frame --> invrt the read sign of the curvature.
            flip = (node_a.idx > node_b.idx)

            if key_AB in edge_curvature:
                # It means that the sign of the curvature is already known
                # --> Read the dictionary
                sign_y_S_try = edge_curvature[key_AB]
                # Correct the sign if the local frame is inverted wrt ad the usual framework
                sign_y_S = -sign_y_S_try if flip else sign_y_S_try
            else:
                # We impose that the source need to be on the other side of the edge
                # wrt the third point C of the triangle
                sign_y_S = -np.sign(y_C)
                if sign_y_S == 0:
                    sign_y_S = 1.0   # degenerate case: C exactly on AB
            
            # We can finally determine the position of the source S
            y_S = sign_y_S * np.sqrt(sq_y_S)

            
            # STEP 5 --> Determine the distane of C from the source S
            t = np.sqrt( (x_C-x_S)**2 + (y_C-y_S)**2  )

            dist = min(dist, t)


            # STEP 6  --> update the curvature on the new edges AC and BC

            # Update only if the distance t is compted
            if t < float('inf'):
                
                # Reconstruct S position in the global 2D coordinates
                S = node_a.coords + x_S*x_hat + y_S*y_hat

                # update the edge AC
                store_sign(node_a, node, S, edge_curvature, y_hat)

                # Update the edge BC
                store_sign(node_b, node, S, edge_curvature, y_hat)


        # CASE 2 --> Degenerate cases - only one node alive
        # in these cases we need to compute the 1d distance
        elif node_a.state == 'ALIVE':
            dist_1d = node_a.dist + node.distance_to_other_node(node_a)   # T(A) + b*F
            dist = min(dist, dist_1d)

        elif node_b.state == 'ALIVE':
            dist_1d = node_b.dist + node.distance_to_other_node(node_b)  # T(B) + c*F
            dist = min(dist, dist_1d)
       
    return dist



def edge_key(node_1, node_2):
    """
    Function needed to confront the same edge in the different iterations of the algorithm.

    In fact the edge AB can be re-updated different times and can happen that dB becomes 
    smaller than dA, in that case the local frame of reference would flip. 

    In that case the curvature sign would not be consistent anymore with the reference
    """
    return (min(node_1.idx, node_2.idx), max(node_1.idx, node_2.idx))


def store_sign(node_1, node_2, S, edge_curvature, y_hat):
    key = edge_key(node_1, node_2)
    if key in edge_curvature:
        return

    if node_1.idx < node_2.idx:
        node_A, node_B = node_1, node_2
    else:
        node_A, node_B = node_2, node_1

    edge_vec = node_B.coords - node_A.coords
    L_edge = np.linalg.norm(edge_vec)
    if L_edge < 1e-14:
        return

    S_local = S - node_A.coords
    y_S_canonical = np.dot(S_local, y_hat)


    flip = (node_1.idx > node_2.idx)
    if flip:
        y_S_canonical = -y_S_canonical

    sign = float(np.sign(y_S_canonical))
    if sign == 0.0:
        sign = 1.0

    edge_curvature[key] = sign