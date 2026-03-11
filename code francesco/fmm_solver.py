import numpy as np
import heapq

def fmm_algorithm(node_list, source_nodes):
    """
        From the source nodes, fmm_algoritm runs the FMM
    """

    print('FMM running ...')
    print()

    # ---- Step 1 --> Initialization ----
    trial_nodes = []
    best_dist_in_heap = {}

    # Mark source nodes as ALIVE before anything else
    for node in source_nodes:
        assert node.dist < float('inf'), f"Source node {node.node_tag} has dist=inf!"
        node.state = 'ALIVE'

    # Internal function to avoid multiple push and get a very long list of values
    def heap_push(node, dist):
        if dist < best_dist_in_heap.get(node.idx, float('inf')):
            best_dist_in_heap[node.idx] = dist
            heapq.heappush(trial_nodes, (dist, node.idx, node))

    # Set the sources neighbor nodes as trial - Creation of the narrow band
    for source in source_nodes:
        for neighbor in source.neighbors:
            # Computation of preliminary distance 
            #! is it ok if we compute it as the euclidean one for starting?
            #! To be asked to the professor
            if neighbor.state != 'ALIVE':
                new_dist = eikonal_sol(neighbor)
                #! is it used correctly?
                if new_dist == float('inf'):
                 #! is this if necessary? to be asked the professor
                    new_dist = neighbor.distance_to_other_node(source) + source.dist

                if new_dist < neighbor.dist:
                    neighbor.dist = new_dist
                    neighbor.state = 'TRIAL'
                    heap_push(neighbor, new_dist)
    
    # ---- Step 2 --> Principal Loop - FMM ---- 
    while trial_nodes:
        # Point 1 -> Extract the lower distance node
        #! is it used correctly? Using this command the node should be removed from the trial list, to be checked
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

    print('FMM running ...')
    print()

    # ---- Step 1 --> Initialization ----
    trial_nodes = []
    best_dist_in_heap = {}

    # Mark source nodes as ALIVE before anything else
    for node in source_nodes:
        assert node.dist < float('inf'), f"Source node {node.node_tag} has dist=inf!"
        node.state = 'ALIVE'

    # Internal function to avoid multiple push and get a very long list of values
    def heap_push(node, dist):
        if dist < best_dist_in_heap.get(node.idx, float('inf')):
            best_dist_in_heap[node.idx] = dist
            heapq.heappush(trial_nodes, (dist, node.idx, node))

    # Set the sources neighbor nodes as trial - Creation of the narrow band
    for source in source_nodes:
        for neighbor in source.neighbors:
            # Computation of preliminary distance 
            #! is it ok if we compute it as the euclidean one for starting?
            #! To be asked to the professor
            if neighbor.state != 'ALIVE':
                new_dist = eikonal_sol_circ(neighbor)
                #! is it used correctly?
                if new_dist == float('inf'):
                 #! is this if necessary? to be asked the professor
                    new_dist = neighbor.distance_to_other_node(source) + source.dist

                if new_dist < neighbor.dist:
                    neighbor.dist = new_dist
                    neighbor.state = 'TRIAL'
                    heap_push(neighbor, new_dist)
    
    # ---- Step 2 --> Principal Loop - FMM ---- 
    while trial_nodes:
        # Point 1 -> Extract the lower distance node
        #! is it used correctly? Using this command the node should be removed from the trial list, to be checked
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
                new_dist = eikonal_sol_circ(neighbor)

                # Update the trial distance only if it is lower than the previously computed (always true if the node was FAR)
                # see pag 4
                if new_dist < old_dist:
                    neighbor.dist = new_dist
                    neighbor.state = 'TRIAL' # Set the node status as TRIAL, in case it was FAR
                    heap_push(neighbor, new_dist)




def eikonal_sol_circ(node):
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

        #! I think it is from here the part we need to change it
        # CASE 1 - Both other nodes are ALIVE
        if node_a.state == 'ALIVE' and node_b.state == 'ALIVE':

            # We need to have T(A) < T(B)
            if node_a.dist > node_b.dist:
                # If it not the case swap the nodes
                node_a, node_b = node_b, node_a

            d_A = node_a.dist
            d_B = node_b.dist

            # STEP 1 ---> obtain all the necessary info about the element AB

            AB = node_b.coords[:2] - node_a.coords[:2]
            L = np.linalg.norm(AB) # compute the distance AB
            # Local frame centered in A with x parallel to AB
            x_hat = AB / L
            y_hat = np.array([-x_hat[1], x_hat[0]])


            # STEP 3 ---> C Coordinates on the local frame

            AC = node.coords[:2] - node_a.coords[:2]
            x_C = np.dot(AC,x_hat)
            y_C = np.dot(AC,y_hat)


            # STEP 4 ---> Virtual Source S coordinates
            x_S = (d_A^2 - d_B^2 + L) / (2*L)
            sq_y_s = d_A**2 - x_S**2

            # Check - if the value under the square root is under lower than zero: fall back to the 1D case
            if sq_y_s < 0:
            # fallback 1D
                t = min(d_A + node.distance_to_other_node(node_a),
                        d_B + node.distance_to_other_node(node_b))
                dist = min(dist, t)
                continue

            # To decide the sign of ys, so which of the two virtual source is correct, use the sign of the curvature memorised in the 
            if tri.curvature_is_set:
                sign_ys = np.sign(tri.curvature)
            else:
                sign_ys = -np.sign(y_C)  #! I am not sure, this part is not mine

            y_s = sign_ys * np.sqrt(val)



























          


            dist = min(dist, t)

        # CASE 2 --> Degenerate cases
        # in these cases we need to compute the 1d distance
        elif node_a.state == 'ALIVE':
            dist_1d = node_a.dist + node.distance_to_other_node(node_a)   # T(A) + b*F
            dist = min(dist, dist_1d)

        elif node_b.state == 'ALIVE':
            dist_1d = node_b.dist + node.distance_to_other_node(node_b)  # T(B) + c*F
            dist = min(dist, dist_1d)
       
       #! Can we delete these two elif (and consequently the if at the start) and put them alltogether without checking if the two nodes are alive?
    return dist












