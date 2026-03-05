import numpy as np
import heapq

def fmm_algorithm(node_list, source_nodes):
    """
        From the source nodes, fmm_algoritm runs the FMM
    """

    # ---- Step 1 --> Initialization ----
    
    trial_nodes = []

    # Source nodes
    for node in source_nodes:
        node.dist = 0.0
        node.state = 'ALIVE'

    # Set the sources neighbor nodes as trial - Creation of the narrow band
    for node in source_nodes:
        for neighbor in node.neighbors:
            # Computation of preliminary distance 
            #! is it ok if we compute it as the euclidean one for starting?
            #! To be asked to the professor
            dist_initial = node.distance_to_other_node(neighbor)
            #! is it used correctly?

            #! is this if necessary? to be asked the professor
            if dist_initial < neighbor.dist:
                neighbor.dist = dist_initial
                neighbor.state = 'TRIAL'
                heapq.heappush(trial_nodes, (neighbor.dist, neighbor.idx, neighbor))
    
    # ---- Step 2 --> Principal Loop - FMM ---- 
    while trial_nodes:
        # Point 1 -> Extract the lower distance node
        #! is it used correctly? Using this command the node should be removed from the trial list, to be checked
        d_min, _, current_node = heapq.heappop(trial_nodes)

        # Security check: a node could be inserted many times in the list with different distances
        if current_node.state == 'ALIVE':
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
                    heapq.heappush(trial_nodes, (neighbor.dist, neighbor.idx, neighbor))


# It should work, but I am not sure

# I am following the document, but I think the code can be upgraded
def eikonal_sol(node, F=1.0):
    """"
    Compute Locally the approximate solution of the Eikonal Equation
    For acute triangles only, for the moment
    """

    # First - Distance initialization
    dist_c = float('inf')

    # For compute the distance D from all adjacents triangles in 
    for tri in node.adjacent_triangles:
        
        nodes_in_tri = tri.nodes 
        
        others = [n for n in nodes_in_tri if n.node_tag != node.node_tag]
        
        # Ora puoi assegnare node_a e node_b in sicurezza
        node_a, node_b = others[0], others[1]
        node_a, node_b = others[0], others[1]

        #! I think it is from here the part we need to change it
        # Case 1 - Both other nodes are ALIVE
        if node_a.state == 'ALIVE' and node_b.state == 'ALIVE':

            # I think the document assumes T(A) < T(B), I am following it, but I think we can update it to work faster without the if
            if node_a.dist > node_b.dist:
                # If it not the case swap the nodes to be in the condition of the document
                node_a, node_b = node_b, node_a

            # From definitions
            u = node_b.dist - node_a.dist
            # Computations of the lengths of the sides of the triangles
            #! Is it more efficient to compute the distance each time or compute them once and store them?
            #! To be asked to the professor, I do not know, maybe if we store them it is more efficient, but it can take a lot of memory, especially with a very big grid
            # For now I'll just compute them each time
            a = node.distance_to_other_node(node_b)    # BC
            b = node.distance_to_other_node(node_a)    # AC
            c = node_a.distance_to_other_node(node_b)  # AB

            # Computation of the angle theta using the cosine theorem
            #          a^2 = b^2 + c^2 - 2bc cos_theta 
            cos_theta = (b**2 + c**2 - a**2) / (2*b*c) 
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

                # Validity Conditions
                c1 = u < t_sol
                c2 = a * cos_theta  <  b * (t_sol - u) / t_sol < a / cos_theta

                if c1 and c2:
                    # use the solution only if it is acceptable
                    t = t_sol
                    dist_c = t + node_a.dist
                else:
                    # If the computation fails, it means that the front runs along a border, so it is easy to update the distance
                    dist_c = min(b * F + node_a.dist, c * F + node_a.dist)

        # Degenerate cases - useful to implement
        # in these cases we need to compute the 1d distance
        elif node_a.state == 'ALIVE':
            dist_1d = node_a.dist + node.distance_to_other_node(node_a) * F  # T(A) + b*F
            dist_c = min(dist_c, dist_1d)

        elif node_b.state == 'ALIVE':
            dist_1d = node_b.dist + node.distance_to_other_node(node_b) * F # T(B) + c*F
            dist_c = min(dist_c, dist_1d)
       
       #! Can we delete these two elif (and consequently the if at the start) and put them alltogether without checking if the two nodes are alive?
    return dist_c

# To run the code faster it would be ideal, as said before, to compute the distances 
# of the elements of the triangles diretly in element and here only access them. 
# I need to search how to do it
