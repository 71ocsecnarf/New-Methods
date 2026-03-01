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
                heapq.heappush(trial_nodes, (neighbor.dis, neighbor.idx, neighbor))
    
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
                    neighbor.dist = node.dist
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
    new_dist = float('inf')

    # For compute the distance D from all adjacents triangles in 
    for tri in node.adjacent_triangles:
        
        # Identify the other two nodes
        nodes = [tri.node_A, tri.node_B, tri.node_C]
        others = [n for n in nodes if n.node_tag != node.node_tag]
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
            









    #! TO BE DONE --> FMM: pag 4 of the document "Computing Geodesic Paths on Manifolds" - Author(s): R. Kimmel and J. A. Sethian
    #! I have some ideas, ill work on it sunday or monday
    return new_dist