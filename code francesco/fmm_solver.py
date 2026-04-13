from platform import node

from matplotlib.pylab import normal
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
            sin_theta = np.sqrt(max(0, 1 - cos_theta**2))   # Avoid numerical errors
            #sin_theta = np.sqrt( 1 - cos_theta**2)

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
                    
                    #!!! Here I put the fix
                    # Upper bound protected against division by zero
                    upper_bound = (a / cos_theta) if cos_theta > 1e-12 else float('inf')
                    
                    # Removed premature return, using non-strict inequalities (<=)
                    if a * cos_theta <= cond <= upper_bound:
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
    trial_nodes       = []
    best_dist_in_heap = {}
    node_virtual_source = {}

    # Mark source nodes as ALIVE and register them as their own virtual source
    for node in source_nodes:
        assert node.dist < float('inf'), f"Source node {node.node_tag} has dist=inf!"
        node.state                    = 'ALIVE'
        node.virtual_source           = node
        node_virtual_source[node.idx] = node

    # Push initial TRIAL neighbours into the heap
    for node in node_list:
        if node.state == 'TRIAL':
            node.virtual_source = source_nodes[0]
            node_virtual_source[node.idx] = node.virtual_source
            heap_push(node, node.dist)

    
    # ---- Step 2 --> Principal Loop - FMM ---- 
    while trial_nodes:
        # Point 1 -> Extract the lowest-distance node
        d_min, _, current_node = heapq.heappop(trial_nodes)

        # A node can appear multiple times in the heap with different distances.
        # Skip it if it is already frozen (ALIVE).
        if current_node.state == 'ALIVE':
            continue

        # Point 2 -> Freeze this node: mark it ALIVE and commit its virtual source
        current_node.state = 'ALIVE'
        # Commit the virtual source to the global dictionary now that the
        # distance is final (a TRIAL node may have been updated several times
        # before becoming ALIVE, so we only register here).
        node_virtual_source[current_node.idx] = current_node.virtual_source

        # Point 3 -> Update all non-ALIVE neighbours
        for neighbor in current_node.neighbors:
            if neighbor.state != 'ALIVE':
                old_dist = neighbor.dist

                new_dist, node_vs = eikonal_sol_circ(neighbor, node_virtual_source)

                if new_dist < old_dist:
                    neighbor.dist           = new_dist
                    neighbor.virtual_source = node_vs
                    neighbor.state          = 'TRIAL'
                    heap_push(neighbor, new_dist)

        #Save_Debug_Frame(node_list, current_node, node_virtual_source, frame_id)
        #frame_id += 1

    #Make_GIF(folder="debug_frames", gif_name="front_source.gif")



def eikonal_sol_circ(node, node_virtual_source):
    """
    Compute the updated distance for 'node' using the circular-wavefront FMM.

    For each adjacent triangle where at least one other vertex is ALIVE, we
    attempt both a 1D update (distance along an edge) and a 2D circular update
    (exact distance from a virtual point source located behind the shared edge).

    When the two ALIVE vertices of a triangle carry DIFFERENT virtual sources,
    the wavefront has wrapped around a concave corner.  In that case we use the
    isCplus classification to select only the geometrically consistent source,
    avoiding spurious backward updates.

    Parameters
    ----------
    node               : NODE   -- the node whose distance we want to update
    node_virtual_source: dict   -- maps node.idx --> NODE (virtual source object)
                                   Guaranteed to contain an entry for every node
                                   (pre-populated with the original source at init).

    Returns
    -------
    dist        : float -- best distance found
    best_node_vs: NODE  -- virtual source associated with that distance
    """

    SAME_LINE_TOL = 1e-1   # Cross-product threshold for collinearity check

    dist         = float('inf')
    best_node_vs = node_virtual_source.get(node.idx)  # fallback: keep current VS
    
    for tri in node.adjacent_triangles:

        others = [n for n in tri.nodes if n.node_tag != node.node_tag]
        node_a, node_b = others[0], others[1]

        normal = tri.compute_outward_normal()

        # ------------------------------------------------------------------
        # CASE 1: Both neighbours are ALIVE --> full 2D update possible
        # ------------------------------------------------------------------
        if node_a.state == 'ALIVE' and node_b.state == 'ALIVE':

            # Convention: d(A) <= d(B)
            if node_a.dist > node_b.dist:
                node_a, node_b = node_b, node_a

            d_A = node_a.dist
            d_B = node_b.dist

            # Retrieve virtual sources -- always valid because the dictionary
            # was pre-populated with the original source for every node.
            corner_a = node_virtual_source[node_a.idx]
            corner_b = node_virtual_source[node_b.idx]

            # best_local_dist starts from the current global best so that local
            # candidates from this triangle are only accepted when strictly better.
            best_local_dist = dist
            best_local_vs   = best_node_vs

            # ---- 1D fallback distances ----
            tA_1d = d_A + node.distance_to_other_node(node_a)
            tB_1d = d_B + node.distance_to_other_node(node_b)

            # Choose the smaller 1D candidate and decide whether the wavefront
            # is propagating straight (collinear with the virtual source) or
            # bending at the ALIVE vertex (which then becomes the new VS).
            if tA_1d <= tB_1d and tA_1d < best_local_dist:
                best_local_dist = tA_1d
                collinear = consistent_update(node_a, node, normal, corner_a, SAME_LINE_TOL)
                best_local_vs = corner_a if collinear else node_a

            elif tB_1d < tA_1d and tB_1d < best_local_dist:
                best_local_dist = tB_1d
                collinear = consistent_update(node_b, node, normal, corner_b, SAME_LINE_TOL)
                best_local_vs = corner_b if collinear else node_b

            # ---- 2D circular update ----
            if corner_a is corner_b:
                # Same virtual source for both A and B: standard circular update.
                # The wavefront is a perfect circle centred on this shared source.
                t_2d = compute_2d_eikonal(node_a, node_b, node, d_A, d_B, corner_a)
                if t_2d < best_local_dist:
                    best_local_dist = t_2d
                    best_local_vs   = corner_a

            else:
                # Different virtual sources: the wavefront has wrapped around a
                # concave corner.  Use isCplus to select the consistent source.
                #
                # Semantics (matching the colleague's implementation):
                #   other_corner = the concave corner node (new virtual source),
                #                  identified by having the SMALLER dist value
                #                  (it is between the original source and C).
                #   close_corner = the original / previous virtual source,
                #                  which has the LARGER dist value.
                #   mynode       = the ALIVE mesh vertex associated with
                #                  other_corner (used as one end of the edge
                #                  passed to compute_2d_eikonal).
                if corner_b.dist < corner_a.dist:
                    # corner_b is closer to the original source --> it is the
                    # concave corner that re-emits the wave.
                    other_corner = corner_b   # concave corner (new VS), smaller dist
                    close_corner = corner_a   # previous VS, larger dist
                    mynode       = node_b     # mesh vertex whose VS is other_corner
                    mynode2      = node_a
                else:
                    other_corner = corner_a
                    close_corner = corner_b
                    mynode       = node_a
                    mynode2      = node_b

                # Build a local axis x_hat pointing from other_corner to
                # close_corner, projected onto the surface plane.
                x_hat = close_corner.coords - other_corner.coords
                x_hat = x_hat - np.dot(x_hat, normal) * normal
                axis_len = np.linalg.norm(x_hat)

                if axis_len > 1e-14:
                    x_hat = x_hat / axis_len
                    y_hat = np.cross(normal, x_hat)
                    y_hat = y_hat / np.linalg.norm(y_hat)

                    # Project (C - other_corner) and (mynode - other_corner)
                    # onto y_hat to classify which side of the axis they are on.
                    vec_C  = (node.coords  - other_corner.coords)
                    vec_C  = vec_C  - np.dot(vec_C,  normal) * normal
                    vec_mn = (mynode.coords - other_corner.coords)
                    vec_mn = vec_mn - np.dot(vec_mn, normal) * normal

                    c_side  = np.dot(vec_C,  y_hat)
                    mn_side = np.dot(vec_mn, y_hat)

                    # isCplus: C is on the SAME side as mynode relative to the
                    # corner axis --> use other_corner (concave corner) as VS.
                    # isCminus: C is on the OPPOSITE side --> use close_corner.
                    
                    #isCplus = (c_side >= 0.0) and (mn_side > 0.0)
                    isCplus = (c_side * mn_side >= 0.0) 

                    if isCplus:
                        t_2d = compute_2d_eikonal(mynode, mynode2, node,
                                                  mynode.dist, mynode2.dist,
                                                  other_corner)
                        if t_2d < best_local_dist:
                            best_local_dist = t_2d
                            best_local_vs   = other_corner
                    else:
                        t_2d = compute_2d_eikonal(mynode, mynode2, node,
                                                  mynode.dist, mynode2.dist,
                                                  close_corner)
                        if t_2d < best_local_dist:
                            best_local_dist = t_2d
                            best_local_vs   = close_corner

            # Commit this triangle's result to the global best
            if best_local_dist < dist:
                dist         = best_local_dist
                best_node_vs = best_local_vs

        # ------------------------------------------------------------------
        # CASE 2: Only node_a is ALIVE --> pure 1D update
        # ------------------------------------------------------------------
        elif node_a.state == 'ALIVE':
            t_1d = node_a.dist + node.distance_to_other_node(node_a)
            if t_1d < dist:
                dist = t_1d
                corner_a = node_virtual_source[node_a.idx]
                # Collinearity check: is C on the same ray as the VS of A?
                collinear = consistent_update(node_a, node, normal, corner_a, SAME_LINE_TOL)
                best_node_vs = corner_a if collinear else node_a

        # ------------------------------------------------------------------
        # CASE 3: Only node_b is ALIVE --> pure 1D update
        # ------------------------------------------------------------------
        elif node_b.state == 'ALIVE':
            t_1d = node_b.dist + node.distance_to_other_node(node_b)
            if t_1d < dist:
                dist = t_1d
                corner_b = node_virtual_source[node_b.idx]
                collinear = consistent_update(node_b, node, normal, corner_b, SAME_LINE_TOL)
                best_node_vs = corner_b if collinear else node_b

    return dist, best_node_vs


# ========== Helper Functions ============
def compute_2d_eikonal(node_a, node_b, node_c, d_A_raw, d_B_raw, S_prime):
    """
    Compute the circular-wavefront distance to node_c given that the wavefront
    passes through the edge AB with arrival times d_A_raw and d_B_raw, and
    originates from the virtual point source S_prime.

    The idea is to place a virtual source S in the plane of the triangle such
    that its distances to A and B match d_A_raw - S_prime.dist and
    d_B_raw - S_prime.dist respectively.  The distance to C is then the
    Euclidean distance from S to C, plus S_prime.dist.

    Returns float('inf') if the upwind condition fails or geometry is degenerate.
    """

    # Extract the distance from the original source S to the virtual source S'
    if S_prime is None:
        S_prime_dist = 0.0
    else:
        S_prime_dist = S_prime.dist

    # Distances from S' to A and B, measured in the local frame
    d_A = d_A_raw - S_prime_dist
    d_B = d_B_raw - S_prime_dist

    if d_A < 0.0 or d_B < 0.0:
        # The virtual source S' is farther from A or B than the current node C,
        # which violates the upwind condition.  This can happen due to
        # numerical errors or in non-convex geometries.  Reject this update.
        
        return float('inf')

    # ====== STEP 1 ---> Local frame for the edge AB ====
    AB = node_b.coords - node_a.coords
    AC = node_c.coords - node_a.coords
    normal = np.cross(AB, AC)

    # Degenerate triangle check (normalised to be scale-invariant)
    norm_AB = np.linalg.norm(AB)
    norm_AC = np.linalg.norm(AC)
    if np.linalg.norm(normal) / (norm_AB * norm_AC + 1e-30) < 1e-10:
        return float('inf')

    normal = normal / np.linalg.norm(normal)
    L = norm_AB
    if L < 1e-14:
        return float('inf')

    # Local frame: x_hat along AB, y_hat in the surface plane perpendicular to AB
    x_hat = AB / L
    y_hat = np.cross(normal, x_hat)
    y_hat = y_hat / np.linalg.norm(y_hat)

    # ====== STEP 2 ---> Coordinates of C in the local frame ====
    x_C = np.dot(AC, x_hat)
    y_C = np.dot(AC, y_hat)

    # ====== STEP 3 ---> Position of the virtual source S in the local frame ====
    # S is placed so that |SA| = d_A and |SB| = d_B.
    # From the two circle equations: x_S = (d_A^2 - d_B^2 + L^2) / (2L)
    x_S    = (d_A**2 - d_B**2 + L**2) / (2.0 * L)
    sq_y_S = d_A**2 - x_S**2

    # Reject if the system has no real solution
    if sq_y_S < 0.0:
        return float('inf')

    # S must be on the opposite side of AB from C (upwind convention)
    sign_y_S = -np.sign(y_C)
    if sign_y_S == 0.0:
        sign_y_S = 1.0    # C exactly on AB: arbitrary but consistent choice
    y_S = sign_y_S * np.sqrt(sq_y_S)

    # ====== STEP 4 ---> Upwind condition (shadow-zone rejection) ====
    # The straight ray from S to C must cross the interior of edge AB.
    # Intersect the ray with the local x-axis (y = 0).
    if abs(y_C - y_S) > 1e-12:
        x_int = x_S - y_S * (x_C - x_S) / (y_C - y_S)
    else:
        x_int = -1.0    # Ray parallel to AB: force rejection

    if not (-1e-10 <= x_int <= L + 1e-10):
        return float('inf')

    # ====== STEP 5 ---> Distance from S to C ====
    t_local = np.sqrt((x_C - x_S)**2 + (y_C - y_S)**2)
    return t_local + S_prime_dist


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



def consistent_update(node1, node2, normal, vs_node1, samelinetol = 1e-1):
    """
    It checks if the 1D update of the edge curvature is on the same line (i.e. the front is 
    moving along a boundary) or the update is not consistent, i.e the front has to cross a
    corner node. In the latter case the virtual source has to be re-initialized.
    """
    if vs_node1 is None:
        return True
    
    v1 = node2.coords - node1.coords
    v2 = (node2.coords - vs_node1.coords)
    v2 = v2 - np.dot(v2, normal) * normal
                
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
                
    if n1 > 1e-14 and n2 > 1e-14:
        isOnTheSameLine = np.linalg.norm(np.cross(v1/n1, v2/n2)) < samelinetol
    else:
        isOnTheSameLine = True

    return isOnTheSameLine






# ===================================================
# =============== Debugging functions ===============
# ===================================================

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

    # --- All nodes in gray ---
    x_all = [n.coords[0] for n in node_list]
    y_all = [n.coords[1] for n in node_list]

    plt.figure(figsize=(6,6))
    plt.scatter(x_all, y_all, c="lightgray", s=5)

    # --- Current node (ALIVE) ---
    plt.scatter(current_node.coords[0], current_node.coords[1],
                c="blue", s=40, label="Current ALIVE")

    # --- Associated source ---
    source = node_virtual_source.get(current_node.idx)

    if source is not None:
        plt.scatter(source.coords[0], source.coords[1],
                    c="red", s=40, label="Virtual Source")

    # --- Formatting ---
    plt.gca().set_aspect("equal")

    # Fix the limits (important)
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

    # --- all nodes ---
    x_all = [n.coords[0] for n in node_list]
    y_all = [n.coords[1] for n in node_list]
    z_all = [n.coords[2] for n in node_list]
    ax.scatter(x_all, y_all, z_all, c="lightgray", s=5)

    # --- ALIVE nodes ---
    alive_nodes = [n for n in node_list if n.state == "ALIVE"]  # adapt if needed
    if alive_nodes:
        x_alive = [n.coords[0] for n in alive_nodes]
        y_alive = [n.coords[1] for n in alive_nodes]
        z_alive = [n.coords[2] for n in alive_nodes]
        ax.scatter(x_alive, y_alive, z_alive, c="blue", s=10)

    # --- current node ---
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

    # --- limits ---
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