import numpy as np
import gmsh
import matplotlib.pyplot as plt

from structures import NODE
from structures import ELEMENT
from structures import TRIAL_BAND



def Make_NodeList_NodeDictionary(filename, tag_to_node_obj_dic):

    if not gmsh.isInitialized():
        gmsh.initialize()
    
    print("\n")
    print("=================================================")
    print("================Making node list=================")
    print("=================================================")

    node_tags, coords, _ = gmsh.model.mesh.getNodes()
    points = coords.reshape(-1, 3)

    nodes_list = []
    
    for i, tag in enumerate(node_tags):
        new_node = NODE(points[i], tag, i)
        nodes_list.append(new_node)
        tag_to_node_obj_dic[tag] = new_node
    
    _, _, node_tags_per_elem = gmsh.model.mesh.getElements(2)
    triangles_tags = node_tags_per_elem[0].reshape(-1, 3)

    for tri in triangles_tags:
        node_a = tag_to_node_obj_dic[tri[0]]
        node_b = tag_to_node_obj_dic[tri[1]]
        node_c = tag_to_node_obj_dic[tri[2]]
        
        node_a.neighbors.update([node_b, node_c])
        node_b.neighbors.update([node_a, node_c])
        node_c.neighbors.update([node_a, node_b])

    print("Node list is done --> ok")
    return np.array(nodes_list)



def Make_ElementList(filename, tag_to_node_obj_dic):

    if not gmsh.isInitialized():
        gmsh.initialize()
    
    print("\n")
    print("=================================================")
    print("===============Making element list===============")
    print("=================================================")

    _, _, node_tags_per_elem = gmsh.model.mesh.getElements(2)
    triangles_tags = node_tags_per_elem[0].reshape(-1, 3)

    element_list = []
    for tri in triangles_tags:
        node_a = tag_to_node_obj_dic[tri[0]]
        node_b = tag_to_node_obj_dic[tri[1]]
        node_c = tag_to_node_obj_dic[tri[2]]
        nodes = [node_a, node_b, node_c]
        element = ELEMENT(nodes)
        element_list.append(element)
        node_a.adjacent_triangles.append(element)
        node_b.adjacent_triangles.append(element)
        node_c.adjacent_triangles.append(element)

    print("Element list is done --> ok")
    return np.array(element_list)



def Check_Obtuse_triangles(element_list):

    print("\n")
    print("========================================================")
    print("=============Cheking for obtuse element=================")
    print("========================================================")
    counting = 0
    for e in element_list:
        AB = np.array([e.nodes[0].coords[0] - e.nodes[1].coords[0], e.nodes[0].coords[1] - e.nodes[1].coords[1]])
        AC = np.array([e.nodes[0].coords[0] - e.nodes[2].coords[0], e.nodes[0].coords[1] - e.nodes[2].coords[1]])
        BC = np.array([e.nodes[1].coords[0] - e.nodes[2].coords[0], e.nodes[1].coords[1] - e.nodes[2].coords[1]])

        AB_AC = AB @ AC
        BC_BA = BC @ (-AB)
        CA_CB = (-AC) @ (-BC)
        if min(AB_AC, BC_BA, CA_CB) < 0:
            e.IsObtuse = True
            counting += 1
    
    print(f"There are {counting} obtuse elements")
    print("Checking for obtuse element is done --> ok\n")



def Compute_Dist_Point_Triangle(P, element):

    A, B, C = [n.coords for n in element.nodes]

    normal = np.cross(B - A, C - A)
    normal /= np.linalg.norm(normal)

    vec_AP = P - A
    dist_ortho = np.abs(np.dot(vec_AP, normal))
    return dist_ortho, normal



def Innit_Origin_Point(target_coords, node_list):

    all_coords = np.array([n.coords for n in node_list])
    distances_to_target = np.linalg.norm(all_coords - target_coords, axis=1)
    closest_node_idx = np.argmin(distances_to_target)
    closest_node = node_list[closest_node_idx]

    snap_err = np.linalg.norm(closest_node.coords - target_coords)

    # Set the snapped node as the true origin with dist = 0
    # This removes the O(h) snapping pollution from the convergence
    closest_node.dist  = 0.0
    closest_node.state = 'ALIVE'

    # Initialize neighbors with Euclidean distance FROM THE SNAPPED NODE
    # (not from the theoretical source) so the FMM starts from a consistent origin
    for neighbor in closest_node.neighbors:
        neighbor.dist  = neighbor.distance_to_other_node(closest_node)
        neighbor.state = 'TRIAL'

    print(f"Source node {closest_node.node_tag} @ {closest_node.coords}")
    print(f"  Theoretical source : {target_coords}")
    print(f"  Snapping error     : {snap_err:.6e}")

    return closest_node.coords, [closest_node]



def Reset_Node_State(node_list):
    for node in node_list:
        node.dist  = float('inf')
        node.state = 'FAR'



def compute_err(node_list, source_coord):
    """
    Compute max, relative, pointwise, and L2 errors for the standard geometry.
    Returns: (max_error, max_relative_error, pointwise_errors_array, L2_error)
    """
    errors = []
    errors_rel = []

    for node in node_list:
        # Avoid plotting crashes or math errors if a node is not reached by the FMM
        if node.dist == float('inf'):
            errors.append(0.0)  
            continue
            
        ex_dist = np.linalg.norm(node.coords - source_coord)
        err_i = abs(ex_dist - node.dist)
        errors.append(err_i)
        
        if ex_dist > 1e-14:
            errors_rel.append(err_i / ex_dist)

    err = np.max(errors) if errors else 0.0
    err_rel = np.max(errors_rel) if errors_rel else 0.0
    e_l2 = np.sqrt(np.mean(np.array(errors)**2)) if errors else 0.0

    # Returning the errors array before e_l2 ensures that result[-1] is still the L2 error
    return err, err_rel, np.array(errors), e_l2


def compute_err_cylinder(node_list, true_source, R):
    """
    Compute errors for the cylinder geometry.
    Returns: (max_error, pointwise_errors_array, L2_error)
    """
    x0, y0, z0 = true_source
    R_source = np.sqrt(x0**2 + y0**2)
    theta0   = np.arctan2(y0, x0)

    max_err = 0.0
    sum_sq  = 0.0
    count   = 0
    errors  = []  # List needed for the error plot

    for node in node_list:
        if node.dist == float('inf'):
            errors.append(0.0)
            continue
            
        x, y, z = node.coords
        theta   = np.arctan2(y, x)
        d_theta = (theta - theta0 + np.pi) % (2 * np.pi) - np.pi
        d_exact = np.sqrt((R_source * d_theta)**2 + (z - z0)**2)
        err     = abs(node.dist - d_exact)
        errors.append(err)
        
        max_err = max(max_err, err)
        sum_sq += err**2
        count  += 1

    l2_err = np.sqrt(sum_sq / count) if count > 0 else 0.0
    
    return max_err, np.array(errors), l2_err


def compute_err_l_shape(node_list, source_coord, L=1.0):
    """
    Compute geodesic errors for the L-shape geometry.
    Returns: (max_error, max_relative_error, pointwise_errors_array, L2_error)
    """
    errors     = []  
    errors_rel = []  

    for node in node_list:
        if node.dist == float('inf'):
            errors.append(0.0)
            continue
            
        d_exact = geodesic_lshape(node.coords, source_coord, L)
        err_i = abs(d_exact - node.dist)
        errors.append(err_i)

        if d_exact > 1e-14:
            errors_rel.append(err_i / d_exact)

    err     = np.max(errors) if errors else 0.0
    err_rel = np.max(errors_rel) if errors_rel else 0.0
    e_l2    = np.sqrt(np.mean(np.array(errors) ** 2)) if errors else 0.0 

    return err, err_rel, np.array(errors), e_l2

def geodesic_lshape(node_coords, source_coords, L=1.0):
    """
    Exact geodesic distance on the L-shape domain.

    The missing quadrant is (x > L/2, y < L/2).
    The concave corner is at C = (L/2, L/2).

    If the straight line S->P crosses the boundary of the missing quadrant,
    the geodesic must detour through the corner C:
        d = dist(S, C) + dist(C, P)

    Otherwise the Euclidean distance is exact.
    """
    corner = np.array([L / 2.0, L / 2.0, 0.0])

    # Direct Euclidean distance
    d_direct = np.linalg.norm(node_coords - source_coords)

    # Check if the straight segment crosses the missing quadrant boundary
    if segment_crosses_missing_quadrant(source_coords, node_coords, L):
        d_via_corner = (np.linalg.norm(source_coords - corner) +
                        np.linalg.norm(node_coords   - corner))
        # min() is a safety net for points exactly on the corner boundary
        return d_via_corner

    return d_direct

def segment_crosses_missing_quadrant(S, P, L):
    """
    Check whether the straight segment S->P passes through the missing
    quadrant of the L-shape, i.e. the rectangle (L/2 < x < L, 0 < y < L/2).

    Strategy: check if the segment crosses either of the two boundary edges
    of the missing quadrant that are interior to the bounding box:
      - Vertical edge  : x = L/2,  y in [0,   L/2]
      - Horizontal edge: y = L/2,  x in [L/2, L  ]

    If it crosses one of them going INTO the missing quadrant, return True.
    """
    x_S, y_S = S[0], S[1]
    x_P, y_P = P[0], P[1]
    dx = x_P - x_S
    dy = y_P - y_S

    # --- Check intersection with vertical edge x = L/2, y in [0, L/2] ---
    if abs(dx) > 1e-14:
        t = (L/2 - x_S) / dx          # Parameter t in [0,1] along segment S->P
        if 0.0 < t < 1.0:
            y_int = y_S + t * dy       # y coordinate at the intersection
            if 0.0 <= y_int <= L/2:    # Intersection is on the interior edge
                # The segment crosses x=L/2 in the lower half
                # --> it is entering or exiting the missing quadrant
                return True

    # --- Check intersection with horizontal edge y = L/2, x in [L/2, L] ---
    if abs(dy) > 1e-14:
        t = (L/2 - y_S) / dy
        if 0.0 < t < 1.0:
            x_int = x_S + t * dx
            if L/2 <= x_int <= L:      # Intersection is on the interior edge
                return True

    return False



def compute_err_hole(node_list, source_coord, L=1.0, R=0.2):
    """
    Compute geodesic errors for the square with hole geometry.
    Returns: (max_error, max_relative_error, pointwise_errors_array, L2_error)
    """
    errors     = []  
    errors_rel = []  

    for node in node_list:
        if node.dist == float('inf'):
            errors.append(0.0)
            continue
            
        d_exact = geodesic_hole(node.coords, source_coord, L, R)
        err_i = abs(d_exact - node.dist)
        errors.append(err_i)

        if d_exact > 1e-14:
            errors_rel.append(err_i / d_exact)

    err     = np.max(errors) if errors else 0.0
    err_rel = np.max(errors_rel) if errors_rel else 0.0
    e_l2    = np.sqrt(np.mean(np.array(errors) ** 2)) if errors else 0.0 

    return err, err_rel, np.array(errors), e_l2



def geodesic_hole(node_coords, source_coords, L=1.0, R=0.2):
    """
    Exact geodesic distance on the square with hole domain.

    The hole is a disk of radius R centered at (L/2, L/2).
    The source is outside the hole.

    If the straight line S->P intersects the hole, the geodesic must detour
    around the hole. The exact distance can be computed by considering all
    possible tangent points on the hole and taking the minimum path:
        d = min_{tangent points T} [dist(S, T) + dist(T, P)]

    Otherwise the Euclidean distance is exact.
    """

    # Supposing the hole is centered at (0,0)
    C_coords = np.array([L/2, L/2, 0.0])

    # Determine the vectors starting from the center and pointing towards the 
    # source S (source_coords) and the target P (node_coords)
    v_S = source_coords - C_coords
    v_P = node_coords   - C_coords

    # Compute their norms, i.e. the distances from the center C
    d_S = np.linalg.norm(v_S)
    d_P = np.linalg.norm(v_P)
    # Avoid any problems due to mesh not properly corrected
    d_S = max(d_S, R)
    d_P = max(d_P, R)

    # Compute the angles from the vectors to the tangency points
    alpha_S = np.arccos(np.clip(R / d_S, -1.0, 1.0))
    alpha_P = np.arccos(np.clip(R / d_P, -1.0, 1.0))

    # Lowest angle separating the vectors v_S and v_P
    """
    dot_prod = np.dot(v_S, v_P) / (d_S * d_P)
    phi = np.arccos(np.clip(dot_prod, -1.0, 1.0))   
    # Dot Product can cause problems of signs
    """
    # Use atan2 to get the signed angle and then take absolute value for the separation
    phi = abs(np.arctan2(v_S[1], v_S[0]) - np.arctan2(v_P[1], v_P[0]))
    if phi > np.pi: 
        # The absolute difference must be always [0 pi]
        phi = 2 * np.pi - phi


    # Visibility condition
    if phi <= alpha_S + alpha_P:
        # P is directly visible from S --> Euclidean distance 
        return np.linalg.norm(node_coords - source_coords)
    
    else:
        # P is not visible from S --> geodesic goes through the tangents to the hole
        t_S = np.sqrt(d_S**2 - R**2)
        t_P = np.sqrt(d_P**2 - R**2)
        arc_angle = phi - (alpha_S + alpha_P)
        return t_S + t_P + R * arc_angle

    

# =============================================
# ------------------- Plots -------------------
# =============================================


def Plot_Isolines(node_list, element_list):

    x = np.array([n.coords[0] for n in node_list])
    y = np.array([n.coords[1] for n in node_list])
    z = np.array([n.dist for n in node_list])

    levels = 20
    triangles = []
    for e in element_list:
        triangles.append([n.idx for n in e.nodes])
    triangles = np.array(triangles)

    plt.figure(figsize=(8, 6))
    plt.gca().set_aspect('equal')

    cntr = plt.tricontourf(x, y, triangles, z, levels=levels, cmap="viridis")
    plt.colorbar(cntr, label="Distance")

    lines = plt.tricontour(x, y, triangles, z, levels=levels, colors='white', linewidths=0.5)
    plt.clabel(lines, inline=True, fontsize=8)

    plt.triplot(x, y, triangles, color='black', alpha=0.1, linewidth=0.5)
    x_line = np.linspace(0.2, 0.8, 100)
    y_line = (5/3) * (x_line - 0.2)

    plt.plot(x_line, y_line, 'r--', linewidth=2)
    plt.title("FMM - Plot of the levelsets")
    plt.xlabel("X")
    plt.ylabel("Y")
    #plt.show()


def Plot_Isolines_3D(node_list, element_list, source_coords=None):

    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    import matplotlib.cm as cm

    x   = np.array([n.coords[0] for n in node_list])
    y   = np.array([n.coords[1] for n in node_list])
    z   = np.array([n.coords[2] for n in node_list])
    val = np.array([n.dist      for n in node_list])

    triangles = np.array([[n.idx for n in e.nodes] for e in element_list])
    face_val  = val[triangles].mean(axis=1)

    norm_c = Normalize(vmin=val.min(), vmax=val.max())
    cmap   = cm.viridis
    colors = cmap(norm_c(face_val))

    fig = plt.figure(figsize=(12, 8))
    ax  = fig.add_subplot(111, projection='3d')

    # Semitransparent surface
    surf = ax.plot_trisurf(x, y, z, triangles=triangles,
                           shade=False, antialiased=False, alpha=0.85)
    surf.set_facecolors(colors)

    # ---- Source Point ----
    if source_coords is not None:
        ax.scatter(source_coords[0], source_coords[1], source_coords[2],
                   color='red', s=100, zorder=5, label='Point source')
        ax.legend()

    # ---- Isolines ----
    # Minumum Radial offset to avoid z-fighting with the sourface 
    R_mean = np.sqrt(x**2 + y**2).mean()
    R_iso  = R_mean * 1.008

    n_levels = 15
    eps      = (val.max() - val.min()) * 0.001
    levels   = np.linspace(val.min() + eps, val.max() - eps, n_levels)

    for level in levels:
        label_pts = []

        for tri in triangles:
            i0, i1, i2 = tri
            pts = []
            for ea, eb in [(i0, i1), (i1, i2), (i2, i0)]:
                va, vb = val[ea], val[eb]
                if (va - level) * (vb - level) < 0:
                    t = (level - va) / (vb - va)
                    # Interpolation in 3D cartesian coordinates
                    px = x[ea] + t * (x[eb] - x[ea])
                    py = y[ea] + t * (y[eb] - y[ea])
                    pz = z[ea] + t * (z[eb] - z[ea])
                    # Re-project on the cylinder with R_iso
                    r_pt = np.sqrt(px**2 + py**2)
                    if r_pt > 1e-10:
                        px = px / r_pt * R_iso
                        py = py / r_pt * R_iso
                    pts.append((px, py, pz))

            if len(pts) == 2:
                ax.plot([pts[0][0], pts[1][0]],
                        [pts[0][1], pts[1][1]],
                        [pts[0][2], pts[1][2]],
                        color='white', linewidth=1.2, alpha=1.0, zorder=10)
                label_pts.append(pts)

        # Label at the mid point of each segment
        if label_pts:
            p0, p1 = label_pts[len(label_pts) // 2]
            mx = (p0[0] + p1[0]) / 2
            my = (p0[1] + p1[1]) / 2
            mz = (p0[2] + p1[2]) / 2
            ax.text(mx, my, mz, f'{level:.2f}',
                    color='yellow', fontsize=6.5, fontweight='bold',
                    zorder=20, ha='center')

    # ---- Colorbar ----
    sm = ScalarMappable(cmap=cmap, norm=norm_c)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, shrink=0.5, label="Geodesic distance")

    ax.set_title("FMM - Distance field on cylinder")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    plt.tight_layout()
    #plt.show()


def Plot_Isolines_L_Cylinder(node_list, element_list, source_coords=None):
    """
    Plot the geodesic distance field and isolines on the L-shaped cylinder surface.

    This function is identical in structure to Plot_Isolines_3D (used for the
    full half-cylinder), but adapted for the L-shaped variant:
      - The radial re-projection of isoline points uses the LOCAL radius of each
        interpolated point instead of a fixed global R_mean. This is necessary
        because the L-cylinder surface is not a surface of constant radius from
        a single axis -- different parts of the surface may have slightly
        different radii due to meshing.
      - The title and labels reflect the L-cylinder geometry.

    Parameters
    ----------
    node_list    : array of NODE objects -- all mesh nodes with .coords and .dist
    element_list : array of ELEMENT objects -- all triangles with .nodes
    source_coords: array-like of shape (3,), optional -- 3D coords of the source
                   point to mark on the plot with a red dot
    """

    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    import matplotlib.cm as cm

    # ------------------------------------------------------------------
    # STEP 1: Extract node coordinates and FMM distance values
    #
    # List comprehensions in Python: [expression for item in iterable]
    # This is equivalent to a for-loop that builds an array.
    # np.array([...]) converts the Python list to a NumPy array (like a MATLAB vector).
    # ------------------------------------------------------------------
    x   = np.array([n.coords[0] for n in node_list])   # x-coordinates of all nodes
    y   = np.array([n.coords[1] for n in node_list])   # y-coordinates of all nodes
    z   = np.array([n.coords[2] for n in node_list])   # z-coordinates of all nodes
    val = np.array([n.dist      for n in node_list])   # FMM distance at each node

    # Build the triangle connectivity array: shape (n_triangles, 3)
    # Each row contains the indices (idx) of the 3 nodes of one triangle.
    triangles = np.array([[n.idx for n in e.nodes] for e in element_list])

    # ------------------------------------------------------------------
    # STEP 2: Compute per-face colors for the surface plot
    #
    # face_val: average distance over the 3 nodes of each triangle.
    # val[triangles] is fancy indexing: for each triangle row [i, j, k],
    # it returns [val[i], val[j], val[k]]. Shape: (n_tri, 3).
    # .mean(axis=1) averages along axis 1 (the 3 nodes), giving one value per face.
    # ------------------------------------------------------------------
    face_val = val[triangles].mean(axis=1)

    # Normalize maps the raw distance values to [0, 1] for colormap lookup
    norm_c = Normalize(vmin=val.min(), vmax=val.max())
    cmap   = cm.viridis
    colors = cmap(norm_c(face_val))   # RGBA color for each triangle face

    # ------------------------------------------------------------------
    # STEP 3: Create the 3D figure and plot the colored surface
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(12, 8))
    ax  = fig.add_subplot(111, projection='3d')   # '111' = 1 row, 1 col, subplot 1

    # plot_trisurf draws a triangulated surface in 3D.
    # shade=False + set_facecolors() lets us assign our own per-face colors
    # instead of using matplotlib's default shading.
    surf = ax.plot_trisurf(x, y, z, triangles=triangles,
                           shade=False, antialiased=False, alpha=0.85)
    surf.set_facecolors(colors)

    # ------------------------------------------------------------------
    # STEP 4: Mark the source point
    # ------------------------------------------------------------------
    if source_coords is not None:
        ax.scatter(source_coords[0], source_coords[1], source_coords[2],
                   color='red', s=100, zorder=5, label='Point source')
        ax.legend()

    # ------------------------------------------------------------------
    # STEP 5: Draw isolines manually by linear interpolation on each triangle
    #
    # matplotlib has no built-in tricontour for 3D surfaces, so we do it
    # manually: for each distance level, we scan every triangle edge and check
    # if the level crosses that edge. If it does, we interpolate the crossing
    # point in 3D, then draw a segment between the two crossing points of
    # the same triangle.
    #
    # The isoline points are slightly offset radially (R_iso = R * 1.008)
    # to avoid z-fighting (visual artifacts where two surfaces overlap exactly).
    # Here we use the LOCAL radius of each interpolated point, so the offset
    # is correct everywhere on the L-shaped surface (not just at a fixed R).
    # ------------------------------------------------------------------

    n_levels = 15
    eps      = (val.max() - val.min()) * 0.001   # small margin to avoid boundary levels
    levels   = np.linspace(val.min() + eps, val.max() - eps, n_levels)

    RADIAL_OFFSET = 1.008   # push isolines 0.8% outward to avoid z-fighting

    for level in levels:
        label_pts = []   # will store one representative segment per level for labelling

        for tri in triangles:
            i0, i1, i2 = tri   # unpack the 3 node indices of this triangle

            pts = []   # crossing points found on this triangle (0, 1, or 2)

            # Check the 3 edges: (i0,i1), (i1,i2), (i2,i0)
            for ea, eb in [(i0, i1), (i1, i2), (i2, i0)]:
                va, vb = val[ea], val[eb]

                # The level crosses this edge if val changes sign relative to level
                # i.e. one endpoint is above and the other is below the level
                if (va - level) * (vb - level) < 0:

                    # Linear interpolation parameter t in [0, 1] along the edge
                    # t=0 at node ea, t=1 at node eb
                    t = (level - va) / (vb - va)

                    # Interpolated 3D coordinates on the edge
                    px = x[ea] + t * (x[eb] - x[ea])
                    py = y[ea] + t * (y[eb] - y[ea])
                    pz = z[ea] + t * (z[eb] - z[ea])

                    # Radial offset: push the point slightly outward from the
                    # cylinder axis so isolines are drawn on top of the surface.
                    # We use the LOCAL in-plane radius of this specific point.
                    r_local = np.sqrt(px**2 + py**2)
                    if r_local > 1e-10:
                        px = px / r_local * (r_local * RADIAL_OFFSET)
                        py = py / r_local * (r_local * RADIAL_OFFSET)

                    pts.append((px, py, pz))

            # A triangle crossed by an isoline should give exactly 2 points
            if len(pts) == 2:
                ax.plot([pts[0][0], pts[1][0]],
                        [pts[0][1], pts[1][1]],
                        [pts[0][2], pts[1][2]],
                        color='white', linewidth=1.2, alpha=1.0, zorder=10)
                label_pts.append(pts)

        # Place one text label per isoline level, at the midpoint of a
        # representative segment (the middle one in the list)
        if label_pts:
            p0, p1 = label_pts[len(label_pts) // 2]   # '//' = integer division
            mx = (p0[0] + p1[0]) / 2
            my = (p0[1] + p1[1]) / 2
            mz = (p0[2] + p1[2]) / 2
            ax.text(mx, my, mz, f'{level:.2f}',
                    color='yellow', fontsize=6.5, fontweight='bold',
                    zorder=20, ha='center')

    # ------------------------------------------------------------------
    # STEP 6: Colorbar and axis labels
    # ------------------------------------------------------------------
    sm = ScalarMappable(cmap=cmap, norm=norm_c)
    sm.set_array([])   # required boilerplate for standalone ScalarMappable colorbars
    fig.colorbar(sm, ax=ax, shrink=0.5, label="Geodesic distance")

    ax.set_title("FMM - Distance field on L-cylinder")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    plt.tight_layout()


def plot_convergence(h_values, errors_l2_fmm, errors_l2_circ, title, save_name):

    h_arr       = np.array(h_values[::-1])
    l2_fmm_arr  = np.array(errors_l2_fmm[::-1])
    l2_circ_arr = np.array(errors_l2_circ[::-1])

    print("\n\nConvergence order (L2) - Standard FMM:")
    for i in range(1, len(h_values)):
        order = np.log(errors_l2_fmm[i] / errors_l2_fmm[i-1]) / \
                np.log(h_values[i] / h_values[i-1])
        print(f"  h={h_values[i]:.4f}  |  L2={errors_l2_fmm[i]:.2e}  |  order ≈ {order:.2f}")

    print("\n\nConvergence order (L2) - Circular FMM:")
    for i in range(1, len(h_values)):
        order = np.log(errors_l2_circ[i] / errors_l2_circ[i-1]) / \
                np.log(h_values[i] / h_values[i-1])
        print(f"  h={h_values[i]:.4f}  |  L2={errors_l2_circ[i]:.2e}  |  order ≈ {order:.2f}")

    plt.figure(figsize=(8, 6))
    plt.loglog(h_arr, l2_fmm_arr,  'o-', label='FMM (L2)')
    plt.loglog(h_arr, l2_circ_arr, 'o-', label='HFMM (L2)')
    plt.loglog(h_arr, h_arr,       '--', color='gray',  label='O(h)')
    plt.loglog(h_arr, h_arr**2,    '--', color='black', label='O(h²)')
    plt.xlabel('h (mesh size)')
    plt.ylabel('L2 Error')
    plt.title(title)
    plt.legend()
    plt.grid(True, which='both')
    plt.gca().invert_xaxis()
    plt.tight_layout()
    plt.savefig(save_name)
    #plt.show()




def Plot_Error_Field(node_list, element_list, mesh_type, true_source, R=0.5, L=1.0):
    import matplotlib.cm as cm
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    import matplotlib.pyplot as plt
    
    x = np.array([n.coords[0] for n in node_list])
    y = np.array([n.coords[1] for n in node_list])
    z_coord = np.array([n.coords[2] for n in node_list])
    
    # Extraction of the error array using the existing mathematical functions
    if mesh_type == 'square_surface':
        _, _, errors, _ = compute_err(node_list, true_source)
    elif mesh_type == 'cylinder':
        _, errors, _ = compute_err_cylinder(node_list, true_source, R)
    elif mesh_type == 'l_shape':
        _, _, errors, _ = compute_err_l_shape(node_list, true_source, L)
    elif mesh_type == 'hole':
        # Passiamo R correttamente, sia che sia 0.2 o il default
        _, _, errors, _ = compute_err_hole(node_list, true_source, L, R)
        
    triangles = np.array([[n.idx for n in e.nodes] for e in element_list])
    
    # --- 3D PLOT for Cylinder ---
    if mesh_type == 'cylinder':
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        face_errors = errors[triangles].mean(axis=1)
        norm_c = Normalize(vmin=errors.min(), vmax=errors.max())
        cmap = cm.inferno  
        colors = cmap(norm_c(face_errors))
        
        surf = ax.plot_trisurf(x, y, z_coord, triangles=triangles, shade=False, antialiased=False, alpha=0.9)
        surf.set_facecolors(colors)
        
        sm = ScalarMappable(cmap=cmap, norm=norm_c)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, shrink=0.5, label="Absolute Error")
        
        ax.set_title(f"Error Distribution - {mesh_type}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        
    # --- 2D PLOT for L-Shape, Square and Hole ---
    else:
        plt.figure(figsize=(8, 6))
        plt.gca().set_aspect('equal')
        
        cntr = plt.tricontourf(x, y, triangles, errors, levels=40, cmap="inferno")
        plt.colorbar(cntr, label="Absolute Error")
        
        plt.triplot(x, y, triangles, color='black', alpha=0.1, linewidth=0.5)
        
        plt.title(f"Error Distribution - {mesh_type}")
        plt.xlabel("X")
        plt.ylabel("Y")


