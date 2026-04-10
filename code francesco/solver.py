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
        #todo innit la premiere couche de trial

    # for neighbor in closest_node.neighbors:
    #     for n in neighbor.neighbors:
    #         if n.state != 'ALIVE':
    #             n.dist = n.distance_to_other_node(closest_node)
    #             n.state = 'TRIAL'

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
        err     = node.dist - d_exact
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
    if _segment_crosses_missing_quadrant(source_coords, node_coords, L):
        d_via_corner = (np.linalg.norm(source_coords - corner) +
                        np.linalg.norm(node_coords   - corner))
        # min() is a safety net for points exactly on the corner boundary
        return d_via_corner

    return d_direct

def _segment_crosses_missing_quadrant(S, P, L):
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



# =============================================
# ------------------- Plots -------------------
# =============================================


def Plot_Isolines(node_list, element_list, source = 0.2):

    x = np.array([n.coords[0] for n in node_list])
    y = np.array([n.coords[1] for n in node_list])
    z = np.array([n.dist for n in node_list])

    levels = 40
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

    middle = [0.5, 0.5]
    m = (middle[1] - 0) / (middle[0]-source)
    p = -m*source

    xf = (1 - p) / m
    x_line = np.linspace(source, xf, 100)
    y_line = m*x_line + p

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

    n_levels = 20
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
                    # r_pt = np.sqrt(px**2 + py**2)
                    # if r_pt > 1e-10:
                    #     px = px / r_pt * R_iso
                    #     py = py / r_pt * R_iso
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
    
    # Estrazione dell'array degli errori tramite le funzioni matematiche già esistenti
    if mesh_type == 'square_surface':
        _, _, errors, _ = compute_err(node_list, true_source)
    elif mesh_type == 'cylinder':
        _, errors, _ = compute_err_cylinder(node_list, true_source, R)
    elif mesh_type == 'l_shape':
        _, _, errors, _ = compute_err_l_shape(node_list, true_source, L)
        
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
        
    # --- 2D PLOT for L-Shape and Square ---
    else:
        plt.figure(figsize=(8, 6))
        plt.gca().set_aspect('equal')
        
        cntr = plt.tricontourf(x, y, triangles, errors, levels=40, cmap="Wistia")
        plt.colorbar(cntr, label="Absolute Error")
        
        plt.triplot(x, y, triangles, color='black', alpha=0.1, linewidth=0.5)
        x_line = np.linspace(0.2, 0.8, 100)
        y_line = (5/3) * (x_line - 0.2)

        plt.plot(x_line, y_line, 'r--', linewidth=2)
        plt.title(f"Error Distribution - {mesh_type}")
        plt.xlabel("X")
        plt.ylabel("Y")


def ComputeNodeError(node, source_node, xc=0.5, yc=0.5, r=0.2):
    """Calcule l'erreur absolue pour un seul nœud"""
    xs = source_node.coords[0]
    ys = source_node.coords[1]
    Xc = np.array([xc, yc, 0])

    theta = np.arcsin(r / np.linalg.norm(Xc - source_node.coords))
    alpha = np.arctan2(yc - ys, xc - xs)
    m1 = np.tan(alpha - theta)
    m2 = np.tan(alpha + theta)
    m_source_center = (yc - ys) / (xc - xs)

    dy = node.coords[1] - ys
    dx = node.coords[0] - xs
    m = dy / dx if abs(dx) > 1e-12 else np.nan
    n_r = np.linalg.norm(node.coords - source_node.coords)
    dist_to_center = np.linalg.norm(Xc - node.coords)

    # Pas dans la zone d'ombre
    if m <= m1 or m >= m2:
        return abs(n_r - node.dist)

    # Avant le cercle
    if n_r <= np.linalg.norm(Xc - source_node.coords):
        return abs(n_r - node.dist)

    # Dans le cercle
    if dist_to_center < r:
        return 0.0

    # Zone d'ombre → chemin qui contourne
    ratio_n = np.clip(r / dist_to_center, -1.0, 1.0)
    theta_n = np.arcsin(ratio_n)

    true_d = r / np.tan(theta_n) + r / np.tan(theta)

    v1 = (node.coords - Xc) / dist_to_center
    v2 = (Xc - source_node.coords) / np.linalg.norm(Xc - source_node.coords)
    gamma = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))  # angle réel

    p_line = yc - m_source_center * xc
    above = node.coords[1] >= m_source_center * node.coords[0] + p_line

    beta1 = np.pi - (np.pi/2 - theta) - (np.pi/2 - theta_n - gamma) if above \
            else np.pi - (np.pi/2 - theta) - (np.pi/2 - theta_n + gamma)
    beta2 = 2*np.pi - 2*(np.pi/2 - theta) - 2*(np.pi/2 - theta_n) - beta1
    beta = min(abs(beta1), abs(beta2))
    true_d += r * beta

    return abs(true_d - node.dist)


def ConvergenceStudy(mesh_sizes, source_nodes, node_lists, target_coords=(1.0, 1.0),
                     xc=0.5, yc=0.5, r=0.2):
    """
    Étude de convergence sur le nœud le plus proche de target_coords.
    
    mesh_sizes   : liste des tailles de maille [h1, h2, h3, ...]
    source_nodes : liste des noeuds source (un par maillage)
    node_lists   : liste des node_list (une par maillage)
    """
    errors = []

    for node_list, source_node in zip(node_lists, source_nodes):
        # Trouver le nœud le plus proche de (1, 1)
        target = np.array([target_coords[0], target_coords[1], 0])
        closest = min(node_list, key=lambda n: np.linalg.norm(n.coords - target))

        err = ComputeNodeError(closest, source_node, xc, yc, r)
        errors.append(err)
        print(f"h = {mesh_sizes[len(errors)-1]:.4f} | error = {err:.6f}")

    # Plot
    plt.figure()
    plt.loglog(mesh_sizes, errors, 'bo-', label='Error at (1,1)')
    # Référence ordre 1 et ordre 2
    h = np.array(mesh_sizes)
    plt.loglog(h, h * errors[0]/mesh_sizes[0], 'k--', label='O(h)')
    plt.loglog(h, h**2 * errors[0]/mesh_sizes[0]**2, 'r--', label='O(h²)')
    plt.xlabel("Mesh size h")
    plt.ylabel("Absolute Error")
    plt.title("Convergence at node (1, 1)")
    plt.legend()
    plt.grid(True)
    plt.show()

    return errors


def Get_error_map_hole(node_list, source_node, element_list, r):

    #* First i get the nodes that are not in the shadow zone (true dist = euclidean dist)
    xs = source_node.coords[0]
    ys = source_node.coords[1]

    #*Center of the circle
    xc = 0.5
    yc = 0.5
    Xc = np.array([xc, yc, 0])
    r = r

    #*Theta is the angle btween the tangeante and the line connecting source and center
    theta = np.arcsin(r / np.linalg.norm(Xc - source_node.coords))
    m_source_center = (yc- ys) / (xc - xs)
    
    alpha = np.atan2(yc - ys, xc - xs)

    m1 = np.tan(alpha - theta)
    m2 = np.tan(alpha + theta)

    errors = []
    OnenodeErr = 0
    for n in node_list:
        
        dy = (n.coords[1] - ys) 
        dx = (n.coords[0] - xs)
        m = dy / dx if abs(dx) > 1e-12 else np.inf
        n_r = np.linalg.norm(n.coords - source_node.coords)
        dist_to_center = np.linalg.norm(Xc - n.coords)

        if m <= m1 or m >= m2:
            errors.append(abs(np.linalg.norm(n.coords - source_node.coords) - n.dist))

        elif n_r <= (r**2 + np.linalg.norm(Xc - source_node.coords)**2)**(1/2):
            errors.append(abs(np.linalg.norm(n.coords - source_node.coords) - n.dist))

        else:
            # Nœud trop proche du cercle (dedans ou sur le bord)
            if dist_to_center < r:
                errors.append(0.0) 
                continue

            true_d = 0
            ratio_n = r / dist_to_center
            theta_n = np.arcsin(ratio_n)

            true_d += r / np.tan(theta_n)
            true_d += r / np.tan(theta)

            v1 = n.coords - Xc
            v1 /= np.linalg.norm(v1)
            v2 = -(Xc - source_node.coords)
            v2 /= np.linalg.norm(v2)
            gamma = np.pi - np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))

            mline  = m_source_center
            p = yc - xc * mline
            isAboveLine = n.coords[1] >= (mline) * n.coords[0] + p

            beta1 = np.pi - (np.pi/2 - theta) - ((np.pi/2 - theta_n) + gamma) 
            beta2 = 2*np.pi - 2*(np.pi/2 - theta) - 2*(np.pi/2 - theta_n) - beta1
            print(beta1*180/np.pi, beta2*180/np.pi)
            beta = min(beta1, beta2)

            true_d += r * beta
            errors.append(abs(true_d - n.dist))
        if n.coords[0] == 1 and n.coords[1] == 1: OnenodeErr = errors[-1]

    triangles = np.array([[n.idx for n in e.nodes] for e in element_list])
    x = np.array([n.coords[0] for n in node_list])
    y = np.array([n.coords[1] for n in node_list])
    
    plt.figure(figsize=(8, 6))
    plt.gca().set_aspect('equal')
    
    cntr = plt.tricontourf(x, y, triangles, errors, levels=40, cmap="inferno")#Wistia
    plt.colorbar(cntr, label="Absolute Error")
    
    plt.triplot(x, y, triangles, color='black', alpha=0.1, linewidth=0.5)
    middle = [0.5, 0.5]
    source = source_node.coords[0]
    m = (middle[1] - 0) / (middle[0]-source)
    p = -m*source

    xf = (1 - p) / m
    x_line = np.linspace(source, xf, 100)
    y_line = m*x_line + p
    def clip_line(xs, ys, phi):
        t_vals = []
        
        if abs(np.cos(phi)) > 1e-12:
            t_vals += [(0 - xs)/np.cos(phi), (1 - xs)/np.cos(phi)]
        if abs(np.sin(phi)) > 1e-12:
            t_vals += [(0 - ys)/np.sin(phi), (1 - ys)/np.sin(phi)]
        
        t_vals = [t for t in t_vals if t > 0]
        tmax = min(t_vals) if t_vals else 1
        
        t = np.linspace(0, tmax, 100)
        return xs + t*np.cos(phi), ys + t*np.sin(phi)
    dx = xc - xs
    dy = yc - ys
    d = np.sqrt(dx**2 + dy**2)

    alpha = np.arctan2(dy, dx)
    theta = np.arcsin(r / d)

    phi1 = alpha - theta
    phi2 = alpha + theta
    x_t1, y_t1 = clip_line(xs, ys, phi1)
    x_t2, y_t2 = clip_line(xs, ys, phi2)

    plt.plot(x_t1, y_t1, 'cyan', linewidth=1)
    plt.plot(x_t2, y_t2, 'cyan', linewidth=1)

    plt.plot(x_line, y_line, 'r--', linewidth=1)
    plt.title(f"Error Distribution - hole")
    plt.xlabel("X")
    plt.ylabel("Y")

    