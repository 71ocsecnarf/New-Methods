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

    errors = []
    errors_rel = []

    for node in node_list:
        ex_dist = np.linalg.norm(node.coords - source_coord)
        err_i = abs(ex_dist - node.dist)
        errors.append(err_i)
        if ex_dist > 1e-14:
            errors_rel.append(err_i / ex_dist)

    err = np.max(errors) 
    err_rel = np.max(errors_rel)
    e_l2 = np.sqrt(np.mean(np.array(errors)**2))

    return err, err_rel, e_l2


"""
def compute_err_cylinder(node_list, true_source, R):
    #Geodesic distance on a cylinder = straight line on the unrolled rectangle:
        #d = sqrt( (R * delta_theta)^2 + delta_z^2 )
    #Returns (max_err, l2_err) — stesso formato di solver.compute_err()
    x0, y0, z0 = true_source
    theta0 = np.arctan2(y0, x0)

    max_err = 0.0
    sum_sq  = 0.0
    count   = 0

    for node in node_list:
        if node.dist == float('inf'):
            continue
        x, y, z   = node.coords
        theta     = np.arctan2(y, x)
        d_theta   = (theta - theta0 + np.pi) % (2 * np.pi) - np.pi
        d_exact   = np.sqrt((R * d_theta)**2 + (z - z0)**2)
        err       = abs(node.dist - d_exact)
        max_err   = max(max_err, err)
        sum_sq   += err**2
        count    += 1

    l2_err = np.sqrt(sum_sq / count) if count > 0 else 0.0
    return max_err, l2_err

"""



def compute_err_cylinder(node_list, true_source, R):
    """
    Geodesic distance on a cylinder (unrolled rectangle):
        d = sqrt( (R * delta_theta)^2 + delta_z^2 )

    true_source must be the SNAPPED source coordinates (coords of the
    closest mesh node), not the theoretical source point. This ensures
    the FMM distance and the exact distance share the same origin.
    """
    x0, y0, z0 = true_source
    # Recompute R from the snapped point to be consistent
    # (the snapped node is ON the mesh, which may differ slightly from R)
    R_source = np.sqrt(x0**2 + y0**2)
    theta0   = np.arctan2(y0, x0)

    max_err = 0.0
    sum_sq  = 0.0
    count   = 0

    for node in node_list:
        if node.dist == float('inf'):
            continue
        x, y, z = node.coords
        theta   = np.arctan2(y, x)
        d_theta = (theta - theta0 + np.pi) % (2 * np.pi) - np.pi
        # Use R_source (from snapped point) for arc length calculation
        d_exact = np.sqrt((R_source * d_theta)**2 + (z - z0)**2)
        err     = abs(node.dist - d_exact)
        max_err = max(max_err, err)
        sum_sq += err**2
        count  += 1

    l2_err = np.sqrt(sum_sq / count) if count > 0 else 0.0
    return max_err, l2_err



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

    plt.title("FMM - Plot of the levelsets")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()


"""
def plot_convergence(h_values, err_inf, method_name):

    h_arr = np.array(h_values)

    plt.figure(100, figsize=(8,6))
    
    plt.loglog(h_arr, err_inf, 'o-', label = method_name)

    plt.xlabel('h (mesh size)')
    plt.ylabel('Error')
    plt.title('FMM Convergence')
    plt.legend()
    plt.grid(True, which='both')

"""






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
    plt.show()


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
    plt.show()