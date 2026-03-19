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
    #! There was a problem when the source was not on a node of a mesh, 
    #! I used AI to fix it

    all_coords = np.array([n.coords for n in node_list])
    distances_to_target = np.linalg.norm(all_coords - target_coords, axis=1)
    closest_node_idx = np.argmin(distances_to_target)
    closest_node = node_list[closest_node_idx]
 
    snap_err = np.linalg.norm(closest_node.coords - target_coords)
    closest_node.dist  = snap_err
    closest_node.state = 'ALIVE'
 
    for neighbor in closest_node.neighbors:
        neighbor.dist  = np.linalg.norm(neighbor.coords - target_coords)
        neighbor.state = 'TRIAL'
 
    print(f"Source node {closest_node.node_tag} @ {closest_node.coords[:2]}")
    print(f"  Snapping error = {snap_err:.6e}")
 
    return target_coords, [closest_node]


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

def plot_convergence(h_values, err_inf, method_name):

    h_arr = np.array(h_values)

    plt.figure(100, figsize=(8,6))
    
    plt.loglog(h_arr, err_inf, 'o-', label = method_name)

    plt.xlabel('h (mesh size)')
    plt.ylabel('Error')
    plt.title('FMM Convergence')
    plt.legend()
    plt.grid(True, which='both')
