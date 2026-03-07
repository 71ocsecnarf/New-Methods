import numpy as np
import gmsh
import matplotlib.pyplot as plt

from structures import NODE
from structures import ELEMENT
from structures import TRIAL_BAND

#!==================================================================== |
#!==================================================================== |
#!========================= Preprocessing============================= |
#!==================================================================== |
#!==================================================================== V

def Make_NodeList_NodeDictionary(filename, tag_to_node_obj_dic):

    if not gmsh.isInitialized():
        gmsh.initialize()
    
    print("\n")
    print("=================================================")
    print("================Making node list=================")
    print("=================================================")

    #!node_tags, it is like ids, written as (1, 2, 3...) --> starts at 1!
    #!coords, written as [x1, y1, z1, x2, y2, z2...]
    node_tags, coords, _ = gmsh.model.mesh.getNodes() #* third return is parametricCoords, if you write "_" it means that you don't want it

    #*New dimensions are (N, 3)
    points = coords.reshape(-1, 3)

    nodes_list = []
    
    for i, tag in enumerate(node_tags):
        new_node = NODE(points[i], tag, i)
        nodes_list.append(new_node)
        tag_to_node_obj_dic[tag] = new_node
    
    #*node_tags_per_elem looks like [ [n1, n2, n3, n4, n5, n6...] ]
    _, _, node_tags_per_elem = gmsh.model.mesh.getElements(2)
    triangles_tags = node_tags_per_elem[0].reshape(-1, 3)

    #*tri = [n1 n2 n3], [n4 n5 n6] and so on
    for tri in triangles_tags:

        #*Get back the NODE object with the dictionary
        node_a = tag_to_node_obj_dic[tri[0]]
        node_b = tag_to_node_obj_dic[tri[1]]
        node_c = tag_to_node_obj_dic[tri[2]]
        
        #*In a triangle, A has B and C as neighboors and so on
        node_a.neighbors.update([node_b, node_c])
        node_b.neighbors.update([node_a, node_c])
        node_c.neighbors.update([node_a, node_b])

    print("Node list is done --> ok")
    return np.array(nodes_list)


def Make_ElementList(filename, tag_to_node_obj_dic):

    if not gmsh.isInitialized():
        gmsh.initialize()
    
    gmsh.open(filename)
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
        # Add the triangle to the adjacent triangle list of each node
        #! I do not know if we will need it
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

    #*Step 1, i look for the closest node 
    all_coords = np.array([n.coords for n in node_list])
    distances = np.linalg.norm(all_coords - target_coords, axis=1)
    closest_node_idx = np.argmin(distances)
    closest_node = node_list[closest_node_idx]

    #*Then, i look for the closest element in the neighbors of the closest node
    best_dist = np.inf
    target_elem = None
    best_normal = None
    for elem in closest_node.adjacent_triangles:
        dist_to_elem, normal = Compute_Dist_Point_Triangle(target_coords, elem)
        if dist_to_elem < best_dist:
            best_dist = dist_to_elem
            target_elem = elem
            best_normal = normal

    
    for node in target_elem.nodes:
        node.dist = np.linalg.norm(node.coords - target_coords)
        print(f"Node {node.node_tag} innitialized at d={node.dist:.4f}")

    projected_point = target_coords - best_dist*best_normal
    return projected_point, target_elem


def compute_err (node_list, source_coord):
    # I thought to compute the error between the source and a random point, 
    # but AI suggested me to compute the global error, which makes sense

    #! Now the error is computed using direct lines since we are on a plane,
    #! when we will go to 3D surfaces we will need to modify this function

    errors = []
    errors_rel = []

    for node in node_list:
        # Exact distance computation
        ex_dist = np.linalg.norm(node.coords - source_coord)

        # Error computation - we can compute it in many ways
        err_i = abs(ex_dist - node.dist)

        errors.append(err_i)
        errors_rel.append(err_i/ex_dist)

    # Max error - absolute value
    err = np.max(errors) 
    # Max relative error
    err_rel = np.max(errors_rel)
    # Norm L2 err
    e_l2 = np.sqrt(np.mean(np.array(errors)**2))

    return err, err_rel, e_l2
    


















######################################
## ------------- PLOTS ------------ ##
######################################

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
    plt.clabel(lines, inline=True, fontsize=8) # Ajoute les valeurs sur les lignes

    plt.triplot(x, y, triangles, color='black', alpha=0.1, linewidth=0.5)

    plt.title("FMM - Plot of the levelsets")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

def plot_convergence(h_values, err_inf, method_name):
    # Plot of the convergence graphs
    # input the name to use it multiple times when we will need to compare multiple methods

    h_arr = np.array(h_values) #convert it to an array from a phtyon list
    # I do not know why it is necessary

    plt.figure(100, figsize=(8,6))
    
    plt.loglog(h_arr, err_inf, 'o-', label = method_name)

    # Print some reference slopes
    plt.loglog(h_arr, h_arr,      '--', color='gray', label='O(h)')
    plt.loglog(h_arr, h_arr**2,   '--', color='black', label='O(h²)')

    plt.xlabel('h (mesh size)')
    plt.ylabel('Error')
    plt.title('FMM Convergence')
    plt.legend()
    plt.grid(True, which='both')
    plt.show()
