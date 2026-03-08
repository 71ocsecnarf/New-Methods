import numpy as np
import gmsh
import matplotlib.pyplot as plt

from structures import NODE
from structures import ELEMENT
from structures import TRIAL_BAND
from structures import EDGE

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

        #*Here, i add all the elements a node belong to
        node_a.adjacent_triangles.append(element)
        node_b.adjacent_triangles.append(element)
        node_c.adjacent_triangles.append(element)

    print("Element list is done --> ok")
    return np.array(element_list)

def Make_EdgeList(element_list):
    edge_dict = {}  # key = (min_tag, max_tag)
    
    for elem in element_list:
        nodes = elem.nodes
        pairs = [(nodes[0], nodes[1]), 
                 (nodes[1], nodes[2]), 
                 (nodes[0], nodes[2])]
        
        for na, nb in pairs:
            key = (min(na.node_tag, nb.node_tag), 
                   max(na.node_tag, nb.node_tag))
            
            if key not in edge_dict:
                edge = EDGE(na, nb)
                edge_dict[key] = edge
            
            edge_dict[key].adjacent_triangles.append(elem)
            elem.edges.append(edge_dict[key])
    
    ##return list(edge_dict.values())
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
        if min(AB_AC, BC_BA, CA_CB) < -1e-6:
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


#!===================================================================== |
#!===================================================================== |
#!======================== FFM BELOW ================================== |
#!===================================================================== |
#!===================================================================== V

def Innit_FFM(target_coords, target_elem):
    trial_band = TRIAL_BAND()
    for n in target_elem.nodes:
        n.dist = np.linalg.norm(n.coords - target_coords)
        n.state = 'TRIAL'
        trial_band.add_node(n)
    return trial_band

def FMM(trial_band):
    print("=================================================")
    print("========== Running Fast Marching Method =========")
    print("=================================================\n")

    while not trial_band.is_empty():

        ui = trial_band.pop_closest()
        if ui is None: break
        ui.state = 'ALIVE'

        #* Looping over the neighbors
        for uj in ui.neighbors:
            if uj.state != 'ALIVE':

                new_dist = Update_Node_Distance(uj)
                
                if new_dist < uj.dist:
                    uj.dist = new_dist
                    trial_band.add_node(uj)

    print("FMM succesfully finished.\n")


def Update_Node_Distance(node):
    t_min = np.inf

    for elem in node.adjacent_triangles:
        
        C = node
        others = [n for n in elem.nodes if n != node]
        A, B = others[0], others[1]


        if A.state == 'ALIVE' and B.state == 'ALIVE':
            #!I make sure to use the paper convention, We want Ta < Tb and we look for Tc
            if(A.dist < B.dist):
                #t_local = Solve_Eikonal_Triangle(A.coords, B.coords, C.coords, A.dist, B.dist)
                t_local = Solve_Eikonal_Triangle_Order2(A.coords, B.coords, C.coords, A.dist, B.dist, 2)
            else:
                t_local = Solve_Eikonal_Triangle_Order2(A.coords, B.coords, C.coords, A.dist, B.dist, 2)
                t_local = Solve_Eikonal_Triangle(B.coords, A.coords, C.coords, B.dist, A.dist)
        elif A.state == 'ALIVE':
            t_local = A.dist + np.linalg.norm(C.coords - A.coords)
        elif B.state == 'ALIVE':
            t_local = B.dist + np.linalg.norm(C.coords - B.coords)
        else:   
            continue

        if t_local < t_min:
            t_min = t_local
                
    return t_min

def Solve_Eikonal_Triangle(A_coords, B_coords, C_coords, Ta, Tb):
    F = 1

    La = np.linalg.norm(B_coords - C_coords) 
    Lb = np.linalg.norm(A_coords - C_coords)  
    Lc = np.linalg.norm(A_coords - B_coords)  

    u = Tb - Ta  #* Ta <= Tb is always true when i call Solve_Eikonal_Triangle()

    CA = A_coords - C_coords
    CB = B_coords - C_coords
    cosTheta = np.dot(CA, CB) / (Lb * La)#*Cos of angle C
    sinTheta = np.sqrt(1 - cosTheta**2)

    #*2nd order equation from the paper
    coef_a = La**2 + Lb**2 - 2*La*Lb*cosTheta  # = Lc²
    coef_b = 2 * Lb * u * (La * cosTheta - Lb)
    coef_c = Lb**2 * (u**2 - F**2 * La**2 * sinTheta**2)

    delta = coef_b**2 - 4*coef_a*coef_c

    if delta >= 0:
        t = (-coef_b + np.sqrt(delta)) / (2 * coef_a)
        if t > u:  # upwind
            b_t_u_over_t = Lb * (t - u) / t
            lower = La * cosTheta
            upper = La / cosTheta if abs(cosTheta) > 1e-10 else np.inf
            if lower < b_t_u_over_t < upper:
                return Ta + t

    return min(Ta + Lb*F, Tb + La*F)   
    #*Otherwise, we move along the edge
    return min(Ta + Lb*F, Tb + La*F)

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

    plt.title("Isolignes de la distance (FMM)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()



# def Solve_Eikonal_Triangle_Sphere(A_coords, B_coords, C_coords, Ta, Tb):
#     F = 1

#     La = np.linalg.norm(B_coords - C_coords) 
#     Lb = np.linalg.norm(A_coords - C_coords)  
#     Lc = np.linalg.norm(A_coords - B_coords)  

#     u = Tb - Ta  #* Ta <= Tb is always true when i call Solve_Eikonal_Triangle_Sphere()
#     if u < 1e-6: u = -1e-12
#     CA = A_coords - C_coords
#     CB = B_coords - C_coords

#     CA_norm = np.linalg.norm(CA)

#     xa = Lb
#     ya = 0.0

#     xb = np.dot(CB, CA) / Lb
#     yb = np.sqrt(La**2 - xb**2)

#     z0 = (xb**2 + yb**2 + u**2 - xa**2 - ya**2) / (2 * u)
    
#     coef_a = 1.0
#     coef_b = -2 * z0
#     coef_c = -(xa**2 + ya**2)

#     delta = coef_b**2 - 4*coef_a*coef_c

#     if delta >= 0:
#         t = (-coef_b + np.sqrt(delta)) / (2*coef_a)
#         Tc = Ta + t
#         print(Tc, Ta, Tb)

#         if u < t:
#             return Tc
#         else:
#             return min(Lb*F + Ta, La*F + Tb)

#     #*Otherwise, we move along the edge
#     return min(Ta + Lb*F, Tb + La*F)

def Solve_Eikonal_Triangle_Order2(A_coords, B_coords, C_coords, Ta, Tb, kappa):
    F = 1.0
    
    CA = A_coords - C_coords
    CB = B_coords - C_coords
    Lb = np.linalg.norm(CA)  # AC
    La = np.linalg.norm(CB)  # BC
    
    xa = Lb
    xb = np.dot(CB, CA) / Lb
    yb_sq = La**2 - xb**2
    if yb_sq < 1e-12:
        return min(Ta + Lb*F, Tb + La*F)
    yb = np.sqrt(yb_sq)
    
    A_eff = Ta - (kappa/2) * xa**2
    B_eff = Tb - (kappa/2) * (xb**2 + yb**2)
    
    N = 1.0 - xb/xa
    M = B_eff - A_eff * (xb/xa)
    
    coef_a = 1/xa**2 + N**2/yb**2
    coef_b = -2*A_eff/xa**2 - 2*M*N/yb**2
    coef_c = A_eff**2/xa**2 + M**2/yb**2 - 1
    
    delta = coef_b**2 - 4*coef_a*coef_c
    
    if delta < 0:
        return min(Ta + Lb*F, Tb + La*F)
    
    Tc = (-coef_b + np.sqrt(delta)) / (2*coef_a)
    
    t = Tc - Ta
    u = Tb - Ta
    cosTheta = np.dot(CA, CB) / (Lb * La)

    if t > u:
        b_t_u_over_t = Lb * (t - u) / t
        lower = La * cosTheta
        upper = La / cosTheta if abs(cosTheta) > 1e-10 else np.inf
        if lower < b_t_u_over_t < upper:
            return Tc
    
    return min(Ta + Lb*F, Tb + La*F)