import numpy as np
import gmsh

from structures import NODE
from structures import ELEMENT

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
        AB = np.array([e.node_A.coords[0] - e.node_B.coords[0], e.node_A.coords[1] - e.node_B.coords[1]])
        AC = np.array([e.node_A.coords[0] - e.node_C.coords[0], e.node_A.coords[1] - e.node_C.coords[1]])
        BC = np.array([e.node_B.coords[0] - e.node_C.coords[0], e.node_B.coords[1] - e.node_C.coords[1]])

        AB_AC = AB @ AC
        BC_BA = BC @ (-AB)
        CA_CB = (-AC) @ (-BC)
        if min(AB_AC, BC_BA, CA_CB) < 0:
            e.IsObtuse = True
            counting += 1
    
    print(f"There are {counting} obtuse elements")
    print("Checking for obtuse element is done --> ok")

