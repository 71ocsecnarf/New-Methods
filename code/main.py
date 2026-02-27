import numpy as np
import os
import matplotlib.pyplot as plt

import gmsh
import solver

def main():
    gmsh.initialize()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    filename = current_dir + "/inputfiles/square_surface.msh"

    #todo--> do we need to keep this dictionnary ??
    #*The node tag is the key, the value is the NODE object of that node
    tag_to_node_obj_dic = {}#*dictionary, works with GMSH tag
    node_list = solver.Make_NodeList_NodeDictionary(filename, tag_to_node_obj_dic)
    element_list = solver.Make_ElementList(filename, tag_to_node_obj_dic)

    #*======================================= |
    #*====USED TO TEST THE NODE LIST========= |
    #*======================================= V
    plt.figure()
    for n in node_list:
        plt.scatter(n.coords[0], n.coords[1], color = 'black')

        #! it is just for checking, each edge is plotted twice... frome node A --> B then B-->A but whatever, it is just to see 
        #!if my node_list works well...
        for neighbor in n.neighbors:
            plt.plot([n.coords[0], neighbor.coords[0]], [n.coords[1], neighbor.coords[1]], color = "red")
    plt.show()
    #*======================================= ^
    #*====USED TO TEST THE NODE LIST========= |
    #*======================================= |


    #*========================================== |
    #*====USED TO TEST THE ELEMENT LIST========= |
    #*========================================== V
    plt.figure()
    for e in element_list:
        plt.scatter(e.node_A.coords[0], e.node_A.coords[1], color = 'black')
        plt.scatter(e.node_B.coords[0], e.node_B.coords[1], color = 'black')
        plt.scatter(e.node_C.coords[0], e.node_C.coords[1], color = 'black')

        plt.plot([e.node_A.coords[0], e.node_B.coords[0]], [e.node_A.coords[1], e.node_B.coords[1]], color = 'red')
        plt.plot([e.node_A.coords[0], e.node_C.coords[0]], [e.node_A.coords[1], e.node_C.coords[1]], color = 'red')
        plt.plot([e.node_C.coords[0], e.node_B.coords[0]], [e.node_C.coords[1], e.node_B.coords[1]], color = 'red')
        

    plt.show()
    #*========================================== ^
    #*====USED TO TEST THE ELEMENT LIST========= |
    #*========================================== |
    gmsh.finalize()

if __name__ == "__main__":
    main()