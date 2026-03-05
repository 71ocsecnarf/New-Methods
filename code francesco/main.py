import numpy as np
import os
import matplotlib.pyplot as plt

import gmsh
import solver
import fmm_solver

def main():
    gmsh.initialize()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    filename = current_dir + "/inputfiles/square_surface.msh"
    gmsh.open(filename)
    #*To avoid having gmsh msg in the terminal
    gmsh.option.setNumber("General.Terminal", 0)

    #todo--> do we need to keep this dictionnary ??
    #*The node tag is the key, the value is the NODE object of that node
    tag_to_node_obj_dic = {}#*dictionary, works with GMSH tag
    node_list = solver.Make_NodeList_NodeDictionary(filename, tag_to_node_obj_dic)
    element_list = solver.Make_ElementList(filename, tag_to_node_obj_dic)

    #*======================================= |
    #*====USED TO TEST THE NODE LIST========= |
    #*======================================= V
    # plt.figure()
    # for n in node_list:
    #     plt.scatter(n.coords[0], n.coords[1], color = 'black')

    #     #! it is just for checking, each edge is plotted twice... frome node A --> B then B-->A but whatever, it is just to see 
    #     #!if my node_list works well...
    #     for neighbor in n.neighbors:
    #         plt.plot([n.coords[0], neighbor.coords[0]], [n.coords[1], neighbor.coords[1]], color = "red")
    # plt.show()
    #*======================================= ^
    #*====USED TO TEST THE NODE LIST========= |
    #*======================================= |

    solver.Check_Obtuse_triangles(element_list)
    target_coords = np.array([0.6, 0.85, 0.1])
    source_point, target_elem = solver.Innit_Origin_Point(target_coords, node_list)

    #*========================================== |
    #*====USED TO TEST THE ELEMENT LIST========= |
    #*========================================== V
    # plt.figure()
    # size = 0.5
    # for e in element_list:
    #     x = []
    #     y = []
    #     for n in e.nodes:
    #         plt.scatter(n.coords[0], n.coords[1], color = 'black')
    #         x.append(n.coords[0])
    #         y.append(n.coords[1])
            
    #     plt.plot([x[0], x[1]], [y[0], y[1]], color = 'red', linewidth = size)
    #     plt.plot([x[0], x[2]], [y[0], y[2]], color = 'red', linewidth = size)
    #     plt.plot([x[1], x[2]], [y[1], y[2]], color = 'red', linewidth = size)
    
    # plt.scatter(source_point[0], source_point[1], color = 'gold')

    # plt.show()
    #*========================================== ^
    #*====USED TO TEST THE ELEMENT LIST========= |
    #*========================================== |


    #! Innit the FFM algorithm:
    trial_band = solver.Innit_FFM(target_coords, target_elem)
    solver.FMM(trial_band)
    
    print("Calcul terminé, génération du tracé...")
    solver.Plot_Isolines(node_list, element_list)
    

    gmsh.finalize()

if __name__ == "__main__":
    main()