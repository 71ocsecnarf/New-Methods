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
    gmsh.option.setNumber("General.Terminal", 0)

    tag_to_node_obj_dic = {}
    node_list = solver.Make_NodeList_NodeDictionary(filename, tag_to_node_obj_dic)
    element_list = solver.Make_ElementList(filename, tag_to_node_obj_dic)

    solver.Check_Obtuse_triangles(element_list)
    target_coords = np.array([0.0, 0.0, 0.0])

    source_point, source_nodes = solver.Innit_Origin_Point(target_coords, node_list)
    snapped_coord = source_nodes[0].coords

    fmm_solver.fmm_algorithm_circ(node_list, source_nodes)
    
    print("Computation Completed, Trace Generation ...")
    solver.Plot_Isolines(node_list, element_list)
    
    gmsh.finalize()

if __name__ == "__main__":
    main()
