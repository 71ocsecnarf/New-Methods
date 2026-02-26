import numpy as np
import gmsh
import solver

def main():
    gmsh.initialize()
    filename = "inputfiles/cylinder_surface.msh"

    #todo--> do we need to keep this dictionnary ??
    #*The node tag is the key, the value is the NODE object of that node
    tag_to_node_obj_dic = {}#*dictionary, works with GMSH tag
    node_list = solver.Make_NodeList_NodeDictionary(filename, tag_to_node_obj_dic)



    gmsh.finalize()

if __name__ == "__main__":
    main()