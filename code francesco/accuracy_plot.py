import numpy as np
import os
import matplotlib.pyplot as plt

import gmsh
import solver
import fmm_solver
import mesh_generation


def main():

    # ----> A - CLOSE ALL TABS - <----
    plt.close('all')


    # ----> B - INITIALIZATION - <----
    N_values = [5 * 2**i for i in range(6)]
    source_coord = np.array([0.0, 0.0, 0.0])
    mesh_type = 'square_surface'    

    current_dir  = os.path.dirname(os.path.abspath(__file__))
    output_path  = os.path.join(current_dir, "square_surface.msh")

    h_values = []
    errors = []
    rel_errors = []
    errors_l2 = []

    # ----> C - LOOP - <----

    for N in N_values:

        # Print the current computation
        print(f"\n{'='*50}")
        print(f"Running N={N}  -->  h={1.0/(N-1):.4f}")
        print(f"{'='*50}")

        # 1 ---> Mesh Generation
        L = 1     # Square size
        mesh_generation.generate_mesh(N, L, mesh_type, output_path)

        # 2 ---> Mesh Opening
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0) # To avoid having gmsh msg in the terminal
        gmsh.open(output_path)

        # 3 ---> Constraction of node and element lists
        #*The node tag is the key, the value is the NODE object of that node
        tag_to_node_obj_dic = {}#*dictionary, works with GMSH tag
        node_list = solver.Make_NodeList_NodeDictionary(output_path, tag_to_node_obj_dic)
        element_list = solver.Make_ElementList(output_path, tag_to_node_obj_dic)

        solver.Check_Obtuse_triangles(element_list)

        # 4 ---> Source initialization
        source_point, target_elem = solver.Innit_Origin_Point(source_coord, node_list)
        source_nodes = target_elem.nodes

        # 5 ---> FMM Algorithm
        fmm_solver.fmm_algorithm(node_list, source_nodes)

        # 6 ---> Error Computation
        err, err_rel, e_l2 = solver.compute_err(node_list,source_coord)
        print(f"Max error = {err:.6e}  | Max Relative Error = {err_rel}   | L2 error = {e_l2:.6e}")

        # 7 ---> Save the results
        h_values.append(1.0/(N-1)) #! I do not know how to properly compute h, AI says like this
        errors.append(err)
        rel_errors.append(err_rel)
        errors_l2.append(e_l2)

        # 8 ---> Mesh Reset
        gmsh.finalize()


    # ----> D - POST PROCESS - <----
    '''
    print("\n\nConvergence order (L∞):")
    for i in range(1, len(h_values)):
        order = np.log(errors_inf[i] / errors_inf[i-1]) / np.log(h_values[i] / h_values[i-1])
        print(f"  h={h_values[i]:.4f}  |  e={errors_inf[i]:.2e}  |  order ≈ {order:.2f}")
    '''

    method_name = 'FMM'
    solver.plot_convergence(h_values, errors_l2, method_name)
    




if __name__ == "__main__":
    main()