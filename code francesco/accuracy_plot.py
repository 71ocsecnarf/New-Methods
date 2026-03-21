import numpy as np
import os
import matplotlib.pyplot as plt
import time

import gmsh
import solver
import fmm_solver
import mesh_generation


def main():

    # ----> A - CLOSE ALL TABS - <----
    plt.close('all')

    # ----> B - INITIALIZATION - <----
    N_values = [5 * 2**i for i in range(8)]
    source_coord = np.array([0.0, 0.0, 0.0])
    mesh_type = 'square_surface'    

    current_dir  = os.path.dirname(os.path.abspath(__file__))
    output_path  = os.path.join(current_dir, "square_surface.msh")

    h_values = []
    # Standard FMM
    errors_fmm = []
    rel_errors_fmm = []
    errors_l2_fmm = []
    # High Order FMM
    errors_circ = []
    rel_errors_circ = []
    errors_l2_circ = []

    # ----> C - LOOP - <----

    for i, N in enumerate(N_values):

        print(f"\n{'='*50}")
        print(f"Running N={N}  -->  h={1.0/(N-1):.4f}, Iteration {i+1}")
        print(f"{'='*50}")

        # 1 ---> Mesh Generation
        L = 1
        mesh_generation.generate_mesh(N, L, mesh_type, output_path)

        # 2 ---> Mesh Opening
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.open(output_path)

        # 3 ---> Constraction of node and element lists
        tag_to_node_obj_dic = {}
        node_list = solver.Make_NodeList_NodeDictionary(output_path, tag_to_node_obj_dic)
        element_list = solver.Make_ElementList(output_path, tag_to_node_obj_dic)

        # 4 ---> Source initialization
        true_source, source_nodes = solver.Innit_Origin_Point(source_coord, node_list)

        # 5a ---> Standard FMM Algorithm
        print()
        print('------------ Standard FMM ------------')
        t_start = time.time()
        fmm_solver.fmm_algorithm(node_list, source_nodes)
        t_end = time.time()
        print(f"Elapsed time: {t_end - t_start:.4f} s")

        err_fmm, err_rel_fmm, e_l2_fmm = solver.compute_err(node_list, true_source)
        print(f"Max error = {err_fmm:.6e}  | Max Relative Error = {err_rel_fmm}   | L2 error = {e_l2_fmm:.6e}")

        errors_fmm.append(err_fmm)
        rel_errors_fmm.append(err_rel_fmm)
        errors_l2_fmm.append(e_l2_fmm)

        # 5b ---> Reset the nodes state and use the higher order fmm
        solver.Reset_Node_State(node_list)
        true_source, source_nodes = solver.Innit_Origin_Point(source_coord, node_list)

        print()
        print('------------ Circular FMM ------------')
        t_start = time.time()
        fmm_solver.fmm_algorithm_circ(node_list, source_nodes)
        t_end = time.time()
        print(f"Elapsed time: {t_end - t_start:.4f} s")

        err_circ, err_rel_circ, e_l2_circ = solver.compute_err(node_list, true_source)
        print(f"Max error = {err_circ:.6e}  | Max Relative Error = {err_rel_circ}   | L2 error = {e_l2_circ:.6e}")

        errors_circ.append(err_circ)
        rel_errors_circ.append(err_rel_circ)
        errors_l2_circ.append(e_l2_circ)

        # 8 ---> Mesh Reset
        h_values.append(L/(N-1))
        gmsh.finalize()

    h_arr       = np.array(h_values[::-1])
    l2_fmm_arr  = np.array(errors_l2_fmm[::-1])
    l2_circ_arr = np.array(errors_l2_circ[::-1])

    # ----> D - POST PROCESS - <----
    
    print("\n\nConvergence order (L2) - Standard FMM:")
    for i in range(1, len(h_values)):
        order = np.log(errors_fmm[i] / errors_fmm[i-1]) / np.log(h_values[i] / h_values[i-1])
        print(f"  h={h_values[i]:.4f}  |  e={errors_fmm[i]:.2e}  |  order ≈ {order:.2f}")
 
    print("\n\nConvergence order (L2) - High Order FMM:")
    for i in range(1, len(h_values)):
        order = np.log(errors_circ[i] / errors_circ[i-1]) / np.log(h_values[i] / h_values[i-1])
        print(f"  h={h_values[i]:.4f}  |  e={errors_circ[i]:.2e}  |  order ≈ {order:.2f}")

    plt.figure(100, figsize=(8, 6))
    plt.loglog(h_arr, l2_fmm_arr,  'o-', label='FMM (L2)')
    plt.loglog(h_arr, l2_circ_arr, 'o-', label='HFMM (L2)')
    plt.loglog(h_arr, h_arr,       '--', color='gray',  label='O(h)')
    plt.loglog(h_arr, h_arr**2,    '--', color='black', label='O(h\u00b2)')
    plt.xlabel('h (mesh size)')
    plt.ylabel('L2 Error')
    plt.title('FMM Convergence')
    plt.legend()
    plt.grid(True, which='both')
    plt.gca().invert_xaxis()
    plt.show()

if __name__ == "__main__":
    main()
