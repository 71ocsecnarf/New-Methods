import numpy as np
import os
import matplotlib.pyplot as plt
import time

import gmsh
import solver
import fmm_solver
import mesh_generation


# ============================================================
#   PARAMETRI DA MODIFICARE
# ============================================================
MESH_TYPE = 'cylinder'   # 'square_surface' oppure 'cylinder'
N_VALUES  = [5 * 2**i for i in range(8)]
# ============================================================


def get_config(mesh_type, current_dir):
    """
    Restituisce un dizionario con tutti i parametri dipendenti dalla geometria.
    Per aggiungere una nuova geometria, aggiungere un nuovo elif qui.
    """

    if mesh_type == 'square_surface':
        L = 1.0
        return {
            'source_coord'   : np.array([0.2, 0.3, 0.0]),
            'generate_mesh'  : lambda N: mesh_generation.generate_mesh(
                                    N, L, mesh_type,
                                    os.path.join(current_dir, "square_surface.msh")),
            'get_output_path': lambda N: os.path.join(current_dir, "square_surface.msh"),
            'get_h'          : lambda N: L / (N - 1),
            'compute_err'    : lambda node_list, true_source: solver.compute_err(
                                    node_list, true_source),
            'plot_title'     : 'FMM Convergence - Square',
            'save_name'      : 'ConvStudySquare.pdf',
        }

    elif mesh_type == 'cylinder':
        R = 0.5
        H = 1.0
        return {
            'source_coord'   : np.array([R, 0.0, 0.0]),
            'generate_mesh'  : lambda N: mesh_generation.generate_cylinder_mesh(
                                    N, R, H, current_dir),
            'get_output_path': lambda N: os.path.join(current_dir, "cylinder_surface.msh"),
            'get_h'          : lambda N: np.pi * R / (N - 1),
            'compute_err'    : lambda node_list, true_source: compute_err_cylinder(
                                    node_list, true_source, R),
            'plot_title'     : 'FMM Convergence - Cylinder',
            'save_name'      : 'ConvStudyCylinder.pdf',
        }

    else:
        raise ValueError(f"Unknown mesh_type: '{mesh_type}'")


def main(mesh_type=MESH_TYPE, N_values=N_VALUES):

    plt.close('all')
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # ----> A - Carica la configurazione della geometria ----
    cfg = get_config(mesh_type, current_dir)

    # ----> B - Inizializzazione liste errori ----
    h_values       = []
    errors_fmm     = [];  errors_l2_fmm  = []
    errors_circ    = [];  errors_l2_circ = []

    # ----> C - Loop su N ----
    for i, N in enumerate(N_values):

        print(f"\n{'='*50}")
        print(f"[{mesh_type}]  N={N}  |  Iteration {i+1}/{len(N_values)}")
        print(f"{'='*50}")

        # 1 ---> Generazione mesh
        cfg['generate_mesh'](N)
        output_path = cfg['get_output_path'](N)

        # 2 ---> Apertura mesh
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.open(output_path)

        # 3 ---> Node list ed element list
        tag_to_node_obj_dic = {}
        node_list    = solver.Make_NodeList_NodeDictionary(output_path, tag_to_node_obj_dic)
        element_list = solver.Make_ElementList(output_path, tag_to_node_obj_dic)

        # 4 ---> Inizializzazione sorgente
        true_source, source_nodes = solver.Innit_Origin_Point(
                                        cfg['source_coord'], node_list)

        # 5a ---> Standard FMM
        print('\n------------ Standard FMM ------------')
        t_start = time.time()
        fmm_solver.fmm_algorithm(node_list, source_nodes)
        t_end = time.time()
        print(f"Elapsed time: {t_end - t_start:.4f} s")

        # err_fmm, e_l2_fmm = cfg['compute_err'](node_list, true_source)[:2]
        result_fmm  = cfg['compute_err'](node_list, true_source)
        err_fmm,  e_l2_fmm  = result_fmm[0],  result_fmm[-1]
        print(f"Max error = {err_fmm:.6e}  |  L2 error = {e_l2_fmm:.6e}")
        errors_fmm.append(err_fmm)
        errors_l2_fmm.append(e_l2_fmm)

        # 5b ---> Circular FMM
        solver.Reset_Node_State(node_list)
        true_source, source_nodes = solver.Innit_Origin_Point(
                                        cfg['source_coord'], node_list)

        print('\n------------ Circular FMM ------------')
        t_start = time.time()
        fmm_solver.fmm_algorithm_circ(node_list, source_nodes)
        t_end = time.time()
        print(f"Elapsed time: {t_end - t_start:.4f} s")

        # err_circ, e_l2_circ = cfg['compute_err'](node_list, true_source)[:2]
        result_circ = cfg['compute_err'](node_list, true_source)
        err_circ, e_l2_circ = result_circ[0], result_circ[-1]
        print(f"Max error = {err_circ:.6e}  |  L2 error = {e_l2_circ:.6e}")
        errors_circ.append(err_circ)
        errors_l2_circ.append(e_l2_circ)

        # 6 ---> h e cleanup
        h_values.append(cfg['get_h'](N))
        gmsh.finalize()

    # ----> D - Post process ----
    plot_convergence(h_values, errors_l2_fmm, errors_l2_circ,
                     cfg['plot_title'], cfg['save_name'])


def plot_convergence(h_values, errors_l2_fmm, errors_l2_circ, title, save_name):

    h_arr       = np.array(h_values[::-1])
    l2_fmm_arr  = np.array(errors_l2_fmm[::-1])
    l2_circ_arr = np.array(errors_l2_circ[::-1])

    print("\n\nConvergence order (L2) - Standard FMM:")
    for i in range(1, len(h_values)):
        order = np.log(errors_l2_fmm[i] / errors_l2_fmm[i-1]) / \
                np.log(h_values[i] / h_values[i-1])
        print(f"  h={h_values[i]:.4f}  |  L2={errors_l2_fmm[i]:.2e}  |  order ≈ {order:.2f}")

    print("\n\nConvergence order (L2) - Circular FMM:")
    for i in range(1, len(h_values)):
        order = np.log(errors_l2_circ[i] / errors_l2_circ[i-1]) / \
                np.log(h_values[i] / h_values[i-1])
        print(f"  h={h_values[i]:.4f}  |  L2={errors_l2_circ[i]:.2e}  |  order ≈ {order:.2f}")

    plt.figure(figsize=(8, 6))
    plt.loglog(h_arr, l2_fmm_arr,  'o-', label='FMM (L2)')
    plt.loglog(h_arr, l2_circ_arr, 'o-', label='HFMM (L2)')
    plt.loglog(h_arr, h_arr,       '--', color='gray',  label='O(h)')
    plt.loglog(h_arr, h_arr**2,    '--', color='black', label='O(h²)')
    plt.xlabel('h (mesh size)')
    plt.ylabel('L2 Error')
    plt.title(title)
    plt.legend()
    plt.grid(True, which='both')
    plt.gca().invert_xaxis()
    plt.tight_layout()
    plt.savefig(save_name)
    plt.show()


def compute_err_cylinder(node_list, true_source, R):
    """
    Geodesic distance on a cylinder = straight line on the unrolled rectangle:
        d = sqrt( (R * delta_theta)^2 + delta_z^2 )
    Returns (max_err, l2_err) — stesso formato di solver.compute_err()
    """
    x0, y0, z0 = true_source
    theta0 = np.arctan2(y0, x0)

    max_err = 0.0
    sum_sq  = 0.0
    count   = 0

    for node in node_list:
        if node.dist == float('inf'):
            continue
        x, y, z   = node.coords
        theta     = np.arctan2(y, x)
        d_theta   = (theta - theta0 + np.pi) % (2 * np.pi) - np.pi
        d_exact   = np.sqrt((R * d_theta)**2 + (z - z0)**2)
        err       = abs(node.dist - d_exact)
        max_err   = max(max_err, err)
        sum_sq   += err**2
        count    += 1

    l2_err = np.sqrt(sum_sq / count) if count > 0 else 0.0
    return max_err, l2_err


if __name__ == "__main__":
    main()