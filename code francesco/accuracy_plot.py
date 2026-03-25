import numpy as np
import os
import matplotlib.pyplot as plt
import time

import gmsh
import solver
import fmm_solver
import mesh_generation


# ============================================================
# ------------------------ Parameters ------------------------
# ============================================================

MESH_TYPE = 'square_surface'   # 'square_surface' or 'cylinder'
N_VALUES  = [5 * 2**i for i in range(8)]

# For cylinder: source specified as (theta_deg, z) in the unrolled domain
# theta_deg in [-90, +90],  z in [0, H]
SOURCE_THETA_DEG = 90    # angle in degrees: 0 = front of cylinder (x=R, y=0)
SOURCE_Z         = 0.1   # height along the cylinder

# For square_surface: source specified as (x, y)
SOURCE_XY = np.array([0.2, 0.3])

# ------------------------------------------------------------


def make_source_coords(mesh_type, R=0.5, H=1.0):
    """
    Convert user-friendly source specification to 3D coordinates.
    - Cylinder: (theta_deg, z) --> (R*cos(theta), R*sin(theta), z)
      theta_deg in [-90, +90] degrees
    - Square:   (x, y)        --> (x, y, 0)
    """
    if mesh_type == 'cylinder':
        theta = np.deg2rad(SOURCE_THETA_DEG)   # convert degrees to radians
        x = R * np.cos(theta)
        y = R * np.sin(theta)
        z = SOURCE_Z
        return np.array([x, y, z])

    elif mesh_type == 'square_surface':
        return np.array([SOURCE_XY[0], SOURCE_XY[1], 0.0])

    else:
        raise ValueError(f"Unknown mesh_type: '{mesh_type}'")


def get_config(mesh_type, current_dir):
    """
    It gives a dictionary with all parameter that depends on geometry.
    To add a new geometry, add a new elif here.
    """

    if mesh_type == 'square_surface':
        L = 1.0
        return {
            'source_coord'   : make_source_coords(mesh_type),
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
            'source_coord'   : make_source_coords(mesh_type, R=R, H=H),
            'generate_mesh'  : lambda N: mesh_generation.generate_cylinder_mesh(
                                    N, R, H, current_dir),
            'get_output_path': lambda N: os.path.join(current_dir, "cylinder_surface.msh"),
            'get_h'          : lambda N: np.pi * R / (N - 1),
            'compute_err'    : lambda node_list, true_source: solver.compute_err_cylinder(
                                    node_list, true_source, R),
            'plot_title'     : 'FMM Convergence - Cylinder',
            'save_name'      : 'ConvStudyCylinder.pdf',
        }

    else:
        raise ValueError(f"Unknown mesh_type: '{mesh_type}'")


def main(mesh_type=MESH_TYPE, N_values=N_VALUES):

    plt.close('all')
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # ----> A - Load Geometry configuration ----
    cfg = get_config(mesh_type, current_dir)

    # ----> B - Error Lists initialization ----
    h_values       = []
    errors_fmm     = [];  errors_l2_fmm  = []
    errors_circ    = [];  errors_l2_circ = []

    # ----> C - Loop over N ----
    for i, N in enumerate(N_values):

        print(f"\n{'='*50}")
        print(f"[{mesh_type}]  N={N}  |  Iteration {i+1}/{len(N_values)}")
        print(f"{'='*50}")

        # 1 ---> Mesh generation
        cfg['generate_mesh'](N)
        output_path = cfg['get_output_path'](N)

        # 2 ---> Mesh Opening
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.open(output_path)

        # 3 ---> Node list and element list
        tag_to_node_obj_dic = {}
        node_list    = solver.Make_NodeList_NodeDictionary(output_path, tag_to_node_obj_dic)
        element_list = solver.Make_ElementList(output_path, tag_to_node_obj_dic)

        # 4 ---> Source Initialization
        true_source, source_nodes = solver.Innit_Origin_Point(
                                        cfg['source_coord'], node_list)
        ## !!! true_source contains the coordinates snapped at the nearest
        ## !   node - used to compute the error in a standard way

        # 5a ---> Standard FMM
        print('\n------------ Standard FMM ------------')
        t_start = time.time()
        fmm_solver.fmm_algorithm(node_list, source_nodes)
        t_end = time.time()
        print(f"Elapsed time: {t_end - t_start:.4f} s")

        # Use the snapped coordinated for the error computations
        snapped_source = source_nodes[0].coords
        result_fmm  = cfg['compute_err'](node_list, snapped_source)
        err_fmm,  e_l2_fmm  = result_fmm[0],  result_fmm[-1]
        print(f"Max error = {err_fmm:.6e}  |  L2 error = {e_l2_fmm:.6e}")
        errors_fmm.append(err_fmm)
        errors_l2_fmm.append(e_l2_fmm)

        # 5b ---> Circular FMM
        solver.Reset_Node_State(node_list)
        true_source, source_nodes = solver.Innit_Origin_Point(
                                        cfg['source_coord'], node_list)
        snapped_source = source_nodes[0].coords

        print('\n------------ Circular FMM ------------')
        t_start = time.time()
        fmm_solver.fmm_algorithm_circ(node_list, source_nodes)
        t_end = time.time()
        print(f"Elapsed time: {t_end - t_start:.4f} s")

        result_circ = cfg['compute_err'](node_list, snapped_source)
        err_circ, e_l2_circ = result_circ[0], result_circ[-1]
        print(f"Max error = {err_circ:.6e}  |  L2 error = {e_l2_circ:.6e}")
        errors_circ.append(err_circ)
        errors_l2_circ.append(e_l2_circ)

        # 6 ---> h and cleanup
        h_values.append(cfg['get_h'](N))
        gmsh.finalize()

    # ----> D - Post process ----
    solver.plot_convergence(h_values, errors_l2_fmm, errors_l2_circ,
                     cfg['plot_title'], cfg['save_name'])



if __name__ == "__main__":
    main()