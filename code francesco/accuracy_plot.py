import numpy as np
import os
import json
import matplotlib.pyplot as plt
import time

import gmsh
import solver
import fmm_solver


# ============================================================
# ------------------------ Parameters ------------------------
# ============================================================

MESH_TYPE = 'square_surface'   # 'square_surface', 'l_shape', 'hole', 'cylinder', 'l_cylinder'
#N_VALUES = [81]

# -------------------------------------------------------------------------
# ----------------------- Cylinder and square meshes ----------------------
# -------------------------------------------------------------------------
N_VALUES = [5 * 2**i for i in range(8)] # Generated till 8
# -------------------------------------------------------------------------


# -------------------------------------------------------------------------
# --- Non-obtuse meshes for L-shape: [5, 11, 19, 40, 81, 160, 319, 641] ---
# -------------------------------------------------------------------------
#N_VALUES = [5, 11, 19, 40, 81, 160, 319]
# -------------------------------------------------------------------------

# -------------------------------------------------------------------------
# --- Non-obtuse meshes for hole: [6, 9, 21, 34, 76, 152, 312]  ---
# -------------------------------------------------------------------------
#N_VALUES = [6, 9, 21, 34, 76, 152, 312] # 312 has 2 obtuse triangles
# -------------------------------------------------------------------------


# -------------------------------------------------------------------------
# --- Non-obtuse meshes for L_cylinder: [9, 10, 18, 41, 77, 161, 321, 647]  --- 
# -------------------------------------------------------------------------
#N_VALUES = [9, 10, 18, 41, 77, 161, 321] 
# -------------------------------------------------------------------------


# Cylinder / l_cylinder source: (theta_deg, z)
SOURCE_THETA_DEG = 0.0
SOURCE_Z         = 0.0

# 2-D geometries source: (x, y)
SOURCE_XY = np.array([0.0, 0.0])

# Probe points for pointwise convergence tracking (2-D geometries only)
PROBE_POINTS = [
    np.array([1.0, 1.0]),
    np.array([0.8, 1.0]),
    np.array([1.0, 0.8]),
]

# ============================================================


# ============================================================
# -------------------- Mesh index helpers --------------------
# ============================================================

def load_index(mesh_type, current_dir):
    """
    Load the index.json file for the given mesh_type.
    Raises FileNotFoundError with a clear message if it does not exist.
    """
    index_path = os.path.join(current_dir, 'meshes', mesh_type, 'index.json')
    if not os.path.exists(index_path):
        raise FileNotFoundError(
            f"No mesh index found at '{index_path}'.\n"
            f"Run generate_meshes.py first to generate meshes for '{mesh_type}'."
        )
    with open(index_path, 'r') as f:
        return json.load(f)


def get_mesh_path(mesh_type, N, index, current_dir):
    """
    Return the absolute path to the .msh file for a given N.
    Raises KeyError with a clear message if N is not in the index.
    """
    key = str(N)
    if key not in index:
        raise KeyError(
            f"N={N} not found in the mesh index for '{mesh_type}'.\n"
            f"Available N values: {sorted(int(k) for k in index.keys())}\n"
            f"Run generate_meshes.py with N={N} to generate this mesh."
        )
    filename = index[key]['filename']
    return os.path.join(current_dir, 'meshes', mesh_type, filename)


# ============================================================
# -------------------- Source coordinates --------------------
# ============================================================

def make_source_coords(mesh_type, R=0.5, H=1.0):
    """
    Convert user-friendly source specification to 3D coordinates.

    cylinder / l_cylinder : (SOURCE_THETA_DEG, SOURCE_Z) -> (R*cos, R*sin, z)
    square_surface / l_shape / hole : SOURCE_XY -> (x, y, 0)
    """
    if mesh_type in ('cylinder', 'l_cylinder'):
        theta = np.deg2rad(SOURCE_THETA_DEG)
        return np.array([R * np.cos(theta), R * np.sin(theta), SOURCE_Z])
    elif mesh_type in ('square_surface', 'l_shape', 'hole'):
        return np.array([SOURCE_XY[0], SOURCE_XY[1], 0.0])
    else:
        raise ValueError(f"Unknown mesh_type: '{mesh_type}'")


# ============================================================
# -------------------- Geometry config -----------------------
# ============================================================

def get_config(mesh_type, current_dir, index):
    """
    Return a geometry-specific configuration dictionary.

    The 'get_output_path' key now reads from meshes/<mesh_type>/ using the
    index, instead of generating the mesh on the fly.

    Keys
    ----
    source_coord    : np.ndarray (3,)
    get_output_path : callable(N) -> str
    get_h           : callable(N) -> float
    compute_err     : callable(node_list, snapped_source) -> tuple
    plot_title      : str
    save_name       : str
    L_val           : float or None
    R_val           : float or None
    """
    if mesh_type == 'square_surface':
        L = 1.0
        return {
            'source_coord'   : make_source_coords(mesh_type),
            'get_output_path': lambda N: get_mesh_path(mesh_type, N, index, current_dir),
            'get_h'          : lambda N: index[str(N)]['h'],
            'compute_err'    : lambda node_list, src: solver.compute_err(node_list, src),
            'plot_title'     : 'FMM Convergence - Square',
            'save_name'      : 'ConvStudySquare.pdf',
            'L_val'          : L,
            'R_val'          : None,
        }

    elif mesh_type == 'l_shape':
        L = 1.0
        return {
            'source_coord'   : make_source_coords(mesh_type),
            'get_output_path': lambda N: get_mesh_path(mesh_type, N, index, current_dir),
            'get_h'          : lambda N: index[str(N)]['h'],
            'compute_err'    : lambda node_list, src: solver.compute_err_l_shape(
                                    node_list, src, L),
            'plot_title'     : 'FMM Convergence - L-shape',
            'save_name'      : 'ConvStudyLshape.pdf',
            'L_val'          : L,
            'R_val'          : None,
        }

    elif mesh_type == 'cylinder':
        R = 0.5
        H = 1.0
        return {
            'source_coord'   : make_source_coords(mesh_type, R=R, H=H),
            'get_output_path': lambda N: get_mesh_path(mesh_type, N, index, current_dir),
            'get_h'          : lambda N: index[str(N)]['h'],
            'compute_err'    : lambda node_list, src: solver.compute_err_cylinder(
                                    node_list, src, R),
            'plot_title'     : 'FMM Convergence - Cylinder',
            'save_name'      : 'ConvStudyCylinder.pdf',
            'L_val'          : None,
            'R_val'          : R,
        }

    elif mesh_type == 'hole':
        L = 1.0
        R = 0.2
        return {
            'source_coord'   : make_source_coords(mesh_type),
            'get_output_path': lambda N: get_mesh_path(mesh_type, N, index, current_dir),
            'get_h'          : lambda N: index[str(N)]['h'],
            'compute_err'    : lambda node_list, src: solver.compute_err_hole(
                                    node_list, src, L, R),
            'plot_title'     : 'FMM Convergence - Square with Hole',
            'save_name'      : 'ConvStudyHole.pdf',
            'L_val'          : L,
            'R_val'          : R,
        }

    elif mesh_type == 'l_cylinder':
        R = 0.5
        H = 1.0
        return {
            'source_coord'   : make_source_coords(mesh_type, R=R, H=H),
            'get_output_path': lambda N: get_mesh_path(mesh_type, N, index, current_dir),
            'get_h'          : lambda N: index[str(N)]['h'],
            'compute_err'    : lambda node_list, src: solver.compute_err_l_cylinder(
                                    node_list, src, R, H),
            'plot_title'     : 'FMM Convergence - L-Cylinder',
            'save_name'      : 'ConvStudyLCylinder.pdf',
            'L_val'          : H,
            'R_val'          : R,
        }

    else:
        raise ValueError(f"Unknown mesh_type: '{mesh_type}'")


# ============================================================
# -------------------------- Main ----------------------------
# ============================================================

def main(mesh_type=MESH_TYPE, N_values=N_VALUES):

    plt.close('all')
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # ----> A - Load mesh index and geometry config ----
    index = load_index(mesh_type, current_dir)
    cfg   = get_config(mesh_type, current_dir, index)

    # ----> B - Error list initialisation ----
    h_values              = []
    errors_fmm            = [];  errors_l2_fmm  = []
    errors_circ           = [];  errors_l2_circ = []
    point_errors_fmm_all  = []
    point_errors_circ_all = []

    # ----> C - Loop over refinement levels ----
    for i, N in enumerate(N_values):

        print(f"\n{'='*50}")
        print(f"[{mesh_type}]  N={N}  |  Iteration {i+1}/{len(N_values)}")
        print(f"{'='*50}")

        # 1 ---> Locate pre-generated mesh (raises KeyError if missing)
        output_path = cfg['get_output_path'](N)
        print(f"  Loading mesh from: {output_path}")

        # 2 ---> Open mesh
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.open(output_path)

        # 3 ---> Node and element lists
        tag_to_node_obj_dic = {}
        node_list    = solver.Make_NodeList_NodeDictionary(output_path, tag_to_node_obj_dic)
        element_list = solver.Make_ElementList(output_path, tag_to_node_obj_dic)

        # 4 ---> Source initialisation
        true_source, source_nodes = solver.Innit_Origin_Point(cfg['source_coord'], node_list)

        # 5a ---> Standard FMM
        print('\n------------ Standard FMM ------------')
        t_start = time.time()
        fmm_solver.fmm_algorithm(node_list, source_nodes)
        t_end = time.time()
        print(f"Elapsed time: {t_end - t_start:.4f} s")

        snapped_source = source_nodes[0].coords
        result_fmm     = cfg['compute_err'](node_list, snapped_source)
        err_fmm, e_l2_fmm = result_fmm[0], result_fmm[-1]
        print(f"Max error = {err_fmm:.6e}  |  L2 error = {e_l2_fmm:.6e}")
        errors_fmm.append(err_fmm)
        errors_l2_fmm.append(e_l2_fmm)

        # Pointwise tracking (2-D geometries only)
        if mesh_type not in ('cylinder', 'l_cylinder'):
            pt_errs_fmm, _ = solver.Track_Point_Errors(
                PROBE_POINTS, node_list, mesh_type, snapped_source,
                L=cfg['L_val'] or 1.0,
                R=cfg['R_val'] or 0.2)
            point_errors_fmm_all.append(pt_errs_fmm)

        # 5b ---> Circular (higher-order) FMM
        solver.Reset_Node_State(node_list)
        true_source, source_nodes = solver.Innit_Origin_Point(cfg['source_coord'], node_list)
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

        if mesh_type not in ('cylinder', 'l_cylinder'):
            pt_errs_circ, snapped = solver.Track_Point_Errors(
                PROBE_POINTS, node_list, mesh_type, snapped_source,
                L=cfg['L_val'] or 1.0,
                R=cfg['R_val'] or 0.2)
            point_errors_circ_all.append(pt_errs_circ)

            if i == 0:
                print("\nProbe point snap report (first mesh only):")
                for k, pt in enumerate(PROBE_POINTS):
                    print(f"  ({pt[0]:.2f},{pt[1]:.2f}) --> snapped to {snapped[k][:2]}")

        # Error field plot on the finest mesh only
        if i == len(N_values) - 1:
            print("\nGenerating error field plot for the finest mesh...")
            solver.Plot_Error_Field(node_list, element_list, mesh_type,
                                    snapped_source, R=cfg['R_val'], L=cfg['L_val'])

        # 6 ---> Record h and release gmsh
        h_values.append(cfg['get_h'](N))
        gmsh.finalize()

    # ----> D - Post-processing plots ----
    solver.plot_convergence(h_values, errors_l2_fmm, errors_l2_circ,
                            cfg['plot_title'], cfg['save_name'])

    if mesh_type not in ('cylinder', 'l_cylinder') and point_errors_fmm_all:
        solver.Plot_Point_Convergence(
            h_values,
            point_errors_fmm_all,
            point_errors_circ_all,
            PROBE_POINTS,
            cfg['plot_title'],
            cfg['save_name'])

    plt.show()


if __name__ == "__main__":
    main()
