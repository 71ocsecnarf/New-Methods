import numpy as np
import os
import json
import matplotlib.pyplot as plt

import gmsh
import solver
import fmm_solver


# =================================================
# ---- Parameters to change geometry/algorithm ----
# =================================================

MESH_TYPE = 'l_shape'   # 'square_surface', 'l_shape', 'hole', 'cylinder', 'l_cylinder'
FMM_TYPE  = 'circ'      # 'standard' or 'circ' (higher-order circular-wavefront FMM)

# N value to load from the pre-generated mesh library
N = 11

# -------------------------------------------------

# Cylinder source: (theta_deg, z)
SOURCE_THETA_DEG    = 0
SOURCE_Z            = 0.5

# L-cylinder source: (theta_deg, z)
SOURCE_THETA_DEG_LC = 0
SOURCE_Z_LC         = 0.0

# 2-D geometries source: (x, y)
SOURCE_XY = np.array([0.0, 0.0])

# -------------------------------------------------


def load_index(mesh_type, current_dir):
    """
    Load the index.json file for the given mesh_type from meshes/<mesh_type>/.
    Raises FileNotFoundError with a clear message if the index does not exist.
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


def make_source_coords(mesh_type, R=0.5, H=1.0):
    """
    Convert user-friendly source specification to 3D coordinates.

    cylinder    : (SOURCE_THETA_DEG,    SOURCE_Z)    -> (R*cos, R*sin, z)
    l_cylinder  : (SOURCE_THETA_DEG_LC, SOURCE_Z_LC) -> (R*cos, R*sin, z)
    square_surface / l_shape / hole : SOURCE_XY      -> (x, y, 0)
    """
    if mesh_type == 'cylinder':
        theta = np.deg2rad(SOURCE_THETA_DEG)
        return np.array([R * np.cos(theta), R * np.sin(theta), SOURCE_Z])

    elif mesh_type == 'l_cylinder':
        theta = np.deg2rad(SOURCE_THETA_DEG_LC)
        return np.array([R * np.cos(theta), R * np.sin(theta), SOURCE_Z_LC])

    elif mesh_type in ('square_surface', 'l_shape', 'hole'):
        return np.array([SOURCE_XY[0], SOURCE_XY[1], 0.0])

    else:
        raise ValueError(f"Unknown mesh_type: '{mesh_type}'")


def main(mesh_type=MESH_TYPE, fmm_type=FMM_TYPE, n=N):

    current_dir = os.path.dirname(os.path.abspath(__file__))

    # ---- Step 1: Load mesh index and locate the file ----
    index       = load_index(mesh_type, current_dir)
    output_path = get_mesh_path(mesh_type, n, index, current_dir)

    # Read geometry parameters back from the index so they are consistent
    # with the mesh that was actually generated.
    entry = index[str(n)]
    L = entry.get('L', 1.0)
    R = entry.get('R', 0.5)
    H = entry.get('H', 1.0)

    print(f"Loading mesh: {output_path}")
    source_coords = make_source_coords(mesh_type, R=R, H=H)

    # ---- Step 2: Open mesh ----
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.open(output_path)

    # ---- Step 3: Node and element lists ----
    tag_to_node_obj_dic = {}
    node_list    = solver.Make_NodeList_NodeDictionary(output_path, tag_to_node_obj_dic)
    element_list = solver.Make_ElementList(output_path, tag_to_node_obj_dic)
    solver.Check_Obtuse_triangles(element_list)

    # ---- Step 4: Source initialisation ----
    true_source, source_nodes = solver.Innit_Origin_Point(source_coords, node_list)

    # ---- Step 5: FMM ----
    if fmm_type == 'standard':
        fmm_solver.fmm_algorithm(node_list, source_nodes)
    elif fmm_type == 'circ':
        fmm_solver.fmm_algorithm_circ(node_list, source_nodes)
    else:
        raise ValueError(f"Unknown fmm_type: '{fmm_type}'")

    # ---- Step 6: Plots ----
    print("Computation completed, generating plots ...")
    snapped_coords = source_nodes[0].coords

    if mesh_type in ('square_surface', 'l_shape', 'hole'):
        solver.Plot_Isolines(node_list, element_list)
        solver.Plot_Error_Field(node_list, element_list, mesh_type,
                                snapped_coords, L=L, R=R)

    elif mesh_type == 'cylinder':
        solver.Plot_Isolines_3D(node_list, element_list, source_coords=snapped_coords)
        solver.Plot_Error_Field(node_list, element_list, mesh_type,
                                snapped_coords, R=R)

    elif mesh_type == 'l_cylinder':
        solver.Plot_Isolines_L_Cylinder(node_list, element_list, source_coords=snapped_coords)
        solver.Plot_Error_Field(node_list, element_list, mesh_type,
                                snapped_coords, R=R, L=H)

    plt.show()
    gmsh.finalize()


if __name__ == "__main__":
    main()
