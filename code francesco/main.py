import numpy as np
import os
import matplotlib.pyplot as plt

import gmsh
import solver
import fmm_solver
import mesh_generation



# -------------------------------------------------
# ---- Parameter to modify to change geometry -----
# -------------------------------------------------

MESH_TYPE = 'cylinder'   # 'square_surface' or 'cylinder'
FMM_TYPE = 'circ'              # 'standard for FMM of 'circ' for higher order FMM

# -------------------------------------------------


def main(mesh_type=MESH_TYPE, fmm_type=FMM_TYPE):

    current_dir = os.path.dirname(os.path.abspath(__file__))

    # ---- Step 1: Mesh opening or mesh generation ----

    if mesh_type == 'square_surface':
        N = 20
        L = 1.0
        output_path    = os.path.join(current_dir, "square_surface.msh")
        source_coords  = np.array([0.0, 0.0, 0.0])
        mesh_generation.generate_mesh(N, L, mesh_type, output_path)

    elif mesh_type == 'cylinder':
        N = 20
        R = 0.5
        H = 1.0
        output_path    = os.path.join(current_dir, "cylinder_surface.msh")
        source_coords  = np.array([R, 0.0, 0.5])
        mesh_generation.generate_cylinder_mesh(N, R, H, current_dir)

    else:
        raise ValueError(f"Unknown mesh_type: '{mesh_type}'")

    # ---- Step 2: Mesh opening with GMSH ----
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.open(output_path)

    # ---- Step 3: node list and element list generation ----
    tag_to_node_obj_dic = {}
    node_list    = solver.Make_NodeList_NodeDictionary(output_path, tag_to_node_obj_dic)
    element_list = solver.Make_ElementList(output_path, tag_to_node_obj_dic)
    solver.Check_Obtuse_triangles(element_list)

    # ---- Step 4: Source Initialization ----
    source_point, source_nodes = solver.Innit_Origin_Point(source_coords, node_list)

    # ---- Step 5: FMM ----
    if fmm_type == 'standard':
        fmm_solver.fmm_algorithm(node_list, source_nodes)
    elif fmm_type == 'circ':
        fmm_solver.fmm_algorithm_circ(node_list, source_nodes)
    else:
        raise ValueError(f"Unknown fmm_type: '{fmm_type}'")

    # ---- Step 6: Plot ----
    print("Computation Completed, Trace Generation ...")

    if mesh_type == 'square_surface':
        solver.Plot_Isolines(node_list, element_list)
    elif mesh_type == 'cylinder':
        solver.Plot_Isolines_3D(node_list, element_list, source_coords=source_coords)

    gmsh.finalize()


if __name__ == "__main__":
    main()
