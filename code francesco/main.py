import numpy as np
import os
import matplotlib.pyplot as plt

import gmsh
import solver
import fmm_solver
import mesh_generation



# =================================================
# ---- Parameter to modify to change geometry -----
# =================================================

MESH_TYPE = 'hole'  # 'square_surface', 'cylinder', 'hole', 'l_shape' or 'l_cylinder'
FMM_TYPE = 'circ'              # 'standard' for FMM of 'circ' for higher order FMM

# -------------------------------------------------

# For cylinder: source specified as (theta, z) in the unrolled domain
# theta in [-90, +90],  z in [0, H]
SOURCE_THETA_DEG = 0    # angle in degrees: 0 = front of cylinder (x=R, y=0)
SOURCE_Z     = 0.5    # height along the cylinder

# For square_surface: source specified as (x, y)
SOURCE_XY = np.array([0.0, 0.0])

# -------------------------------------------------


def make_source_coords(mesh_type, R=0.5, H=1.0):
    """
    Convert user-friendly source specification to 3D coordinates.
    - Cylinder  : (theta_deg, z) --> (R*cos(theta), R*sin(theta), z)
      theta_deg in [-90, +90] degrees
    - Square    : (x, y)        --> (x, y, 0)
    - L-shape   : (x, y)        --> (x, y, 0)
      Recommended: place source in the visible arm, e.g. (0.2, 0.8) for L=1
      to exercise the shadow zone around the concave corner at (L/2, L/2)
    """
    if mesh_type == 'cylinder':
        theta = np.deg2rad(SOURCE_THETA_DEG)   # convert degrees to radians
        x = R * np.cos(theta)
        y = R * np.sin(theta)
        z = SOURCE_Z
        return np.array([x, y, z])

    elif mesh_type in ('square_surface', 'l_shape'):
        return np.array([SOURCE_XY[0], SOURCE_XY[1], 0.0])
    elif mesh_type == 'hole':
        return np.array([SOURCE_XY[0], SOURCE_XY[1], 0.0])
    else:
        raise ValueError(f"Unknown mesh_type: '{mesh_type}'")


def main(mesh_type=MESH_TYPE, fmm_type=FMM_TYPE):

    current_dir = os.path.dirname(os.path.abspath(__file__))

    # ---- Step 1: Mesh generation ----

    if mesh_type == 'square_surface':
        N = 10
        L = 1.0
        output_path   = os.path.join(current_dir, "square_surface.msh")
        source_coords = make_source_coords(mesh_type)
        mesh_generation.generate_mesh(N, L, mesh_type, output_path)

    elif mesh_type == 'l_shape':
        N = 20     # number of points along the longest edge
        L = 1.0      # bounding box side length; concave corner is at (L/2, L/2)
        output_path   = os.path.join(current_dir, "l_shape.msh")
        source_coords = make_source_coords(mesh_type)
        # generate_mesh handles both 'square_surface' and 'l_shape' internally
        mesh_generation.generate_mesh(N, L, mesh_type, output_path)    

    elif mesh_type == 'cylinder':
        N = 20
        R = 0.5
        H = 1.0
        output_path   = os.path.join(current_dir, "cylinder_surface.msh")
        source_coords = make_source_coords(mesh_type, R=R, H=H)
        mesh_generation.generate_cylinder_mesh(N, R, H, current_dir)

    elif mesh_type == 'hole':
        N = 50
        L = 1
        R = 0.2

        source_coords = make_source_coords(mesh_type, R = R, H = L)

        output_path = mesh_generation.generate_square_with_hole(N, L, R, current_dir)

    elif mesh_type == 'l_cylinder':
        N = 20
        R = 0.5
        H = 1.0
        output_path   = os.path.join(current_dir, "l_cylinder_surface.msh")
        source_coords = np.array([-R, 0.0, 0.3])   # theta=-pi/2 (left edge), z=0.3
        mesh_generation.generate_l_cylinder_mesh(N, R, H, current_dir)

    else:
        raise ValueError(f"Unknown mesh_type: '{mesh_type}'")

    # ---- Step 2: Open mesh ----
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.open(output_path)

    # ---- Step 3: Node and element lists ----
    tag_to_node_obj_dic = {}
    node_list    = solver.Make_NodeList_NodeDictionary(output_path, tag_to_node_obj_dic)
    element_list = solver.Make_ElementList(output_path, tag_to_node_obj_dic)
    solver.Check_Obtuse_triangles(element_list)

    # ---- Step 4: Source initialization ----
    # Innit_Origin_Point snaps the source to the closest mesh node
    # and returns the snapped coordinates as true_source
    true_source, source_nodes = solver.Innit_Origin_Point(source_coords, node_list)

    # ---- Step 5: FMM ----
    if fmm_type == 'standard':
        fmm_solver.fmm_algorithm(node_list, source_nodes)
    elif fmm_type == 'circ':
        fmm_solver.fmm_algorithm_circ(node_list, source_nodes)
    else:
        raise ValueError(f"Unknown fmm_type: '{fmm_type}'")

    # ---- Step 6: Plot ----
    print("Computation Completed, Trace Generation ...")

    # Use the snapped source coordinates for the plot marker
    snapped_coords = source_nodes[0].coords

    if mesh_type in ('square_surface', 'l_shape', 'hole'):
        solver.Plot_Isolines(node_list, element_list)
        solver.Plot_Error_Field(node_list, element_list, mesh_type, snapped_coords, L=1.0, R = R)
    
    elif mesh_type == 'cylinder':
        solver.Plot_Isolines_3D(node_list, element_list,
                                source_coords=snapped_coords)
        solver.Plot_Error_Field(node_list, element_list, mesh_type, snapped_coords, R=0.5)

    elif mesh_type == 'l_cylinder':
        solver.Plot_Isolines_L_Cylinder(node_list, element_list, source_coords=snapped_coords)
        #solver.Plot_Error_Field(node_list, element_list, 'cylinder',snapped_coords, R=0.5)

    plt.show()
    
    gmsh.finalize()


if __name__ == "__main__":
    main()
