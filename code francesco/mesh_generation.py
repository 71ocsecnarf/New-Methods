import gmsh
import os
import numpy as np

def generate_mesh(N, L, mesh_type, output_dir):
    gmsh.initialize()
    gmsh.model.add(mesh_type)
 
    
    # I'll do it in this way to incorporate other meshes in future
    if mesh_type == 'square_surface':
        gmsh.model.occ.add_rectangle(0, 0, 0, L, L)
        gmsh.model.occ.synchronize()
 
        curves = gmsh.model.getEntities(1)
        for c in curves:
            gmsh.model.mesh.setTransfiniteCurve(c[1], N)
 
        surface = gmsh.model.getEntities(2)[0][1]
        gmsh.model.mesh.setTransfiniteSurface(surface)
    
    # elif in case of other mesh type 
    elif mesh_type == 'l_shape':
        lc = L / (N - 1)
        # Create a rotated L-shape (180° CCW rotation)
        # The 'gap' is located in the bottom-right quadrant (L/2 < x < L, 0 < y < L/2).
        # Corner P3 (L/2, L/2) acts as the shadow corner.
        p1 = gmsh.model.occ.addPoint(0, 0, 0)           # Bottom-left
        p2 = gmsh.model.occ.addPoint(L/2, 0, 0)
        p3 = gmsh.model.occ.addPoint(L/2, L/2, 0)     # Inner concave corner (Shadow trigger)
        p4 = gmsh.model.occ.addPoint(L, L/2, 0)
        p5 = gmsh.model.occ.addPoint(L, L, 0)           # Top-right
        p6 = gmsh.model.occ.addPoint(0, L, 0)           # Top-left
 
        # Define boundary segments
        lines = []
        lines.append(gmsh.model.occ.addLine(p1, p2))
        lines.append(gmsh.model.occ.addLine(p2, p3))
        lines.append(gmsh.model.occ.addLine(p3, p4))
        lines.append(gmsh.model.occ.addLine(p4, p5))
        lines.append(gmsh.model.occ.addLine(p5, p6))
        lines.append(gmsh.model.occ.addLine(p6, p1))
 
        # Define the surface from the curve loop
        loop = gmsh.model.occ.addCurveLoop(lines)
        surface = gmsh.model.occ.addPlaneSurface([loop])
        gmsh.model.occ.synchronize()
 
        # --- APPLY lc ---
        # Set the characteristic size to all points for unstructured mesh generation
        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), lc)
        
    # gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)
    # gmsh.option.setNumber("Mesh.RecombineAll", 1)
    #!Used for quads:::::::::::::::::::::::::::::::::::::::::
    else:
        raise ValueError(f"Unknown mesh_type: '{mesh_type}'")
    
    gmsh.model.mesh.generate(2)
 
    # Use mesh_type as the filename so that different geometries do not
    # overwrite each other (e.g. 'l_shape.msh', 'square_surface.msh')
    filename = f"{mesh_type}.msh"
 
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(current_dir, filename)
 
    gmsh.write(output_path)
    # gmsh.fltk.run()
    gmsh.finalize()
 
    return output_path
 
# In mesh_generation.py
 
def generate_cylinder_mesh(N, R, H, output_dir):
    """
    Generate a structured triangular mesh of a half-cylinder lateral surface.
    N controls the number of points along both the arc and the height.
    """
    gmsh.initialize()
    gmsh.model.add("cylinder_surface")
 
    p_center = gmsh.model.occ.addPoint(0, 0, 0)
    p_left   = gmsh.model.occ.addPoint(-R, 0, 0)
    p_right  = gmsh.model.occ.addPoint( R, 0, 0)
 
    arc_bot  = gmsh.model.occ.addCircleArc(p_right, p_center, p_left)
    extruded = gmsh.model.occ.extrude([(1, arc_bot)], 0, 0, H)
    gmsh.model.occ.synchronize()
 
    surf_tag       = [e[1] for e in extruded if e[0] == 2][0]
    lines          = [e[1] for e in extruded if e[0] == 1]
    arc_top_tag    = lines[1]
    line_right_tag = lines[0]
    line_left_tag  = lines[2]
 
    gmsh.model.mesh.setTransfiniteCurve(arc_bot,        N)
    gmsh.model.mesh.setTransfiniteCurve(arc_top_tag,    N)
    gmsh.model.mesh.setTransfiniteCurve(line_left_tag,  N)
    gmsh.model.mesh.setTransfiniteCurve(line_right_tag, N)
    gmsh.model.mesh.setTransfiniteSurface(surf_tag)
    # No setRecombine -> triangles
 
    gmsh.model.addPhysicalGroup(2, [surf_tag], tag=1)
    gmsh.model.setPhysicalName(2, 1, "lateral_face")
 
    gmsh.model.mesh.generate(2)
 
    output_path = os.path.join(output_dir, "cylinder_surface.msh")
    gmsh.write(output_path)
    gmsh.finalize()
 
    return output_path


def generate_square_with_hole(N, L, R, output_dir):

    gmsh.initialize()
    gmsh.model.add("square_with_hole")

    # --- Geometry ---
    # Square
    rect = gmsh.model.occ.addRectangle(0, 0, 0, L, L)

    # Circle (hole)
    circle = gmsh.model.occ.addDisk(L/2, L/2, 0, R, R)

    # Boolean subtraction
    domain = gmsh.model.occ.cut([(2, rect)], [(2, circle)])
    gmsh.model.occ.synchronize()

    # --- Mesh size ---
    lc = L / (N - 1)
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), lc)

    # Refinement near the hole (optional but recommended)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 20)

    # --- Physical groups ---
    surfaces = gmsh.model.getEntities(2)
    gmsh.model.addPhysicalGroup(2, [s[1] for s in surfaces], 1)
    gmsh.model.setPhysicalName(2, 1, "domain")

    # Outer boundary / hole
    curves = gmsh.model.getEntities(1)
    gmsh.model.addPhysicalGroup(1, [c[1] for c in curves], 2)
    gmsh.model.setPhysicalName(1, 2, "boundary")

    # --- Mesh ---
    gmsh.model.mesh.generate(2)

    output_path = os.path.join(output_dir, "square_with_hole.msh")
    gmsh.write(output_path)
    gmsh.finalize()

    return output_path

def generate_l_cylinder_mesh(N, R, H, output_dir):
    """
    Generate a triangular mesh of an L-shaped half-cylinder lateral surface.

    The full half-cylinder spans theta in [-pi/2, +pi/2] and z in [0, H].
    The missing quadrant is (theta in [0, pi/2], z in [H/2, H]),
    which creates a concave corner at (theta=0, z=H/2) -- the shadow corner.

    Layout (unrolled view):
        z=H   [left arm only]  *------*
                               |      |   <- this quadrant is MISSING
        z=H/2 *------*---------*------*
              |               |
        z=0   *---------------*
            theta=-pi/2     theta=0    theta=+pi/2

    The concave corner node is at theta=0, z=H/2.

    Parameters
    ----------
    N          : int   -- number of mesh points along the longest edge
    R          : float -- cylinder radius
    H          : float -- total cylinder height
    output_dir : str   -- directory where the .msh file is saved

    Returns
    -------
    output_path : str -- full path to the written .msh file
    """

    gmsh.initialize()
    gmsh.model.add("l_cylinder_surface")

    # ------------------------------------------------------------------
    # STEP 1: Define the 6 corner points of the L-shape on the cylinder
    #
    # The L-shape boundary has 6 vertices (like the flat L-shape):
    #   P1 = (-R, 0, 0)    theta=-pi/2, z=0   (bottom-left)
    #   P2 = ( R, 0, 0)    theta=+pi/2, z=0   (bottom-right)
    #   P3 = ( R, 0, H/2)  theta=+pi/2, z=H/2 (mid-right)
    #   P4 = ( 0, R, H/2)  theta=0,     z=H/2 (concave corner -- shadow trigger)
    #   P5 = ( 0, R, H  )  theta=0,     z=H   (top-mid)  <- NOT NEEDED: boundary goes to P4
    #   ...
    # Actually we define them by (theta, z) and convert to 3D:
    #   x = R * cos(theta),  y = R * sin(theta),  z = z
    # ------------------------------------------------------------------

    def cyl_point(theta_rad, z_val, lc=0.0):
        """Add a point on the cylinder surface and return its gmsh tag."""
        x = R * np.cos(theta_rad)
        y = R * np.sin(theta_rad)
        # lc=0 means gmsh uses the global mesh size (set later with setSize)
        return gmsh.model.occ.addPoint(x, y, z_val, lc)

    # Center of the cylinder base (needed for arc definitions)
    p_center_bot = gmsh.model.occ.addPoint(0.0, 0.0, 0.0)
    p_center_mid = gmsh.model.occ.addPoint(0.0, 0.0, H / 2.0)

    # The 6 boundary corners of the L-shape (theta in radians):
    #   left  = -pi/2,  mid = 0,  right = +pi/2
    theta_left  = -np.pi / 2.0
    theta_mid   =  0.0
    theta_right = +np.pi / 2.0

    p1 = cyl_point(theta_left,  0.0  )   # bottom-left
    p2 = cyl_point(theta_right, 0.0  )   # bottom-right
    p3 = cyl_point(theta_right, H / 2)   # mid-right
    p4 = cyl_point(theta_mid,   H / 2)   # concave corner (shadow corner)
    p5 = cyl_point(theta_mid,   H    )   # top-mid
    p6 = cyl_point(theta_left,  H    )   # top-left

    # ------------------------------------------------------------------
    # STEP 2: Define the boundary curves
    #
    # Bottom arc  : P1 -> P2  along z=0      (full half-circle arc)
    # Right edge  : P2 -> P3  vertical line  (theta=+pi/2)
    # Mid-low arc : P3 -> P4  along z=H/2    (right quarter-arc, REVERSED)
    # Inner vert  : P4 -> P5  vertical line  (theta=0, inner concave edge)
    # Top arc     : P5 -> P6  along z=H      (left quarter-arc, REVERSED)
    # Left edge   : P6 -> P1  vertical line  (theta=-pi/2)
    # ------------------------------------------------------------------

    # gmsh.model.occ.addCircleArc(start, center, end)
    # The arc goes from 'start' to 'end' passing through the short way around
    # the circle centred at 'center'.

    arc_bottom = gmsh.model.occ.addCircleArc(p1, p_center_bot, p2)  # z=0, full half
    line_right = gmsh.model.occ.addLine(p2, p3)                      # right vertical
    arc_mid    = gmsh.model.occ.addCircleArc(p3, p_center_mid, p4)  # z=H/2, right quarter
    line_inner = gmsh.model.occ.addLine(p4, p5)                      # inner concave vertical
    arc_top    = gmsh.model.occ.addCircleArc(p5, p_center_mid, p6)  # z=H, left quarter
    line_left  = gmsh.model.occ.addLine(p6, p1)                      # left vertical

    # ------------------------------------------------------------------
    # STEP 3: Create the surface from the closed boundary loop
    #
    # addCurveLoop expects an ordered list of curve tags that form a
    # closed, oriented boundary.  Negative tag = reversed orientation.
    # addSurfaceFilling creates a surface that follows the curved boundary
    # (unlike addPlaneSurface which is only for flat surfaces).
    # ------------------------------------------------------------------

    boundary_loop = gmsh.model.occ.addCurveLoop([
        arc_bottom,
        line_right,
        arc_mid,
        line_inner,
        arc_top,
        line_left
    ])

    # addSurfaceFilling: creates a surface that interpolates the boundary curves.
    # It handles non-planar (curved) surfaces correctly.
    surface = gmsh.model.occ.addSurfaceFilling(boundary_loop)

    # synchronize() "compiles" all OCC geometry into the gmsh model
    # -- must be called before any meshing or physical group operation
    gmsh.model.occ.synchronize()

    # ------------------------------------------------------------------
    # STEP 4: Set mesh size and generate mesh
    #
    # lc = characteristic mesh size = approx. edge length of triangles.
    # getEntities(0) returns all points (dimension 0) in the model.
    # setSize applies the same size to all of them.
    # ------------------------------------------------------------------

    lc = (np.pi * R) / (N - 1)   # arc length of full half-circle / (N-1) intervals
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), lc)

    gmsh.model.mesh.generate(2)   # 2 = surface mesh (triangles)

    # ------------------------------------------------------------------
    # STEP 5: Write to file and clean up
    # ------------------------------------------------------------------

    output_path = os.path.join(output_dir, "l_cylinder_surface.msh")
    gmsh.write(output_path)
    gmsh.finalize()

    return output_path