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
    
    # --- OPTIMIZATION FOR FMM ---
    # Alogorithm 6 (Frontal-Delaunay) - Limits obtuse triangles
    gmsh.option.setNumber("Mesh.Algorithm", 6)
    # Elevate smoothing to 100 iterations to further improve mesh quality (especially for the l_shape)
    gmsh.option.setNumber("Mesh.Smoothing", 100)

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
    
    # --- OPTIMIZATION FOR FMM ---
    # Alogorithm 6 (Frontal-Delaunay) - Limits obtuse triangles
    gmsh.option.setNumber("Mesh.Algorithm", 6)
    # Elevate smoothing to 100 iterations to further improve mesh quality (especially for the l_shape)
    gmsh.option.setNumber("Mesh.Smoothing", 100)

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

    # --- OPTIMIZATION FOR FMM ---
    # Alogorithm 6 (Frontal-Delaunay) - Limits obtuse triangles
    gmsh.option.setNumber("Mesh.Algorithm", 6)
    # Elevate smoothing to 100 iterations to further improve mesh quality (especially for the l_shape)
    gmsh.option.setNumber("Mesh.Smoothing", 100)

    # --- Mesh ---
    gmsh.model.mesh.generate(2)

    output_path = os.path.join(output_dir, "square_with_hole.msh")
    gmsh.write(output_path)
    gmsh.finalize()

    return output_path

def generate_l_cylinder_mesh(N, R, H, output_dir):
    """
    Generate a triangular mesh of an L-shaped half-cylinder lateral surface.
 
    The geometry is the half-cylinder lateral surface with one quadrant removed.
    Angle parametrisation:
      - theta=0    -> point ( R, 0, z)   right edge
      - theta=pi/2 -> point ( 0, R, z)   front of cylinder  (concave corner)
      - theta=pi   -> point (-R, 0, z)   left edge
 
    The domain is the union of two patches:
      - Bottom arm : theta in [0,   pi],    z in [0,   H/2]
      - Left arm   : theta in [pi/2, pi],   z in [H/2, H  ]
 
    The missing quadrant is theta in [0, pi/2], z in [H/2, H].
    The concave (shadow) corner is at (0, R, H/2).
 
    KEY FIX (compared to the previous version):
        The two patches are built separately and then merged with
        gmsh.model.occ.fragment() BEFORE meshing.  fragment() forces gmsh
        to detect the shared edge (the arc theta in [pi/2, pi] at z=H/2)
        and to assign a SINGLE set of mesh nodes to it.  Without this call,
        gmsh treats the two arcs as independent curves and generates duplicate
        nodes on the shared boundary, breaking the FMM connectivity.
 
    Parameters
    ----------
    N          : int   -- number of mesh points along the longest edge
                          (bottom arc, arc-length pi*R).
    R          : float -- cylinder radius
    H          : float -- total cylinder height
    output_dir : str   -- directory where the .msh file is saved
 
    Returns
    -------
    output_path : str -- full path to the written .msh file
    """
 
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("l_cylinder_surface")
 
    # ------------------------------------------------------------------
    # Angle constants
    # ------------------------------------------------------------------
    theta_right = 0.0
    theta_mid   = np.pi / 2.0
    theta_left  = np.pi
 
    def cyl_pt(theta, z):
        """Add a point on the cylinder surface and return its gmsh tag."""
        return gmsh.model.occ.addPoint(R * np.cos(theta), R * np.sin(theta), z)
 
    # ------------------------------------------------------------------
    # STEP 1: Corner points
    #
    #   p_right_bot = ( R,  0, 0  )   bottom-right
    #   p_left_bot  = (-R,  0, 0  )   bottom-left
    #   p_right_mid = ( R,  0, H/2)   mid-right  (top-right of bottom arm)
    #   p_front_mid = ( 0,  R, H/2)   concave corner
    #   p_left_mid  = (-R,  0, H/2)   mid-left   (shared by both arms)
    #   p_front_top = ( 0,  R, H  )   top of left arm, right side
    #   p_left_top  = (-R,  0, H  )   top-left
    #
    # p_axis_bot and p_axis_mid are helper points for addCircleArc only;
    # they are removed before meshing.
    # ------------------------------------------------------------------
    p_right_bot = cyl_pt(theta_right, 0.0)
    p_left_bot  = cyl_pt(theta_left,  0.0)
    p_right_mid = cyl_pt(theta_right, H / 2.0)   # created but used implicitly by extrude
    p_front_mid = cyl_pt(theta_mid,   H / 2.0)
    p_left_mid  = cyl_pt(theta_left,  H / 2.0)
    p_front_top = cyl_pt(theta_mid,   H)
    p_left_top  = cyl_pt(theta_left,  H)
 
    p_axis_bot = gmsh.model.occ.addPoint(0.0, 0.0, 0.0)
    p_axis_mid = gmsh.model.occ.addPoint(0.0, 0.0, H / 2.0)
 
    # ------------------------------------------------------------------
    # STEP 2: Bottom arm -- extrude the full bottom arc upward by H/2
    #
    # The arc spans theta in [0, pi] at z=0.
    # extrude() returns a list of (dim, tag) pairs:
    #   (2, surf_bot)         -- the new lateral surface
    #   (1, right_lateral)    -- vertical edge at theta=0
    #   (1, top_arc_bot)      -- arc at z=H/2, theta in [0, pi]
    #   (1, left_lateral_bot) -- vertical edge at theta=pi
    # ------------------------------------------------------------------
    arc_bot = gmsh.model.occ.addCircleArc(p_right_bot, p_axis_bot, p_left_bot)
    ext_bot  = gmsh.model.occ.extrude([(1, arc_bot)], 0, 0, H / 2.0)
    gmsh.model.occ.synchronize()
 
    surf_bot = next(e[1] for e in ext_bot if e[0] == 2)
 
    # ------------------------------------------------------------------
    # STEP 3: Left arm -- extrude the left-quarter arc upward by H/2
    #
    # The arc spans theta in [pi/2, pi] at z=H/2.
    # We create it with the SAME endpoint tags (p_front_mid, p_left_mid)
    # that were used as corner points in STEP 1.  After fragment() in
    # STEP 4, gmsh will recognise that this arc coincides geometrically
    # with the left half of top_arc_bot and merge them.
    # ------------------------------------------------------------------
    arc_left_mid = gmsh.model.occ.addCircleArc(p_front_mid, p_axis_mid, p_left_mid)
    ext_top      = gmsh.model.occ.extrude([(1, arc_left_mid)], 0, 0, H / 2.0)
    gmsh.model.occ.synchronize()
 
    surf_top = next(e[1] for e in ext_top if e[0] == 2)
 
    # ------------------------------------------------------------------
    # STEP 4: Merge the two patches with fragment()
    #
    # fragment(A, B) computes the Boolean fragmentation of A and B:
    # it splits both objects along their intersection and produces a
    # conforming topology where shared boundaries carry a single curve tag.
    # This is the key step that forces shared mesh nodes on the common
    # edge (theta in [pi/2, pi], z=H/2) so that the FMM can propagate
    # across the two patches.
    #
    # We do NOT use fuse() because that would merge the two surfaces into
    # one, losing the separate patch structure.  fragment() keeps both
    # surfaces intact while enforcing conformity at the shared boundary.
    # ------------------------------------------------------------------
    gmsh.model.occ.fragment([(2, surf_bot)], [(2, surf_top)])
    gmsh.model.occ.synchronize()
 
    # ------------------------------------------------------------------
    # STEP 5: Remove axis helper points
    #
    # p_axis_bot and p_axis_mid were needed only as arc-centre arguments.
    # Removing them prevents isolated nodes from appearing in the mesh.
    # recursive=False: remove the point only, not the curves attached to it
    # (those curves are the arcs we already built).
    # ------------------------------------------------------------------
    for pt in [p_axis_bot, p_axis_mid]:
        try:
            gmsh.model.occ.remove([(0, pt)], recursive=False)
        except Exception:
            pass   # already removed implicitly by fragment() -- safe to ignore
    gmsh.model.occ.synchronize()
 
    # ------------------------------------------------------------------
    # STEP 6: Mesh size and generation
    #
    # lc is chosen so that the longest edge (bottom arc, arc-length pi*R)
    # has N-1 intervals, consistent with the other mesh generators.
    # ------------------------------------------------------------------
    lc = np.pi * R / (N - 1)
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), lc)
    
    # --- OPTIMIZATION FOR FMM ---
    # Alogorithm 6 (Frontal-Delaunay) - Limits obtuse triangles
    gmsh.option.setNumber("Mesh.Algorithm", 6)
    # Elevate smoothing to 100 iterations to further improve mesh quality (especially for the l_shape)
    gmsh.option.setNumber("Mesh.Smoothing", 100)
    
    gmsh.model.mesh.generate(2)
 
    # ------------------------------------------------------------------
    # STEP 7: Write and finalise
    # ------------------------------------------------------------------
    output_path = os.path.join(output_dir, "l_cylinder_surface.msh")
    gmsh.write(output_path)
    gmsh.finalize()
 
    return output_path
    """
    Generate a triangular mesh of an L-shaped half-cylinder lateral surface.

    The geometry is the half-cylinder lateral surface with one quadrant removed.
    Angle parametrisation (consistent with generate_cylinder_mesh):
      - theta=0    -> point ( R, 0, z)   right edge
      - theta=pi/2 -> point ( 0, R, z)   front of cylinder
      - theta=pi   -> point (-R, 0, z)   left edge

    The domain is the union of two patches:
      - Bottom arm : theta in [0,    pi],  z in [0,   H/2]  (full half-cylinder)
      - Left arm   : theta in [pi/2, pi],  z in [H/2, H  ]  (left quarter only)

    The missing quadrant is theta in [0, pi/2], z in [H/2, H].
    The concave (shadow) corner is at theta=pi/2, z=H/2 -> 3D coords (0, R, H/2).

    Unrolled layout (theta increases right-to-left, z upward):

        z=H    P6-------P5
               |  LEFT   |
        z=H/2  P_lm--P4--P3
               |  BOTTOM  |
        z=0    P_lb-------P2
             theta=pi  pi/2  0
            (-R,0)   (0,R)  (R,0)

    Each patch is built by extruding an arc upward (dz = H/2), which places
    nodes exactly on the cylinder surface (no surface interpolation).

    Parameters
    ----------
    N          : int   -- number of mesh points along the longest edge (bottom arc,
                          arc-length pi*R). Other edges are scaled proportionally.
    R          : float -- cylinder radius
    H          : float -- total cylinder height
    output_dir : str   -- directory where the .msh file is saved

    Returns
    -------
    output_path : str -- full path to the written .msh file
    """

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("l_cylinder_surface")

    # ------------------------------------------------------------------
    # Angle constants
    # ------------------------------------------------------------------
    theta_right = 0.0
    theta_mid   = np.pi / 2.0
    theta_left  = np.pi

    def cyl_pt(theta, z):
        """Add a point on the cylinder surface and return its gmsh tag."""
        return gmsh.model.occ.addPoint(R * np.cos(theta), R * np.sin(theta), z)

    # ------------------------------------------------------------------
    # STEP 1: Define the 7 boundary corner points
    #
    #   p_right_bot = (R,  0, 0)     bottom-right
    #   p_left_bot  = (-R, 0, 0)     bottom-left
    #   p_right_mid = (R,  0, H/2)   mid-right   (bottom arm top-right)
    #   p_front_mid = (0,  R, H/2)   concave corner (shadow corner)
    #   p_left_mid  = (-R, 0, H/2)   mid-left    (shared by both arms)
    #   p_front_top = (0,  R, H)     top-mid     (left arm top-right)
    #   p_left_top  = (-R, 0, H)     top-left
    # ------------------------------------------------------------------
    p_right_bot = cyl_pt(theta_right, 0.0)
    p_left_bot  = cyl_pt(theta_left,  0.0)
    p_right_mid = cyl_pt(theta_right, H / 2)
    p_front_mid = cyl_pt(theta_mid,   H / 2)
    p_left_mid  = cyl_pt(theta_left,  H / 2)
    p_front_top = cyl_pt(theta_mid,   H)
    p_left_top  = cyl_pt(theta_left,  H)

    # Axis points: needed only as arc-centre arguments for addCircleArc.
    # They are NOT part of the surface; removed before meshing.
    p_axis_bot = gmsh.model.occ.addPoint(0.0, 0.0, 0.0)
    p_axis_mid = gmsh.model.occ.addPoint(0.0, 0.0, H / 2.0)

    # ------------------------------------------------------------------
    # STEP 2: Build BOTTOM ARM by extruding the full bottom arc upward by H/2
    #
    # Arc at z=0: p_right_bot -> p_left_bot (passes through (0, R, 0), the front).
    # extrude returns [(dim, tag), ...]:
    #   dim=2 -> new surface
    #   dim=1 -> new curves: [right_lateral, top_arc, left_lateral]
    # ------------------------------------------------------------------
    arc_bot = gmsh.model.occ.addCircleArc(p_right_bot, p_axis_bot, p_left_bot)
    ext_bot  = gmsh.model.occ.extrude([(1, arc_bot)], 0, 0, H / 2)
    gmsh.model.occ.synchronize()

    # ------------------------------------------------------------------
    # STEP 3: Build LEFT ARM by extruding the left quarter arc upward by H/2
    #
    # Arc at z=H/2: p_front_mid -> p_left_mid (left quarter, theta pi/2 -> pi).
    # We define this arc explicitly rather than reusing the extruded top arc
    # of the bottom arm (which spans the full half-circle).
    # ------------------------------------------------------------------
    arc_left_mid = gmsh.model.occ.addCircleArc(p_front_mid, p_axis_mid, p_left_mid)
    ext_top      = gmsh.model.occ.extrude([(1, arc_left_mid)], 0, 0, H / 2)
    gmsh.model.occ.synchronize()

    # ------------------------------------------------------------------
    # STEP 4: Remove axis helper points so they do not appear as isolated
    # nodes in the output .msh file.
    # ------------------------------------------------------------------
    gmsh.model.occ.remove([(0, p_axis_bot), (0, p_axis_mid)], recursive=False)
    gmsh.model.occ.synchronize()

    # ------------------------------------------------------------------
    # STEP 5: Set uniform mesh size and generate the surface mesh
    #
    # lc = arc-length of the bottom arc / (N-1) intervals.
    # ------------------------------------------------------------------
    lc = np.pi * R / (N - 1)
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), lc)

    gmsh.model.mesh.generate(2)

    # ------------------------------------------------------------------
    # STEP 6: Write to file and clean up
    # ------------------------------------------------------------------
    output_path = os.path.join(output_dir, "l_cylinder_surface.msh")
    gmsh.write(output_path)
    gmsh.finalize()

    return output_path