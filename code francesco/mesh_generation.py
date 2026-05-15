import gmsh
import os
import numpy as np


# ---------------------------------------------------------------------------
# Public constants: output filenames keyed by mesh_type.
# Used by callers (main.py, accuracy_plot.py) to locate the written .msh file
# without reconstructing the name themselves.
# ---------------------------------------------------------------------------
MESH_FILENAMES = {
    'square_surface'   : 'square_surface.msh',
    'l_shape'          : 'l_shape.msh',
    'cylinder'         : 'cylinder_surface.msh',
    'hole'             : 'square_with_hole.msh',
    'l_cylinder'       : 'l_cylinder_surface.msh',
}


def generate_mesh(N, mesh_type, output_dir, L=1.0, R=0.5, H=1.0):
    """
    Generate a 2-D triangular surface mesh and write it to <output_dir>/<filename>.msh.

    Parameters
    ----------
    N           : int   -- mesh resolution (meaning depends on geometry, see below)
    mesh_type   : str   -- one of 'square_surface', 'l_shape', 'cylinder',
                           'hole', 'l_cylinder'
    output_dir  : str   -- directory where the .msh file is saved
    L           : float -- side length of the bounding square
                           (square_surface, l_shape, hole)
    R           : float -- cylinder radius OR hole radius
                           (cylinder, l_cylinder, hole)
    H           : float -- cylinder height
                           (cylinder, l_cylinder)

    Meaning of N per geometry
    -------------------------
    square_surface : N points along each transfinite edge  -> (N-1)^2 quads / 2*(N-1)^2 tris
    l_shape        : characteristic length lc = L/(N-1), unstructured Delaunay
    cylinder       : N points along arc and height, transfinite -> structured
    hole           : characteristic length lc = L/(N-1), unstructured Delaunay
    l_cylinder     : N points along the longest arc (pi*R), lc = pi*R/(N-1)

    Returns
    -------
    output_path : str -- absolute path to the written .msh file
    """
    if mesh_type not in MESH_FILENAMES:
        raise ValueError(
            f"Unknown mesh_type: '{mesh_type}'. "
            f"Valid options: {list(MESH_FILENAMES.keys())}"
        )

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add(mesh_type)

    if mesh_type == 'square_surface':
        build_square_surface(N, L)

    elif mesh_type == 'l_shape':
        build_l_shape(N, L)

    elif mesh_type == 'cylinder':
        build_cylinder(N, R, H)

    elif mesh_type == 'hole':
        build_hole(N, L, R)

    elif mesh_type == 'l_cylinder':
        build_l_cylinder(N, R, H)

    # --- Shared meshing options (applied to every geometry) ---
    # Algorithm 6 (Frontal-Delaunay) minimises obtuse triangles, which is
    # required for the FMM upwind condition to hold on all triangles.
    gmsh.option.setNumber("Mesh.Algorithm", 6)
    # 100 Laplacian smoothing passes further improve triangle quality,
    # especially important for the l_shape concave corner region.
    gmsh.option.setNumber("Mesh.Smoothing", 100)

    gmsh.model.mesh.generate(2)

    # Post-processing optimization for hole geometry only:
    # edge-swaps fix connectivity (obtuse triangles), then Laplacian re-positions nodes
    if mesh_type == 'hole':
        gmsh.model.mesh.optimize("")           # edge swaps
        gmsh.model.mesh.optimize("Laplace2D") # Laplacian after swaps

    output_path = os.path.join(output_dir, MESH_FILENAMES[mesh_type])
    gmsh.write(output_path)
    gmsh.finalize()

    return output_path


# ---------------------------------------------------------------------------
# Private geometry builders
# Each function assumes gmsh is already initialised and the model is open.
# It only builds the CAD geometry + sets mesh sizes; it does NOT call
# generate(2) or write() -- those are handled by the public function above.
# ---------------------------------------------------------------------------

def build_square_surface(N, L):
    """
    Unit square [0,L]^2 with a structured transfinite mesh.
    N points along each edge -> 2*(N-1)^2 right triangles.
    """
    gmsh.model.occ.add_rectangle(0, 0, 0, L, L)
    gmsh.model.occ.synchronize()

    for c in gmsh.model.getEntities(1):
        gmsh.model.mesh.setTransfiniteCurve(c[1], N)

    surface = gmsh.model.getEntities(2)[0][1]
    gmsh.model.mesh.setTransfiniteSurface(surface)


def build_l_shape(N, L):
    """
    L-shaped domain: full [0,L]^2 square minus the bottom-right quadrant
    (L/2 < x < L, 0 < y < L/2).
    The concave (shadow) corner is at (L/2, L/2).
    Unstructured Delaunay mesh with characteristic length lc = L/(N-1).
    """
    lc = L / (N - 1)

    p1 = gmsh.model.occ.addPoint(0,   0,   0)   # bottom-left
    p2 = gmsh.model.occ.addPoint(L/2, 0,   0)
    p3 = gmsh.model.occ.addPoint(L/2, L/2, 0)   # concave corner
    p4 = gmsh.model.occ.addPoint(L,   L/2, 0)
    p5 = gmsh.model.occ.addPoint(L,   L,   0)   # top-right
    p6 = gmsh.model.occ.addPoint(0,   L,   0)   # top-left

    lines = [
        gmsh.model.occ.addLine(p1, p2),
        gmsh.model.occ.addLine(p2, p3),
        gmsh.model.occ.addLine(p3, p4),
        gmsh.model.occ.addLine(p4, p5),
        gmsh.model.occ.addLine(p5, p6),
        gmsh.model.occ.addLine(p6, p1),
    ]

    loop    = gmsh.model.occ.addCurveLoop(lines)
    surface = gmsh.model.occ.addPlaneSurface([loop])
    gmsh.model.occ.synchronize()

    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), lc)


def build_cylinder(N, R, H):
    """
    Half-cylinder lateral surface (front half: theta in [0, pi], y >= 0).
    Structured transfinite mesh: N points along the arc and along the height.

    Extrusion of the bottom arc (theta in [0,pi] at z=0) by H in the z-direction
    produces a conforming structured mesh on the curved surface.
    """
    p_center = gmsh.model.occ.addPoint(0,  0, 0)
    p_left   = gmsh.model.occ.addPoint(-R, 0, 0)
    p_right  = gmsh.model.occ.addPoint( R, 0, 0)

    arc_bot  = gmsh.model.occ.addCircleArc(p_right, p_center, p_left)
    extruded = gmsh.model.occ.extrude([(1, arc_bot)], 0, 0, H)
    gmsh.model.occ.synchronize()

    surf_tag       = next(e[1] for e in extruded if e[0] == 2)
    lines          = [e[1] for e in extruded if e[0] == 1]
    arc_top_tag    = lines[1]
    line_right_tag = lines[0]
    line_left_tag  = lines[2]

    gmsh.model.mesh.setTransfiniteCurve(arc_bot,        N)
    gmsh.model.mesh.setTransfiniteCurve(arc_top_tag,    N)
    gmsh.model.mesh.setTransfiniteCurve(line_left_tag,  N)
    gmsh.model.mesh.setTransfiniteCurve(line_right_tag, N)
    gmsh.model.mesh.setTransfiniteSurface(surf_tag)

    gmsh.model.addPhysicalGroup(2, [surf_tag], tag=1)
    gmsh.model.setPhysicalName(2, 1, "lateral_face")

"""
def build_hole(N, L, R):
    
    Square [0,L]^2 with a circular hole of radius R centred at (L/2, L/2).
    Boolean cut followed by unstructured Delaunay mesh, lc = L/(N-1).
    Curvature-based refinement near the hole boundary is also enabled.
    
    lc = L / (N - 1)

    rect   = gmsh.model.occ.addRectangle(0, 0, 0, L, L)
    circle = gmsh.model.occ.addDisk(L/2, L/2, 0, R, R)
    gmsh.model.occ.cut([(2, rect)], [(2, circle)])
    gmsh.model.occ.synchronize()

    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), lc)
    # Extra refinement near the curved hole boundary
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 20)

    surfaces = gmsh.model.getEntities(2)
    gmsh.model.addPhysicalGroup(2, [s[1] for s in surfaces], 1)
    gmsh.model.setPhysicalName(2, 1, "domain")

    curves = gmsh.model.getEntities(1)
    gmsh.model.addPhysicalGroup(1, [c[1] for c in curves], 2)
    gmsh.model.setPhysicalName(1, 2, "boundary")
"""

def build_hole(N, L, R):
    """
    Square [0,L]^2 with a circular hole of radius R centred at (L/2, L/2).
    Boolean cut followed by unstructured Delaunay mesh, lc = L/(N-1).
    Uniform characteristic length everywhere -- no curvature-based refinement,
    which caused obtuse triangles at the domain boundary far from the hole.
    """
    lc = L / (N - 1)

    rect   = gmsh.model.occ.addRectangle(0, 0, 0, L, L)
    circle = gmsh.model.occ.addDisk(L/2, L/2, 0, R, R)
    gmsh.model.occ.cut([(2, rect)], [(2, circle)])
    gmsh.model.occ.synchronize()

    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), lc)

    surfaces = gmsh.model.getEntities(2)
    gmsh.model.addPhysicalGroup(2, [s[1] for s in surfaces], 1)
    gmsh.model.setPhysicalName(2, 1, "domain")

    curves = gmsh.model.getEntities(1)
    gmsh.model.addPhysicalGroup(1, [c[1] for c in curves], 2)
    gmsh.model.setPhysicalName(1, 2, "boundary")


def build_l_cylinder(N, R, H):
    """
    L-shaped half-cylinder lateral surface.

    Geometry: half-cylinder (theta in [0, pi]) with one quadrant removed.
      - Bottom arm : theta in [0,    pi],  z in [0,   H/2]
      - Left arm   : theta in [pi/2, pi],  z in [H/2, H  ]
      - Missing quadrant: theta in [0, pi/2], z in [H/2, H]
      - Concave corner: (0, R, H/2)

    Construction strategy:
      1. Build the bottom arm by extruding the full bottom arc up by H/2.
      2. Build the left arm by extruding the left-quarter arc up by H/2.
      3. Call fragment() to force a conforming shared edge at z=H/2,
         theta in [pi/2, pi], so that the FMM can propagate across both patches.

    Characteristic length: lc = pi*R / (N-1)  (based on the longest arc).
    """
    theta_right = 0.0
    theta_mid   = np.pi / 2.0
    theta_left  = np.pi

    def cyl_pt(theta, z):
        return gmsh.model.occ.addPoint(R * np.cos(theta), R * np.sin(theta), z)

    # Corner points
    p_right_bot = cyl_pt(theta_right, 0.0)
    p_left_bot  = cyl_pt(theta_left,  0.0)
    _           = cyl_pt(theta_right, H / 2.0)   # p_right_mid: created implicitly by extrude
    p_front_mid = cyl_pt(theta_mid,   H / 2.0)
    p_left_mid  = cyl_pt(theta_left,  H / 2.0)
    _           = cyl_pt(theta_mid,   H)          # p_front_top
    _           = cyl_pt(theta_left,  H)          # p_left_top

    # Axis helper points (arc centres only, removed after fragment)
    p_axis_bot = gmsh.model.occ.addPoint(0.0, 0.0, 0.0)
    p_axis_mid = gmsh.model.occ.addPoint(0.0, 0.0, H / 2.0)

    # Bottom arm: extrude full arc [0, pi] at z=0 upward by H/2
    arc_bot = gmsh.model.occ.addCircleArc(p_right_bot, p_axis_bot, p_left_bot)
    ext_bot  = gmsh.model.occ.extrude([(1, arc_bot)], 0, 0, H / 2.0)
    gmsh.model.occ.synchronize()
    surf_bot = next(e[1] for e in ext_bot if e[0] == 2)

    # Left arm: extrude left-quarter arc [pi/2, pi] at z=H/2 upward by H/2
    arc_left_mid = gmsh.model.occ.addCircleArc(p_front_mid, p_axis_mid, p_left_mid)
    ext_top      = gmsh.model.occ.extrude([(1, arc_left_mid)], 0, 0, H / 2.0)
    gmsh.model.occ.synchronize()
    surf_top = next(e[1] for e in ext_top if e[0] == 2)

    # Merge patches: fragment() creates a single shared edge at the interface,
    # guaranteeing conforming (no-duplicate) mesh nodes across both patches.
    gmsh.model.occ.fragment([(2, surf_bot)], [(2, surf_top)])
    gmsh.model.occ.synchronize()

    # Remove axis helper points (not part of the surface)
    for pt in [p_axis_bot, p_axis_mid]:
        try:
            gmsh.model.occ.remove([(0, pt)], recursive=False)
        except Exception:
            pass  # already removed implicitly by fragment() -- safe to ignore
    gmsh.model.occ.synchronize()

    lc = np.pi * R / (N - 1)
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), lc)
