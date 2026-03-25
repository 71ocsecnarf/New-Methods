import gmsh
import os

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
        # Corner P3 (L/2, L/2) acts as the shadow corner[cite: 37, 38].
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
 
# Dans mesh_generation.py
 
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
    # Pas de setRecombine -> triangles
 
    gmsh.model.addPhysicalGroup(2, [surf_tag], tag=1)
    gmsh.model.setPhysicalName(2, 1, "lateral_face")
 
    gmsh.model.mesh.generate(2)
 
    output_path = os.path.join(output_dir, "cylinder_surface.msh")
    gmsh.write(output_path)
    gmsh.finalize()
 
    return output_path