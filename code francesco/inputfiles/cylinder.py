import gmsh
import os

gmsh.initialize()
gmsh.model.add("half_cylinder_surface")

R = 0.50
H = 1.0
N_arc = 200   # points le long de l'arc
N_z   = 200   # points le long de la hauteur

# --- Points du demi-cercle bas (z=0) ---
p_center = gmsh.model.occ.addPoint(0, 0, 0)
p_left   = gmsh.model.occ.addPoint(-R, 0, 0)
p_right  = gmsh.model.occ.addPoint( R, 0, 0)

# --- Arc bas + extrusion ---
arc_bot  = gmsh.model.occ.addCircleArc(p_right, p_center, p_left)
extruded = gmsh.model.occ.extrude([(1, arc_bot)], 0, 0, H)
gmsh.model.occ.synchronize()

# --- Récupérer les entités ---
surf_tag      = [e[1] for e in extruded if e[0] == 2][0]
arc_top_tag   = [e[1] for e in extruded if e[0] == 1][1]  # arc du haut
line_left_tag = [e[1] for e in extruded if e[0] == 1][2]  # ligne verticale gauche
line_right_tag= [e[1] for e in extruded if e[0] == 1][0]  # ligne verticale droite

# --- Mesh structuré ---
gmsh.model.mesh.setTransfiniteCurve(arc_bot,       N_arc)
gmsh.model.mesh.setTransfiniteCurve(arc_top_tag,   N_arc)
gmsh.model.mesh.setTransfiniteCurve(line_left_tag, N_z)
gmsh.model.mesh.setTransfiniteCurve(line_right_tag,N_z)
gmsh.model.mesh.setTransfiniteSurface(surf_tag)

# Pour avoir des quads au lieu de triangles rectangles :
#gmsh.model.mesh.setRecombine(2, surf_tag)

# --- Groupe physique ---
gmsh.model.addPhysicalGroup(2, [surf_tag], tag=1)
gmsh.model.setPhysicalName(2, 1, "lateral_face")

gmsh.model.mesh.generate(2)

current_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(current_dir, "cylinder_surface.msh")
gmsh.write(output_path)
gmsh.fltk.run()
gmsh.finalize()