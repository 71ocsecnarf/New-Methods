import gmsh
import os

gmsh.initialize()
gmsh.model.add("half_cylinder_surface")

R = 0.50
H = 1.0
N_arc = 20   # points le long de l'arc
N_z   = 20   # points le long de la hauteur
N = 20
gmsh.initialize()
gmsh.model.add("cylinder_surface")

p_center = gmsh.model.occ.addPoint(0, 0, 0)
p_left   = gmsh.model.occ.addPoint(-R, 0, 0)
p_right  = gmsh.model.occ.addPoint( R, 0, 0)

arc_bot  = gmsh.model.occ.addCircleArc(p_right, p_center, p_left)
extruded = gmsh.model.occ.extrude([(1, arc_bot)], 0, 0, H)
gmsh.model.occ.synchronize()

surf_tag = [e[1] for e in extruded if e[0] == 2][0]

# 🔹 Taille de maille (clé pour non structuré)
lc = H / (N - 1)
gmsh.model.mesh.setSize(gmsh.model.getEntities(0), lc)

gmsh.model.addPhysicalGroup(2, [surf_tag], tag=1)
gmsh.model.setPhysicalName(2, 1, "lateral_face")

gmsh.model.mesh.generate(2)

# output_path = os.path.join(output_dir, "cylinder_surface.msh")
# gmsh.write(output_path)
gmsh.fltk.run()
gmsh.finalize()