import gmsh
import os

gmsh.initialize()
gmsh.model.add("square_surface")

L = 1.0 
lc = 0.025 #Mesh size

gmsh.model.occ.add_rectangle(0, 0, 0, L, L)
gmsh.model.occ.synchronize()

#*the number of point along the edge
N = 20

curves = gmsh.model.getEntities(1)
for c in curves:
    gmsh.model.mesh.setTransfiniteCurve(c[1], N)

# Récupérer la surface
surface = gmsh.model.getEntities(2)[0][1]
gmsh.model.mesh.setTransfiniteSurface(surface)

#!Used for quads:::::::::::::::::::::::::::::::::::::::::
# gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)
# gmsh.option.setNumber("Mesh.RecombineAll", 1)
#!Used for quads:::::::::::::::::::::::::::::::::::::::::


gmsh.model.mesh.setSize(gmsh.model.getEntities(0), lc)
gmsh.model.mesh.generate(2)

current_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(current_dir, "square_surface.msh")

gmsh.write(output_path)
gmsh.fltk.run()
gmsh.finalize()