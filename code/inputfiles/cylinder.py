import gmsh

gmsh.initialize()
gmsh.model.add("cylinder_surface")

R = 0.50
H = 1.0
lc = 0.05

gmsh.model.occ.addCylinder(0, 0, 0, 0, 0, H, R)
gmsh.model.occ.synchronize()

#!Used for quads:::::::::::::::::::::::::::::::::::::::::
# gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)
# gmsh.option.setNumber("Mesh.RecombineAll", 1)
#!Used for quads:::::::::::::::::::::::::::::::::::::::::


gmsh.model.mesh.setSize(gmsh.model.getEntities(0), lc)
gmsh.model.mesh.generate(2)

gmsh.write("cylinder_surface.msh")
gmsh.fltk.run()
gmsh.finalize()