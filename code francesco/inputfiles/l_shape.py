import gmsh
import os

gmsh.initialize()
gmsh.model.add("l_shape")

L = 1.0 
# Number of points along the long outer edges
N = 20
# Characteristic size based on N
lc = L / (N - 1)

# --- Rotated 180 degrees L-shape geometry (pillar on the left, arm on top) ---
# Let's define the 6 points for the boundary. The missing gap is in the bottom-right.
# Points in order: (0,0) -> (L/2, 0) -> (L/2, L/2) -> (L, L/2) -> (L, L) -> (0, L)
p1 = gmsh.model.occ.addPoint(0, 0, 0)           # Outer bottom-left corner
p2 = gmsh.model.occ.addPoint(L/2, 0, 0)
p3 = gmsh.model.occ.addPoint(L/2, L/2, 0)     # Inner corner (concave) - P_c
p4 = gmsh.model.occ.addPoint(L, L/2, 0)
p5 = gmsh.model.occ.addPoint(L, L, 0)           # Outer top-right corner
p6 = gmsh.model.occ.addPoint(0, L, 0)           # Outer top-left corner

# Creation of the perimeter segments
l1 = gmsh.model.occ.addLine(p1, p2)
l2 = gmsh.model.occ.addLine(p2, p3)
l3 = gmsh.model.occ.addLine(p3, p4)
l4 = gmsh.model.occ.addLine(p4, p5)
l5 = gmsh.model.occ.addLine(p5, p6)
l6 = gmsh.model.occ.addLine(p6, p1)

# Surface definition
loop = gmsh.model.occ.addCurveLoop([l1, l2, l3, l4, l5, l6])
surface = gmsh.model.occ.addPlaneSurface([loop])

gmsh.model.occ.synchronize()

# --- Mesh Settings ---
# Apply the size lc defined above to all points
gmsh.model.mesh.setSize(gmsh.model.getEntities(0), lc)

# Generation of the mesh (unstructured triangles)
gmsh.model.mesh.generate(2)

# Save the .msh file
current_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(current_dir, "l_shape.msh")

gmsh.write(output_path)

# Start the graphical interface to view the result
gmsh.fltk.run()
gmsh.finalize()