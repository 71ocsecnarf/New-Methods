import gmsh
import os

def generate_square_with_hole(N, L, R, output_dir):

    gmsh.initialize()
    gmsh.model.add("square_with_hole")

    # --- Géométrie ---
    # Carré
    rect = gmsh.model.occ.addRectangle(0, 0, 0, L, L)

    # Cercle (trou)
    circle = gmsh.model.occ.addDisk(L/2, L/2, 0, R, R)

    # Soustraction booléenne
    domain = gmsh.model.occ.cut([(2, rect)], [(2, circle)])
    gmsh.model.occ.synchronize()

    # --- Taille de maille ---
    lc = L / (N - 1)
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), lc)

    # Raffinement proche du trou (optionnel mais recommandé)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 20)

    # --- Physical groups ---
    surfaces = gmsh.model.getEntities(2)
    gmsh.model.addPhysicalGroup(2, [s[1] for s in surfaces], 1)
    gmsh.model.setPhysicalName(2, 1, "domain")

    # Bord extérieur / trou
    curves = gmsh.model.getEntities(1)
    gmsh.model.addPhysicalGroup(1, [c[1] for c in curves], 2)
    gmsh.model.setPhysicalName(1, 2, "boundary")

    # --- Mesh ---
    gmsh.model.mesh.generate(2)

    output_path = os.path.join(output_dir, "square_with_hole.msh")
    gmsh.write(output_path)
    gmsh.fltk.run()
    gmsh.finalize()

    return output_path

current_dir = os.path.dirname(os.path.abspath(__file__))
generate_square_with_hole(30, 1, 0.2, current_dir)