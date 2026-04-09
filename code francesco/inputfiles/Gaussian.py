import gmsh
import numpy as np
import os

def generate_gaussian_surface_open(N, Rmax, A, sigma, lc, output_dir):

    gmsh.initialize()
    gmsh.model.add("gaussian_surface_open")

    rs = np.linspace(0, Rmax, N)
    pts = []

    for r in rs:
        z = A * np.exp(-r**2 / sigma**2)
        pts.append(gmsh.model.occ.addPoint(r, 0, z, lc))

    spline = gmsh.model.occ.addSpline(pts)

    gmsh.model.occ.synchronize()

    # --- révolution → surface ouverte ---
    angle = 2 * np.pi
    gmsh.model.occ.revolve([(1, spline)], 0, 0, 0, 0, 0, 1, angle)

    gmsh.model.occ.synchronize()

    # --- contrôle global du mesh ---
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", lc)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc)

    # optionnel: meilleur mesh
    # gmsh.option.setNumber("Mesh.Algorithm", 6)  # frontal

    # --- mesh surface ---
    gmsh.model.mesh.generate(2)

    output_path = os.path.join(output_dir, "gaussian_surface_open.msh")
    gmsh.write(output_path)

    gmsh.fltk.run()
    gmsh.finalize()

    return output_path


current_dir = os.path.dirname(os.path.abspath(__file__))
generate_gaussian_surface_open(
    N=100,
    Rmax=2,
    A=1,
    sigma=0.3,
    lc=0.05,
    output_dir=current_dir
)