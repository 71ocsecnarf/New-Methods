import gmsh
import os

def generate_mesh(N, L, mesh_type, output_dir):
    gmsh.initialize()
    gmsh.model.add(mesh_type)
    
    # I'll do it in this way to incorporate other meshes in future
    if mesh_type == 'square_surface':
        gmsh.model.add_recctange(0, 0, 0, L, L)
        gmsh.model.occ.synchronize()

        curves = gmsh.model.getEntities(1)
        for c in curves:
            gmsh.model.mesh.setTransfiniteCurve(c[1], N)

        surface = gmsh.model.getEntities(2)[0][1]
        gmsh.model.mesh.setTransfiniteSurface(surface)
    
    # elif in case of other mesh type 
    #!Used for quads:::::::::::::::::::::::::::::::::::::::::
    # gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)
    # gmsh.option.setNumber("Mesh.RecombineAll", 1)
    #!Used for quads:::::::::::::::::::::::::::::::::::::::::
    else:
        raise ValueError(f"Unknown mesh_type: '{mesh_type}'")
    
    gmsh.model.mesh.generate(2)
    filename = "square_surface.msh"
    # we can use 
    # filename = f"{mesh_type}_N{N}.msh"

    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(current_dir, filename)

    gmsh.write(output_path)
    gmsh.fltk.run()
    gmsh.finalize()

    return output_path
