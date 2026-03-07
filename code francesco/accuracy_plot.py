import numpy as np
import os
import matplotlib.pyplot as plt

import gmsh
import solver
import fmm_solver
import mesh_generation


def main():

    # ----> A - CLOSE ALL TABS - <----
    plt.close('all')


    # ----> B - INITIALIZATION - <----
    N_values = [5 * 2**i for i in range(6)]
    source_coord = np.array([0.5, 0.5, 0.0])
    mesh_type = 'square_surface'    

    current_dir  = os.path.dirname(os.path.abspath(__file__))
    output_path  = os.path.join(current_dir, "square_surface.msh")

    h_values = []
    errors = []
    rel_errors = []
    errors_l2 = []

    # ----> C - LOOP - <----

    for N in N_values:

        # Print the current computation
        print(f"\n{'='*50}")
        print(f"Running N={N}  -->  h={1.0/(N-1):.4f}")
        print(f"{'='*50}")

        # 1 ---> Mesh Generation
        generate_mesh(N, mesh_type, output_path)


        # 2 ---> Mesh Opening



    # ----> D - POST PROCESS - <----






if __name__ == "__main__":
    main()