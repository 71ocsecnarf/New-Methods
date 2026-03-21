import numpy as np
import os
import matplotlib.pyplot as plt

import gmsh
import solver

def main():
    gmsh.initialize()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    filename = current_dir + "/inputfiles/square_surface.msh"
    gmsh.open(filename)
    #*To avoid having gmsh msg in the terminal
    gmsh.option.setNumber("General.Terminal", 0)

    #todo--> do we need to keep this dictionnary ??
    #*The node tag is the key, the value is the NODE object of that node
    tag_to_node_obj_dic = {}#*dictionary, works with GMSH tag
    node_list = solver.Make_NodeList_NodeDictionary(filename, tag_to_node_obj_dic)
    element_list = solver.Make_ElementList(filename, tag_to_node_obj_dic)
    edge_dict = solver.Make_EdgeList(element_list)

    #*======================================= |
    #*====USED TO TEST THE NODE LIST========= |
    #*======================================= V
    # plt.figure()
    # for n in node_list:
    #     plt.scatter(n.coords[0], n.coords[1], color = 'black')

    #     #! it is just for checking, each edge is plotted twice... frome node A --> B then B-->A but whatever, it is just to see 
    #     #!if my node_list works well...
    #     for neighbor in n.neighbors:
    #         plt.plot([n.coords[0], neighbor.coords[0]], [n.coords[1], neighbor.coords[1]], color = "red")
    # plt.show()
    #*======================================= ^
    #*====USED TO TEST THE NODE LIST========= |
    #*======================================= |

    solver.Check_Obtuse_triangles(element_list)
    target_coords = np.array([0.50, 0.50, 0.])
    source_point, target_elem = solver.Innit_Origin_Point(target_coords, node_list)

    #*========================================== |
    #*====USED TO TEST THE ELEMENT LIST========= |
    #*========================================== V
    # plt.figure()
    # size = 0.5
    # for e in element_list:
    #     x = []
    #     y = []
    #     for n in e.nodes:
    #         plt.scatter(n.coords[0], n.coords[1], color = 'black')
    #         x.append(n.coords[0])
    #         y.append(n.coords[1])
            
    #     plt.plot([x[0], x[1]], [y[0], y[1]], color = 'red', linewidth = size)
    #     plt.plot([x[0], x[2]], [y[0], y[2]], color = 'red', linewidth = size)
    #     plt.plot([x[1], x[2]], [y[1], y[2]], color = 'red', linewidth = size)
    
    # plt.scatter(source_point[0], source_point[1], color = 'gold')

    # plt.show()
    #*========================================== ^
    #*====USED TO TEST THE ELEMENT LIST========= |
    #*========================================== |


    #! Innit the FFM algorithm:
    trial_band = solver.Innit_FFM(target_coords, target_elem)
    #solver.plot_kappa_edges(edge_dict)
    solver.FMM(trial_band, edge_dict, node_list)
    print("node list size = ", len(node_list))
    
    print("Calcul terminé, génération du tracé...")
    solver.Plot_Isolines(node_list, element_list)
    solver.plot_Txy(element_list, 50, edge_dict)
    #solver.plot_kappa_edges(edge_dict)
    gmsh.finalize()

def generate_mesh(N, current_dir):
    """Regénère square_surface.msh avec N points par edge."""
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("square_surface")

    L = 1.0
    gmsh.model.occ.add_rectangle(0, 0, 0, L, L)
    gmsh.model.occ.synchronize()

    curves = gmsh.model.getEntities(1)
    for c in curves:
        gmsh.model.mesh.setTransfiniteCurve(c[1], N)

    surface = gmsh.model.getEntities(2)[0][1]
    gmsh.model.mesh.setTransfiniteSurface(surface)

    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), 1.0 / (N - 1))
    gmsh.model.mesh.generate(2)

    output_path = os.path.join(current_dir, "inputfiles", "square_surface.msh")
    gmsh.write(output_path)
    gmsh.finalize()


def run_fmm_for_mesh(filename, target_coords):
    """Lance le pipeline FMM complet et retourne node_list."""
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.open(filename)

    tag_to_node_obj_dic = {}
    node_list    = solver.Make_NodeList_NodeDictionary(filename, tag_to_node_obj_dic)
    element_list = solver.Make_ElementList(filename, tag_to_node_obj_dic)
    edge_dict    = solver.Make_EdgeList(element_list)

    source_point, target_elem = solver.Innit_Origin_Point(target_coords, node_list)
    trial_band = solver.Innit_FFM(target_coords, target_elem)
    solver.FMM(trial_band, edge_dict, node_list)

    gmsh.finalize()
    return node_list, edge_dict


def convergence_study():
    current_dir  = os.path.dirname(os.path.abspath(__file__))
    filename     = os.path.join(current_dir, "inputfiles", "square_surface.msh")
    target_coords = np.array([.5, .5, 0.])


    N_values = [21, 41, 81, 161, 201, 501, 701, 801]

    h_list        = []
    err_max_list  = []
    err_l2_list   = []

    for N in N_values:
        print(f"\n{'='*50}")
        print(f"  N = {N}  (h = {1.0/(N-1):.4f})")
        print(f"{'='*50}")

        generate_mesh(N, current_dir)
        node_list, edge_dict = run_fmm_for_mesh(filename, target_coords)

        h = 1.0 / (N - 1)

        errors = []
        for node in node_list:
            d_exact = np.linalg.norm(node.coords[:2] - target_coords[:2])
            d_fmm   = node.dist
            if d_exact > 1e-10:   # on exclut le point source lui-même
                errors.append(abs(d_fmm - d_exact))

        err_max = np.max(errors)
        err_l2  = np.sqrt(np.mean(np.array(errors)**2))

        print(f"  err_max = {err_max:.6f}   err_L2 = {err_l2:.6f}")

        h_list.append(h)
        err_max_list.append(err_max)
        err_l2_list.append(err_l2)

    
    log_h   = np.log(h_list)
    slope_max, _ = np.polyfit(log_h, np.log(err_max_list), 1)
    slope_l2,  _ = np.polyfit(log_h, np.log(err_l2_list),  1)

    print(f"\nPente log-log  err_max : {slope_max:.2f}")
    print(f"Pente log-log  err_L2  : {slope_l2:.2f}")

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.loglog(h_list, err_max_list, 'o-', label=f"err max  (pente ≈ {slope_max:.2f})")
    ax.loglog(h_list, err_l2_list,  's-', label=f"err L2   (pente ≈ {slope_l2:.2f})")

    # Références de pente
    h_arr = np.array(h_list)
    ax.loglog(h_arr, h_arr**1 * err_max_list[0]/h_list[0]**1,
              'k--', linewidth=0.8, label="ordre 1 (ref)")
    ax.loglog(h_arr, h_arr**2 * err_max_list[0]/h_list[0]**2,
              'k:',  linewidth=0.8, label="ordre 2 (ref)")

    ax.set_xlabel("Taille de maille  h = 1/(N-1)")
    ax.set_ylabel("Erreur")
    ax.set_title("Convergence FMM ordre 2")
    ax.legend()
    ax.grid(True, which='both', linestyle=':', linewidth=0.5)
    plt.tight_layout()
    plt.show()

def plot_error_map(N, current_dir):
    """Plot la carte d'erreur spatiale pour un N donné."""
    filename      = os.path.join(current_dir, "inputfiles", "square_surface.msh")
    target_coords = np.array([.5, .5, 0.])

    generate_mesh(N, current_dir)
    node_list, edge_dict = run_fmm_for_mesh(filename, target_coords)

    x      = np.array([n.coords[0] for n in node_list])
    y      = np.array([n.coords[1] for n in node_list])
    d_fmm  = np.array([n.dist      for n in node_list])
    d_exact= np.linalg.norm(
                np.stack([n.coords[:2] for n in node_list]) - target_coords[:2],
                axis=1)

    err = np.abs(d_fmm - d_exact)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 1. Distance FMM
    sc0 = axes[0].scatter(x, y, c=d_fmm, cmap='viridis', s=8)
    plt.colorbar(sc0, ax=axes[0])
    axes[0].set_title("d_FMM")
    axes[0].set_aspect('equal')

    # 2. Distance exacte
    sc1 = axes[1].scatter(x, y, c=d_exact, cmap='viridis', s=8)
    plt.colorbar(sc1, ax=axes[1])
    axes[1].set_title("d_exact")
    axes[1].set_aspect('equal')

    # 3. Erreur absolue — logscale
    err_safe = np.where(d_exact > 1e-10, err, np.nan)
    sc2 = axes[2].scatter(x, y, c=np.log10(err_safe), cmap='hot', s=8)
    plt.colorbar(sc2, ax=axes[2], label="log10(|err|)")
    axes[2].set_title("log10(erreur absolue)")
    axes[2].set_aspect('equal')

    plt.suptitle(f"N={N},  h={1/(N-1):.4f},  err_max={np.nanmax(err_safe):.2e}")
    plt.tight_layout()
    plt.show()

    # Stats par zone
    r = d_exact
    mask_near = (r < 0.1) & (d_exact > 1e-10)
    mask_mid  = (r >= 0.1) & (r < 0.35)
    mask_far  = (r >= 0.35)

    print(f"\nErreur par zone (N={N}):")
    print(f"  proche  (r<0.10)  : max={np.max(err[mask_near]):.2e}  mean={np.mean(err[mask_near]):.2e}")
    print(f"  milieu  (0.1-0.35): max={np.max(err[mask_mid ]):.2e}  mean={np.mean(err[mask_mid ]):.2e}")
    print(f"  loin    (r>0.35)  : max={np.max(err[mask_far ]):.2e}  mean={np.mean(err[mask_far ]):.2e}")

if __name__ == "__main__":
    # convergence_study()   # <-- remplace main() pour l'étude de convergence
    main()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    plot_error_map(101, current_dir)
