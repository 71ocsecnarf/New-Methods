import numpy as np
import os
import gmsh

import mesh_generation
import solver

def find_non_obtuse_meshes_exp():
    mesh_type = 'l_shape'
    L = 1.0
    
    # I tuoi target ideali: [10, 20, 40, 80, 160, 320]
    targets = [10 * 2**i for i in range(6)]
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    valid_results = []

    print(f"{'='*50}")
    print("Ricerca mesh non ottuse per L-shape (Spaziatura Esponenziale)")
    print(f"{'='*50}\n")

    for target in targets:
        found_for_target = False
        
        # Cerchiamo in un raggio intorno al target (es: +/- 5 nodi)
        # Per i target più grandi, possiamo ampliare dinamicamente la ricerca
        search_radius = max(5, int(target * 0.1)) 
        
        # Creiamo un ordine di ricerca che parta dal target e si allarghi:
        # [target, target+1, target-1, target+2, target-2, ...]
        offsets = [0]
        for step in range(1, search_radius + 1):
            offsets.extend([step, -step])
            
        print(f"--- Cerco una mesh valida intorno a N = {target} ---")
        
        for offset in offsets:
            N = target + offset
            if N < 3: # N deve essere almeno 3 per fare un triangolo
                continue
                
            h = L / (N - 1)
            
            try:
                # 1. Generazione della mesh
                mesh_generation.generate_mesh(N, L, mesh_type, current_dir)
                actual_output_path = os.path.join(current_dir, f"{mesh_type}.msh")

                # 2. Riapertura
                gmsh.initialize()
                gmsh.option.setNumber("General.Terminal", 0)
                gmsh.open(actual_output_path)

                # 3. Node list e element list
                tag_to_node_obj_dic = {}
                node_list = solver.Make_NodeList_NodeDictionary(actual_output_path, tag_to_node_obj_dic)
                element_list = solver.Make_ElementList(actual_output_path, tag_to_node_obj_dic)
                
                # 4. Controllo dei triangoli ottusi
                solver.Check_Obtuse_triangles(element_list)
                obtuse_count = sum(1 for el in element_list if el.IsObtuse)
                
                gmsh.finalize()

                if obtuse_count == 0:
                    print(f"✅ TROVATA per target ~{target}: N={N:3d} (offset {offset:+d}) | h={h:.6f}\n")
                    valid_results.append((N, h))
                    found_for_target = True
                    break  # Usciamo dal loop degli offset, passiamo al prossimo target!
                    
            except Exception as e:
                # print(f"⚠️ Errore con N={N}: {e}")
                if gmsh.isInitialized():
                    gmsh.finalize()

        if not found_for_target:
            print(f"❌ Nessuna mesh valida trovata nei dintorni di {target} (raggio ±{search_radius}).\n")

    # 5. Salvataggio
    print(f"{'='*50}")
    if valid_results:
        save_path = os.path.join(current_dir, "valid_h_l_shape_exp.txt")
        with open(save_path, "w") as f:
            f.write("N\th\n")
            for res in valid_results:
                f.write(f"{res[0]}\t{res[1]:.6f}\n")
        print(f"🎉 Ricerca completata! I risultati sono stati salvati in: {save_path}\n")
        
        # Stampa la riga Python pronta da incollare nel tuo file accuracy_plot.py
        n_array_str = "[" + ", ".join([str(res[0]) for res in valid_results]) + "]"
        print(f"👉 INCOLLA QUESTO IN accuracy_plot.py:")
        print(f"N_VALUES = {n_array_str}")
        print(f"{'='*50}")

if __name__ == "__main__":
    find_non_obtuse_meshes_exp()