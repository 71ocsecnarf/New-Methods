import numpy as np
import gmsh
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import imageio
import glob


from structures import NODE
from structures import ELEMENT
from structures import TRIAL_BAND
from structures import EDGE

#!==================================================================== |
#!==================================================================== |
#!========================= Preprocessing============================= |
#!==================================================================== |
#!==================================================================== V

def Make_NodeList_NodeDictionary(filename, tag_to_node_obj_dic):

    if not gmsh.isInitialized():
        gmsh.initialize()
    
    print("\n")
    print("=================================================")
    print("================Making node list=================")
    print("=================================================")

    #!node_tags, it is like ids, written as (1, 2, 3...) --> starts at 1!
    #!coords, written as [x1, y1, z1, x2, y2, z2...]
    node_tags, coords, _ = gmsh.model.mesh.getNodes() #* third return is parametricCoords, if you write "_" it means that you don't want it

    #*New dimensions are (N, 3)
    points = coords.reshape(-1, 3)

    nodes_list = []
    
    for i, tag in enumerate(node_tags):
        new_node = NODE(points[i], tag, i)
        nodes_list.append(new_node)
        tag_to_node_obj_dic[tag] = new_node
    
    #*node_tags_per_elem looks like [ [n1, n2, n3, n4, n5, n6...] ]
    _, _, node_tags_per_elem = gmsh.model.mesh.getElements(2)
    triangles_tags = node_tags_per_elem[0].reshape(-1, 3)

    #*tri = [n1 n2 n3], [n4 n5 n6] and so on
    for tri in triangles_tags:

        #*Get back the NODE object with the dictionary
        node_a = tag_to_node_obj_dic[tri[0]]
        node_b = tag_to_node_obj_dic[tri[1]]
        node_c = tag_to_node_obj_dic[tri[2]]
        
        #*In a triangle, A has B and C as neighboors and so on
        node_a.neighbors.update([node_b, node_c])
        node_b.neighbors.update([node_a, node_c])
        node_c.neighbors.update([node_a, node_b])

    print("Node list is done --> ok")
    return np.array(nodes_list)


def Make_ElementList(filename, tag_to_node_obj_dic):

    if not gmsh.isInitialized():
        gmsh.initialize()
    
    gmsh.open(filename)
    print("\n")
    print("=================================================")
    print("===============Making element list===============")
    print("=================================================")

    _, _, node_tags_per_elem = gmsh.model.mesh.getElements(2)
    triangles_tags = node_tags_per_elem[0].reshape(-1, 3)

    element_list = []
    for tri in triangles_tags:
        node_a = tag_to_node_obj_dic[tri[0]]
        node_b = tag_to_node_obj_dic[tri[1]]
        node_c = tag_to_node_obj_dic[tri[2]]
        nodes = [node_a, node_b, node_c]
        element = ELEMENT(nodes)
        element_list.append(element)
        # Add the triangle to the adjacent triangle list of each node
        #! I do not know if we will need it

        #*Here, i add all the elements a node belong to
        node_a.adjacent_triangles.append(element)
        node_b.adjacent_triangles.append(element)
        node_c.adjacent_triangles.append(element)

    print("Element list is done --> ok")
    return np.array(element_list)

def Make_EdgeList(element_list):
    edge_dict = {}  # key = (min_tag, max_tag)
    
    for elem in element_list:
        nodes = elem.nodes
        pairs = [(nodes[0], nodes[1]), 
                 (nodes[1], nodes[2]), 
                 (nodes[0], nodes[2])]
        
        for na, nb in pairs:
            key = (min(na.node_tag, nb.node_tag), 
                   max(na.node_tag, nb.node_tag))
            
            if key not in edge_dict:
                edge = EDGE(na, nb)
                edge_dict[key] = edge
            
            edge_dict[key].adjacent_triangles.append(elem)
            elem.edges.append(edge_dict[key])
    
    return edge_dict
def Check_Obtuse_triangles(element_list):

    print("\n")
    print("========================================================")
    print("=============Cheking for obtuse element=================")
    print("========================================================")
    counting = 0
    for e in element_list:
        AB = np.array([e.nodes[0].coords[0] - e.nodes[1].coords[0], e.nodes[0].coords[1] - e.nodes[1].coords[1]])
        AC = np.array([e.nodes[0].coords[0] - e.nodes[2].coords[0], e.nodes[0].coords[1] - e.nodes[2].coords[1]])
        BC = np.array([e.nodes[1].coords[0] - e.nodes[2].coords[0], e.nodes[1].coords[1] - e.nodes[2].coords[1]])

        AB_AC = AB @ AC
        BC_BA = BC @ (-AB)
        CA_CB = (-AC) @ (-BC)
        if min(AB_AC, BC_BA, CA_CB) < -1e-6:
            e.IsObtuse = True
            counting += 1
    
    print(f"There are {counting} obtuse elements")
    print("Checking for obtuse element is done --> ok\n")

def Compute_Dist_Point_Triangle(P, element):

    A, B, C = [n.coords for n in element.nodes]

    normal = np.cross(B - A, C - A)
    normal /= np.linalg.norm(normal)

    vec_AP = P - A
    dist_ortho = np.abs(np.dot(vec_AP, normal))
    return dist_ortho, normal

def Innit_Origin_Point(target_coords, node_list):

    #*Step 1, i look for the closest node 
    all_coords = np.array([n.coords for n in node_list])
    distances = np.linalg.norm(all_coords - target_coords, axis=1)
    closest_node_idx = np.argmin(distances)
    closest_node = node_list[closest_node_idx]

    #*Then, i look for the closest element in the neighbors of the closest node
    best_dist = np.inf
    target_elem = None
    best_normal = None
    for elem in closest_node.adjacent_triangles:
        dist_to_elem, normal = Compute_Dist_Point_Triangle(target_coords, elem)
        if dist_to_elem < best_dist:
            best_dist = dist_to_elem
            target_elem = elem
            best_normal = normal

    
    for node in target_elem.nodes:
        node.dist = np.linalg.norm(node.coords - target_coords)
        print(f"Node {node.node_tag} innitialized at d={node.dist:.4f}")

    for edge in target_elem.edges:
        edge.kappa = 2 / (edge.nodes[0].dist + edge.nodes[1].dist)
        edge.Is_innit_in_target_elem = True

    projected_point = target_coords - best_dist*best_normal
    return projected_point, target_elem


#!===================================================================== |
#!===================================================================== |
#!======================== FFM BELOW ================================== |
#!===================================================================== |
#!===================================================================== V

def Innit_FFM(target_coords, target_elem):
    trial_band = TRIAL_BAND()
    for n in target_elem.nodes:
        n.dist = np.linalg.norm(n.coords - target_coords)
        n.state = 'TRIAL'
        trial_band.add_node(n)
    return trial_band

def FMM(trial_band, edge_dict, node_list):
    print("=================================================")
    print("========== Running Fast Marching Method =========")
    print("=================================================\n")
    frame_id = 0
    counter = 0
    Z = [0]
    while not trial_band.is_empty():

        ui = trial_band.pop_closest()
        if ui is None: break
        ui.state = 'ALIVE'
        counter += 1 

        #* Looping over the neighbors
        for uj in ui.neighbors:
            if uj.state != 'ALIVE':

                new_dist = Update_Node_Distance(uj, edge_dict, Z)
                
                if new_dist < uj.dist:
                    uj.dist = new_dist
                    trial_band.add_node(uj)
                    #!I innit all the kappa's on the edges coming from the first closest node
                    if counter == 1:
                        key = (min(ui.node_tag, uj.node_tag), max(ui.node_tag, uj.node_tag))
                        edge_dict[key].kappa = 2 / (ui.dist + uj.dist)
                        edge_dict[key].Is_innit_in_target_elem = True

        #Save_Front_Frame(node_list, frame_id)
        frame_id += 1
    #Make_GIF()
    print("FMM succesfully finished.\n")

def Update_Node_Distance(node, edge_dict, Z):
    t_min = np.inf
    
    for elem in node.adjacent_triangles:
        
        C = node
        others = [n for n in elem.nodes if n != node]
        A, B = others[0], others[1]
        key_AB = (min(A.node_tag, B.node_tag), max(A.node_tag, B.node_tag))
        kappa = edge_dict[key_AB].kappa  

        key_AB = (min(A.node_tag, B.node_tag), max(A.node_tag, B.node_tag))
        key_CB = (min(C.node_tag, B.node_tag), max(C.node_tag, B.node_tag))
        key_AC = (min(A.node_tag, C.node_tag), max(A.node_tag, C.node_tag))
        edge_AB = edge_dict[key_AB]
        edge_CB = edge_dict[key_CB]
        edge_AC = edge_dict[key_AC]

        if A.state == 'ALIVE' and B.state == 'ALIVE':
            if True:
                mid_AB = (A.coords + B.coords) / 2
                mid_CB = (C.coords + B.coords) / 2
                mid_AC = (A.coords + C.coords) / 2

                s_AB_BC = np.linalg.norm(mid_AB-mid_CB)
                s_AB_AC = np.linalg.norm(mid_AB-mid_AC)

                
                
            #!I make sure to use the paper convention, We want Ta < Tb and we look for Tc
            if(A.dist < B.dist):
                #t_local = Solve_Eikonal_Triangle(A.coords, B.coords, C.coords, A.dist, B.dist)
                t_local, s = Solve_Eikonal_Triangle_Order2(A.coords, B.coords, C.coords, A.dist, B.dist, kappa, Z)
            else:
                t_local, s = Solve_Eikonal_Triangle_Order2(B.coords, A.coords, C.coords, B.dist, A.dist, kappa, Z)
                #t_local = Solve_Eikonal_Triangle(B.coords, A.coords, C.coords, B.dist, A.dist)
            
            #!Here if the grad doesn't come from within the triangle, i choose s as the the median
            #!from C
            if s is None: 
                # fallback géométrique : C reçoit depuis le milieu de AB
                mid_AB = (A.coords + B.coords) / 2
                s = np.linalg.norm(C.coords - mid_AB)   
                 
            
            if edge_CB.Is_innit_in_target_elem is False and s is not None:
                edge_CB.kappa = max(edge_CB.kappa, kappa / (1 + kappa*s))
            if edge_AC.Is_innit_in_target_elem is False and s is not None:
                edge_AC.kappa = max(edge_AC.kappa, kappa / (1 + kappa*s))
            #if  kappa == 0.0: print("kappa is Zero!:", kappa)
            # edge_CB.kappa = max(edge_CB.kappa, 2 / (t_local + B.dist))
            # edge_AC.kappa = max(edge_AC.kappa, 2 / (t_local + A.dist))

        elif A.state == 'ALIVE':
            t_local = A.dist + np.linalg.norm(C.coords - A.coords)
            edge_AC.kappa = max(edge_AC.kappa, 2 / (t_local + A.dist))
        elif B.state == 'ALIVE':
            t_local = B.dist + np.linalg.norm(C.coords - B.coords)
            edge_CB.kappa = max(edge_CB.kappa, 2 / (t_local + B.dist))
        else:   
            continue

        if t_local < t_min:
            t_min = t_local
                
    return t_min

def Solve_Eikonal_Triangle(A_coords, B_coords, C_coords, Ta, Tb):
    F = 1

    La = np.linalg.norm(B_coords - C_coords) 
    Lb = np.linalg.norm(A_coords - C_coords)  
    Lc = np.linalg.norm(A_coords - B_coords)  

    u = Tb - Ta  #* Ta <= Tb is always true when i call Solve_Eikonal_Triangle()

    CA = A_coords - C_coords
    CB = B_coords - C_coords
    cosTheta = np.dot(CA, CB) / (Lb * La)#*Cos of angle C
    sinTheta = np.sqrt(1 - cosTheta**2)

    #*2nd order equation from the paper
    coef_a = La**2 + Lb**2 - 2*La*Lb*cosTheta  # = Lc²
    coef_b = 2 * Lb * u * (La * cosTheta - Lb)
    coef_c = Lb**2 * (u**2 - F**2 * La**2 * sinTheta**2)

    delta = coef_b**2 - 4*coef_a*coef_c

    if delta >= 0:
        t = (-coef_b + np.sqrt(delta)) / (2 * coef_a)
        if t > u:  # upwind
            b_t_u_over_t = Lb * (t - u) / t
            cosTheta = abs(cosTheta)
            lower = La * cosTheta
            upper = La / cosTheta if abs(cosTheta) > 1e-10 else np.inf
            if lower <= b_t_u_over_t <= upper:
                return Ta + t

    #*Otherwise, we move along the edge
    return min(Ta + Lb*F, Tb + La*F)

def Plot_Isolines(node_list, element_list):

    x = np.array([n.coords[0] for n in node_list])
    y = np.array([n.coords[1] for n in node_list])
    z = np.array([n.dist for n in node_list])

    levels = 20
    triangles = []
    for e in element_list:
        triangles.append([n.idx for n in e.nodes])
    triangles = np.array(triangles)

    plt.figure(figsize=(8, 6))
    plt.gca().set_aspect('equal')

    cntr = plt.tricontourf(x, y, triangles, z, levels=levels, cmap="viridis")
    plt.colorbar(cntr, label="Distance")

    lines = plt.tricontour(x, y, triangles, z, levels=levels, colors='white', linewidths=0.5)
    plt.clabel(lines, inline=True, fontsize=8) # Ajoute les valeurs sur les lignes

    plt.triplot(x, y, triangles, color='black', alpha=0.1, linewidth=0.5)

    plt.title("Isolignes de la distance (FMM)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

def Solve_Eikonal_Triangle_Order2(A_coords, B_coords, C_coords, Ta, Tb, kappa, Z):
    F = 1.0
    s = None
    
    CA = A_coords - C_coords
    CB = B_coords - C_coords
    BA = A_coords - B_coords
    Lb = np.linalg.norm(CA)  # AC
    La = np.linalg.norm(CB)  # BC
    
    xa = Lb
    xb = np.dot(CB, CA) / Lb
    yb_sq = La**2 - xb**2
    if yb_sq < 1e-12:
        return min(Ta + Lb*F, Tb + La*F), None
    yb = np.sqrt(yb_sq)
    
    
    A_eff = Ta - (kappa/4) * xa**2
    B_eff = Tb - (kappa/4) * (xb**2 + yb**2)
    
    N = 1.0 - xb/xa
    M = B_eff - A_eff * (xb/xa)
    
    coef_a = 1/xa**2 + N**2/yb**2
    coef_b = -2*A_eff/xa**2 - 2*M*N/yb**2
    coef_c = A_eff**2/xa**2 + M**2/yb**2 - 1
    
    delta = coef_b**2 - 4*coef_a*coef_c
    
    if delta < 0:
        return min(Ta + Lb*F, Tb + La*F), None
    
    Tc = (-coef_b + np.sqrt(delta)) / (2*coef_a)
    
    Ra = A_eff - Tc 
    Rb = B_eff - Tc
    p = (Ra) / xa
    q = (Rb - Ra*xb/xa)/yb
    gradT_C = np.array([p, q])

    minus_gradT_C = - gradT_C
 

    if Tc >= Ta and Tc >= Tb and minus_gradT_C[1] >=0  and minus_gradT_C[0] >=0  and minus_gradT_C[1]/minus_gradT_C[0]*xb <= yb:
        gx, gy = minus_gradT_C[0], minus_gradT_C[1]
    
        # Intersection de la demi-droite C + t*(gx,gy) avec le segment AB
        # dans le repère local où C=(0,0), A=(xa,0), B=(xb,yb)
        denom = gx * yb - gy * (xb - xa)
        if abs(denom) < 1e-21:
            return min(Ta + Lb*F, Tb + La*F), None
        
        t_intersect = xa * yb / denom
        u = xa * gy / denom
        if t_intersect >= 0 and 0 <= u <= 1:
            P = np.array([t_intersect * gx, t_intersect * gy])
            # s = distance de C à P (C est à l'origine du repère local)
            s = np.linalg.norm(P)
        else:
            s = None
        
        Z[0] +=1 
        #print(Z[0])
        return Tc, s
    
    return min(Ta + Lb*F, Tb + La*F), None

def plot_Txy(element_list, idx_el, edge_dict):
    elem = element_list[idx_el]
    max_dist = 0
    

    for n in elem.nodes:
        if n.dist > max_dist:
            max_dist = n.dist
            C_node = n
    bool_A_found = False
    for idx, n in enumerate(elem.nodes):
        if n == C_node: continue
        if n != C_node and bool_A_found == False:
            A_node = n
            bool_A_found = True
        else:
            B_node = n

    CA = A_node.coords - C_node.coords
    CB = B_node.coords - C_node.coords
    Lb = np.linalg.norm(CA)  # AC
    La = np.linalg.norm(CB)  # BC
    
    xa = Lb
    xb = np.dot(CB, CA) / Lb
    yb_sq = La**2 - xb**2
    yb = np.sqrt(yb_sq)

    Tc = C_node.dist
    Ta = A_node.dist
    Tb = B_node.dist

    key = (min(A_node.node_tag, B_node.node_tag), 
                   max(A_node.node_tag, B_node.node_tag))
    edge = edge_dict[key]
    kappa = edge.kappa
    A_eff = Ta - (kappa/4) * xa**2
    B_eff = Tb - (kappa/4) * (xb**2 + yb**2)
    Ra = A_eff - Tc 
    Rb = B_eff - Tc
    p = (Ra) / xa
    q = (Rb - Ra*xb/xa)/yb

    #* T(x,y) = Tc + px + qy + 1/4(kappa)(x**2 + y**2)

    # grille dans le plan local
    nx = 200
    ny = 200

    x = np.linspace(-0.2, 0.2, nx)
    y = np.linspace(-0.2, 0.2, ny)
    X, Y = np.meshgrid(x, y)

    # champ T(x,y)
    T = Tc + p*X + q*Y + 0.25*kappa*(X**2 + Y**2)
    # masque pour garder seulement le triangle

    if xb != 0: mask = (X >= 0) & (Y >= 0) & (Y <= yb/xb * X) & (Y <= yb - (yb/(xa-xb))*(X-xb))
    else:
        mask = (X >= 0) & (Y >= 0) & (Y <= yb - (yb/(xa-xb))*(X-xb))
    T_masked = np.where(mask, T, np.nan)

    # triangle
    triangle_x = [0, xa, xb, 0]
    triangle_y = [0, 0, yb, 0]

    plt.figure()

    plt.plot(triangle_x, triangle_y, 'k-')

    # isolignes
    cs = plt.contour(X, Y, T, 15)
    plt.clabel(cs)

    plt.scatter([0, xa, xb], [0, 0, yb])
    # plt.xlim(-xa, 2*xa)
    # plt.ylim(-yb, 2*yb)


    plt.gca().set_aspect('equal')
    plt.show()



def plot_kappa_edges(edge_dict):

    edges = list(edge_dict.values())
    kappas = [e.kappa for e in edges]


    norm = mpl.colors.Normalize(vmin=min(kappas), vmax=max(kappas))
    cmap = plt.cm.coolwarm

    fig, ax = plt.subplots()

    for e in edges:
        n1, n2 = e.nodes

        x = [n1.coords[0], n2.coords[0]]
        y = [n1.coords[1], n2.coords[1]]

        color = cmap(norm(e.kappa))
        #if e.kappa != 0.0: print(e.kappa)
        #print(e.kappa)
        ax.plot(x, y, color=color, linewidth=2)

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(sm, ax=ax, label="kappa")

    ax.set_aspect('equal')
    plt.show()


def Save_Front_Frame(node_list, frame_id, folder="frames"):

    if not os.path.exists(folder):
        os.makedirs(folder)

    x_alive, y_alive = [], []
    x_trial, y_trial = [], []
    x_far, y_far = [], []

    for n in node_list:

        if n.state == 'ALIVE':
            x_alive.append(n.coords[0])
            y_alive.append(n.coords[1])

        elif n.state == 'TRIAL':
            x_trial.append(n.coords[0])
            y_trial.append(n.coords[1])

        else:
            x_far.append(n.coords[0])
            y_far.append(n.coords[1])

    plt.figure(figsize=(6,6))

    plt.scatter(x_far, y_far, c="lightgray", s=5)
    plt.scatter(x_trial, y_trial, c="orange", s=10)
    plt.scatter(x_alive, y_alive, c="blue", s=10)

    plt.gca().set_aspect("equal")

    filename = f"{folder}/frame_{frame_id:05d}.png"
    plt.savefig(filename, dpi=150)
    plt.close()

def Make_GIF(folder="frames", gif_name="front.gif"):

    files = sorted(glob.glob(folder + "/*.png"))

    images = []
    for f in files:
        images.append(imageio.imread(f))

    imageio.mimsave(gif_name, images, duration=0.05)
