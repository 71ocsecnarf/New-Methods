import numpy as np
import gmsh
import matplotlib.pyplot as plt

from structures import NODE
from structures import ELEMENT
from structures import TRIAL_BAND



def Make_NodeList_NodeDictionary(filename, tag_to_node_obj_dic):

    if not gmsh.isInitialized():
        gmsh.initialize()
    
    print("\n")
    print("=================================================")
    print("================Making node list=================")
    print("=================================================")

    node_tags, coords, _ = gmsh.model.mesh.getNodes()
    points = coords.reshape(-1, 3)

    nodes_list = []
    
    for i, tag in enumerate(node_tags):
        new_node = NODE(points[i], tag, i)
        nodes_list.append(new_node)
        tag_to_node_obj_dic[tag] = new_node
    
    _, _, node_tags_per_elem = gmsh.model.mesh.getElements(2)
    triangles_tags = node_tags_per_elem[0].reshape(-1, 3)

    for tri in triangles_tags:
        node_a = tag_to_node_obj_dic[tri[0]]
        node_b = tag_to_node_obj_dic[tri[1]]
        node_c = tag_to_node_obj_dic[tri[2]]
        
        node_a.neighbors.update([node_b, node_c])
        node_b.neighbors.update([node_a, node_c])
        node_c.neighbors.update([node_a, node_b])

    print("Node list is done --> ok")
    return np.array(nodes_list)



def Make_ElementList(filename, tag_to_node_obj_dic):

    if not gmsh.isInitialized():
        gmsh.initialize()
    
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
        node_a.adjacent_triangles.append(element)
        node_b.adjacent_triangles.append(element)
        node_c.adjacent_triangles.append(element)

    print("Element list is done --> ok")
    return np.array(element_list)


""" 
# THIS FUNCTION DOES NOT WORK IN 3D
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
        if min(AB_AC, BC_BA, CA_CB) < 0:
            e.IsObtuse = True
            counting += 1
    
    print(f"There are {counting} obtuse elements")
    print("Checking for obtuse element is done --> ok\n")

"""
def Check_Obtuse_triangles(element_list):
    """
    Checks for obtuse triangles in the mesh.
    An obtuse triangle has one angle strictly greater than 90 degrees.
    This check is crucial for the Eikonal equation's upwind condition in FMM.
    """
    print("\n")
    print("========================================================")
    print("============= Checking for obtuse elements =============")
    print("========================================================")
    
    counting = 0
    for e in element_list:
        # Extract 3D coordinates for all 3 nodes of the triangle
        A = e.nodes[0].coords
        B = e.nodes[1].coords
        C = e.nodes[2].coords

        # Compute vectors representing the sides of the triangle
        # These are 3D vectors (x, y, z)
        vAB = B - A
        vAC = C - A
        vBC = C - B

        # Check for obtuse angles using the dot product:
        # If dot(v1, v2) < 0, the angle between them is > 90 degrees.
        
        # Angle at vertex A: dot product of AB and AC
        angle_A = np.dot(vAB, vAC)
        # Angle at vertex B: dot product of BA and BC
        angle_B = np.dot(-vAB, vBC)
        # Angle at vertex C: dot product of CA and CB
        angle_C = np.dot(-vAC, -vBC)

        # Using a small epsilon to avoid false positives due to floating point precision
        if min(angle_A, angle_B, angle_C) < -1e-12:
            e.IsObtuse = True
            counting += 1
    
    print(f"There are {counting} obtuse elements")
    print("Checking for obtuse elements is done --> ok\n")


def Compute_Dist_Point_Triangle(P, element):

    A, B, C = [n.coords for n in element.nodes]

    normal = np.cross(B - A, C - A)
    normal /= np.linalg.norm(normal)

    vec_AP = P - A
    dist_ortho = np.abs(np.dot(vec_AP, normal))
    return dist_ortho, normal



def Innit_Origin_Point(target_coords, node_list):

    all_coords = np.array([n.coords for n in node_list])
    distances_to_target = np.linalg.norm(all_coords - target_coords, axis=1)
    closest_node_idx = np.argmin(distances_to_target)
    closest_node = node_list[closest_node_idx]

    snap_err = np.linalg.norm(closest_node.coords - target_coords)

    # Set the snapped node as the true origin with dist = 0
    # This removes the O(h) snapping pollution from the convergence
    closest_node.dist  = 0.0
    closest_node.state = 'ALIVE'

    # Initialize neighbors with Euclidean distance FROM THE SNAPPED NODE
    # (not from the theoretical source) so the FMM starts from a consistent origin
    for neighbor in closest_node.neighbors:
        neighbor.dist  = neighbor.distance_to_other_node(closest_node)
        neighbor.state = 'TRIAL'

    print(f"Source node {closest_node.node_tag} @ {closest_node.coords}")
    print(f"  Theoretical source : {target_coords}")
    print(f"  Snapping error     : {snap_err:.6e}")

    return closest_node.coords, [closest_node]



def Reset_Node_State(node_list):
    for node in node_list:
        node.dist  = float('inf')
        node.state = 'FAR'



def compute_err(node_list, source_coord):
    """
    Compute max, relative, pointwise, and L2 errors for the standard geometry.
    Returns: (max_error, max_relative_error, pointwise_errors_array, L2_error)
    """
    errors = []
    errors_rel = []

    for node in node_list:
        # Avoid plotting crashes or math errors if a node is not reached by the FMM
        if node.dist == float('inf'):
            errors.append(0.0)  
            continue
            
        ex_dist = np.linalg.norm(node.coords - source_coord)
        err_i = (ex_dist - node.dist)
        errors.append(err_i)
        
        if ex_dist > 1e-14:
            errors_rel.append(err_i / ex_dist)

    err = np.max(errors) if errors else 0.0
    err_rel = np.max(errors_rel) if errors_rel else 0.0
    e_l2 = np.sqrt(np.mean(np.array(errors)**2)) if errors else 0.0

    # Returning the errors array before e_l2 ensures that result[-1] is still the L2 error
    return err, err_rel, np.array(errors), e_l2


def compute_err_cylinder(node_list, true_source, R):
    """
    Compute errors for the cylinder geometry.
    Returns: (max_error, pointwise_errors_array, L2_error)
    """
    x0, y0, z0 = true_source
    R_source = np.sqrt(x0**2 + y0**2)
    theta0   = np.arctan2(y0, x0)

    max_err = 0.0
    sum_sq  = 0.0
    count   = 0
    errors  = []  # List needed for the error plot

    for node in node_list:
        if node.dist == float('inf'):
            errors.append(0.0)
            continue
            
        x, y, z = node.coords
        theta   = np.arctan2(y, x)
        d_theta = (theta - theta0 + np.pi) % (2 * np.pi) - np.pi
        d_exact = np.sqrt((R_source * d_theta)**2 + (z - z0)**2)
        err     = (node.dist - d_exact)
        errors.append(err)
        
        max_err = max(max_err, err)
        sum_sq += err**2
        count  += 1

    l2_err = np.sqrt(sum_sq / count) if count > 0 else 0.0
    
    return max_err, np.array(errors), l2_err


def compute_err_l_shape(node_list, source_coord, L=1.0):
    """
    Compute geodesic errors for the L-shape geometry.
    Returns: (max_error, max_relative_error, pointwise_errors_array, L2_error)
    """
    errors     = []  
    errors_rel = []  

    for node in node_list:
        if node.dist == float('inf'):
            errors.append(0.0)
            continue
            
        d_exact = geodesic_lshape(node.coords, source_coord, L)
        err_i = (d_exact - node.dist)
        """
                # DEBUG
        if err_i < -1e-6:  # errore negativo significativo
            print(f"OVERESTIMATE: node=({node.coords[0]:.4f},{node.coords[1]:.4f})"
                  f" | d_exact={d_exact:.6f} | d_fmm={node.dist:.6f}"
                  f" | err={err_i:.2e}"
                  f" | crosses={segment_crosses_missing_quadrant(source_coord, node.coords, L)}")
        
        """
        errors.append(err_i)
        

        if d_exact > 1e-14:
            errors_rel.append(err_i / d_exact)

    err     = np.max(errors) if errors else 0.0
    err_rel = np.max(errors_rel) if errors_rel else 0.0
    e_l2    = np.sqrt(np.mean(np.array(errors) ** 2)) if errors else 0.0 

    return err, err_rel, np.array(errors), e_l2

def geodesic_lshape(node_coords, source_coords, L=1.0):
    """
    Exact geodesic distance on the L-shape domain.

    The missing quadrant is (x > L/2, y < L/2).
    The concave corner is at C = (L/2, L/2).

    If the straight line S->P crosses the boundary of the missing quadrant,
    the geodesic must detour through the corner C:
        d = dist(S, C) + dist(C, P)

    Otherwise the Euclidean distance is exact.
    """
    corner = np.array([L / 2.0, L / 2.0, 0.0])

    # Direct Euclidean distance
    d_direct = np.linalg.norm(node_coords - source_coords)

    # Check if the straight segment crosses the missing quadrant boundary
    if segment_crosses_missing_quadrant(source_coords, node_coords, L):
        d_via_corner = (np.linalg.norm(source_coords - corner) +
                        np.linalg.norm(node_coords   - corner))
        # min() is a safety net for points exactly on the corner boundary
        return d_via_corner

    return d_direct

def segment_crosses_missing_quadrant(S, P, L):
    """
    Check whether the straight segment S->P passes through the missing
    quadrant of the L-shape, i.e. the rectangle (L/2 < x < L, 0 < y < L/2).

    Strategy: check if the segment crosses either of the two boundary edges
    of the missing quadrant that are interior to the bounding box:
      - Vertical edge  : x = L/2,  y in [0,   L/2]
      - Horizontal edge: y = L/2,  x in [L/2, L  ]

    If it crosses one of them going INTO the missing quadrant, return True.
    """
    x_S, y_S = S[0], S[1]
    x_P, y_P = P[0], P[1]
    dx = x_P - x_S
    dy = y_P - y_S

    # --- Check intersection with vertical edge x = L/2, y in [0, L/2] ---
    if abs(dx) > 1e-14:
        t = (L/2 - x_S) / dx          # Parameter t in [0,1] along segment S->P
        if 0.0 < t < 1.0:
            y_int = y_S + t * dy       # y coordinate at the intersection
            if 0.0 <= y_int <= L/2:    # Intersection is on the interior edge
                # The segment crosses x=L/2 in the lower half
                # --> it is entering or exiting the missing quadrant
                return True

    # --- Check intersection with horizontal edge y = L/2, x in [L/2, L] ---
    if abs(dy) > 1e-14:
        t = (L/2 - y_S) / dy
        if 0.0 < t < 1.0:
            x_int = x_S + t * dx
            if L/2 <= x_int <= L:      # Intersection is on the interior edge
                return True

    return False



def compute_err_hole(node_list, source_coord, L=1.0, R=0.2):
    """
    Compute geodesic errors for the square with hole geometry.
    Returns: (max_error, max_relative_error, pointwise_errors_array, L2_error)
    """
    errors     = []  
    errors_rel = []  

    for node in node_list:
        if node.dist == float('inf'):
            errors.append(0.0)
            continue
            
        d_exact = geodesic_hole(node.coords, source_coord, L, R)
        err_i = (d_exact - node.dist)
        errors.append(err_i)

        if d_exact > 1e-14:
            errors_rel.append(err_i / d_exact)

    err     = np.max(errors) if errors else 0.0
    err_rel = np.max(errors_rel) if errors_rel else 0.0
    e_l2    = np.sqrt(np.mean(np.array(errors) ** 2)) if errors else 0.0 

    return err, err_rel, np.array(errors), e_l2



def geodesic_hole(node_coords, source_coords, L=1.0, R=0.2):
    """
    Exact geodesic distance on the square-with-hole domain.
    The hole is a disk of radius R centered at (L/2, L/2).
    If the straight segment S->P intersects the hole, the geodesic
    detours around the hole via one of the two tangent paths.
    We compute BOTH (clockwise and counter-clockwise) and return the minimum.
    """
    C_coords = np.array([L/2, L/2, 0.0])

    v_S = source_coords - C_coords
    v_P = node_coords   - C_coords

    d_S = max(np.linalg.norm(v_S), R)
    d_P = max(np.linalg.norm(v_P), R)

    # Half-angle of the tangent cone from S and from P
    alpha_S = np.arccos(np.clip(R / d_S, -1.0, 1.0))
    alpha_P = np.arccos(np.clip(R / d_P, -1.0, 1.0))

    # Tangent lengths from S and P to the circle
    t_S = np.sqrt(max(d_S**2 - R**2, 0.0))
    t_P = np.sqrt(max(d_P**2 - R**2, 0.0))

    # Signed angles of v_S and v_P
    angle_S = np.arctan2(v_S[1], v_S[0])
    angle_P = np.arctan2(v_P[1], v_P[0])

    # Signed angular difference (from S to P)
    dphi = angle_P - angle_S
    # Wrap to (-pi, pi]
    dphi = (dphi + np.pi) % (2 * np.pi) - np.pi

    # Visibility condition: |dphi| < alpha_S + alpha_P  means direct line-of-sight
    if abs(dphi) <= alpha_S + alpha_P:
        return np.linalg.norm(node_coords - source_coords)

    # Two detour paths: CCW (positive arc) and CW (negative arc)
    # Arc angle for CCW path (going the short way in the positive direction)
    if dphi > 0:
        arc_CCW = dphi - (alpha_S + alpha_P)
        arc_CW  = (2 * np.pi - dphi) - (alpha_S + alpha_P)
    else:
        arc_CW  = (-dphi) - (alpha_S + alpha_P)
        arc_CCW = (2 * np.pi - (-dphi)) - (alpha_S + alpha_P)

    # Both arcs must be non-negative (they represent a real detour)
    arc_CCW = max(arc_CCW, 0.0)
    arc_CW  = max(arc_CW,  0.0)

    d_CCW = t_S + t_P + R * arc_CCW
    d_CW  = t_S + t_P + R * arc_CW

    return min(d_CCW, d_CW)




# =============================================
# ------- L-cylinder geodesic error -----------
# =============================================

def compute_err_l_cylinder(node_list, source_coord, R, H):
    """
    Compute geodesic errors for the L-shaped half-cylinder surface.

    Returns: (max_error, max_relative_error, pointwise_errors_array, L2_error)
    Same signature as compute_err_l_shape so it can be used identically in
    accuracy_plot.py and Plot_Error_Field.
    """
    errors     = []
    errors_rel = []

    for node in node_list:
        if node.dist == float('inf'):
            errors.append(0.0)
            continue

        d_exact = geodesic_l_cylinder(node.coords, source_coord, R, H)
        err_i   = d_exact - node.dist
        errors.append(err_i)

        if d_exact > 1e-14:
            errors_rel.append(err_i / d_exact)

    err     = np.max(errors)     if errors else 0.0
    err_rel = np.max(errors_rel) if errors_rel else 0.0
    e_l2    = np.sqrt(np.mean(np.array(errors) ** 2)) if errors else 0.0

    return err, err_rel, np.array(errors), e_l2


def geodesic_l_cylinder(node_coords, source_coords, R, H):
    """
    Exact geodesic distance on the L-shaped half-cylinder lateral surface.

    KEY IDEA -- isometric unrolling:
        A cylinder has zero Gaussian curvature, so its surface is locally
        isometric to the plane.  The unrolling map

            phi : (x, y, z)  -->  (s, z)   with  s = R * arctan2(y, x)

        preserves all lengths.  Therefore every geodesic on the cylinder
        corresponds to a straight line in the unrolled 2D domain, and the
        geodesic LENGTH equals the Euclidean distance in that 2D domain.

    UNROLLED DOMAIN:
        Horizontal axis: s = R * theta,  theta in [0, pi]
                         s in [0, pi*R]
                         s=0      <-> theta=0   <-> ( R,  0, z)  right edge
                         s=pi*R/2 <-> theta=pi/2 <-> ( 0,  R, z)  front
                         s=pi*R   <-> theta=pi   <-> (-R,  0, z)  left edge
        Vertical axis: z in [0, H]

        Bottom arm: s in [0, pi*R],   z in [0,   H/2]
        Left  arm:  s in [pi*R/2, pi*R], z in [H/2, H]
        Missing quadrant: s in [0, pi*R/2], z in [H/2, H]
        Concave corner: (s_c, z_c) = (pi*R/2, H/2)

    GEODESIC RULE (identical structure to geodesic_lshape):
        - Compute unrolled 2D coordinates S' and P' of source and target.
        - If the straight segment S'->P' does NOT cross the boundary of the
          missing quadrant: geodesic = Euclidean distance ||S' - P'||.
        - If it DOES cross: the geodesic must detour through the concave
          corner C' = (pi*R/2, H/2):
              d = ||S' - C'|| + ||C' - P'||

    NOTE on theta convention:
        arctan2(y, x) returns values in (-pi, pi].  For our domain
        theta in [0, pi] (the front half of the cylinder, y >= 0), this
        gives values in [0, pi] -- no wrap-around needed, unlike the full
        cylinder case in compute_err_cylinder.

    Parameters
    ----------
    node_coords   : array of shape (3,) -- 3D coords of the target node
    source_coords : array of shape (3,) -- 3D coords of the snapped source
    R             : float -- cylinder radius
    H             : float -- cylinder height

    Returns
    -------
    d : float -- exact geodesic distance
    """
    # -- Unroll source and target to 2D --
    # theta = arctan2(y, x) is in [0, pi] for the front half (y >= 0)
    theta_S = np.arctan2(source_coords[1], source_coords[0])
    theta_P = np.arctan2(node_coords[1],   node_coords[0])

    S2 = np.array([R * theta_S, source_coords[2]])   # (s_S, z_S)
    P2 = np.array([R * theta_P, node_coords[2]])     # (s_P, z_P)

    # -- Concave corner in 2D --
    # theta=pi/2 -> s = R*pi/2;  z = H/2
    s_corner = R * np.pi / 2.0
    C2 = np.array([s_corner, H / 2.0])

    # -- Check if the direct segment S'->P' crosses the missing quadrant --
    # Missing quadrant in 2D: s in [0, pi*R/2], z in [H/2, H]
    # Its interior boundary consists of two edges:
    #   Horizontal: z = H/2,  s in [0,     pi*R/2]
    #   Vertical:   s = pi*R/2, z in [H/2, H     ]
    if _segment_crosses_missing_quadrant_2d(S2, P2, s_corner, H):
        d_via_corner = (np.linalg.norm(S2 - C2) + np.linalg.norm(P2 - C2))
        return d_via_corner

    return np.linalg.norm(P2 - S2)


def _segment_crosses_missing_quadrant_2d(S, P, s_corner, H):
    """
    Check whether the 2D segment S->P crosses the interior boundary of the
    missing quadrant  {s in [0, s_corner], z in [H/2, H]}.

    The two interior boundary edges are:
      - Horizontal edge: z = H/2,     s in [0, s_corner]
      - Vertical edge:   s = s_corner, z in [H/2, H    ]

    A crossing on either edge means the straight path enters the missing
    quadrant and the geodesic must detour via the concave corner.

    This is the direct 2D analogue of segment_crosses_missing_quadrant
    used for the flat L-shape.
    """
    s_S, z_S = S[0], S[1]
    s_P, z_P = P[0], P[1]
    ds = s_P - s_S
    dz = z_P - z_S
    z_half = H / 2.0

    # -- Horizontal edge: z = H/2,  s in [0, s_corner] --
    if abs(dz) > 1e-14:
        t = (z_half - z_S) / dz        # t in (0,1) for a proper interior crossing
        if 0.0 < t < 1.0:
            s_int = s_S + t * ds
            if 0.0 <= s_int <= s_corner:
                return True

    # -- Vertical edge: s = s_corner,  z in [H/2, H] --
    if abs(ds) > 1e-14:
        t = (s_corner - s_S) / ds
        if 0.0 < t < 1.0:
            z_int = z_S + t * dz
            if z_half <= z_int <= H:
                return True

    return False


# =============================================
# ------- Point-wise convergence tracking -----
# =============================================

def Find_Closest_Node(target_xy, node_list):
    """
    Find the mesh node closest to a given 2D target point.

    The search is done in the (x, y) plane only (ignoring z), which is correct
    for flat 2D geometries (square_surface, l_shape, hole).

    Parameters
    ----------
    target_xy : array-like of shape (2,)
        The (x, y) coordinates of the target point.
    node_list : array of NODE objects
        All mesh nodes.

    Returns
    -------
    closest_node : NODE
        The mesh node whose (x, y) coordinates are closest to target_xy.
    snap_distance : float
        The Euclidean distance between target_xy and the snapped node.
    """
    target = np.array([target_xy[0], target_xy[1]])

    # Build a (N_nodes, 2) array of (x, y) coordinates for vectorised search.
    # node.coords is a 3-element array [x, y, z]; we take only the first two.
    xy_coords = np.array([n.coords[:2] for n in node_list])

    # np.linalg.norm with axis=1 computes the 2D distance from target
    # to every node in one vectorised operation (no Python loop needed).
    distances = np.linalg.norm(xy_coords - target, axis=1)

    # np.argmin returns the index of the minimum value in the array
    idx = np.argmin(distances)

    return node_list[idx], distances[idx]


def Track_Point_Errors(probe_points, node_list, mesh_type, source_coord, L=1.0, R=0.2):
    """
    Compute the pointwise relative error at a set of probe points after one FMM run.

    For each probe point, the closest mesh node is found (snap), the exact
    geodesic distance is evaluated, and the relative error is computed as:
        rel_err = |T_FMM - T_exact| / T_exact

    This function is meant to be called once after fmm_algorithm() or
    fmm_algorithm_circ() has already populated node.dist for all nodes.

    Parameters
    ----------
    probe_points : list of array-like, each of shape (2,)
        The (x, y) coordinates of the points to track.
        Example: [np.array([1.0, 1.0]), np.array([0.8, 1.0])]
    node_list    : array of NODE objects
        All mesh nodes (with .dist already filled by the FMM).
    mesh_type    : str
        One of 'square_surface', 'l_shape', 'hole'.
        Used to select the correct exact geodesic formula.
    source_coord : array-like of shape (3,)
        The (x, y, z) coordinates of the snapped source node.
    L : float
        Domain side length (used by l_shape and hole).
    R : float
        Hole radius (used by hole only).

    Returns
    -------
    rel_errors : list of float
        Relative error at each probe point, in the same order as probe_points.
        Returns 0.0 for a point if T_exact is too small (avoids division by zero).
    snapped_coords : list of np.ndarray
        The actual (x, y, z) coordinates of the snapped nodes,
        useful for reporting how far the snap was.
    """
    rel_errors     = []
    snapped_coords = []

    for pt in probe_points:
        # --- Step 1: snap to the closest mesh node ---
        node, snap_dist = Find_Closest_Node(pt, node_list)
        snapped_coords.append(node.coords.copy())

        # --- Step 2: FMM value at the snapped node ---
        T_fmm = node.dist
        if T_fmm == float('inf'):
            # Node was never reached by the FMM (should not happen on a connected mesh)
            rel_errors.append(0.0)
            continue

        # --- Step 3: exact geodesic distance at the snapped node ---
        # We pass node.coords (3D) to the geodesic functions because they
        # internally use .coords[0] and .coords[1] (x and y).
        if mesh_type == 'square_surface':
            T_exact = np.linalg.norm(node.coords - source_coord)

        elif mesh_type == 'l_shape':
            T_exact = geodesic_lshape(node.coords, source_coord, L)

        elif mesh_type == 'hole':
            T_exact = geodesic_hole(node.coords, source_coord, L, R)

        else:
            # Cylinder and l_cylinder are 3D geometries; point tracking
            # is not supported here. Return 0.0 as a safe fallback.
            rel_errors.append(0.0)
            continue

        # --- Step 4: relative error ---
        # Guard against division by zero when the probe point is the source itself
        if T_exact < 1e-14:
            rel_errors.append(0.0)
        else:
            rel_errors.append(abs(T_fmm - T_exact) / T_exact)

    return rel_errors, snapped_coords



# =============================================
# ------------------- Plots -------------------
# =============================================


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
    plt.clabel(lines, inline=True, fontsize=8)

    plt.triplot(x, y, triangles, color='black', alpha=0.1, linewidth=0.5)
    x_line = np.linspace(0.2, 0.8, 100)
    y_line = (5/3) * (x_line - 0.2)

    plt.plot(x_line, y_line, 'r--', linewidth=2)
    plt.title("FMM - Plot of the levelsets")
    plt.xlabel("X")
    plt.ylabel("Y")
    #plt.show()


def Plot_Isolines_3D(node_list, element_list, source_coords=None):

    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    import matplotlib.cm as cm

    x   = np.array([n.coords[0] for n in node_list])
    y   = np.array([n.coords[1] for n in node_list])
    z   = np.array([n.coords[2] for n in node_list])
    val = np.array([n.dist      for n in node_list])

    triangles = np.array([[n.idx for n in e.nodes] for e in element_list])
    face_val  = val[triangles].mean(axis=1)

    norm_c = Normalize(vmin=val.min(), vmax=val.max())
    cmap   = cm.viridis
    colors = cmap(norm_c(face_val))

    fig = plt.figure(figsize=(12, 8))
    ax  = fig.add_subplot(111, projection='3d')

    # Semitransparent surface
    surf = ax.plot_trisurf(x, y, z, triangles=triangles,
                           shade=False, antialiased=False, alpha=0.85)
    surf.set_facecolors(colors)

    # ---- Source Point ----
    if source_coords is not None:
        ax.scatter(source_coords[0], source_coords[1], source_coords[2],
                   color='red', s=100, zorder=5, label='Point source')
        ax.legend()

    # ---- Isolines ----
    # Minumum Radial offset to avoid z-fighting with the sourface 
    R_mean = np.sqrt(x**2 + y**2).mean()
    R_iso  = R_mean * 1.008

    n_levels = 15
    eps      = (val.max() - val.min()) * 0.001
    levels   = np.linspace(val.min() + eps, val.max() - eps, n_levels)

    for level in levels:
        label_pts = []

        for tri in triangles:
            i0, i1, i2 = tri
            pts = []
            for ea, eb in [(i0, i1), (i1, i2), (i2, i0)]:
                va, vb = val[ea], val[eb]
                if (va - level) * (vb - level) < 0:
                    t = (level - va) / (vb - va)
                    # Interpolation in 3D cartesian coordinates
                    px = x[ea] + t * (x[eb] - x[ea])
                    py = y[ea] + t * (y[eb] - y[ea])
                    pz = z[ea] + t * (z[eb] - z[ea])
                    # Re-project on the cylinder with R_iso
                    r_pt = np.sqrt(px**2 + py**2)
                    if r_pt > 1e-10:
                        px = px / r_pt * R_iso
                        py = py / r_pt * R_iso
                    pts.append((px, py, pz))

            if len(pts) == 2:
                ax.plot([pts[0][0], pts[1][0]],
                        [pts[0][1], pts[1][1]],
                        [pts[0][2], pts[1][2]],
                        color='white', linewidth=1.2, alpha=1.0, zorder=10)
                label_pts.append(pts)

        # Label at the mid point of each segment
        if label_pts:
            p0, p1 = label_pts[len(label_pts) // 2]
            mx = (p0[0] + p1[0]) / 2
            my = (p0[1] + p1[1]) / 2
            mz = (p0[2] + p1[2]) / 2
            ax.text(mx, my, mz, f'{level:.2f}',
                    color='yellow', fontsize=6.5, fontweight='bold',
                    zorder=20, ha='center')

    # ---- Colorbar ----
    sm = ScalarMappable(cmap=cmap, norm=norm_c)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, shrink=0.5, label="Geodesic distance")

    ax.set_title("FMM - Distance field on cylinder")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    plt.tight_layout()
    #plt.show()


def Plot_Isolines_L_Cylinder(node_list, element_list, source_coords=None):
    """
    Plot the geodesic distance field and isolines on the L-shaped cylinder surface.

    This function is identical in structure to Plot_Isolines_3D (used for the
    full half-cylinder), but adapted for the L-shaped variant:
      - The radial re-projection of isoline points uses the LOCAL radius of each
        interpolated point instead of a fixed global R_mean. This is necessary
        because the L-cylinder surface is not a surface of constant radius from
        a single axis -- different parts of the surface may have slightly
        different radii due to meshing.
      - The title and labels reflect the L-cylinder geometry.

    Parameters
    ----------
    node_list    : array of NODE objects -- all mesh nodes with .coords and .dist
    element_list : array of ELEMENT objects -- all triangles with .nodes
    source_coords: array-like of shape (3,), optional -- 3D coords of the source
                   point to mark on the plot with a red dot
    """

    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    import matplotlib.cm as cm

    # ------------------------------------------------------------------
    # STEP 1: Extract node coordinates and FMM distance values
    #
    # List comprehensions in Python: [expression for item in iterable]
    # This is equivalent to a for-loop that builds an array.
    # np.array([...]) converts the Python list to a NumPy array (like a MATLAB vector).
    # ------------------------------------------------------------------
    x   = np.array([n.coords[0] for n in node_list])   # x-coordinates of all nodes
    y   = np.array([n.coords[1] for n in node_list])   # y-coordinates of all nodes
    z   = np.array([n.coords[2] for n in node_list])   # z-coordinates of all nodes
    val = np.array([n.dist      for n in node_list])   # FMM distance at each node

    # Build the triangle connectivity array: shape (n_triangles, 3)
    # Each row contains the indices (idx) of the 3 nodes of one triangle.
    triangles = np.array([[n.idx for n in e.nodes] for e in element_list])

    # ------------------------------------------------------------------
    # STEP 2: Compute per-face colors for the surface plot
    #
    # face_val: average distance over the 3 nodes of each triangle.
    # val[triangles] is fancy indexing: for each triangle row [i, j, k],
    # it returns [val[i], val[j], val[k]]. Shape: (n_tri, 3).
    # .mean(axis=1) averages along axis 1 (the 3 nodes), giving one value per face.
    # ------------------------------------------------------------------
    face_val = val[triangles].mean(axis=1)

    # Normalize maps the raw distance values to [0, 1] for colormap lookup
    norm_c = Normalize(vmin=val.min(), vmax=val.max())
    cmap   = cm.viridis
    colors = cmap(norm_c(face_val))   # RGBA color for each triangle face

    # ------------------------------------------------------------------
    # STEP 3: Create the 3D figure and plot the colored surface
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(12, 8))
    ax  = fig.add_subplot(111, projection='3d')   # '111' = 1 row, 1 col, subplot 1

    # plot_trisurf draws a triangulated surface in 3D.
    # shade=False + set_facecolors() lets us assign our own per-face colors
    # instead of using matplotlib's default shading.
    surf = ax.plot_trisurf(x, y, z, triangles=triangles,
                           shade=False, antialiased=False, alpha=0.85)
    surf.set_facecolors(colors)

    # ------------------------------------------------------------------
    # STEP 4: Mark the source point
    # ------------------------------------------------------------------
    if source_coords is not None:
        ax.scatter(source_coords[0], source_coords[1], source_coords[2],
                   color='red', s=100, zorder=5, label='Point source')
        ax.legend()

    # ------------------------------------------------------------------
    # STEP 5: Draw isolines manually by linear interpolation on each triangle
    #
    # matplotlib has no built-in tricontour for 3D surfaces, so we do it
    # manually: for each distance level, we scan every triangle edge and check
    # if the level crosses that edge. If it does, we interpolate the crossing
    # point in 3D, then draw a segment between the two crossing points of
    # the same triangle.
    #
    # The isoline points are slightly offset radially (R_iso = R * 1.008)
    # to avoid z-fighting (visual artifacts where two surfaces overlap exactly).
    # Here we use the LOCAL radius of each interpolated point, so the offset
    # is correct everywhere on the L-shaped surface (not just at a fixed R).
    # ------------------------------------------------------------------

    n_levels = 15
    eps      = (val.max() - val.min()) * 0.001   # small margin to avoid boundary levels
    levels   = np.linspace(val.min() + eps, val.max() - eps, n_levels)

    RADIAL_OFFSET = 1.008   # push isolines 0.8% outward to avoid z-fighting

    for level in levels:
        label_pts = []   # will store one representative segment per level for labelling

        for tri in triangles:
            i0, i1, i2 = tri   # unpack the 3 node indices of this triangle

            pts = []   # crossing points found on this triangle (0, 1, or 2)

            # Check the 3 edges: (i0,i1), (i1,i2), (i2,i0)
            for ea, eb in [(i0, i1), (i1, i2), (i2, i0)]:
                va, vb = val[ea], val[eb]

                # The level crosses this edge if val changes sign relative to level
                # i.e. one endpoint is above and the other is below the level
                if (va - level) * (vb - level) < 0:

                    # Linear interpolation parameter t in [0, 1] along the edge
                    # t=0 at node ea, t=1 at node eb
                    t = (level - va) / (vb - va)

                    # Interpolated 3D coordinates on the edge
                    px = x[ea] + t * (x[eb] - x[ea])
                    py = y[ea] + t * (y[eb] - y[ea])
                    pz = z[ea] + t * (z[eb] - z[ea])

                    # Radial offset: push the point slightly outward from the
                    # cylinder axis so isolines are drawn on top of the surface.
                    # We use the LOCAL in-plane radius of this specific point.
                    r_local = np.sqrt(px**2 + py**2)
                    if r_local > 1e-10:
                        px = px / r_local * (r_local * RADIAL_OFFSET)
                        py = py / r_local * (r_local * RADIAL_OFFSET)

                    pts.append((px, py, pz))

            # A triangle crossed by an isoline should give exactly 2 points
            if len(pts) == 2:
                ax.plot([pts[0][0], pts[1][0]],
                        [pts[0][1], pts[1][1]],
                        [pts[0][2], pts[1][2]],
                        color='white', linewidth=1.2, alpha=1.0, zorder=10)
                label_pts.append(pts)

        # Place one text label per isoline level, at the midpoint of a
        # representative segment (the middle one in the list)
        if label_pts:
            p0, p1 = label_pts[len(label_pts) // 2]   # '//' = integer division
            mx = (p0[0] + p1[0]) / 2
            my = (p0[1] + p1[1]) / 2
            mz = (p0[2] + p1[2]) / 2
            ax.text(mx, my, mz, f'{level:.2f}',
                    color='yellow', fontsize=6.5, fontweight='bold',
                    zorder=20, ha='center')

    # ------------------------------------------------------------------
    # STEP 6: Colorbar and axis labels
    # ------------------------------------------------------------------
    sm = ScalarMappable(cmap=cmap, norm=norm_c)
    sm.set_array([])   # required boilerplate for standalone ScalarMappable colorbars
    fig.colorbar(sm, ax=ax, shrink=0.5, label="Geodesic distance")

    ax.set_title("FMM - Distance field on L-cylinder")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    plt.tight_layout()


def plot_convergence(h_values, errors_l2_fmm, errors_l2_circ, title, save_name):

    h_arr       = np.array(h_values[::-1])
    l2_fmm_arr  = np.array(errors_l2_fmm[::-1])
    l2_circ_arr = np.array(errors_l2_circ[::-1])

    print("\n\nConvergence order (L2) - Standard FMM:")
    for i in range(1, len(h_values)):
        order = np.log(errors_l2_fmm[i] / errors_l2_fmm[i-1]) / \
                np.log(h_values[i] / h_values[i-1])
        print(f"  h={h_values[i]:.4f}  |  L2={errors_l2_fmm[i]:.2e}  |  order ≈ {order:.2f}")

    print("\n\nConvergence order (L2) - Circular FMM:")
    for i in range(1, len(h_values)):
        order = np.log(errors_l2_circ[i] / errors_l2_circ[i-1]) / \
                np.log(h_values[i] / h_values[i-1])
        print(f"  h={h_values[i]:.4f}  |  L2={errors_l2_circ[i]:.2e}  |  order ≈ {order:.2f}")

    plt.figure(figsize=(8, 6))
    plt.loglog(h_arr, l2_fmm_arr,  'o-', label='FMM (L2)')
    plt.loglog(h_arr, l2_circ_arr, 'o-', label='HFMM (L2)')
    plt.loglog(h_arr, h_arr,       '--', color='gray',  label='O(h)')
    plt.loglog(h_arr, h_arr**2,    '--', color='black', label='O(h²)')
    plt.xlabel('h (mesh size)')
    plt.ylabel('L2 Error')
    plt.title(title)
    plt.legend()
    plt.grid(True, which='both')
    plt.gca().invert_xaxis()
    plt.tight_layout()
    plt.savefig(save_name)
    #plt.show()




def Plot_Error_Field(node_list, element_list, mesh_type, true_source, L=1.0, R=0.5):
    import matplotlib.cm as cm
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    import matplotlib.pyplot as plt
    
    x = np.array([n.coords[0] for n in node_list])
    y = np.array([n.coords[1] for n in node_list])
    z_coord = np.array([n.coords[2] for n in node_list])
    
    # Extraction of the error array using the existing mathematical functions
    if mesh_type == 'square_surface':
        _, _, errors, _ = compute_err(node_list, true_source)
    elif mesh_type == 'cylinder':
        _, errors, _ = compute_err_cylinder(node_list, true_source, R)
    elif mesh_type == 'l_shape':
        _, _, errors, _ = compute_err_l_shape(node_list, true_source, L)
    elif mesh_type == 'hole':
        _, _, errors, _ = compute_err_hole(node_list, true_source, L, R)
    elif mesh_type == 'l_cylinder':
        _, _, errors, _ = compute_err_l_cylinder(node_list, true_source, R, L)
        # NOTE: L is repurposed as H (cylinder height) when called for l_cylinder
        
    triangles = np.array([[n.idx for n in e.nodes] for e in element_list])
    
    # --- 3D PLOT for Cylinder and L-cylinder ---
    if mesh_type in ('cylinder', 'l_cylinder'):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        face_errors = errors[triangles].mean(axis=1)
        norm_c = Normalize(vmin=errors.min(), vmax=errors.max())
        cmap = cm.inferno  
        colors = cmap(norm_c(face_errors))
        
        surf = ax.plot_trisurf(x, y, z_coord, triangles=triangles, shade=False, antialiased=False, alpha=0.9)
        surf.set_facecolors(colors)
        
        sm = ScalarMappable(cmap=cmap, norm=norm_c)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, shrink=0.5, label="Absolute Error")
        
        ax.set_title(f"Error Distribution - {mesh_type}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        
    # --- 2D PLOT for L-Shape, Square and Hole ---
    else:
        plt.figure(figsize=(8, 6))
        plt.gca().set_aspect('equal')
    
        levels = np.linspace(errors.min(), errors.max(), 40)
        
        cntr = plt.tricontourf(x, y, triangles, errors, levels=levels, cmap="viridis", extend='both')
        plt.colorbar(cntr, label="Signed Error")
        
        plt.triplot(x, y, triangles, color='black', alpha=0.1, linewidth=0.5)
        
        plt.title(f"Error Distribution - {mesh_type}")
        plt.xlabel("X")
        plt.ylabel("Y")





def Plot_Point_Convergence(h_values, point_errors_fmm, point_errors_circ,
                           probe_points, title, save_name):
    """
    Plot the pointwise relative error convergence for HFMM only, on a log-log scale.

    One curve is drawn per probe point for HFMM (circular wavefront FMM).
    Reference lines O(h) and O(h^2) are also shown.

    Parameters
    ----------
    h_values : list of float
        Mesh sizes h, one per refinement level (coarse to fine).
    point_errors_fmm : list of list of float
        Not used in the plot. Kept in the signature for compatibility
        with the call in accuracy_plot.py.
    point_errors_circ : list of list of float
        Outer list: one entry per refinement level.
        Inner list: one relative error per probe point.
        point_errors_circ[i][j] = relative error at probe_points[j] for h=h_values[i].
    probe_points : list of array-like
        The (x, y) probe point coordinates, used for legend labels.
    title : str
        Plot title.
    save_name : str
        Filename for saving the figure (e.g. 'ConvStudyLshape.pdf').
    """
    # Reverse so that h goes from coarse to fine (large h to small h) on x-axis,
    # consistent with the global convergence plot in plot_convergence().
    h_arr = np.array(h_values[::-1])

    # One distinct colour per probe point
    colors = plt.cm.tab10(np.linspace(0, 0.5, len(probe_points)))

    plt.figure(figsize=(8, 6))

    for j, pt in enumerate(probe_points):
        # Extract the error at probe point j across all refinement levels,
        # then reverse to match the reversed h array.
        errs_circ = np.array([point_errors_circ[i][j] for i in range(len(h_values))])[::-1]

        label_circ = f'HFMM @ ({pt[0]:.1f}, {pt[1]:.1f})'

        plt.loglog(h_arr, errs_circ, '-o', color=colors[j], label=label_circ)

    # Reference slopes
    plt.loglog(h_arr, h_arr,    '--', color='gray',  linewidth=1.5, label='O(h)')
    plt.loglog(h_arr, h_arr**2, '--', color='black', linewidth=1.5, label='O(h²)')

    plt.xlabel('h (mesh size)')
    plt.ylabel('Relative error  |T_HFMM - T_exact| / T_exact')
    plt.title(title + ' — HFMM pointwise convergence')
    plt.legend(fontsize=8)
    plt.grid(True, which='both')
    plt.gca().invert_xaxis()
    plt.ylim(bottom=1e-7)
    plt.tight_layout()

    # Save under a distinct name to avoid overwriting the global convergence plot
    point_save_name = save_name.replace('.pdf', '_points.pdf')
    plt.savefig(point_save_name)




"""

def Plot_Point_Convergence(h_values, point_errors_fmm, point_errors_circ,
                           probe_points, title, save_name):

    Plot the pointwise relative error convergence on a log-log scale.

    One curve is drawn per probe point, for both FMM and HFMM, giving
    2 * len(probe_points) curves in total.  Reference lines O(h) and O(h^2)
    are also shown.

    Parameters
    ----------
    h_values : list of float
        Mesh sizes h, one per refinement level, in the order they were computed
        (coarse to fine, i.e. decreasing h).
    point_errors_fmm : list of list of float
        Outer list: one entry per refinement level.
        Inner list: one relative error per probe point.
        So point_errors_fmm[i][j] = relative error at probe_points[j] for h=h_values[i].
    point_errors_circ : list of list of float
        Same structure as point_errors_fmm, but for HFMM (circular wavefront).
    probe_points : list of array-like
        The (x, y) probe point coordinates, used only for legend labels.
    title : str
        Plot title.
    save_name : str
        Filename for saving the figure (e.g. 'PointConvSquare.pdf').

    # Reverse the lists so that h goes from coarse to fine on the x-axis,
    # consistent with the global convergence plot in plot_convergence().
    h_arr = np.array(h_values[::-1])

    # Define one colour per probe point so FMM and HFMM share the same colour
    # but are distinguished by line style (solid vs dashed).
    colors = plt.cm.tab10(np.linspace(0, 0.5, len(probe_points)))

    plt.figure(figsize=(8, 6))

    for j, pt in enumerate(probe_points):
        # Extract the error at probe point j across all refinement levels,
        # then reverse to match the reversed h array.
        errs_fmm  = np.array([point_errors_fmm[i][j]  for i in range(len(h_values))])[::-1]
        errs_circ = np.array([point_errors_circ[i][j] for i in range(len(h_values))])[::-1]

        label_fmm  = f'FMM  @ ({pt[0]:.1f},{pt[1]:.1f})'
        label_circ = f'HFMM @ ({pt[0]:.1f},{pt[1]:.1f})'

        # Solid line for standard FMM, dashed for HFMM; same colour per point
        plt.loglog(h_arr, errs_fmm,  '-o',  color=colors[j], label=label_fmm)
        plt.loglog(h_arr, errs_circ, '--o', color=colors[j], label=label_circ)

    # Reference slopes
    plt.loglog(h_arr, h_arr,    ':', color='gray',  linewidth=1.5, label='O(h)')
    plt.loglog(h_arr, h_arr**2, ':', color='black', linewidth=1.5, label='O(h²)')

    plt.xlabel('h (mesh size)')
    plt.ylabel('Relative error  |T_FMM - T_exact| / T_exact')
    plt.title(title + ' — pointwise')
    plt.legend(fontsize=7)
    plt.grid(True, which='both')
    plt.gca().invert_xaxis()
    plt.tight_layout()

    # Build a separate save name for this plot to avoid overwriting
    # the global convergence figure saved by plot_convergence().
    point_save_name = save_name.replace('.pdf', '_points.pdf')
    plt.savefig(point_save_name)
"""