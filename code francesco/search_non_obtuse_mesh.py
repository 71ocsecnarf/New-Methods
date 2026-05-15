"""
search_non_obtuse_mesh.py
--------------------------
Search for non-obtuse meshes near a set of target N values for a given geometry.

For each target N, the script generates meshes in a neighbourhood [N-r, N+r]
(expanding outward from N) and keeps the first one that has zero obtuse triangles.

Fallback behaviour
------------------
If no fully non-obtuse mesh is found within the search radius, the script falls
back to the mesh with the FEWEST obtuse triangles in the explored range.
Fallback results are clearly marked in the output and saved to a separate file,
because meshes with even a few obtuse triangles violate the FMM upwind condition
and will degrade the convergence order -- mixing them silently with clean results
would make the convergence plots misleading.

Output files
------------
    non_obtuse_<mesh_type>.txt          -- clean results only (0 obtuse triangles)
    non_obtuse_<mesh_type>_fallback.txt -- fallback results (>0 obtuse triangles)

NOTE: 'square_surface' and 'cylinder' use structured transfinite meshes that
are always non-obtuse by construction -- searching makes no sense for them.
The script warns and exits early for those types.
"""

import numpy as np
import os
import gmsh

import mesh_generation
import solver


# ============================================================
# -------------------- User configuration --------------------
# ============================================================

MESH_TYPE = 'hole'   # 'l_shape', 'hole', 'l_cylinder'
                        # ('square_surface' and 'cylinder' are always non-obtuse)

# Target N values to search around
TARGETS = [5 * 2**i for i in range(6)]


# Geometry parameters
L = 1.0   # side length        (square_surface, l_shape, hole)
H = 1.0   # cylinder height    (cylinder, l_cylinder)

# R depends on mesh type:
#   cylinder / l_cylinder : R = 0.5 (cylinder radius)
#   hole                  : R = 0.2 (hole radius)
#   square_surface / l_shape : R unused (set to 0.5 as dummy)
R_CYLINDER = 0.5
R_HOLE     = 0.2

R_MAP = {
    'square_surface': 0.5,   # unused, dummy value
    'l_shape'       : 0.5,   # unused, dummy value
    'cylinder'      : R_CYLINDER,
    'l_cylinder'    : R_CYLINDER,
    'hole'          : R_HOLE,
}

R = R_MAP[MESH_TYPE]

# ============================================================

# Mesh types where obtuse triangles cannot appear (structured transfinite grids).
# Searching for non-obtuse meshes for these types is meaningless.
ALWAYS_NON_OBTUSE = {'square_surface', 'cylinder'}


def get_h(N, mesh_type):
    """Return the characteristic mesh size h for a given N and geometry."""
    if mesh_type in ('l_shape', 'hole'):
        return L / (N - 1)
    elif mesh_type == 'l_cylinder':
        return np.pi * R / (N - 1)
    else:
        raise ValueError(f"Unsupported mesh_type for h computation: '{mesh_type}'")


def count_obtuse(mesh_type, N, current_dir):
    """
    Generate a mesh for (mesh_type, N), count its obtuse triangles, and
    return (output_path, obtuse_count).

    Raises on any gmsh or solver error; the caller is responsible for
    finalising gmsh in the except branch.
    """
    output_path = mesh_generation.generate_mesh(
        N, mesh_type, current_dir, L=L, R=R, H=H
    )

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.open(output_path)

    tag_to_node_obj_dic = {}
    node_list    = solver.Make_NodeList_NodeDictionary(output_path, tag_to_node_obj_dic)
    element_list = solver.Make_ElementList(output_path, tag_to_node_obj_dic)

    solver.Check_Obtuse_triangles(element_list)
    obtuse_count = sum(1 for el in element_list if el.IsObtuse)

    # Debug: print centroid of obtuse triangles
    for e in element_list:
        if e.IsObtuse:
            centroid = np.mean([n.coords for n in e.nodes], axis=0)
            print(f"  Obtuse triangle centroid: {centroid}")

    gmsh.finalize()
    return output_path, obtuse_count


def find_non_obtuse_meshes(mesh_type=MESH_TYPE, targets=TARGETS):
    """
    For each value in `targets`, find the nearest N (within a dynamic radius)
    that produces a fully non-obtuse mesh of type `mesh_type`.

    If no clean mesh exists in the range, fall back to the N that produced
    the fewest obtuse triangles and report it separately.

    Parameters
    ----------
    mesh_type : str
        One of 'l_shape', 'hole', 'l_cylinder'.
    targets : list of int
        Ideal N values to search around.
    """

    # -- Guard: structured types are always non-obtuse; no search needed --------
    if mesh_type in ALWAYS_NON_OBTUSE:
        print(
            f"[INFO] '{mesh_type}' uses a structured transfinite mesh that is\n"
            f"       non-obtuse by construction. Nothing to search for.\n"
            f"       If you need an N_VALUES list, just use your TARGETS directly."
        )
        return

    if mesh_type not in mesh_generation.MESH_FILENAMES:
        raise ValueError(
            f"Unknown mesh_type: '{mesh_type}'. "
            f"Valid options: {list(mesh_generation.MESH_FILENAMES.keys())}"
        )

    current_dir = os.path.dirname(os.path.abspath(__file__))

    # clean_results   : list of (N, h)               -- zero obtuse triangles
    # fallback_results: list of (N, h, obtuse_count) -- best available otherwise
    clean_results    = []
    fallback_results = []

    print(f"{'='*60}")
    print(f"  Non-obtuse mesh search  |  type = {mesh_type}")
    print(f"  Targets : {targets}")
    print(f"{'='*60}\n")

    for target in targets:

        # Dynamic search radius: at least ±5, or ±10 % of the target
        search_radius = max(10, int(target * 0.10))

        # Build the offset sequence: 0, +1, -1, +2, -2, ...
        # This ensures we always try the exact target first.
        offsets = [0]
        for step in range(1, search_radius + 1):
            offsets.extend([step, -step])

        print(f"--- Searching around N = {target} (radius ±{search_radius}) ---")

        found_clean   = False
        # Tracks the best fallback seen so far for this target.
        # Stored as (obtuse_count, N) so we can minimise obtuse_count.
        best_fallback = None

        for offset in offsets:
            N = target + offset
            if N < 3:   # minimum 3 points to form a triangle
                continue

            try:
                _, obtuse_count = count_obtuse(mesh_type, N, current_dir)

                if obtuse_count == 0:
                    h = get_h(N, mesh_type)
                    print(
                        f"  ✅  Clean mesh found for target ~{target}: "
                        f"N={N:4d} (offset {offset:+d}) | h={h:.6f}\n"
                    )
                    clean_results.append((N, h))
                    found_clean = True
                    break   # stop searching for this target

                else:
                    # Keep track of the candidate with the fewest obtuse triangles.
                    # In case of a tie, the first one found wins (i.e. closest to target).
                    if best_fallback is None or obtuse_count < best_fallback[0]:
                        best_fallback = (obtuse_count, N)

            except Exception as e:
                print(f"  ⚠️  Error with N={N}: {e}")
                if gmsh.isInitialized():
                    gmsh.finalize()

        # If the full range was exhausted without a clean mesh, record the fallback
        if not found_clean:
            if best_fallback is not None:
                best_obtuse, best_N = best_fallback
                h = get_h(best_N, mesh_type)
                print(
                    f"  ⚠️  No clean mesh found for target ~{target}.\n"
                    f"      Best fallback: N={best_N:4d} | "
                    f"obtuse triangles = {best_obtuse} | h={h:.6f}\n"
                )
                fallback_results.append((best_N, h, best_obtuse))
            else:
                print(
                    f"  ❌  No mesh could be generated near target {target} "
                    f"(all attempts failed with errors).\n"
                )

    # -- Summary ----------------------------------------------------------------
    print(f"{'='*60}")
    print(f"  Search complete.")
    print(f"  Clean meshes   : {len(clean_results)}/{len(targets)}")
    print(f"  Fallback meshes: {len(fallback_results)}/{len(targets)}")
    print()

    all_N_clean    = [N for N, _       in clean_results]
    all_N_fallback = [N for N, _, __ in fallback_results]

    if all_N_clean:
        print(f"  ✅  Clean N values   : {all_N_clean}")
    if all_N_fallback:
        print(f"  ⚠️  Fallback N values (obtuse triangles > 0):")
        for N, h, oc in fallback_results:
            print(f"        N={N:4d}  |  obtuse triangles = {oc:5d}  |  h={h:.6f}")
    print()

    # -- Save clean results -----------------------------------------------------
    if clean_results:
        save_path = os.path.join(current_dir, f"non_obtuse_{mesh_type}.txt")
        with open(save_path, "w") as f:
            f.write("N\th\n")
            for N, h in clean_results:
                f.write(f"{N}\t{h:.6f}\n")
        print(f"  Clean results saved to  : {save_path}")

        n_str = "[" + ", ".join(str(N) for N, _ in clean_results) + "]"
        print(f"  Paste into accuracy_plot.py / generate_meshes.py:")
        print(f"  N_VALUES = {n_str}")
        print()

    # -- Save fallback results (separate file, NOT to be mixed with clean ones) -
    if fallback_results:
        fb_path = os.path.join(current_dir, f"non_obtuse_{mesh_type}_fallback.txt")
        with open(fb_path, "w") as f:
            f.write("N\th\tobtuse_count\n")
            for N, h, oc in fallback_results:
                f.write(f"{N}\t{h:.6f}\t{oc}\n")
        print(f"  Fallback results saved to: {fb_path}")
        print(
            f"\n  WARNING: fallback meshes contain obtuse triangles.\n"
            f"           The FMM upwind condition is NOT guaranteed on them.\n"
            f"           Do NOT mix fallback and clean results in convergence studies."
        )

    print(f"{'='*60}")


if __name__ == "__main__":
    find_non_obtuse_meshes()
