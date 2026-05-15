"""
generate_meshes.py
------------------
Standalone script to pre-generate and save meshes for a given geometry and
set of N values.

Meshes are written to:
    meshes/<mesh_type>/<mesh_type>_N<NNN>.msh

An index file is maintained at:
    meshes/<mesh_type>/index.json

The index maps each N (as a string key) to its geometric parameters:
    { "N": int, "h": float, "mesh_type": str, "L": float, "R": float, "H": float }

Usage
-----
Configure MESH_TYPE, N_VALUES, and geometry parameters below, then run:
    python generate_meshes.py

FORCE_REGENERATE controls behaviour when a mesh for a given N already exists:
    False  --> skip existing meshes (safe default)
    True   --> overwrite existing meshes unconditionally
"""

import os
import json
import numpy as np
import mesh_generation


# ============================================================
# -------------------- User configuration --------------------
# ============================================================

MESH_TYPE = 'hole'   # 'square_surface', 'l_shape', 'hole', 'cylinder', 'l_cylinder'

#N_VALUES  = [5, 11, 19, 40, 81, 160, 319, 641] # L_shapes
# N_VALUES = [5 * 2**i for i in range(8)] # Square surface and cylinder
N_VALUES = [6, 9, 21, 34, 76, 152, 312] # hole 
#N_VALUES = [9, 10, 18, 41, 77, 161, 321, 647] # l_cylinder


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

# If False: skip N values whose mesh file already exists in the index.
# If True : regenerate and overwrite every mesh regardless.
FORCE_REGENERATE = True

# ============================================================


def get_h(N, mesh_type, L, R):
    """
    Return the characteristic mesh size h for a given N and geometry.
    Mirrors the get_h lambdas in accuracy_plot.py so the stored value is
    consistent with what the convergence study uses.
    """
    if mesh_type in ('square_surface', 'l_shape', 'hole'):
        return L / (N - 1)
    elif mesh_type in ('cylinder', 'l_cylinder'):
        return np.pi * R / (N - 1)
    else:
        raise ValueError(f"Unknown mesh_type: '{mesh_type}'")


def mesh_filename(mesh_type, N):
    """Return the .msh filename (without directory) for a given mesh_type and N."""
    return f"{mesh_type}_N{N:04d}.msh"


def load_index(index_path):
    """Load the index JSON file, returning an empty dict if it does not exist."""
    if os.path.exists(index_path):
        with open(index_path, 'r') as f:
            return json.load(f)
    return {}


def save_index(index_path, index):
    """Write the index dict to disk as pretty-printed JSON."""
    with open(index_path, 'w') as f:
        json.dump(index, f, indent=2)


def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    mesh_dir    = os.path.join(current_dir, 'meshes', MESH_TYPE)
    os.makedirs(mesh_dir, exist_ok=True)

    index_path = os.path.join(mesh_dir, 'index.json')
    index      = load_index(index_path)

    print(f"{'='*55}")
    print(f"  Mesh generation  |  type = {MESH_TYPE}")
    print(f"  Output dir       : {mesh_dir}")
    print(f"  FORCE_REGENERATE : {FORCE_REGENERATE}")
    print(f"{'='*55}\n")

    skipped   = []
    generated = []
    failed    = []

    for N in N_VALUES:
        key      = str(N)
        filename = mesh_filename(MESH_TYPE, N)
        out_path = os.path.join(mesh_dir, filename)

        # --- Skip logic ---
        if key in index and os.path.exists(out_path) and not FORCE_REGENERATE:
            print(f"  [SKIP]  N={N:4d}  -->  {filename}  (already in index)")
            skipped.append(N)
            continue

        # --- Generate ---
        print(f"  [GEN]   N={N:4d}  -->  {filename} ...", end=' ', flush=True)
        try:
            # generate_mesh() writes to output_dir using MESH_FILENAMES[mesh_type].
            # We then rename the file to our N-specific filename.
            tmp_path = mesh_generation.generate_mesh(
                N, MESH_TYPE, mesh_dir, L=L, R=R, H=H
            )

            # generate_mesh writes to the generic name (e.g. l_shape.msh).
            # Rename to the N-specific name so multiple meshes coexist.
            if tmp_path != out_path:
                if os.path.exists(out_path):
                    os.remove(out_path)
                os.rename(tmp_path, out_path)

            # Update index entry
            index[key] = {
                'N'         : N,
                'h'         : get_h(N, MESH_TYPE, L, R),
                'mesh_type' : MESH_TYPE,
                'filename'  : filename,
                'L'         : L,
                'R'         : R,
                'H'         : H,
            }
            save_index(index_path, index)   # save after each success (crash-safe)

            print('OK')
            generated.append(N)

        except Exception as e:
            print(f'FAILED  ({e})')
            failed.append(N)

    # --- Summary ---
    print(f"\n{'='*55}")
    print(f"  Done.  Generated={len(generated)}  Skipped={len(skipped)}  Failed={len(failed)}")
    if failed:
        print(f"  Failed N values: {failed}")
    print(f"  Index saved to: {index_path}")
    print(f"{'='*55}")


if __name__ == '__main__':
    main()
