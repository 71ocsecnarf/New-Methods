# Fast Marching Method — Geodesic Distance Solver

This project implements a **Fast Marching Method (FMM)** and a higher-order **Circular-Wavefront FMM (HFMM)** for computing geodesic distances on triangulated surfaces. Five mesh geometries are supported: a flat square, an L-shaped domain, a square with a circular hole, a half-cylinder, and an L-shaped half-cylinder.

Developed by Francesco Mauri and Catalin-Ioan Chirita
---

## Project Structure

| File | Role |
|---|---|
| `structures.py` | Core data structures: `NODE`, `ELEMENT`, `TRIAL_BAND` |
| `mesh_generation.py` | Gmsh-based mesh builders for all five geometries |
| `generate_meshes.py` | Pre-generates and indexes meshes on disk |
| `search_non_obtuse_mesh.py` | Finds mesh resolutions with zero obtuse triangles |
| `fmm_solver.py` | Standard FMM and circular-wavefront HFMM solvers |
| `solver.py` | Mesh loading, source initialisation, error computation, plots |
| `main.py` | Single-run script: run FMM on one mesh and visualise results |
| `accuracy_plot.py` | Convergence study: loop over multiple mesh sizes and plot errors |

---

## Workflow Overview

The typical workflow has three steps:

```
1. search_non_obtuse_mesh.py   →   find valid N values for your geometry
2. generate_meshes.py          →   pre-generate and save those meshes
3. main.py  /  accuracy_plot.py  →  run the solver and visualise
```

Steps 1–2 only need to be run once per geometry. After that, `main.py` and `accuracy_plot.py` load meshes from disk and never regenerate them unless forced.

---

## Step 1 — Finding Non-Obtuse Mesh Resolutions

The FMM upwind condition requires that **all triangles in the mesh are non-obtuse** (all angles ≤ 90°). Structured geometries (`square_surface`, `cylinder`) are always non-obtuse by construction. Unstructured geometries (`l_shape`, `hole`, `l_cylinder`) require a search.

Open `search_non_obtuse_mesh.py` and set:

```python
MESH_TYPE = 'l_shape'   # 'l_shape', 'hole', or 'l_cylinder'
TARGETS   = [5 * 2**i for i in range(6)]   # target N values to search around
```

Then run:
```bash
python search_non_obtuse_mesh.py
```

The script searches in a neighbourhood around each target N and prints a list of valid N values with zero obtuse triangles. Copy the resulting `N_VALUES` list into the next step.

> **Note:** The script also writes `non_obtuse_<mesh_type>.txt` with the clean N values, and `non_obtuse_<mesh_type>_fallback.txt` for any N values that still have a small number of obtuse triangles. Do **not** mix fallback meshes with clean meshes in convergence studies.

### Geometry parameters to set in this file

| Parameter | Relevant geometry | Meaning |
|---|---|---|
| `L = 1.0` | `l_shape`, `hole` | Side length of the bounding square |
| `H = 1.0` | `l_cylinder` | Cylinder height |
| `R_CYLINDER = 0.5` | `l_cylinder` | Cylinder radius |
| `R_HOLE = 0.2` | `hole` | Radius of the circular hole |

---

## Step 2 — Pre-generating Meshes

Open `generate_meshes.py` and set:

```python
MESH_TYPE = 'l_shape'   # geometry to generate
N_VALUES  = [5, 11, 19, 40, 81, 160, 319]   # N values from Step 1
```

Then run:
```bash
python generate_meshes.py
```

Meshes are written to `meshes/<mesh_type>/` and indexed in `meshes/<mesh_type>/index.json`. The index stores `N`, `h`, geometry parameters, and the filename for each mesh, so that subsequent scripts can locate the correct file without regenerating it.

Setting `FORCE_REGENERATE = True` overwrites existing meshes; the default (`False`) skips any N already present in the index.

### Geometry parameters to set in this file

The same `L`, `H`, `R_CYLINDER`, `R_HOLE` parameters as in Step 1. Make sure they are consistent between the two files.

### Pre-computed N values per geometry

These are the N values already validated to produce non-obtuse meshes:

| Geometry | Valid N values |
|---|---|
| `square_surface` | `[5 * 2**i for i in range(8)]` (any N, structured) |
| `cylinder` | `[5 * 2**i for i in range(8)]` (any N, structured) |
| `l_shape` | `[5, 11, 19, 40, 81, 160, 319, 641]` |
| `hole` | `[6, 9, 21, 34, 76, 152, 312]` |
| `l_cylinder` | `[9, 10, 18, 41, 77, 161, 321, 647]` |

---

## Step 3a — Single Run (`main.py`)

`main.py` loads one mesh, runs the FMM or HFMM, and displays the distance field and error field.

Open `main.py` and set:

```python
MESH_TYPE = 'l_shape'    # geometry
FMM_TYPE  = 'circ'       # 'standard' (FMM) or 'circ' (HFMM)
N         = 19           # must exist in the mesh index
```

Set the source point coordinates:

```python
# For flat 2D geometries (square_surface, l_shape, hole):
SOURCE_XY = np.array([0.6, 0.7])

# For cylinder (theta in degrees, z coordinate):
SOURCE_THETA_DEG    = -90
SOURCE_Z            = 0.0

# For l_cylinder (theta in degrees, z coordinate):
SOURCE_THETA_DEG_LC = 50
SOURCE_Z_LC         = 0.0
```

Then run:
```bash
python main.py
```

The script produces:
- **Isolines plot** of the computed distance field
- **Error field plot** showing the signed difference between FMM and the exact geodesic

---

## Step 3b — Convergence Study (`accuracy_plot.py`)

`accuracy_plot.py` loops over several mesh resolutions, runs both FMM and HFMM, and plots the L2 error convergence.

Open `accuracy_plot.py` and set:

```python
MESH_TYPE = 'l_shape'
N_VALUES  = [5, 11, 19, 40, 81, 160, 319]   # must all exist in the mesh index
```

Set the source point (same convention as `main.py`):

```python
SOURCE_XY        = np.array([0.6, 0.7])   # 2D geometries
SOURCE_THETA_DEG = 0.0                     # cylinder / l_cylinder
SOURCE_Z         = 0.0
```

Optionally set probe points for pointwise convergence tracking (2D geometries only):

```python
PROBE_POINTS = [
    np.array([1.0, 1.0]),
    np.array([0.8, 1.0]),
    np.array([1.0, 0.8]),
]
```

Then run:
```bash
python accuracy_plot.py
```

The script produces:
- A **log-log convergence plot** of L2 error vs. mesh size `h` for both FMM and HFMM, with O(h) and O(h²) reference lines
- A **pointwise convergence plot** for HFMM at each probe point (2D geometries only)
- An **error field plot** on the finest mesh

Both plots are saved to PDF files in the working directory.

---

## Geometry Reference

### `square_surface`
Unit square `[0, L]² `. Structured transfinite mesh. Exact solution is the Euclidean distance.

### `l_shape`
L-shaped domain: unit square minus the bottom-right quadrant `(x > L/2, y < L/2)`. Concave corner at `(L/2, L/2)`. Geodesics that cross the missing quadrant detour through the corner.

### `hole`
Unit square with a circular hole of radius `R` centred at `(L/2, L/2)`. Geodesics that intersect the hole detour around it along tangent paths.

### `cylinder`
Front half of a cylinder (`theta ∈ [0, π]`, `z ∈ [0, H]`). Structured mesh. Exact solution uses the arc-length metric on the cylindrical surface.

### `l_cylinder`
L-shaped half-cylinder lateral surface: a bottom arm (`theta ∈ [0, π]`, `z ∈ [0, H/2]`) plus a left arm (`theta ∈ [π/2, π]`, `z ∈ [H/2, H]`). The concave corner is at `(theta=π/2, z=H/2)`. Geodesics are computed by isometric unrolling to a 2D L-shaped domain.

---

## Algorithm Notes

**Standard FMM** propagates a wavefront using the first-order Eikonal update on each triangle. It achieves O(h) convergence.

**Circular-wavefront HFMM** tracks a virtual point source for each wave segment, allowing it to reconstruct the exact circular wavefront rather than a piecewise-linear approximation. It achieves O(h²) convergence on non-obtuse meshes.

Both solvers require non-obtuse meshes. On obtuse triangles, the upwind condition fails and the computed distance may be inaccurate.
