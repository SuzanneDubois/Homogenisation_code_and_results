# DEM homogenisation / calibration

## Requirements

The project was developed with **Python 3.11** inside a virtual environment created with `venv`.
All dependencies, with their exact versions, are listed in `requirements.txt`.

Create the virtual environment and install the dependencies from the project root:

```bash
python3.11 -m venv venv            # create the environment in ./venv
source venv/bin/activate           # activate it (Windows: venv\Scripts\activate)
pip install --upgrade pip
pip install -r requirements.txt    # install all dependencies
```

`git` must be installed because one package (ELATE) is installed straight from its GitHub repository.

Activate the environment (`source venv/bin/activate`) every time you open a new terminal to work on the project, and run `deactivate` to leave it.

If you add or update a package, regenerate the file with:

```bash
pip freeze > requirements.txt
```

## Numerical homogenisation

`0_Numerical_homogenisation/homogenisation.py` computes the **effective (homogenised) stiffness matrix** of a spring network..
Use it to check that a set of spring stiffnesses reproduces the target isotropic elastic material.

### How it works

1. The script reads the network (nodes and bonds) and the stiffness of every spring, then assembles the problem with [fedoo].
2. It runs 9 elementary loading cases: 3 tractions (x, y, z) and 6 shears (yz, xz, xy, and zy, zx, yx to check minor symmetry).
   Each case imposes a small macroscopic strain `eps_bar` (a displacement of `0.001` divided by the RVE length) through one of two kinds of boundary conditions, chosen with the `BC` flag at the top of the file:
   - `"pbc"` (default): periodic boundary conditions (the mesh must be periodic)
   - `"kubc"`: kinematic uniform boundary conditions, `u = eps_bar · x` on the boundary nodes
3. For each case, it computes the homogenised stress from the reaction forces and the RVE volume (the volume of the mesh bounding box). Each case gives one column of the 6×6 stiffness matrix `L_eff` (Voigt notation). The matrix is then checked for symmetry and symmetrised.
4. It compares `L_eff` with the isotropic reference material defined by `YoungModulus` and `PoissonRatio` (set in the `__main__` block):
   - **Isotropy error**: the relative Mandel-norm distance between `L_eff` and the reference stiffness tensor
   - **Linearity error**: the relative L2 distance between the nodal displacements and the affine (Cauchy-Born) field `eps_bar · x`, for each loading case
5. It plots the directional elastic properties with [ELATE](https://github.com/coudertlab/elate): Young's modulus, Poisson's ratio, linear compressibility and shear modulus.

### Usage

Run it from the folder that contains the input files, giving the base name of the case without an extension:

```bash
cd R0_Cube_periodic
python ../0_Numerical_homogenisation/homogenisation.py cube_diagonal
```

### Input

| File | Content |
|------|---------|
| `<name>.msh` | Gmsh mesh of the network: nodes, with bonds stored as 2-node line elements |
| `<name>_k.txt` | Spring stiffness of each bond, one value per line, in the same order as the elements of the mesh |
| `<name>_boundary_vertices.txt` *(optional, KUBC only)* | Indices of the boundary nodes, one per line. Without this file, the script finds the boundary nodes from the convex hull, which is only reliable for convex shapes. |

### Output

| File | Content |
|------|---------|
| terminal | `L_eff`, the symmetry check, the isotropy error, the linearity errors and the mean Young's modulus |
| `<name>_errors.txt` | Isotropy error and linearity error of each test, with their mean and max |
| `<name>_plots/<name>_periodic.pdf` | Overview figure: polar plots of Young's modulus, Poisson's ratio, linear compressibility and shear modulus, the shear-modulus error, and the mesh. The suffix is `_non-periodic` with KUBC. |
| `<name>_plots/young_modulus.pdf`, `poisson_ratio.pdf`, `linear_compressibility.pdf`, `shear_modulus.pdf`, `shear_error.pdf`, `mesh.pdf` | Each panel of the overview figure as a separate file |
| `<name>_plots/displacement_*.pdf`, `displacements_vs_cauchy_born.pdf` | Nodal displacements compared with the Cauchy-Born field, one file per test plus an overview |

## Calibration (analytical cell stiffnesses)

The scripts in `1_Anaytical_cell_stiffnesses/` compute the spring stiffness `k_n` of every bond **analytically**, so that the network behaves like an isotropic elastic material with a given Young's modulus `E`.
The result is written to `<name>_k.txt`, which `homogenisation.py` and the tests then read.

### Principle

For a network of central springs with bond vectors `x_n` in a volume `V`, the homogenised stiffness tensor is

```
C_abcd = 1/V · Σ_n k_n · x_a x_b x_c x_d / |x_n|²
```

Calibration chooses the `k_n` so that `C` equals the isotropic tensor `2/5 · E · (δ_ab δ_cd + δ_ac δ_bd + δ_ad δ_bc)`.
Central-force networks obey the Cauchy relations, so the Poisson's ratio is always **ν = 1/4**.
This gives a linear system `M · k = target`, with one column of `M` (81 components) per bond.

### The 4 files

| File | Calibration strategy | Volume `V` used |
|------|----------------------|-----------------|
| `coefficients_computation.py` | **Library and global solve.** Defines the shared functions: `solve_gn` solves the linear system by least squares (SVD, rank, null space and residual), and `find_positive_solution` searches the null space by linear programming for a solution with every `k_n > 0`. Run as a script, it solves **one system for the whole mesh** (one unknown per bond). | Bounding box of the mesh |
| `individual_cell_coefficients.py` | **Cell by cell.** Solves one system for **each cell** (each Gmsh physical group, i.e. the 28 bonds of a hexahedron or the 6 bonds of a tetrahedron), then makes it positive with `find_positive_solution`. Each cell is isotropic on its own. The per-cell solutions are concatenated in the order of the physical groups, which matches the bond order written by the mesh convertors. | Convex hull of the cell |
| `mean_cell_coefficients.py` | **Mean calibration by orientation class.** Groups the bonds into orientation classes on the unit sphere with a HEALPix tessellation (`nside = 4`; `x` and `-x` are the same orientation). It solves **one system with one unknown `k_eq` per class**, using the mean direction of each class, then splits each `k_eq` among the bonds of its class (see below). | Convex hull of the mesh |
| `same_coefficients.py` | **Uniform mean calibration.** Uses the same orientation classes but does **not solve** the system: every class gets `k_eq = 6·E·V / n_classes`. This value is exact when the classes are spread uniformly over the sphere, because HEALPix pixels have equal area. The split among bonds is the same as in `mean_cell_coefficients.py`. | Convex hull of the mesh |

**Splitting `k_eq` among the bonds of a class.** A bond `i` of class `c` contributes `k_i·|x_i|²` to the class, so the bond stiffnesses must satisfy `Σ_i k_i·|x_i|² = k_eq`.
Both mean scripts define three ways to split `k_eq`. You choose one by uncommenting it in the `__main__` block:

| Function | Rule | Result folder in R3 |
|----------|------|---------------------|
| `k_cell_identical` | `k_i = k_eq / Σ_j |x_j|²`: every bond of the class has the same stiffness | `mean_calibration_identical` |
| `k_cell_length` | `k_i = k_eq / (n_c · |x_i|²)`: every bond carries the same share `k_eq / n_c`, where `n_c` is the number of bonds in the class | `mean_calibration_length` |
| `k_cell_volume` *(active)* | `k_i = k_eq · V_cell(i) / (Σ_j V_cell(j) · |x_i|²)`: each bond's share is proportional to the volume of its cell, which suits meshes with cells of different sizes | `mean_calibration_volume` |

### Usage

Run each script from the folder that holds the mesh, giving the base name without an extension:

```bash
python ../1_Anaytical_cell_stiffnesses/mean_cell_coefficients.py <name>
```

The Young's modulus is hardcoded: `E = 1e6` in `coefficients_computation.py` and `E = 210e9` in the three other scripts.

### Input

| File | Content |
|------|---------|
| `<name>.msh` | Gmsh MSH2 (ASCII) bond network produced by the mesh convertors. Bonds are 2-node line elements, each with a physical group `cell_<i>` that identifies the cell it comes from. |

### Output

| File | Content |
|------|---------|
| `<name>_k.txt` | One spring stiffness per line, in the same order as the bonds in `<name>.msh` |
| terminal | Rank and residual of the linear system (a residual of about 0 means exact isotropy is achievable), and whether a positive solution was found |

## Mesh manipulation

Three files at the project root convert a finite element (FEM) mesh into a spring network and read and write the network files.
Every network is saved in the **Gmsh MSH2 ASCII** format:

- each bond is a 2-node line element (Gmsh element type 1)
- each bond has a physical group `cell_<i>` naming the cell it comes from

The calibration scripts use these physical groups.

### Mesh convertors: `hex_mesh_convertor.py` and `tet_mesh_convertor.py`

These scripts turn a solid FEM mesh into the matching spring network. Each cell of the FEM mesh is replaced by the **complete bond graph** of its nodes, meaning every pair of nodes is linked by a spring:

| Script | Input elements | Bonds per cell |
|--------|----------------|----------------|
| `hex_mesh_convertor.py` | 8-node hexahedra (Gmsh type 5) | 28: 12 edges, 12 face diagonals and 4 space diagonals |
| `tet_mesh_convertor.py` | 4-node tetrahedra (Gmsh type 4) | 6: the edges |

Other element types in the file (surfaces, lines, points) are ignored.
Bonds shared by neighbouring cells are **not merged**. They appear once per cell, so each cell stays a self-contained group that can be calibrated on its own.
Bonds are written cell by cell, which gives the bond order that `individual_cell_coefficients.py` relies on.

Both scripts take the base name of the mesh without the `.msh` extension:

```bash
python hex_mesh_convertor.py <name>     # reads <name>.msh
python tet_mesh_convertor.py <name>
```

| | File | Content |
|---|------|---------|
| Input | `<name>.msh` | FEM mesh from Gmsh, saved in MSH2 ASCII format (*File → Export → Version 2 ASCII*, or `-format msh2`) |
| Output | `<name>_spring_network.msh` | Spring network: the same nodes (renumbered from 1 to N), one line element per bond, and one physical group per cell |

The script then opens a 3D matplotlib view of the network. Close the window to end the script.
Pass `<name>_spring_network` as the base name to the calibration and homogenisation scripts.

You can also call `convert(input_path, output_path, plot=False)` from another script.

### Shared helpers: `mesh_utils.py`

The other scripts import these functions, so this file has no command-line interface.

| Function | Role |
|----------|------|
| `parse_msh2(filename)` | Reads a spring-network `.msh` file. Returns the nodes (`{id: xyz}`), the bonds (`[(id_a, id_b)]`), the physical tag of each bond and the physical group names. |
| `write_msh2(filename, node_coords, elements, name_to_physical_id)` | Writes nodes, bonds `(n1, n2, physical_id)` and physical names to an MSH2 ASCII file |
| `volume_bonding_box(nodes)` | Volume of the axis-aligned bounding box of the nodes |
| `volume_convex_hull(nodes)` | Volume of the convex hull of the nodes, used as the volume of a cell or of the whole mesh |
| `plot_mesh(nodes, bonds)` | 3D matplotlib plot of the nodes and bonds |

## Tests

<!-- 2_FEM_LSM_test/ (test_execution.py, post_processing.py, plot_stiffnesses.py) -->

### R0 – Periodic cube

### R1 – Deformed hexahedron

### R2 – Deformed hexahedral mesh

### R3 – Hexahedral beam

### R4 – Tetrahedral beam

### R5 – Final test


