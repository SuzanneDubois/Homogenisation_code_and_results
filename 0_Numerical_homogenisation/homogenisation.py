import fedoo as fd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from ELATE import elastic
import sys
import os

SPHERICAL_PLOT = False
BC = "pbc"  # "kubc" or "pbc"


#=============================================================== PLOT HELPERS ===============================================================

def set_axes_equal_3d(ax, xyz):

    xyz = np.asarray(xyz)
    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)
    centers = (mins + maxs) / 2
    max_range = (maxs - mins).max() / 2
    if max_range == 0:
        max_range = 1.0

    ax.set_xlim(centers[0] - max_range, centers[0] + max_range)
    ax.set_ylim(centers[1] - max_range, centers[1] + max_range)
    ax.set_zlim(centers[2] - max_range, centers[2] + max_range)
    ax.set_box_aspect((1, 1, 1))

def plot_displacement_vs_coord(pb, mesh, i, j, ax=None, title=None, boundary_nodes=None,
                                u_expected=None):
    """
    Plot displacement component u_i as a function of coordinate x_j.

    Parameters
    ----------
    pb    : solved fd.problem.Linear object
    mesh  : fd.Mesh object
    i     : int (0,1,2) — displacement component (0=ux, 1=uy, 2=uz)
    j     : int (0,1,2) — position coordinate   (0=x,  1=y,  2=z)
    ax    : matplotlib Axes to plot on; if None, a new figure is created
    title : str — custom title; if None, a default title is generated
    boundary_nodes : array-like of int, optional
        Indices of the boundary nodes. If None, they are recomputed with
        get_boundary_nodes(mesh) (convex-hull based, no file lookup).
    u_expected : array of shape (n_nodes, 3), optional — expected
        displacement field (see expected_displacement). If given, the
        expected u_i is drawn as a line and the relative L2 error (see
        compute_linearity_error) is annotated on the plot.

    Returns
    -------
    ax    : the matplotlib Axes with the plot
    error : float or None — the relative L2 linearity error, if u_expected
        was provided.
    """

    labels = ['x', 'y', 'z']

    nodes = mesh.nodes                        # (n_nodes, 3)
    U     = pb.get_disp().reshape(3, -1)     # (3, n_nodes)

    if boundary_nodes is None:
        boundary_nodes = get_boundary_nodes(mesh)
    boundary_nodes = np.asarray(boundary_nodes)

    is_boundary = np.zeros(nodes.shape[0], dtype=bool)
    is_boundary[boundary_nodes] = True

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    coord = nodes[:, j]
    order = np.argsort(coord)
    coord_sorted = coord[order]

    if u_expected is not None:
        ax.plot(coord_sorted, u_expected[order, i],
                color='gray', linewidth=2, zorder=1, label='expected linear')

    ax.scatter(nodes[~is_boundary, j], U[i, ~is_boundary],
               s=12, alpha=0.7, color='royalblue', zorder=2, label='interior')

    ax.scatter(nodes[is_boundary, j], U[i, is_boundary],
               s=12, alpha=0.7, color='crimson', zorder=3, label='boundary')

    error = None
    if u_expected is not None:
        error = compute_linearity_error(U.T, u_expected)
        ax.text(0.02, 0.98, f'L2 linearity error = {error:.2%}',
                transform=ax.transAxes, ha='left', va='top', fontsize=10,
                bbox=dict(facecolor='white', edgecolor='gray', alpha=0.85))

    ax.set_xlabel(f'${labels[j]}$ coordinate (m)')
    ax.set_ylabel(f'$u_{{{labels[i]}}}$ (m)')
    ax.set_title(title or f'$u_{{{labels[i]}}}$ vs ${labels[j]}$')
    ax.legend()

    #plt.show()

    return ax, error


#=============================================================== ERROR HELPERS ===============================================================

def expected_displacement(mesh, eps_bar):
    """
    Displacement field the imposed macroscopic strain asks for, i.e. the
    linear field that includes the solicitation, u_i = eps_ij * (x_j - x0_j):

    - "kubc" / "full constraint": eps_bar is applied as is, u_i = eps_bar_ij x_j
      (x0 = 0), exactly as in apply_kubc / apply_full_constraint.
    - "pbc": only the symmetric part of eps_bar is imposed (apply_pbc), and
      the translation is fixed by blocking the node returned by
      pbc_reference_node, so the field is measured from that node
      (x0 = position of the blocked node).

    Returns
    -------
    u_expected : ndarray of shape (n_nodes, 3)
    """
    nodes = mesh.nodes
    if BC == "pbc":
        eps = 0.5 * (eps_bar + eps_bar.T)
        x0 = nodes[pbc_reference_node(mesh)]
    else:
        eps = eps_bar
        x0 = np.zeros(3)
    return ((nodes - x0) @ eps.T)


def compute_linearity_error(u_values, u_expected):
    """
    Relative L2 error between the computed displacement field and the
    expected linear displacement field:

        error = ||u - u_expected||_2 / ||u_expected||_2

    Parameters
    ----------
    u_values   : array of shape (n_nodes, 3), computed nodal displacements
    u_expected : array of shape (n_nodes, 3), expected nodal displacements

    Returns
    -------
    error : float
    """
    u_values = np.asarray(u_values)
    u_expected = np.asarray(u_expected)
    return np.linalg.norm(u_values - u_expected) / np.linalg.norm(u_expected)


def build_isotropic_stiffness_voigt(E, nu):

    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    C_iso = np.zeros((6, 6))
    C_iso[:3, :3] = lam
    for k in range(3):
        C_iso[k, k] += 2 * mu
    for k in range(3, 6):
        C_iso[k, k] = mu
    return C_iso


def mandel_scale(C):
    """
    Rescale a 6x6 Voigt stiffness matrix (engineering-shear convention) to
    Mandel notation (shear-shear block scaled by 2, normal-shear blocks by
    sqrt(2)), so that its Frobenius norm equals the true tensorial norm of
    the underlying rank-4 stiffness tensor and is therefore invariant under
    a change of coordinate frame.
    """
    scale = np.ones(6)
    scale[3:] = np.sqrt(2)
    return C * np.outer(scale, scale)


def compute_stiffness_isotropy_error(C_homog, E, nu):
    """
    Relative distance between a homogenized stiffness matrix (Voigt
    notation) and the isotropic stiffness matrix defined by (E, nu),
    measured with the Mandel-scaled (rotation-invariant) Frobenius norm:

        error = ||C_homog - C_iso||_F / ||C_iso||_F   (Mandel scaling)

    Parameters
    ----------
    C_homog : array (6, 6) — homogenized stiffness matrix, Voigt notation
    E, nu   : reference isotropic material properties

    Returns
    -------
    error : float
    """
    C_iso = build_isotropic_stiffness_voigt(E, nu)
    diff_norm = np.linalg.norm(mandel_scale(C_homog - C_iso))
    ref_norm = np.linalg.norm(mandel_scale(C_iso))
    return diff_norm / ref_norm

#=============================================================== STIFFNESS HELPERS ===============================================================

def compute_homogenized_stress(F_vect, dirichlet_nodes, node_coords, volume):
    """
    Compute the homogenized stress vector (Voigt notation) from reaction forces.

    Parameters
    ----------
    F_vect : array of shape (3, n_nodes_total)
        Reaction force vector for all nodes (e.g. from pb.get_ext_forces()).
    dirichlet_nodes : array-like of int
        Indices of nodes constrained by Dirichlet BC.
    node_coords : array of shape (n_nodes_total, 3)
        Coordinates x, y, z of all nodes.
    volume : float
        Volume of the RVE.
    Returns
    -------
    sigma_voigt : array of shape (6,)
        [sigma_11, sigma_22, sigma_33, sigma_23, sigma_13, sigma_12]
    """
    dirichlet_nodes = np.asarray(dirichlet_nodes)
    n_nodes = F_vect.shape[1]

    # --- Computation ---
    f = F_vect[:, dirichlet_nodes]      # (3, n_dirichlet)
    x = node_coords[dirichlet_nodes, :] # (n_dirichlet, 3)

    sigma = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            sigma[i, j] = np.sum(f[i, :] * x[:, j]) / volume

    sigma_voigt = np.array([
        sigma[0, 0],  # sigma_11
        sigma[1, 1],  # sigma_22
        sigma[2, 2],  # sigma_33
        sigma[1, 2],  # sigma_23
        sigma[0, 2],  # sigma_13
        sigma[0, 1],  # sigma_12
    ])

    return sigma_voigt


def compute_homogenized_stress_pbc(pb, volume):
    """
    Compute the homogenized stress vector (Voigt notation) for periodic boundary conditions (PBC).

    Parameters
    ----------
    pb     : solved fd.problem.Linear object, whose bc list includes a
             PeriodicBC (added by apply_pbc).
    volume : float
        Volume of the RVE.

    Returns
    -------
    sigma_voigt : array of shape (6,)
        [sigma_11, sigma_22, sigma_33, sigma_23, sigma_13, sigma_12]
    """
    sigma_voigt = np.array([
        pb.get_ext_forces('E_xx')[0],
        pb.get_ext_forces('E_yy')[0],
        pb.get_ext_forces('E_zz')[0],
        pb.get_ext_forces('E_yz')[0],
        pb.get_ext_forces('E_xz')[0],
        pb.get_ext_forces('E_xy')[0],
    ]) / volume

    return sigma_voigt


#=============================================================== BC HELPERS ===============================================================

def pbc_reference_node(mesh):
    """
    Index of the node blocked in translation by apply_pbc: the node closest
    to the centre of the RVE.
    """
    nodes = mesh.nodes
    center = nodes.mean(axis=0)
    return int(np.argmin(np.linalg.norm(nodes - center, axis=1)))


def get_boundary_nodes(mesh, basename=None):
    """
    Return indices of nodes on the outer boundary of the mesh, either by
    reading them from the boundary-vertices file or by computing the convex
    hull of the mesh (may be inaccurate if the shape is not convex).
    """

    #try to get the nodes from th file (name)_boundary_vertices.txt
    print(f"Attempting to read boundary nodes from file: {basename}_boundary_vertices.txt")
    full_name = f"{basename}_boundary_vertices" if basename else None
    try:
        with open("{}.txt".format(full_name), "r") as f:
            boundary_nodes = np.array([int(line.strip()) for line in f.readlines()])
            print(f"Boundary nodes read from file: {len(boundary_nodes)} nodes found.")

    except FileNotFoundError:

        nodes = np.asarray(mesh.nodes, dtype=float)
        if nodes.size == 0:
            return np.array([], dtype=int)
        if nodes.ndim != 2 or nodes.shape[1] != 3:
            raise ValueError(f"mesh.nodes must have shape (n_nodes, 3), got {nodes.shape}")

        hull = ConvexHull(nodes)

        bbox_diag = np.linalg.norm(nodes.max(axis=0) - nodes.min(axis=0))
        tol = 1e-8 * max(1.0, bbox_diag)

        # hull.equations: shape (n_facets, 4), each row is [nx, ny, nz, d]
        # A point x is on the facet plane iff nx*x + ny*y + nz*z + d == 0
        # (normals are already unit-length)
        normals = hull.equations[:, :3]  # (n_facets, 3)
        offsets = hull.equations[:, 3]   # (n_facets,)

        # signed distances of all nodes to all facet planes: shape (n_nodes, n_facets)
        signed_dist = nodes @ normals.T + offsets[np.newaxis, :]

        # A node is on the boundary if it lies on at least one hull plane
        # AND is not strictly inside (signed_dist <= 0 for all facets, hull convention)
        on_plane = np.abs(signed_dist) <= tol          # (n_nodes, n_facets)
        not_outside = signed_dist <= tol                # hull interior: all dist <= 0
        is_interior = not_outside.all(axis=1)           # True for all valid mesh nodes
        on_boundary = is_interior & on_plane.any(axis=1)

        boundary_nodes = np.where(on_boundary)[0]
        print(f"Found {len(boundary_nodes)} boundary nodes out of {nodes.shape[0]} total nodes using the convex hull.")

    return boundary_nodes


def flatten_periodic_boundary_nodes(bc_periodic):
    """
    Flatten the dict of face/edge/corner node-index arrays exposed by
    fd.constraint.PeriodicBC.boundary_nodes (keys 'x-', 'x+', 'x-y-', ...)
    into a single array of unique node indices, for post-processing /
    plotting purposes (same role as the boundary_nodes array returned by
    apply_kubc / get_boundary_nodes).
    """
    all_nodes = np.concatenate(
        [np.atleast_1d(v).astype(int) for v in bc_periodic.boundary_nodes.values()]
    )
    return np.unique(all_nodes)


def apply_kubc(pb, mesh, eps_bar, basename=None):
    """
    Imposes the boundary conditions (KUBC):
        u_i(x) = eps_bar_ij * x_j   on all boundary nodes ∂Ω

    Parameters
    ----------
    pb       : fedoo problem (fd.problem.Linear)
    mesh     : fedoo mesh
    eps_bar  : macroscopic strain tensor (3x3 numpy array)

    Returns
    -------
    boundary_nodes : indices of the boundary nodes (useful for post-processing).
        The boundary conditions are added directly to the problem `pb`.
    """
    nodes = mesh.nodes
    boundary_nodes = get_boundary_nodes(mesh, basename)
    print(f"Applying KUBC to {len(boundary_nodes)} boundary nodes.")
    
    x0 = nodes[boundary_nodes]          # reference positions (N_bnd x 3)
    u_imposed = (eps_bar @ x0.T).T      # u_i = eps_bar_ij * x_j  →  (N_bnd x 3)
    
    pb.bc.add('Dirichlet', boundary_nodes, 'DispX', u_imposed[:, 0])
    pb.bc.add('Dirichlet', boundary_nodes, 'DispY', u_imposed[:, 1])
    pb.bc.add('Dirichlet', boundary_nodes, 'DispZ', u_imposed[:, 2])
    return boundary_nodes


def apply_pbc(pb, mesh, eps_bar, basename=None):
    """
    Applies periodic BC:

    Parameters
    ----------
    pb       : fedoo problem (fd.problem.Linear)
    mesh     : fedoo mesh (must be periodic)
    eps_bar  : macroscopic strain tensor (3x3 numpy array)
    basename : unused, kept for the same signature as apply_kubc

    Returns
    -------
    bc_periodic : the periodic BC object, also added directly to the problem `pb`.
    """
    # --- Periodicity conditions ---
    bc_periodic = fd.constraint.PeriodicBC("small_strain")
    pb.bc.add(bc_periodic)

    # --- Imposed macroscopic deformation ---
    pb.bc.add('Dirichlet', 'E_xx', eps_bar[0, 0])
    pb.bc.add('Dirichlet', 'E_yy', eps_bar[1, 1])
    pb.bc.add('Dirichlet', 'E_zz', eps_bar[2, 2])
    pb.bc.add('Dirichlet', 'E_xy', eps_bar[0, 1] + eps_bar[1, 0])
    pb.bc.add('Dirichlet', 'E_xz', eps_bar[0, 2] + eps_bar[2, 0])
    pb.bc.add('Dirichlet', 'E_yz', eps_bar[1, 2] + eps_bar[2, 1])

    # --- Stop rigid body motion ---
    nodes = mesh.nodes
    center = nodes.mean(axis=0)
    center_node = [int(np.argmin(np.linalg.norm(nodes - center, axis=1)))]
    pb.bc.add('Dirichlet', center_node, 'DispX', 0)
    pb.bc.add('Dirichlet', center_node, 'DispY', 0)
    pb.bc.add('Dirichlet', center_node, 'DispZ', 0)

    return bc_periodic

#=============================================================== TESTS ===============================================================

def setup_spring_problem(filename):
    """
    Read the mesh and spring stiffnesses for `filename` and build the fedoo
    assembly once. Each elementary test (traction_test / shear_test) then
    only has to create a fresh Problem and boundary conditions on top of
    it, instead of re-reading files and re-assembling for every test.

    Parameters
    ----------
    filename : str
        Base name of the mesh file (without extension); also used to
        locate the "{filename}_k.txt" spring-stiffness file.

    Returns
    -------
    mesh : fedoo mesh (spring elements)
    assembly_name : str
        Name under which the assembly was registered; pass it to
        fd.problem.Linear(assembly_name) for each test.
    """
    fd.ModelingSpace("3D")
    mesh = fd.Mesh.read("./{}.msh".format(filename), name="Domain")
    mesh.elm_type = 'spring'

    with open(filename + "_k.txt", "r") as f:
        k_values = f.read().splitlines()
    k_values = np.array([float(k) for k in k_values])

    wf = fd.weakform.SpringEquilibrium(k_values, nlgeom=False)
    assembly_name = "GlobalAssembly"
    fd.Assembly.create(wf, mesh, name=assembly_name)

    return mesh, assembly_name


def traction_test(filename, mesh, assembly_name, traction=0.1, direction="z", plot=False):

    pb = fd.problem.Linear(assembly_name)

    nodes = mesh.nodes

    # --- Characteristic length and imposed strain ---
    if direction == "x":
        col = 0
    elif direction == "y":
        col = 1
    else:
        col = 2
    L = nodes[:, col].max() - nodes[:, col].min()
    eps0 = traction / L   # target macroscopic strain

    # --- Macroscopic strain tensor ---
    eps_bar = np.zeros((3, 3))
    eps_bar[col, col] = eps0

    # --- Apply boundary conditions (KUBC, PBC or full constraint, depending on BC) ---
    if BC == "pbc":
        bc_periodic = apply_pbc(pb, mesh, eps_bar, basename=filename)
        boundary_nodes = flatten_periodic_boundary_nodes(bc_periodic)
    else:
        boundary_nodes = apply_kubc(pb, mesh, eps_bar, basename=filename)

    # --- Solve ---
    pb.solve()

    # --- Reaction forces and homogenized stress ---
    x_c, y_c, z_c = nodes[:, 0], nodes[:, 1], nodes[:, 2]
    volume = (x_c.max()-x_c.min()) * (y_c.max()-y_c.min()) * (z_c.max()-z_c.min())

    if BC == "pbc":
        sigma_voigt = compute_homogenized_stress_pbc(pb, volume)
    else:
        F_vect = pb.get_ext_forces().reshape(pb.space.nvar, -1)
        sigma_voigt = compute_homogenized_stress(F_vect, boundary_nodes, nodes, volume)

    stiffness = sigma_voigt / eps0

    # --- Linearity error: relative L2 error between the computed displacement
    #     and the expected linear displacement eps_bar . x ---
    U_full = pb.get_disp().reshape(3, -1)
    u_expected = expected_displacement(mesh, eps_bar)
    linearity_error = compute_linearity_error(U_full.T, u_expected)

    if plot:
        if direction == "x":
            plot_displacement_vs_coord(pb, mesh, i=0, j=0, title=f"u_x vs x (traction {traction})", boundary_nodes=boundary_nodes, u_expected=u_expected)
        elif direction == "y":
            plot_displacement_vs_coord(pb, mesh, i=1, j=1, title=f"u_y vs y (traction {traction})", boundary_nodes=boundary_nodes, u_expected=u_expected)
        else:
            plot_displacement_vs_coord(pb, mesh, i=2, j=2, title=f"u_z vs z (traction {traction})", boundary_nodes=boundary_nodes, u_expected=u_expected)

    return stiffness, linearity_error


def shear_test(filename, mesh, assembly_name, shear=0.1, direction="yz", plot=False):

    pb = fd.problem.Linear(assembly_name)

    nodes = mesh.nodes

    # --- Characteristic length and imposed engineering strain ---
    if direction == "yz":
        normal_col = 2   # z
        slide_col  = 1   # y
    elif direction == "xz":
        normal_col = 2   # z
        slide_col  = 0   # x
    elif direction == "xy":
        normal_col = 1   # y
        slide_col  = 0   # x
    elif direction == "zy":
        normal_col = 1   # y
        slide_col  = 2   # z
    elif direction == "zx":
        normal_col = 0   # x
        slide_col  = 2   # z
    elif direction == "yx":
        normal_col = 0   # x
        slide_col  = 1   # y
    else:
        raise ValueError("direction must be 'yz', 'xz' or 'xy'")

    L = nodes[:, normal_col].max() - nodes[:, normal_col].min()
    gamma = shear / L         
    eps_ij = gamma       

    # --- Macroscopic strain tensor ---
    eps_bar = np.zeros((3, 3))
    eps_bar[slide_col, normal_col] = eps_ij
 
    # --- Boundary conditions ---
    if BC == "pbc":
        bc_periodic = apply_pbc(pb, mesh, eps_bar, basename=filename)
        boundary_nodes = flatten_periodic_boundary_nodes(bc_periodic)

    else:
        boundary_nodes = apply_kubc(pb, mesh, eps_bar, basename=filename)

    # --- Solve ---
    pb.solve()

    # --- Reaction forces and homogenized stress ---
    x_c, y_c, z_c = nodes[:, 0], nodes[:, 1], nodes[:, 2]
    volume = (x_c.max()-x_c.min()) * (y_c.max()-y_c.min()) * (z_c.max()-z_c.min())

    if BC == "pbc":
        sigma_voigt = compute_homogenized_stress_pbc(pb, volume)
    else:
        F_vect = pb.get_ext_forces().reshape(pb.space.nvar, -1)
        sigma_voigt = compute_homogenized_stress(F_vect, boundary_nodes, nodes, volume)

    stiffness = sigma_voigt / gamma
    # --- Linearity error: relative L2 error between the computed displacement
    #     and the expected linear displacement eps_bar . x ---
    U_full = pb.get_disp().reshape(3, -1)
    u_expected = expected_displacement(mesh, eps_bar)
    linearity_error = compute_linearity_error(U_full.T, u_expected)

    if plot:
        if direction == "yz":
            plot_displacement_vs_coord(pb, mesh, i=1, j=2, title=f"u_y vs z (shear {shear})", boundary_nodes=boundary_nodes, u_expected=u_expected)
        elif direction == "xz":
            plot_displacement_vs_coord(pb, mesh, i=0, j=2, title=f"u_x vs z (shear {shear})", boundary_nodes=boundary_nodes, u_expected=u_expected)
        elif direction == "xy":
            plot_displacement_vs_coord(pb, mesh, i=0, j=1, title=f"u_x vs y (shear {shear})", boundary_nodes=boundary_nodes, u_expected=u_expected)

    return stiffness, linearity_error

def get_stiffness_matrix(filename, traction=0.1, shear=0.1, plot=False):
    """
    Compute the homogenized stiffness matrix (Voigt notation) from a given mesh file.

    Parameters
    ----------
    filename : str
        Base name of the mesh file (without extension).
    traction : float
        Traction value for normal tests.
    shear : float
        Shear value for shear tests.
    plot : bool
        Whether to plot the results.

    Returns
    -------
    stiffness_matrix : np.ndarray
        The homogenized stiffness matrix (Voigt notation).
    linearity_errors : dict
        Relative L2 linearity error (see compute_linearity_error) for
        each of the 9 elementary tests, keyed by test direction.
    """
    mesh, assembly_name = setup_spring_problem(filename)

    c1, e1 = traction_test(filename, mesh, assembly_name, traction=traction, direction="x", plot=True)
    c2, e2 = traction_test(filename, mesh, assembly_name, traction=traction, direction="y", plot=True)
    c3, e3 = traction_test(filename, mesh, assembly_name, traction=traction, direction="z", plot=True)

    c4, e4 = shear_test(filename, mesh, assembly_name, shear=shear, direction="yz", plot=True)
    c5, e5 = shear_test(filename, mesh, assembly_name, shear=shear, direction="xz", plot=True)
    c6, e6 = shear_test(filename, mesh, assembly_name, shear=shear, direction="xy", plot=True)

    c4_bis, e4_bis = shear_test(filename, mesh, assembly_name, shear=shear, direction="zy", plot=False)
    c5_bis, e5_bis = shear_test(filename, mesh, assembly_name, shear=shear, direction="zx", plot=False)
    c6_bis, e6_bis = shear_test(filename, mesh, assembly_name, shear=shear, direction="yx", plot=False)

    # Check symmetry of shear tests
    if not (np.allclose(c4, c4_bis) and np.allclose(c5, c5_bis) and np.allclose(c6, c6_bis)):
        print("Warning: Shear tests are not symmetric. Minor symetry is lost.")
        print(f"c4: {c4}, c4_bis: {c4_bis}")
        print(f"c5: {c5}, c5_bis: {c5_bis}")
        print(f"c6: {c6}, c6_bis: {c6_bis}")

    stiffness_matrix = np.array([c1, c2, c3, c4, c5, c6]).T

    linearity_errors = {
        "x": e1, "y": e2, "z": e3,
        "yz": e4, "xz": e5, "xy": e6,
        "zy": e4_bis, "zx": e5_bis, "yx": e6_bis,
    }
    print("Linearity errors:")
    print(f"  mean: {np.mean(list(linearity_errors.values())):.4%}")
    print(f"  max: {max(linearity_errors.values()):.4%}")

    return stiffness_matrix, linearity_errors




#=============================================================== MAIN PLOT HELPERS ===============================================================

def _draw_mean_circle(ax, phi, mean_value, unit=""):
    """
    Draw a grey dashed circle at the given mean value on a polar axis, and
    add it to the legend.
    """
    label = f'mean = {mean_value:.4g}' + (f' {unit}' if unit else '')
    ax.plot(phi, np.full_like(phi, mean_value), '--', color='gray', linewidth=2, label=label)
    ax.legend(loc='upper left', bbox_to_anchor=(0.9, 1.15), fontsize=15)


def _sample_over_theta(phi, theta, prop_fn):
    """
    Sample a vectorized (theta, phi) -> value property along a few
    colatitudes theta. Returns the list of sampled curves and their max.
    """
    curves = [prop_fn(tt, phi) for tt in theta]
    return curves, np.concatenate(curves).max()


def _sample_over_theta_chi(phi, theta, chi, prop_fn):
    """
    Sample a (theta, phi, chi) -> value property (called as prop_fn([theta,
    phi, chi])) over a grid of (theta, chi) pairs. Returns the list of
    sampled curves, their max and their mean.
    """
    curves = []
    for tt in theta:
        for c in chi:
            f = np.vectorize(lambda x, tt=tt, c=c: prop_fn([tt, x, c]))
            curves.append(f(phi))
    all_values = np.concatenate(curves)
    return curves, all_values.max(), all_values.mean()


def draw_young_modulus_polar(ax, phi, curves, young_mean, rmax, standalone=False):
    """Polar plot of the directional Young modulus E(theta, phi). Units: Pa."""
    for r in curves:
        ax.plot(phi, r, linewidth=2)
    _draw_mean_circle(ax, phi, young_mean, unit="Pa")
    ax.set_rmin(0.)
    ax.set_rmax(rmax * 1.2)
    ax.grid(False)
    ax.yaxis.grid(True, color='gray', linestyle='--', linewidth=0.7, alpha=0.7)
    ax.text(0, 0, r'$E$', fontsize=20, weight='bold', ha='center', va='center')
    if not standalone:
        ax.set_title('Young modulus')


def draw_poisson_polar(ax, phi, curves, nu_mean, rmax, standalone=False):
    """Polar plot of the directional Poisson ratio nu(theta, phi, chi). Dimensionless."""
    for r in curves:
        ax.plot(phi, r, linewidth=2)
    _draw_mean_circle(ax, phi, nu_mean)
    ax.grid(axis='x')
    ax.set_rmax(rmax * 1.2)
    ax.text(0, 0, r'$\nu$', fontsize=20, weight='bold', ha='center', va='center')
    if not standalone:
        ax.set_title('Poisson ratio')


def draw_linear_compressibility_polar(ax, phi, curves, lc_mean, rmax, standalone=False):
    """Polar plot of the directional linear compressibility beta(theta, phi). Units: Pa^-1."""
    for r in curves:
        ax.plot(phi, r, linewidth=2)
    _draw_mean_circle(ax, phi, lc_mean, unit=r"Pa$^{-1}$")
    ax.grid(axis='x')
    ax.set_rmax(rmax * 1.2)
    ax.text(0, 0, r'$\beta$', fontsize=20, weight='bold', ha='center', va='center')
    if not standalone:
        ax.set_title('Linear compressibility')


def draw_shear_modulus_polar(ax, phi, curves, G_mean, rmax, standalone=False):
    """Polar plot of the directional shear modulus G(theta, phi, chi). Units: Pa."""
    for r in curves:
        ax.plot(phi, r, linewidth=2)
    _draw_mean_circle(ax, phi, G_mean, unit="Pa")
    ax.grid(axis='x')
    ax.set_rmax(rmax * 1.2)
    ax.text(0, 0, r'$G$', fontsize=20, weight='bold', ha='center', va='center')
    if not standalone:
        ax.set_title('Shear modulus')


def compute_shear_error(material):
    """
    Relative deviation between the computed shear modulus G and the
    isotropic relation G = E / (2*(1+nu)), in percent, sampled on a
    (theta, phi) grid covering the full sphere.
    """
    n = 200
    theta = np.linspace(0, 2 * np.pi, n)
    phi = np.linspace(0, 2 * np.pi, n)
    theta_grid, phi_grid = np.meshgrid(theta, phi)

    E = np.vectorize(material.Young_2)(theta_grid, phi_grid)
    nu = np.vectorize(lambda a, b, c: material.Poisson([a, b, c]))(theta_grid, phi_grid, 0.)
    G = np.vectorize(lambda a, b, c: material.shear([a, b, c]))(theta_grid, phi_grid, 0.)

    G_error = (G - E / (2 * (1 + nu))) / G * 100
    return theta, phi, G_error


def draw_shear_error(fig, position, theta, phi, G_error, spherical, standalone=False):
    """
    Draw the shear-error map (see compute_shear_error) either as a colored
    sphere (spherical=True) or as a flat heat map. Values are in percent.
    """
    if spherical:
        ax = fig.add_subplot(*position, projection='3d')
        err = np.abs(G_error)
        err_min, err_max = err.min(), err.max()
        norm_err = (err - err_min) / (err_max - err_min + 1e-12)
        radius = 0.5 + 0.5 * norm_err

        x = radius * np.sin(theta) * np.cos(phi)
        y = radius * np.sin(theta) * np.sin(phi)
        z = radius * np.cos(theta)

        cmap = plt.cm.viridis
        ax.plot_surface(x, y, z, facecolors=cmap(norm_err), rstride=1, cstride=1,
                         linewidth=0, antialiased=True, shade=False)
        cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=err_min, vmax=err_max)),
                             ax=ax, shrink=0.8, pad=0.08)
        ax.set_box_aspect((1, 1, 1))
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
    else:
        ax = fig.add_subplot(*position)
        cs = ax.imshow(np.abs(G_error))
        cbar = fig.colorbar(cs)
        ax.axis('equal')

    cbar.set_label('|G - E/(2(1+nu))| / G [%]', rotation=270, labelpad=18)
    if not standalone:
        ax.set_title(r'$||G - \dfrac{E}{2(1+\nu)}||$ [%]')
    return ax


def draw_mesh(fig, position, xyz, bonds, standalone=False):
    """Draw the RVE mesh as a 3D wireframe of springs/bonds. Units: m."""
    ax = fig.add_subplot(*position, projection='3d')
    for b in bonds:
        ax.plot(xyz[b, 0], xyz[b, 1], xyz[b, 2], 'o-k')
    set_axes_equal_3d(ax, xyz)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('z (m)')
    if not standalone:
        ax.set_title('RVE mesh')
    return ax


#=============================================================== MAIN ===============================================================

if __name__ == '__main__':

    filename = sys.argv[1]

    YoungModulus = 210e9
    PoissonRatio = 0.25

    L_eff, linearity_errors = get_stiffness_matrix(filename, traction=0.001, shear=0.001, plot=True)
    L_eff[np.isclose(L_eff,1e-10)]=0.

    # Compute the maximal relative difference between L_eff and its transpose (symmetry check)
    max_percentage_diff = 0
    for i in range(6):
        for j in range(6):
            if L_eff[i,j] != 0:
                percentage_diff = abs(L_eff[i,j] - L_eff[j,i]) / abs(L_eff[i,j])
                if percentage_diff > max_percentage_diff:
                    max_percentage_diff = percentage_diff
    print(f"Max percentage difference between L_eff and its transpose: {max_percentage_diff:.3e}")

    # Check symmetry
    SymmetryWarrning = False
    if max_percentage_diff > 1e-6:
        print("The homogenized stiffness matrix is not symmetric.")
        SymmetryWarrning = True
    else:
        print("The homogenized stiffness matrix is symmetric.",)

    L_eff += L_eff.T
    L_eff*=0.5

    print("The homogenized stiffness matrix is : \n", L_eff)

    # --- Isotropy error: Mandel-norm (rotation-invariant) distance between
    #     L_eff and the reference isotropic tensor defined by
    #     (YoungModulus, PoissonRatio) ---
    isotropy_error = compute_stiffness_isotropy_error(L_eff, YoungModulus, PoissonRatio)
    print(f"Isotropy error vs reference (E={YoungModulus:.3g}, nu={PoissonRatio}): {isotropy_error:.4%}")

    # --- Linearity errors (mean and max) over all tests ---
    mean_linearity_error = np.mean(list(linearity_errors.values()))
    max_linearity_error = max(linearity_errors.values())
    print(f"Mean linearity error over all tests: {mean_linearity_error:.4%}")
    print(f"Max linearity error over all tests: {max_linearity_error:.4%}")

    # --- Write both errors to (filename)_errors.txt ---
    with open(f"{filename}_errors.txt", "w") as f:
        f.write(f"Isotropy error vs reference (E={YoungModulus:.6g}, nu={PoissonRatio}): {isotropy_error:.6e}\n")
        f.write("Linearity errors per test:\n")
        for direction, err in linearity_errors.items():
            f.write(f"  {direction}: {err:.6e}\n")
        f.write(f"  mean: {mean_linearity_error:.6e}\n")
        f.write(f"  max: {max_linearity_error:.6e}\n")
    print(f"Errors written to {filename}_errors.txt")

    def add_error_annotations(fig):
        if not SymmetryWarrning:
            return
        fig.text(
            0.5,
            0.01,
            f"The homogenized stiffness matrix is not symmetric "
            f"(max difference {max_percentage_diff:.2%}).",
            color="red",
            ha="center",
            va="bottom",
            fontsize=16,
            bbox=dict(facecolor="white", edgecolor="red", alpha=0.85),
        )

    material = elastic.Elastic(L_eff)

    phi   = np.linspace(0, 2 * np.pi, 200, endpoint=False)
    theta = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi]
    chi   = theta

    # --- Output folder: one global overview pdf + one pdf per panel ---
    bc_suffix = {"pbc": "periodic", "full constraint": "full-constraint"}.get(BC, "non-periodic")
    plot_dir = f"{filename}_plots"
    os.makedirs(plot_dir, exist_ok=True)
    base = os.path.basename(filename)

    # Interactive 3D view of the Young modulus surface.
    phi_3d = np.linspace(0, 2 * np.pi, 180)
    theta_3d = np.linspace(0, np.pi, 90)
    phi_grid, theta_grid = np.meshgrid(phi_3d, theta_3d)
    young_3d = np.vectorize(material.Young_2)(theta_grid, phi_grid)

    # Mean Young modulus over all directions of the unit sphere:
    #   <E> = 1/(4*pi) * int E(theta, phi) sin(theta) dtheta dphi
    young_mean = np.trapezoid(np.trapezoid(young_3d * np.sin(theta_grid), phi_3d, axis=1),
                          theta_3d) / (4 * np.pi)
    print(f"Mean Young modulus over all directions: {young_mean:.6e} ({young_mean / YoungModulus:.4%} of E={YoungModulus:.3g})")

    radius = young_3d / YoungModulus  # Normalize radius for visualization
    x = radius * np.sin(theta_grid) * np.cos(phi_grid)
    y = radius * np.sin(theta_grid) * np.sin(phi_grid)
    z = radius * np.cos(theta_grid)

    fig3d = plt.figure(figsize=(10, 8))
    ax3d = fig3d.add_subplot(111, projection='3d')
    surface = ax3d.plot_surface(x, y, z, facecolors=plt.cm.viridis(radius / radius.max()),
                                rstride=1, cstride=1, linewidth=0, antialiased=True, shade=False)
    fig3d.colorbar(plt.cm.ScalarMappable(cmap='viridis'), ax=ax3d, shrink=0.7, pad=0.1, label='Normalized Young modulus')
    ax3d.set_title('3D representation of the Young modulus')
    ax3d.set_box_aspect((1, 1, 1))
    ax3d.set_xlabel('x')
    ax3d.set_ylabel('y')
    ax3d.set_zlabel('z')
    #plt.show()


    # Interactive 3D view of the linear compressibility surface.
    LC_3d = np.vectorize(material.LC_2)(theta_grid, phi_grid)

    # Mean linear compressibility over all directions of the unit sphere
    # (same integral as young_mean above, applied to LC_3d).
    LC_mean = np.trapezoid(np.trapezoid(LC_3d * np.sin(theta_grid), phi_3d, axis=1),
                            theta_3d) / (4 * np.pi)

    radius = LC_3d / LC_3d.max()
    x = radius * np.sin(theta_grid) * np.cos(phi_grid)
    y = radius * np.sin(theta_grid) * np.sin(phi_grid)
    z = radius * np.cos(theta_grid)

    fig3d = plt.figure(figsize=(10, 8))
    ax3d = fig3d.add_subplot(111, projection='3d')
    surface = ax3d.plot_surface(x, y, z, facecolors=plt.cm.viridis(radius / radius.max()),
                                rstride=1, cstride=1, linewidth=0, antialiased=True, shade=False)
    fig3d.colorbar(plt.cm.ScalarMappable(cmap='viridis'), ax=ax3d, shrink=0.7, pad=0.1, label='Normalized linear compressibility')
    ax3d.set_title('3D representation of the linear compressibility')
    ax3d.set_box_aspect((1, 1, 1))
    ax3d.set_xlabel('x')
    ax3d.set_ylabel('y')
    ax3d.set_zlabel('z')
    #plt.show()

    # --- Sample the directional properties shown in the polar plots ---
    young_curves, young_rmax    = _sample_over_theta(phi, theta, np.vectorize(material.Young_2))
    lc_curves, lc_rmax          = _sample_over_theta(phi, theta, np.vectorize(material.LC_2))
    nu_curves, nu_rmax, nu_mean = _sample_over_theta_chi(phi, theta, chi, material.Poisson)
    G_curves, G_rmax, G_mean    = _sample_over_theta_chi(phi, theta, chi, material.shear)

    shear_err_theta, shear_err_phi, G_error = compute_shear_error(material)

    mesh = fd.Mesh.get_all()['Domain']
    xyz  = mesh.nodes
    bonds = mesh.elements

    # --- Combined overview figure (6 panels) ---
    fig, ax = plt.subplots(2, 3, subplot_kw={'projection': 'polar'}, figsize=(30, 15))

    draw_young_modulus_polar(ax[0, 0], phi, young_curves, young_mean, young_rmax)
    draw_poisson_polar(ax[1, 0], phi, nu_curves, nu_mean, nu_rmax)
    draw_linear_compressibility_polar(ax[0, 1], phi, lc_curves, LC_mean, lc_rmax)
    draw_shear_modulus_polar(ax[1, 1], phi, G_curves, G_mean, G_rmax)

    ax[0, 2].remove()
    draw_shear_error(fig, (2, 3, 3), shear_err_theta, shear_err_phi, G_error, spherical=SPHERICAL_PLOT)

    ax[1, 2].remove()
    draw_mesh(fig, (2, 3, 6), xyz, bonds)

    add_error_annotations(fig)
    global_pdf = os.path.join(plot_dir, f"{base}_{bc_suffix}.pdf")
    plt.savefig(global_pdf)
    plt.close(fig)
    print(f"Saved combined overview to {global_pdf}")

    # --- Individual figures: one per panel, no title, no symmetry warning ---
    polar_plots = [
        ("young_modulus.pdf", lambda ax: draw_young_modulus_polar(ax, phi, young_curves, young_mean, young_rmax, standalone=True)),
        ("poisson_ratio.pdf", lambda ax: draw_poisson_polar(ax, phi, nu_curves, nu_mean, nu_rmax, standalone=True)),
        ("linear_compressibility.pdf", lambda ax: draw_linear_compressibility_polar(ax, phi, lc_curves, LC_mean, lc_rmax, standalone=True)),
        ("shear_modulus.pdf", lambda ax: draw_shear_modulus_polar(ax, phi, G_curves, G_mean, G_rmax, standalone=True)),
    ]
    for fname, draw in polar_plots:
        fig_i, ax_i = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 7))
        draw(ax_i)
        plt.savefig(os.path.join(plot_dir, fname), bbox_inches='tight')
        plt.close(fig_i)

    fig_i = plt.figure(figsize=(8, 7))
    draw_shear_error(fig_i, (1, 1, 1), shear_err_theta, shear_err_phi, G_error, spherical=SPHERICAL_PLOT, standalone=True)
    plt.savefig(os.path.join(plot_dir, "shear_error.pdf"), bbox_inches='tight')
    plt.close(fig_i)

    fig_i = plt.figure(figsize=(8, 7))
    draw_mesh(fig_i, (1, 1, 1), xyz, bonds, standalone=True)
    plt.savefig(os.path.join(plot_dir, "mesh.pdf"), bbox_inches='tight')
    plt.close(fig_i)

    print(f"Saved 6 individual plots and the combined overview to {plot_dir}/")