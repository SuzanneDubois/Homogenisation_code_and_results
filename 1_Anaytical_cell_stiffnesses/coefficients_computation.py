import sys
import os

from matplotlib.pylab import rand
import numpy as np
from scipy.optimize import linprog

# mesh_utils.py lives at the project root, one directory up from here.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mesh_utils import parse_msh2, volume_bonding_box
from collections import namedtuple


def fabric_tensor(vector):
    """
    Given a bond vector (or a list of bond vectors) x_n computes the sum of their 4th order fabric tensor contributions, i.e.
    sum_n x_a x_b x_c x_d / (2 |x_n|^2)

    Returns a (3,3,3,3) array.
    """
    vectors = np.asarray(vector, dtype=float)
    T = np.zeros((3, 3, 3, 3))
    for x in vectors:
        outer4 = np.einsum('a,b,c,d->abcd', x, x, x, x)
        T += outer4
    return T

def isotropic_target(V=1, E=1.0):
    """ V (d_ab d_cd + d_ac d_bd + d_ad d_bc)  -> isotropic stiffness tensor of given YM"""
    d = np.eye(3)
    T = (np.einsum('ab,cd->abcd', d, d)
       + np.einsum('ac,bd->abcd', d, d)
       + np.einsum('ad,bc->abcd', d, d))
    return V * 2/5 * E * T



GnSolution = namedtuple("GnSolution", ["g", "residual_norm", "rank", "n_bonds", "null_space"])

def solve_gn(vectors, V=1, E=1e6, verbose=True, tol=1e-8):
    """
    Solve the GN system for the given vectors and target.

    Parameters
    ----------
    vectors : list of arrays, each of shape (3,)
        The bond vectors.
    V      : volume of the unit cell.
    E      : Young's modulus.
    tol    : relative singular-value threshold used to decide numerical rank.

    Returns
    -------
    GnSolution namedtuple with fields:
        g            : minimum-norm particular solution (one valid choice of g_n's)
        residual_norm: ||M @ g - target|| (should be ~0 if isotropy is achievable)
        rank         : numerical rank of M
        n_bonds      : number of unknowns (= len(vectors))
        null_space   : (n_bonds, n_free) array whose columns span the space of
                        directions g can be perturbed in while leaving M @ g
                        unchanged. n_free = n_shells - rank.
                        The full solution set (when residual_norm ~ 0) is
                            g = g_particular + null_space @ c   for any c in R^n_free
    """
    fabric_tensors = []
    for x in vectors:
        outer4 = np.einsum('a,b,c,d->abcd', x, x, x, x) / (np.dot(x, x))  # shape (3,3,3,3)
        #flatten the 4th order tensor to a 1D array of length 81
        outer4 = outer4.flatten()
        fabric_tensors.append(outer4)

    #stack the flattened tensors to form a matrix M of shape (81, n_bonds)
    fabric_tensors = np.stack(fabric_tensors, axis=1)   # (81, n_bonds)
    M = fabric_tensors#np.stack([T.flatten() for T in fabric_tensors], axis=1)   # (81, n_bonds)
    target = isotropic_target(V, E).flatten()                       # (81,)

    n_bonds = M.shape[1]

    # Full SVD gives us access to the null space of M (not just the solution)
    U, S, Vt = np.linalg.svd(M, full_matrices=True)
    rank = int(np.sum(S > tol * S.max())) if S.size else 0

    # Minimum-norm particular solution (equivalent to lstsq's result)
    g, _, _, _ = np.linalg.lstsq(M, target, rcond=None)

    reconstructed = M @ g
    residual_norm = np.linalg.norm(reconstructed - target)

    # Null space basis = right-singular vectors beyond the rank
    null_space = Vt[rank:].T   # shape (n_shells, n_free)
    n_free = null_space.shape[1]

    if verbose:
        print(f"Matrix rank             : {rank}")
        print(f"Residual norm (should be ~0 if isotropy is achievable): {residual_norm:.3e}")

        if residual_norm < 1e-8 and n_free > 0:
            print(f"\nSystem is underdetermined: solution space is {n_free}-dimensional.")

        elif residual_norm >= 1e-8 and n_free > 0:
            print(f"\nNote: M has a {n_free}-dimensional null space, but the residual is nonzero,")
            print("so exact isotropy isn't achievable here — the null space doesn't add exact solutions,")
            print("only directions along which the (nonzero) residual stays constant.")
        else:
            print("\nSolution is unique (M has full column rank).")

    return GnSolution(g=g, residual_norm=residual_norm, rank=rank,
                       n_bonds=n_bonds, null_space=null_space)


def read_msh_nodes(filepath):
    """
    Reads the $Nodes section of a MSH2 file.
    Returns a (n_nodes, 3) array of coordinates, 0-indexed. and "a" the lattice spacing (maximum distance in a direction)
    """
    nodes = {}
    with open(filepath, 'r') as f:
        in_nodes = False
        for line in f:
            line = line.strip()
            if line == '$Nodes':
                in_nodes = True
                next(f)  # skip the count line
                continue
            if line == '$EndNodes':
                break
            if in_nodes:
                parts = line.split()
                node_id = int(parts[0]) - 1  # MSH2 is 1-indexed → 0-indexed
                coords = [float(parts[1]), float(parts[2]), float(parts[3])]
                nodes[node_id] = coords

    n = max(nodes.keys()) + 1
    coords_array = np.zeros((n, 3))
    for idx, coords in nodes.items():
        coords_array[idx] = coords

    #finding a 
    min_coord = min (min (coords_array[:,0]), min (coords_array[:,1]), min (coords_array[:,2]))
    max_coord = max (max (coords_array[:,0]), max (coords_array[:,1]), max (coords_array[:,2]))
    a = max_coord - min_coord   
    #a = 1
    return coords_array, a


def extract_springs_from_msh(filepath, node_coords):
    """
    filepath    : path to .msh file (MSH2 format)
    node_coords : (n_nodes, 3) array of node coordinates (0-indexed)
    
    Returns: list of (1, 3) arrays -- one shell per bond (2-node element)
             found in the file, ready to pass as shells to solve_gn.
    """
    shells = []

    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Find the $Elements block
    start = lines.index('$Elements\n') + 1
    n_elements = int(lines[start])
    elem_lines = lines[start + 1 : start + 1 + n_elements]

    for line in elem_lines:
        parts = line.split()
        elem_type = int(parts[1])

        # MSH2 element type 1 = 2-node line element
        if elem_type != 1:
            continue

        n_tags = int(parts[2])
        node_start = 3 + n_tags
        # node ids are 1-indexed in .msh -> convert to 0-indexed
        i = int(parts[node_start]) - 1
        j = int(parts[node_start + 1]) - 1

        vec = node_coords[j] - node_coords[i]
        shells.append(np.array([vec]))  # shape (1, 3)

    return shells

def find_positive_solution(sol, tol=1e-8):
    """
    Find c such that g + N @ c > 0 using linear programming.
    
    Reformulation: maximize s subject to  g + N @ c >= s * 1
                   i.e.  N @ c - s * 1 >= -g
    LP minimizes, so minimize -s.
    Variables: x = [c (n_free,), s (scalar)]  →  shape (n_free + 1,)
    """
    g = sol.g
    g_copy = g.copy()
    N = sol.null_space
    n_eq, n_free = N.shape

    #check if g is already positive
    if np.all(g > tol):
        print("g is already positive, no need to solve LP.")
        return g

    # Objective: minimize -s  (i.e. maximize the minimum component)
    c_obj = np.zeros(n_free + 1)
    c_obj[-1] = -1.0          # coefficient of s

    # Constraint:  g + N @ c >= s  →  -N @ c + s <= g
    # linprog expects  A_ub @ x <= b_ub
    # x = [c; s],  A_ub = [-N | 1],  b_ub = g
    A_ub = np.hstack([-N, np.ones((n_eq, 1))])
    b_ub = g

    result = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=[(None, None)] * (n_free + 1))

    if result.status != 0:
        print(f"LP failed: {result.message}")
        return g_copy  # return original g if LP fails

    s_opt = result.x[-1]
    c_opt = result.x[:-1]
    g_positive = g + N @ c_opt

    if s_opt > tol:
        print(f"Found a positive solution (min component = {s_opt:.4e})")
        return g_positive
    else:
        print(f"No positive solution exists (max achievable min = {s_opt:.4e})")
        return g_copy  # return original g if no positive solution found

if __name__ == "__main__":   

    #take the filepath of the msh file as input
    filename = sys.argv[1]
    filepath = filename + ".msh"

    nodes, bonds, bond_physical_tags, physical_names = parse_msh2(filepath)
    vectors = [nodes[i] - nodes[j] for i, j in bonds]
    volume = volume_bonding_box(nodes)
    print(f"Volume of bounding box: {volume:.3f}")
    sol = solve_gn(vectors, V=volume, E=1e6)

    with open(filename + "_k.txt", "w") as f:
        for k in sol.g:
            f.write(f"{k}\n")



