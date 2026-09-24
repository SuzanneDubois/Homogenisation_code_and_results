import os
import sys
import numpy as np
from coefficients_computation import solve_gn, GnSolution, find_positive_solution

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mesh_utils import parse_msh2, volume_convex_hull
import healpy as hp



def sort_springs_by_orientation(springs, N=2):
    """
    Groups springs (bond vectors) into orientation classes, independently of
    their length, using a HEALPix tessellation of the sphere.

    springs : list of bond vectors, one per spring. 
    N       : HEALPix "nside" resolution parameter (must be a power of two).
              Each of the 12 base HEALPix pixels is subdivided into N*N
              pixels, so larger N gives finer orientation classes.

    Returns
    -------
    mean_directions     : list of (3,) unit vectors -- for each non-empty
                           class, the mean (renormalized) orientation of the
                           springs it contains. This is the representative
                           direction of the class.
    spring_class        : list of int, length len(springs) -- spring_class[i]
                           is the index (into mean_directions) of the class
                           that spring i belongs to.
    class_length2_sums  : list of float -- for each class, the sum of the
                           squared lengths (|x|^2) of its springs.
    """
    if not hp.isnsideok(N):
        raise ValueError(f"N={N} is not a valid HEALPix nside (must be a power of two).")

    n_springs = len(springs)
    directions = np.zeros((n_springs, 3))
    lengths2 = np.zeros(n_springs)

    for i, s in enumerate(springs):
        v = np.asarray(s, dtype=float).reshape(-1, 3)[0]
        lengths2[i] = v @ v

        # orientation only: unit vector, independent of length
        u = v / np.linalg.norm(v)

        # a spring's orientation is a line, not an oriented vector: fold u
        # into one canonical hemisphere so that u and -u land in the same
        # HEALPix pixel (ties broken lexicographically on z, then y, then x)
        if u[2] < 0 or (u[2] == 0 and (u[1] < 0 or (u[1] == 0 and u[0] < 0))):
            u = -u
        directions[i] = u

    # bin each orientation into a HEALPix pixel (its orientation class)
    pix = hp.vec2pix(N, directions[:, 0], directions[:, 1], directions[:, 2])

    mean_directions = []
    class_length2_sums = []
    spring_class = np.empty(n_springs, dtype=int)
    for k, p in enumerate(np.unique(pix)):
        idx = np.nonzero(pix == p)[0]

        mean_dir = directions[idx].mean(axis=0)
        mean_dir /= np.linalg.norm(mean_dir)

        mean_directions.append(mean_dir)
        class_length2_sums.append(float(lengths2[idx].sum()))
        spring_class[idx] = k


    return mean_directions, spring_class.tolist(), class_length2_sums


def compute_associated_volume(nodes, bonds, bond_physical_tags):
    """
    Associates to each bond the volume of the cell it belongs to. A cell is
    the set of bonds sharing the same physical tag, and its volume is the
    convex hull volume of the nodes used by those bonds.

    nodes              : dict { node_id : coordinates (3,) }
    bonds              : list of (node_id_a, node_id_b), one per bond
    bond_physical_tags : (N,) array of physical group ids, one per bond

    Returns
    -------
    bond_volumes : list of float, length len(bonds) -- bond_volumes[i] is
                   the volume of the cell that bond i belongs to.
    """
    bond_physical_tags = np.asarray(bond_physical_tags)

    # volume of each cell, computed once per physical tag
    cell_volumes = {}
    for tag in np.unique(bond_physical_tags):
        idx = np.nonzero(bond_physical_tags == tag)[0]
        # nodes used by the bonds of the current cell
        cell_nodes = {nid: nodes[nid] for b in idx for nid in bonds[b]}
        cell_volumes[tag] = volume_convex_hull(cell_nodes)

    return [cell_volumes[tag] for tag in bond_physical_tags]

def k_cell_length (keq, springs, spring_class, nbr_bonds_per_class):
    kcell = np.zeros(len(springs))
    for index in range(len(springs)):
        kcell[index] = keq[spring_class[index]] / (nbr_bonds_per_class[spring_class[index]] * np.linalg.norm(springs[index])**2)

    return kcell

def k_cell_identical (keq, springs, spring_class, class_length2_sums):
    kcell = np.zeros(len(springs))
    for index in range(len(springs)):
        kcell[index] = keq[spring_class[index]] / (class_length2_sums[spring_class[index]])

    return kcell

def k_cell_volume (keq, springs, spring_class, bond_volumes):
    kcell = np.zeros(len(springs))
    sum_volumes = np.zeros(len(keq))
    for index in range(len(springs)):
        sum_volumes[spring_class[index]] += bond_volumes[index]
    for index in range(len(springs)):
        kcell[index] = keq[spring_class[index]] * bond_volumes[index] / (sum_volumes[spring_class[index]] * np.linalg.norm(springs[index])**2)

    return kcell

if __name__ == "__main__":   

    E = 210e9  # Young's modulus in Pascals

    #take the filepath of the msh file as input
    filename = sys.argv[1]
    filepath = filename + ".msh"

    nodes, bonds, bond_physical_tags, physical_names = parse_msh2(filepath)
    volume = volume_convex_hull(nodes)
    springs = [nodes[i] - nodes[j] for i, j in bonds]
    mean_directions, spring_class, class_length2_sums = sort_springs_by_orientation(springs, N=4)
    print(f"Found {len(mean_directions)} orientation classes.") 
    nbr_bonds_per_class = [spring_class.count(i) for i in range(len(mean_directions))]

    g = 6 * E * volume / (len(mean_directions)) * np.ones(len(mean_directions))  # initial guess for the equivalent stiffnesses

    #kcell = k_cell_identical(g, springs, spring_class, class_length2_sums)
    #kcell = k_cell_length(g, springs, spring_class, nbr_bonds_per_class)
    kcell = k_cell_volume(g, springs,spring_class, compute_associated_volume(nodes, bonds, bond_physical_tags))
    with open(filename + "_k.txt", "w") as f:
        for k in kcell:
            f.write(f"{k}\n")
    print(f"Stiffness coefficients written to {filename}_k.txt")
