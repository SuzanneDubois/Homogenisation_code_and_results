import os
import sys
import numpy as np
from coefficients_computation import solve_gn, GnSolution, find_positive_solution

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mesh_utils import parse_msh2, volume_convex_hull


if __name__ == "__main__":   

    #take the filepath of the msh file as input
    filename = sys.argv[1]
    filepath = filename + ".msh"

    nodes, bonds, bond_physical_tags, physical_names = parse_msh2(filepath)

    #solve for each physical group separately, then concatenate the solutions
    all_g = []
    for phys_id, phys_name in physical_names.items():
        group_bonds = [bond for bond, tag in zip(bonds, bond_physical_tags) if tag == phys_id]
        if not group_bonds:
            continue
        #nodes used by the bonds of the current group
        group_nodes = {nid: nodes[nid] for bond in group_bonds for nid in bond}
        volume = volume_convex_hull(group_nodes)
        print(f"Physical group {phys_id} ({phys_name}): {len(group_bonds)} bonds, "
              f"{len(group_nodes)} nodes, convex hull volume {volume:.3f}")
        vectors = [nodes[i] - nodes[j] for i, j in group_bonds]

        sol = solve_gn(vectors, V=volume, E=210e9)
        g_pos = find_positive_solution(sol)
        all_g.append(g_pos)
    g = np.concatenate(all_g)

    with open(filename + "_k.txt", "w") as f:
        for k in g:
            f.write(f"{k}\n")
