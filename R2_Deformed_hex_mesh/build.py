import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mesh_utils import write_msh2, plot_mesh

def _multiple_hex_from_links(N, h, perturbation, ii, jj):
    """
    N : number of cells in each direction (total cells = N^3)
    h : size of one cell (cube)
    perturbation : fraction of h to perturb the interior vertices by
    ii, jj : """

    #Step 1 : generate an h by h cube
    reference_cube = np.array([[0, 0, 0],
                               [h, 0, 0],
                               [h, h, 0],
                               [0, h, 0],
                               [0, 0, h],
                               [h, 0, h],
                               [h, h, h],
                               [0, h, h]])

    #step 2 : duplicate the cube N times in each direction (vectorised)
    i, j, k = np.meshgrid(np.arange(N), np.arange(N), np.arange(N), indexing="ij")

    cell_origins = np.stack([i.ravel(), j.ravel(), k.ravel()], axis=1) * h
    n_cells = cell_origins.shape[0]
    n_bonds_per_cell = ii.shape[0]

    # (n_cells, 8, 3) -> (n_cells*8, 3), 8 vertices per cell in cell order
    vertices = (reference_cube[None, :, :] + cell_origins[:, None, :]).reshape(-1, 3)

    cell_index = np.repeat(np.arange(n_cells), n_bonds_per_cell)
    bond_i = np.tile(ii, n_cells) + cell_index * 8
    bond_j = np.tile(jj, n_cells) + cell_index * 8
    group_names = np.char.add("cell_", np.arange(n_cells).astype(str))
    bond_groups = np.repeat(group_names, n_bonds_per_cell)

    #step 3 : fuse vertices at the same position and identify boundary vertices
    # (np.unique on rounded coordinates replaces the O(V^2) pairwise np.allclose scan)
    decimals = 8
    single_vertices, vertex_map = np.unique(np.round(vertices, decimals), axis=0, return_inverse=True)
    vertex_map = vertex_map.ravel()

    x, y, z = single_vertices[:, 0], single_vertices[:, 1], single_vertices[:, 2]
    boundary_mask = (
        np.isclose(x, 0) | np.isclose(x, N * h) |
        np.isclose(y, 0) | np.isclose(y, N * h) |
        np.isclose(z, 0) | np.isclose(z, N * h)
    )
    bondary_vertices = np.flatnonzero(boundary_mask).tolist()

    #step 4 : remap the bonds to the fused vertex indices
    new_bond_i = vertex_map[bond_i]
    new_bond_j = vertex_map[bond_j]

    #step 5 : perturb the vertices that are not on the bondary of the cube
    interior_mask = ~boundary_mask
    noise = np.random.uniform(-perturbation*h/2, perturbation*h/2, single_vertices.shape)
    single_vertices[interior_mask] += noise[interior_mask]

    #change format of the bonds to match other implementations
    final_bonds = list(zip(new_bond_i.tolist(), new_bond_j.tolist(), bond_groups.tolist()))

    print("number of vertices", len(single_vertices))
    print("number of bonds", len(final_bonds))
    print("number of boundary vertices", len(bondary_vertices))

    return single_vertices, final_bonds, bondary_vertices

def multiple_hex_complete(N, h, perturbation=0.2):
    """
    N =  number of cells per direction
    h =  cell size
    perturbation =  perturbation of the vertices of the hexahedron

    Bonds each cell with its full complete graph (all 28 vertex pairs, see
    complete_graph): edge, face-diagonal and volume bonds alike.

    See _multiple_hex_from_links for the returned values.
    """
    #the 28 unordered vertex-index pairs of a single hexahedron
    ii, jj = np.triu_indices(8, k=1)
    return _multiple_hex_from_links(N, h, perturbation, ii, jj)


if __name__ == "__main__":

    if len(sys.argv) != 5:
        print("Usage: python script.py <name> <N> <h> <perturbation>")
        sys.exit(1)

    name = sys.argv[1]
    N = int(sys.argv[2])
    H = float(sys.argv[3]) #size of the full mesh
    h = H/N #size of one cell
    perturbation = float(sys.argv[4])



    node, bonds, boundary_vertices = multiple_hex_complete(perturbation=perturbation, N=N, h=h)

     #multiply the z component of the nodes by 0.8
    node[:, 2] *= 0.8
    
    #multiply the y component by 1.2
    node[:, 1] *= 1.2
    
    name_to_physical_id = {}
    elements = []
    for i, j, group in bonds:
        if group not in name_to_physical_id:
            name_to_physical_id[group] = len(name_to_physical_id) + 1
        elements.append((i + 1, j + 1, name_to_physical_id[group]))


    write_msh2(name + ".msh", node, elements, name_to_physical_id)
    plot_mesh(node, elements)

    boundary_file = "{}_boundary_vertices.txt".format(name)
    with open(boundary_file, "w") as f:
        for v in boundary_vertices:
            f.write("{}\n".format(v))
