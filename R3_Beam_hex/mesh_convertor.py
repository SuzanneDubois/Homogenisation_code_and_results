"""
The goal of this script is to take a mesh of 3d hex elements and change it to a mesh of the associated spring network. 
"""
import sys
import os
import re

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mesh_utils import write_msh2, plot_mesh
HEXAHEDRON_ELM_TYPE = 5

def read_gmsh_mesh(filepath):
    """Read the $Nodes and $Elements sections of an MSH2 ASCII file.

    Returns
    -------
    node_ids : list[int]
        Node ids as found in the file (1-based, in file order).
    node_coords : dict[int, np.ndarray]
        Mapping from node id to its (x, y, z) coordinates.
    hexahedra : list[tuple[int, ...]]
        One tuple of 8 node ids (1-based, gmsh order) per hexahedron element.
    """
    with open(filepath, "r") as f:
        text = f.read()

    nodes_match = re.search(r"\$Nodes\n(\d+)\n(.*?)\$EndNodes", text, re.S)
    if not nodes_match:
        raise ValueError(f"No $Nodes section found in {filepath}")

    node_ids = []
    node_coords = {}
    for line in nodes_match.group(2).strip().splitlines():
        parts = line.split()
        node_id = int(parts[0])
        x, y, z = (float(v) for v in parts[1:4])
        node_ids.append(node_id)
        node_coords[node_id] = np.array([x, y, z])

    elements_match = re.search(r"\$Elements\n(\d+)\n(.*?)\$EndElements", text, re.S)
    if not elements_match:
        raise ValueError(f"No $Elements section found in {filepath}")

    hexahedra = []
    for line in elements_match.group(2).strip().splitlines():
        vals = [int(v) for v in line.split()]
        elm_type = vals[1]
        if elm_type != HEXAHEDRON_ELM_TYPE:
            continue
        num_tags = vals[2]
        node_list = vals[3 + num_tags:]
        if len(node_list) != 8:
            raise ValueError(
                f"Hexahedron element expected 8 nodes, got {len(node_list)}: {line}"
            )
        hexahedra.append(tuple(node_list))

    return node_ids, node_coords, hexahedra

def hexahedra_to_bonds(hexahedra):
    """Build the complete bond graph of every hexahedral cell (28 bonds per
    cell). Bonds shared by neighbouring cells are kept once per cell, i.e.
    not deduplicated.

    Returns
    -------
    bonds : list[tuple[int, int, str]]
        (node_id_a, node_id_b, bond_type), node ids as found in the file.
    """
    bonds = []
    cell_index = 0
    for hexa in hexahedra:
        #cell name is a physical goup that indicates all the bonds that come from the same cell, this will be used to categorise the bonds later on
        cell_name = "cell_" +str(cell_index)
        #add all 28 bonds of the hexahedron
        for i in range(8):
            for j in range(i + 1, 8):
                pair = tuple(sorted((hexa[i], hexa[j])))
                bonds.append((pair[0], pair[1], cell_name))

        cell_index += 1
    return bonds



def convert(input_path, output_path, plot=False, verbose = True):
    node_ids, node_coords, hexahedra = read_gmsh_mesh(input_path)
    if verbose:
        print(f"Read {len(node_ids)} nodes and {len(hexahedra)} hexahedra from {input_path}")

    bonds = hexahedra_to_bonds(hexahedra)

    # write_msh2 renumbers nodes 1..N in list order, so map the file's node ids onto that numbering
    nodes = np.array([node_coords[nid] for nid in node_ids])
    id_to_index = {nid: k + 1 for k, nid in enumerate(node_ids)}

    name_to_physical_id = {}
    elements = []
    for a, b, group in bonds:
        if group not in name_to_physical_id:
            name_to_physical_id[group] = len(name_to_physical_id) + 1
        elements.append((id_to_index[a], id_to_index[b], name_to_physical_id[group]))

    write_msh2(output_path, nodes, elements, name_to_physical_id)

    if plot:
        plot_mesh(nodes, elements)
        
    return output_path




if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python mesh_convertor.py <mesh_file>")
        sys.exit(1)

    mesh_file = sys.argv[1] + ".msh"
    output_file = sys.argv[1] + "_spring_network.msh"

    convert(mesh_file, output_file, plot=True)

