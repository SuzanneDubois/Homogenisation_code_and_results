import numpy as np
from scipy.spatial import ConvexHull

def parse_msh2(filename: str) -> tuple[dict, list, np.ndarray, dict[int, str]]:
    """
    Parse a Gmsh MSH2 file and extract nodes, bond elements and physical groups.
 
    Supported element types:
        1  — 2-node line  (bond)   ← kept
        15 — 1-node point          ← ignored
        2  — 3-node triangle       ← ignored
        (any other type is silently skipped)
 
    Parameters
    ----------
    filename : path to the .msh file (MSH2 format, ascii)
 
    Returns
    -------
    nodes : dict { node_id (int) : coordinates (np.ndarray shape (3,)) }
    bonds : list of (node_id_a, node_id_b) int tuples, one per line element
    bond_physical_tags : (N,) ndarray of physical group ids, one per bond
    physical_names : dict mapping physical group id to name, if present
    """
    nodes: dict[int, np.ndarray] = {}
    bonds: list[tuple[int, int]] = []
    bond_physical_tags: list[int] = []
    physical_names: dict[int, str] = {}
 
    with open(filename, "r") as f:
        lines = f.readlines()
 
    i = 0
    while i < len(lines):
        tag = lines[i].strip()

        # ── Physical names section ───────────────────────────────────
        if tag == "$PhysicalNames":
            i += 1
            n_phys = int(lines[i].strip())
            i += 1
            for _ in range(n_phys):
                parts = lines[i].strip().split(None, 2)
                if len(parts) >= 3:
                    phys_id = int(parts[1])
                    physical_names[phys_id] = parts[2].strip().strip('"')
                i += 1
            i += 1
 
        # ── Node section ────────────────────────────────────────────────
        elif tag == "$Nodes":
            i += 1
            n_nodes = int(lines[i].strip())
            i += 1
            for _ in range(n_nodes):
                parts = lines[i].strip().split()
                nid = int(parts[0])
                nodes[nid] = np.array([float(parts[1]),
                                       float(parts[2]),
                                       float(parts[3])])
                i += 1
            # skip $EndNodes
            i += 1
 
        # ── Element section ─────────────────────────────────────────────
        elif tag == "$Elements":
            i += 1
            n_elems = int(lines[i].strip())
            i += 1
            for _ in range(n_elems):
                parts = lines[i].strip().split()
                elem_type = int(parts[1])
                n_tags = int(parts[2])
                node_start = 3 + n_tags          # index of first node id
 
                if elem_type == 1:               # 2-node line = bond
                    phys_tag = int(parts[3]) if n_tags >= 1 else 0
                    na = int(parts[node_start])
                    nb = int(parts[node_start + 1])
                    bonds.append((na, nb))
                    bond_physical_tags.append(phys_tag)
                i += 1
            # skip $EndElements
            i += 1
 
        else:
            i += 1
 
    return nodes, bonds, np.asarray(bond_physical_tags, dtype=int), physical_names

def volume_bonding_box(nodes: dict[int, np.ndarray]) -> float:
    """
    Compute the bounding box size of the mesh defined by the nodes.
    """
    max_x = max(coord[0] for coord in nodes.values())
    min_x = min(coord[0] for coord in nodes.values())
    max_y = max(coord[1] for coord in nodes.values())
    min_y = min(coord[1] for coord in nodes.values())
    max_z = max(coord[2] for coord in nodes.values())
    min_z = min(coord[2] for coord in nodes.values())

    volume = (max_x - min_x) * (max_y - min_y) * (max_z - min_z)
    return volume

def volume_convex_hull(nodes: dict[int, np.ndarray]) -> float:
    """
    Compute the volume of the convex hull of the nodes.
    """
    points = np.array(list(nodes.values()))
    hull = ConvexHull(points)
    return hull.volume

def write_msh2(filename, node_coords, elements, name_to_physical_id):
    """Write nodes/line-elements/physical names to an MSH2 ASCII file."""
    with open(filename, "w") as f:
        f.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n")
 
        f.write("$PhysicalNames\n")
        f.write(f"{len(name_to_physical_id)}\n")
        for name, phys_id in name_to_physical_id.items():
            f.write(f'1 {phys_id} "{name}"\n')
        f.write("$EndPhysicalNames\n")
 
        f.write("$Nodes\n")
        f.write(f"{len(node_coords)}\n")
        for i, coord in enumerate(node_coords):
            x, y, z = coord
            f.write(f"{i + 1} {x:.10g} {y:.10g} {z:.10g}\n")
        f.write("$EndNodes\n")
 
        f.write("$Elements\n")
        f.write(f"{len(elements)}\n")
        # elm-type 1 = 2-node line ; 2 tags = (physical id, elementary id)
        for elm_id, (n1, n2, phys_id) in enumerate(elements, start=1):
            f.write(f"{elm_id} 1 2 {phys_id} {phys_id} {n1} {n2}\n")
        f.write("$EndElements\n")

def plot_mesh(nodes, bonds):
    """
    Plot the mesh using matplotlib.
 
    Parameters
    ----------
    nodes : a list of (x, y, z) coordinates for each node
    bonds : list of (node_id_a, node_id_b, physical_id) int tuples, one per line element
    """
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot nodes
    node_coords = np.array(nodes)
    ax.scatter(node_coords[:, 0], node_coords[:, 1], node_coords[:, 2], color='b', s=20, label='Nodes')

    # Plot bonds
    for n1, n2, _ in bonds:
        p1 = nodes[n1 - 1]  # Convert to 0-based indexing
        p2 = nodes[n2 - 1]  # Convert to 0-based indexing
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color='r', label='Bonds')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Mesh Visualization')
    plt.show()