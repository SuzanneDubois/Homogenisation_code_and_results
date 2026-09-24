import fedoo as fd
import numpy as np
import os
import sys
import meshio
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mesh_utils import parse_msh2, volume_convex_hull

# ====================== ELEMENT TYPE HELPERS ======================
# local edges of each supported volume element (node pairs inside one element)
ELEMENT_EDGES = {
    "hex8": [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)],
    "tet4": [(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)],
}
ELEMENT_TYPE_BY_N_NODES = {8: "hex8", 4: "tet4"}


def detect_element_type(mesh_path):
    """Return the volume element type ("hex8" or "tet4") of a gmsh mesh file."""
    raw = meshio.read(mesh_path)
    cell_types = {block.type for block in raw.cells}
    if "hexahedron" in cell_types:
        return "hex8"
    if "tetra" in cell_types:
        return "tet4"
    raise ValueError(f"{mesh_path}: no hexahedron or tetra elements found (cell types: {cell_types})")


def read_volume_mesh(mesh_path, elm_type=None):
    """Read a gmsh file with fedoo and return only its volume mesh (hex8 or tet4)."""
    if elm_type is None:
        elm_type = detect_element_type(mesh_path)
    mesh = fd.Mesh.read(mesh_path)
    if hasattr(mesh, "mesh_dict"):  # mesh with several element types (volume + boundary faces)
        mesh = mesh.mesh_dict[elm_type]
    return mesh


def element_edges(elements):
    """Unique edges (node pairs) of hex8 / tet4 elements; 2-node elements (springs) are their own edge."""
    elements = np.asarray(elements)
    if elements.shape[1] == 2:
        return elements
    local_edges = ELEMENT_EDGES[ELEMENT_TYPE_BY_N_NODES[elements.shape[1]]]
    edges = np.array([(elm[a], elm[b]) for elm in elements for a, b in local_edges])
    return np.unique(np.sort(edges, axis=1), axis=0)  # edges shared by neighbouring elements drawn once


# ========================== PLOT HELPERS ==========================
def plot_mesh_bc(mesh, node_sets, dirichlet_sets=(), neumann_sets=()):
    """Plot the mesh wireframe with nodes colored by BC (matplotlib: no GPU/OpenGL needed).

    node_sets : dict {surface name: array of node indices} (from get_nodes_sets)
    dirichlet_sets / neumann_sets : names of node_sets actually used as
    Dirichlet/Neumann BC, so only those are highlighted.
    """
    nodes = np.asarray(mesh.nodes)
    elements = np.asarray(mesh.elements)

    segments = nodes[element_edges(elements)]

    bc_color = np.full(len(nodes), "lightgray", dtype=object)
    for name in dirichlet_sets:
        bc_color[node_sets[name]] = "red"      # Dirichlet
    for name in neumann_sets:
        bc_color[node_sets[name]] = "blue"     # Neumann

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.add_collection3d(Line3DCollection(segments, colors="gray", linewidths=0.3, alpha=0.5))
    ax.scatter(nodes[:, 0], nodes[:, 1], nodes[:, 2], c=bc_color, s=8)
    ax.set_box_aspect(np.ptp(nodes, axis=0))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(f"gray = free   red = Dirichlet ({', '.join(dirichlet_sets) or 'none'})"
                 f"   blue = Neumann ({', '.join(neumann_sets) or 'none'})")
    plt.show()



# ======================== MESH UTILS ==========================
def get_nodes_sets (fem_mesh_path):
    raw = meshio.read(fem_mesh_path)
    surface_name_by_tag = {tag: name for name, (tag, dim) in raw.field_data.items() if dim == 2}

    # boundary faces are quads for hex meshes and triangles for tet meshes
    node_sets = {}
    for block, tags in zip(raw.cells, raw.cell_data["gmsh:physical"]):
        if block.type not in ("quad", "triangle"):
            continue
        for tag in np.unique(tags):
            name = surface_name_by_tag.get(tag)
            if name is not None:
                nodes = block.data[tags == tag]
                node_sets[name] = np.union1d(node_sets.get(name, []), nodes).astype(int)

    return node_sets


#======================== FEM TEST ==========================

def fem_test(mesh_path, boundary_conditions, node_sets, elm_type=None, E=210e9, nu=0.25, verbose=True):

    fd.ModelingSpace("3D")

    #extract only the volume mesh (hex8 or tet4) from the mesh file
    if elm_type is None:
        elm_type = detect_element_type(mesh_path)
    mesh = read_volume_mesh(mesh_path, elm_type)

    if verbose:
        print(f"Mesh {mesh_path}: {len(mesh.nodes)} nodes, {len(mesh.elements)} {elm_type} elements")

    #define our material
    fd.constitutivelaw.ElasticIsotrop(E=E, nu=nu, name="test_material")
    fd.weakform.StressEquilibrium("test_material", name="wf")

    #define the problem
    assembly = fd.Assembly.create("wf", mesh, name="assembly")
    pb = fd.problem.Linear("assembly")

    #define the BC
    dirichlet_sets, neumann_sets = [], []
    for bc in boundary_conditions:
        nodes = node_sets[bc["surface"]]  # array of node indices
        value = bc["value"]
        if bc["type"] == "Neumann":
            value = np.asarray(value) / len(nodes)  # distribute
        pb.bc.add(bc["type"], nodes, "Disp", value)
        (dirichlet_sets if bc["type"] == "Dirichlet" else neumann_sets).append(bc["surface"])

    plot_mesh_bc(mesh, node_sets, dirichlet_sets=dirichlet_sets, neumann_sets=neumann_sets)
    pb.solve()
    res = pb.get_results(["Disp", "Stress"], "Node")

    displacement = res["Disp"]

    return displacement



# ======================= LSM TEST ==========================

def lsm_test(mesh_path,stiffness_path, boundary_conditions, node_sets, verbose=True):

    fd.ModelingSpace("3D")

    #step 1: load the spring network (bonds) mesh
    mesh = fd.Mesh.read(mesh_path)
    mesh.elm_type = "spring"

    with open(stiffness_path, "r") as f:
        k_values = np.array([float(line) for line in f.read().splitlines()])

    #step 4: weakform + assembly
    wf = fd.weakform.SpringEquilibrium(k_values, nlgeom=False)
    assembly = fd.Assembly.create(wf, mesh, name="assembly")

    #step 5: define the problem
    pb = fd.problem.Linear("assembly")

    #define the BC
    dirichlet_sets, neumann_sets = [], []
    for bc in boundary_conditions:
        nodes = node_sets[bc["surface"]]  # array of node indices
        value = bc["value"]
        if bc["type"] == "Neumann":
            value = np.asarray(value) / len(nodes)  # distribute
        pb.bc.add(bc["type"], nodes, "Disp", value)
        (dirichlet_sets if bc["type"] == "Dirichlet" else neumann_sets).append(bc["surface"])

    #step 7: visualize the mesh with the BC highlighted
    plot_mesh_bc(mesh, node_sets, dirichlet_sets=dirichlet_sets, neumann_sets=neumann_sets)

    #step 8: solve
    pb.solve()

    res = pb.get_results(["Disp"], "Node")
    displacement = res["Disp"]

    return displacement



if __name__ == "__main__":
    name = sys.argv[1]
    mesh_fem = name + ".msh"
    mesh_lsm = name + "_spring_network.msh"

    stiffness_file = name + "_spring_network_k.txt"

    #detect the FEM element type (hex8 or tet4) once, at the beginning
    elm_type = detect_element_type(mesh_fem)
    print(f"{name}: {elm_type} mesh detected")

    node_sets = get_nodes_sets(mesh_fem)
    displacement_fem = fem_test(mesh_fem, elm_type=elm_type, boundary_conditions=[
        {"surface": "x_min", "type": "Dirichlet", "value": [0, 0, 0]},
        {"surface": "x_max", "type": "Neumann", "value": [0, 0, -1e7]}
    ], node_sets=node_sets)

    displacement_lsm = lsm_test(mesh_lsm, stiffness_file, boundary_conditions=[
        {"surface": "x_min", "type": "Dirichlet", "value": [0, 0, 0]},
        {"surface": "x_max", "type": "Neumann", "value": [0, 0, -1e7]}
    ], node_sets=node_sets)

    #save the displacements to files
    np.savetxt(name + "_fem_disp.txt", displacement_fem)
    np.savetxt(name + "_lsm_disp.txt", displacement_lsm)
