import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mesh_utils import parse_msh2, volume_convex_hull


def cell_volumes_per_bond(nodes, bonds, bond_physical_tags):
    # volume of the cell (physical group) each bond belongs to: convex hull of the nodes used by the cell's bonds
    cell_volumes = {}
    for tag in np.unique(bond_physical_tags):
        idx = np.nonzero(bond_physical_tags == tag)[0]
        cell_nodes = {nid: nodes[nid] for b in idx for nid in bonds[b]}
        cell_volumes[tag] = volume_convex_hull(cell_nodes)

    return np.array([cell_volumes[tag] for tag in bond_physical_tags])


def normalised_stiffnesses(name):
    # k * L^2 / V_cell for each bond of name.msh, with k read from name_k.txt (one value per bond, same order)
    nodes, bonds, bond_physical_tags, _ = parse_msh2(name + ".msh")

    with open(name + "_k.txt", "r") as f:
        k_values = np.array([float(line) for line in f.read().splitlines() if line.strip()])

    if len(k_values) != len(bonds):
        raise ValueError(f"{len(k_values)} stiffnesses in {name}_k.txt but {len(bonds)} bonds in {name}.msh")

    lengths = np.array([np.linalg.norm(nodes[a] - nodes[b]) for a, b in bonds])
    volumes = cell_volumes_per_bond(nodes, bonds, bond_physical_tags)

    return k_values * lengths**2 / volumes


def plot_distribution(values, name="", bins=50):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(values, bins=bins, color="#3b6ea5", edgecolor="white", linewidth=0.5)

    ax.axvline(values.mean(), color="black", linestyle="--", linewidth=1,
               label=f"mean = {values.mean():.3e}")
    ax.set_xlabel(r"$k \, L^2 / V_{cell}$")
    ax.set_ylabel("Number of bonds")
    ax.set_title(f"Distribution of $k L^2 / V_{{cell}}$ — {os.path.basename(name)}")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    return fig


if __name__ == "__main__":
    name = sys.argv[1]

    values = normalised_stiffnesses(name)
    print(f"{len(values)} bonds: min {values.min():.3e}, max {values.max():.3e}, "
          f"mean {values.mean():.3e}, std {values.std():.3e}")

    fig = plot_distribution(values, name=name)
    fig.savefig(name + "_kL2_over_V.png", dpi=300, bbox_inches="tight")
    plt.show()
