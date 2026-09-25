import fedoo as fd
import numpy as np
import os
import sys
import meshio
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from test_execution import detect_element_type, read_volume_mesh, element_edges

LENGTH_UNIT = "m"  # SI units (E given in Pa in test_execution.py)
ERR_LIM = 100.0  # fixed color scale -ERR_LIM .. +ERR_LIM % of all error plots, so they can be compared
FIGURES = []  # (figure, filename) of the current case, written to disk by save_figures


def percent(error, reference):
    # error / reference in percent, NaN where the reference is zero (e.g. clamped nodes)
    reference = np.asarray(reference, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(reference != 0, np.asarray(error) / reference * 100, np.nan)


def l2_error(reference, approx):
    # absolute L2 error ||reference - approx|| and relative L2 error ||reference - approx|| / ||reference|| in percent
    reference = np.ravel(reference)
    approx = np.ravel(approx)
    abs_err = np.linalg.norm(reference - approx)
    ref_norm = np.linalg.norm(reference)
    rel_err = abs_err / ref_norm * 100 if ref_norm != 0 else np.nan
    return abs_err, rel_err


def error_extend(values):
    # colorbar arrows for values beyond the fixed -ERR_LIM .. +ERR_LIM % scale
    above = np.nanmax(values) > ERR_LIM
    below = np.nanmin(values) < -ERR_LIM
    return {(True, True): "both", (True, False): "max", (False, True): "min"}.get((above, below), "neither")


def set_xyz_labels(ax):
    ax.set_xlabel(f"x ({LENGTH_UNIT})")
    ax.set_ylabel(f"y ({LENGTH_UNIT})")
    ax.set_zlabel(f"z ({LENGTH_UNIT})")


def set_title(ax, title, short_title=None):
    # full title for the combined pdf, short title (no numbers) for the individual figure files
    ax.set_title(title)
    ax.short_title = short_title if short_title is not None else title


def save_and_show(fig, filename):
    # figures are only collected here, they are written by save_figures once the case is done
    FIGURES.append((fig, filename))
    fig.show()


def save_figures(dir_name, pdf_path):
    # all figures with their full titles in one pdf, then each figure with its short title in dir_name
    os.makedirs(dir_name, exist_ok=True)
    with PdfPages(pdf_path) as pdf:
        for fig, _ in FIGURES:
            pdf.savefig(fig, bbox_inches="tight")
    for fig, filename in FIGURES:
        for ax in fig.axes:
            ax.set_title(getattr(ax, "short_title", ""))
        fig.savefig(os.path.join(dir_name, filename), dpi=300, bbox_inches="tight")
        plt.close(fig)
    FIGURES.clear()


def plot_displacement(mesh, displacement_fem, displacement_lsm, name="", scale=1.0):
    #plot the deformed mesh for both FEM and LSM side by side, then plot the difference between the two displacements
    nodes = np.asarray(mesh.nodes)
    elements = np.asarray(mesh.elements)

    # fedoo returns Disp as (3, n_nodes): put it back as (n_nodes, 3)
    u_fem = np.asarray(displacement_fem)
    u_lsm = np.asarray(displacement_lsm)
    if u_fem.shape[0] == 3 and u_fem.shape[1] != 3:
        u_fem = u_fem.T
    if u_lsm.shape[0] == 3 and u_lsm.shape[1] != 3:
        u_lsm = u_lsm.T

    edges = element_edges(elements)  # works for hex8 and tet4

    u_norm_fem = np.linalg.norm(u_fem, axis=1)
    u_norm_lsm = np.linalg.norm(u_lsm, axis=1)
    vmin = min(u_norm_fem.min(), u_norm_lsm.min())
    vmax = max(u_norm_fem.max(), u_norm_lsm.max())

    # --- deformed meshes side by side, same color scale ---
    fig = plt.figure(figsize=(14, 6))
    for i, (u, u_norm, label) in enumerate([(u_fem, u_norm_fem, "FEM"), (u_lsm, u_norm_lsm, "LSM")]):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        deformed = nodes + scale * u
        ax.add_collection3d(Line3DCollection(nodes[edges], colors="lightgray", linewidths=0.3, alpha=0.4))
        ax.add_collection3d(Line3DCollection(deformed[edges], colors="gray", linewidths=0.3, alpha=0.6))
        sc = ax.scatter(*deformed.T, c=u_norm, cmap="viridis", vmin=vmin, vmax=vmax, s=8)
        ax.set_box_aspect(np.ptp(deformed, axis=0))
        set_xyz_labels(ax)
        set_title(ax, f"{label} deformed mesh (x{scale:.3g})", f"{label} deformed mesh")
    fig.colorbar(sc, ax=fig.axes, shrink=0.6, label=f"|u| ({LENGTH_UNIT})")
    save_and_show(fig, "displacement_deformed.pdf")

    # --- difference FEM - LSM on the undeformed mesh ---
    diff = percent(np.linalg.norm(u_fem - u_lsm, axis=1), u_norm_fem)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.add_collection3d(Line3DCollection(nodes[edges], colors="gray", linewidths=0.3, alpha=0.5))
    # same fixed -ERR_LIM .. +ERR_LIM % scale as the other error plots (this one is always >= 0)
    sc = ax.scatter(*nodes.T, c=diff, cmap="coolwarm", vmin=-ERR_LIM, vmax=ERR_LIM, s=8)
    fig.colorbar(sc, ax=ax, shrink=0.6, extend=error_extend(diff),
                 label="|u_FEM - u_LSM| / |u_FEM| (%)")
    ax.set_box_aspect(np.ptp(nodes, axis=0))
    set_xyz_labels(ax)
    set_title(ax, f"Displacement difference FEM vs LSM (max = {np.nanmax(diff):.2f} %)",
              "Displacement difference FEM vs LSM")
    save_and_show(fig, "displacement_error.pdf")

    # --- signed error per direction (u_FEM - u_LSM) / |u_FEM| in percent, side by side ---
    diff_xyz = percent(u_fem - u_lsm, u_norm_fem[:, None])
    fig = plt.figure(figsize=(18, 6))
    for i, comp in enumerate("xyz"):
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        max_err = np.nanmax(np.abs(diff_xyz[:, i]))
        ax.add_collection3d(Line3DCollection(nodes[edges], colors="gray", linewidths=0.3, alpha=0.5))
        # fixed symmetric color scale -ERR_LIM .. +ERR_LIM %
        sc = ax.scatter(*nodes.T, c=diff_xyz[:, i], cmap="coolwarm", vmin=-ERR_LIM, vmax=ERR_LIM, s=8)
        fig.colorbar(sc, ax=ax, shrink=0.6, extend=error_extend(diff_xyz[:, i]),
                     label=f"(u{comp}_FEM - u{comp}_LSM) / |u_FEM| (%)")
        ax.set_box_aspect(np.ptp(nodes, axis=0))
        set_xyz_labels(ax)
        set_title(ax, f"Error in {comp} (max |.| = {max_err:.2f} %)", f"Error in {comp}")
    save_and_show(fig, "displacement_error_xyz.pdf")


def plot_elongations(lsm_mesh, displacement_fem, displacement_lsm, name="", scale=1.0):
    #plot the elongation of each bond for both FEM and LSM side by side, then plot the difference between the two elongations
    nodes = np.asarray(lsm_mesh.nodes)
    bonds = np.asarray(lsm_mesh.elements)

    u_fem = np.asarray(displacement_fem)
    u_lsm = np.asarray(displacement_lsm)
    if u_fem.shape[0] == 3 and u_fem.shape[1] != 3:
        u_fem = u_fem.T
    if u_lsm.shape[0] == 3 and u_lsm.shape[1] != 3:
        u_lsm = u_lsm.T

    # compute elongation for each bond
    def compute_elongation(bonds, nodes, displacements):
        elongations = []
        for i, j in bonds:
            x0, x1 = nodes[i], nodes[j]
            u0, u1 = displacements[i], displacements[j]
            L0 = np.linalg.norm(x1 - x0)
            L1 = np.linalg.norm((x1 + scale * u1) - (x0 + scale * u0))
            elongations.append((L1 - L0) / L0)
        return np.array(elongations)

    elongation_fem = compute_elongation(bonds, nodes, u_fem)
    elongation_lsm = compute_elongation(bonds, nodes, u_lsm)

    #make the plots
    segments = nodes[bonds]

    def add_bonds(ax, values, cmap, vmin, vmax):
        # draw each bond colored by its value
        lc = Line3DCollection(segments, cmap=cmap, norm=plt.Normalize(vmin, vmax), linewidths=1.0)
        lc.set_array(values)
        ax.add_collection3d(lc)
        ax.set_xlim(nodes[:, 0].min(), nodes[:, 0].max())
        ax.set_ylim(nodes[:, 1].min(), nodes[:, 1].max())
        ax.set_zlim(nodes[:, 2].min(), nodes[:, 2].max())
        ax.set_box_aspect(np.ptp(nodes, axis=0))
        set_xyz_labels(ax)
        return lc

    # --- bond elongations FEM and LSM side by side, same color scale ---
    vmin = min(elongation_fem.min(), elongation_lsm.min())
    vmax = max(elongation_fem.max(), elongation_lsm.max())
    fig = plt.figure(figsize=(14, 6))
    for i, (elong, label) in enumerate([(elongation_fem, "FEM"), (elongation_lsm, "LSM")]):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        lc = add_bonds(ax, elong, "viridis", vmin, vmax)
        set_title(ax, f"{label} bond elongation")
    fig.colorbar(lc, ax=fig.axes, shrink=0.6, label="(L - L0) / L0 (-)")
    save_and_show(fig, "elongation.pdf")

    # --- difference FEM - LSM per bond ---
    diff = percent(elongation_fem - elongation_lsm, elongation_fem)  # relative difference in percent
    max_err = np.nanmax(np.abs(diff))
    # fixed symmetric color scale -ERR_LIM .. +ERR_LIM %: bonds beyond it are drawn in black
    n_out = np.count_nonzero(np.abs(diff) > ERR_LIM)
    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_over("black")
    cmap.set_under("black")
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    lc = add_bonds(ax, diff, cmap, -ERR_LIM, ERR_LIM)
    fig.colorbar(lc, ax=ax, shrink=0.6, extend=error_extend(diff),
                 label="(elongation FEM - LSM) / FEM (%)")
    title = f"Bond elongation difference FEM vs LSM (max |.| = {max_err:.2f} %)"
    if n_out:
        title += f"\n{n_out} bonds beyond ±{ERR_LIM:.0f} % shown in black"
    set_title(ax, title, "Bond elongation difference FEM vs LSM")
    save_and_show(fig, "elongation_error.pdf")

    # --- LSM vs FEM elongation per bond (points on y = x mean perfect agreement) ---
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(elongation_fem, elongation_lsm, s=4, alpha=0.5)
    ax.plot([vmin, vmax], [vmin, vmax], "k--", linewidth=1, label="y = x")
    ax.set_xlabel("FEM elongation (L - L0) / L0 (-)")
    ax.set_ylabel("LSM elongation (L - L0) / L0 (-)")
    set_title(ax, "Bond elongation: LSM vs FEM")
    ax.legend()
    ax.set_aspect("equal")
    save_and_show(fig, "elongation_lsm_vs_fem.pdf")

    return elongation_fem, elongation_lsm



if __name__ == "__main__":
    # usage: python post_processing.py name1 [name2 ...]
    # expects name.msh, name_fem_disp.txt and name_lsm_disp.txt (written by test_execution.py)
    # writes name_all_figures.pdf (all plots with titles) and name_figures/ (one pdf per plot, short titles)
    fd.ModelingSpace("3D")
    for name in sys.argv[1:]:
        figures_dir = name + "_figures"
        elm_type = detect_element_type(name + ".msh")  # hex8 or tet4
        fem_mesh = read_volume_mesh(name + ".msh", elm_type)
        displacement_fem = np.loadtxt(name + "_fem_disp.txt")
        displacement_lsm = np.loadtxt(name + "_lsm_disp.txt")

        print(f"{name}: {elm_type} mesh, {len(fem_mesh.nodes)} nodes")
        plot_displacement(fem_mesh, displacement_fem, displacement_lsm, name=name, scale = 10)

        lsm_mesh = fd.Mesh.read(name + "_spring_network.msh")
        elongation_fem, elongation_lsm = plot_elongations(lsm_mesh, displacement_fem, displacement_lsm, name=name)

        # L2 errors of LSM with respect to FEM, written next to the figures
        disp_abs, disp_rel = l2_error(displacement_fem, displacement_lsm)
        elong_abs, elong_rel = l2_error(elongation_fem, elongation_lsm)
        save_figures(figures_dir, name + "_all_figures.pdf")
        with open(os.path.join(figures_dir, "l2_errors.txt"), "w") as f:
            f.write(f"# L2 errors of LSM with respect to FEM for {name}\n")
            f.write(f"# absolute_L2 in {LENGTH_UNIT} for displacement, dimensionless (-) for elongation\n")
            f.write(f"# quantity      absolute_L2      relative_L2 (%)\n")
            f.write(f"displacement    {disp_abs:.6e}     {disp_rel:.6f}\n")
            f.write(f"elongation      {elong_abs:.6e}     {elong_rel:.6f}\n")
        print(f"L2 error displacement: {disp_rel:.4f} %, elongation: {elong_rel:.4f} %")
