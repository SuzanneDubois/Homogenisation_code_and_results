"""
Post-processing of homogenisation results.

Reads the "{filename}_errors.txt" files written by homogenisation.py and plots
    - the mean linearity error vs the number of springs
    - the isotropy error vs the number of springs

Usage:
    python plot_errors.py name1 name2 ... [-o output.png]

Each name is the base name given to homogenisation.py (without extension);
a path ending in "_errors.txt" is also accepted.
"""
import argparse
import re
import numpy as np
import matplotlib.pyplot as plt


def base_name(name):
    return name[:-len("_errors.txt")] if name.endswith("_errors.txt") else name


def read_errors(filename):
    """
    Parse "{filename}_errors.txt".

    Returns
    -------
    n_springs : int
    isotropy_error : float
    mean_linearity_error : float
    """
    n_springs = None
    isotropy_error = None
    mean_linearity_error = None

    with open(f"{filename}_errors.txt", "r") as f:
        for line in f:
            m = re.search(r"number of bonds \(springs\):\s*(\d+)", line)
            if m:
                n_springs = int(m.group(1))
            elif line.startswith("Isotropy error"):
                isotropy_error = float(line.rsplit(":", 1)[1])
            elif line.strip().startswith("mean:"):
                mean_linearity_error = float(line.split(":", 1)[1])

    if n_springs is None or isotropy_error is None or mean_linearity_error is None:
        raise ValueError(f"Could not read the spring count and errors in {filename}_errors.txt")

    return n_springs, isotropy_error, mean_linearity_error


def plot_error_vs_springs(ax, n_springs, errors, labels, ylabel):
    ax.plot(n_springs, errors, "o-")
    for n, e, label in zip(n_springs, errors, labels):
        ax.annotate(label, (n, e), textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of springs")
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", alpha=0.3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("filenames", nargs="+", help="base names of the homogenised meshes")
    parser.add_argument("-o", "--output", default="errors_vs_springs.png", help="output figure")
    args = parser.parse_args()

    names = [base_name(name) for name in args.filenames]
    results = [read_errors(name) for name in names]

    # sort by number of springs so the curves are monotonic in x
    order = np.argsort([r[0] for r in results])
    names = [names[i] for i in order]
    n_springs = np.array([results[i][0] for i in order])
    isotropy_errors = np.array([results[i][1] for i in order])
    linearity_errors = np.array([results[i][2] for i in order])

    print(f"{'file':<40} {'springs':>10} {'mean linearity':>16} {'isotropy':>12}")
    for name, n, lin, iso in zip(names, n_springs, linearity_errors, isotropy_errors):
        print(f"{name:<40} {n:>10d} {lin:>16.4%} {iso:>12.4%}")

    labels = [name.split("/")[-1] for name in names]
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    plot_error_vs_springs(ax[0], n_springs, linearity_errors, labels, "Mean linearity error")
    ax[0].set_title("Mean linearity error vs number of springs")
    plot_error_vs_springs(ax[1], n_springs, isotropy_errors, labels, "Isotropy error")
    ax[1].set_title("Isotropy error vs number of springs")
    fig.tight_layout()

    fig.savefig(args.output, dpi=150)
    print(f"Figure saved to {args.output}")
    plt.show()
 
