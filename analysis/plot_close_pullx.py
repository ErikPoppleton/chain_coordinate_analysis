#!/usr/bin/env python3
"""Plot all 'close' pull-COM (.xvg) trajectories: Position (nm) vs Time (ps).

One figure per lipid (DOPC, DMPC, DMPG).  Within each figure the three force
fields (CHARMM, AMBER, SLipids) get their own panel with a single-hue
dark->light monochrome gradient across the 10 replicates.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from palette import FORCEFIELDS, CMAPS
from lipiddefs import LIPIDS

# Project layout: this script lives in analysis/, data in data/, output in plots/.
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
PLOTS = ROOT / "plots"

# Lipids to plot; data files are named "{ff}_{lipid}_close{rep}_pullx.xvg".

N_REP = 10          # replicates per force field
STRIDE = 5          # subsample factor (2.5M points/file is more than a plot needs)
LINEWIDTH = 0.3     # fine lines


def load_xvg(path, stride=1):
    """Read x, y columns from a GROMACS/xmgrace .xvg file, skipping metadata."""
    x, y = [], []
    with open(path) as fh:
        for line in fh:
            if not line or line[0] in "#@&\n":
                continue
            cols = line.split()
            x.append(float(cols[0]))
            y.append(float(cols[1]))
    return np.array(x[::stride]), np.array(y[::stride])


def plot_lipid(lipid):
    """Build the three-panel close-trajectory figure for one lipid."""
    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True, sharey=True)

    for ax, (ff, label, _color) in zip(axes, FORCEFIELDS):
        cmap = CMAPS[ff]
        # Sample the colormap dark (rep 1) -> light (rep 10).
        shades = np.linspace(0.9, 0.3, N_REP)
        for i in range(N_REP):
            rep = i + 1
            fname = f"{ff}_{lipid}_close{rep}_pullx.xvg"
            x, y = load_xvg(DATA / fname, stride=STRIDE)
            ax.plot(x, y, color=cmap(shades[i]), lw=LINEWIDTH,
                    rasterized=True, label=f"close{rep}")

        ax.set_ylabel(r"$\xi$")
        ax.set_title(label, loc="left", fontsize=11, fontweight="bold")
        ax.legend(ncol=5, fontsize=6, frameon=False, loc="best")

    axes[-1].set_xlabel("Time (ps)")
    fig.suptitle(lipid, fontsize=13, fontweight="bold")
    fig.tight_layout()
    outpath = PLOTS / f"close_pullx_{lipid}.png"
    fig.savefig(outpath, dpi=300)
    plt.close(fig)
    print(f"wrote {outpath}")


def main():
    for lipid in LIPIDS:
        plot_lipid(lipid)


if __name__ == "__main__":
    main()
