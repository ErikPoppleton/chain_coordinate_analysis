#!/usr/bin/env python3
"""Plot PMF profiles as three horizontal panels -- DOPC, DMPC, DMPG across the
columns of one figure.  Within each panel the three force fields (CHARMM, AMBER,
SLipids) are overlaid, each shifted so its minimum energy is zero; the five 20 ns
partial profiles (if present) are drawn faintly behind the full curve.  Force
fields / lipids with no profile data are simply absent from their panel."""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from palette import FORCEFIELDS

# Project layout: this script lives in analysis/, data in data/, output in plots/.
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
PLOTS = ROOT / "plots"

# Lipids across the columns; profiles are named "{ff}_{lipid}_profile.xvg".
LIPIDS = ["DOPC", "DMPC", "DMPG"]

# Each full profile is accompanied by five partial profiles computed from
# consecutive 20 ns windows (0-20, 20-40, 40-60, 60-80, 80-100 ns), sharing the
# full profile's basename, e.g. "charmm_DMPC_profile.xvg" ->
# "charmm_DMPC_part1_profile.xvg" ... "charmm_DMPC_part5_profile.xvg".
N_PARTS = 5


def load_xvg(path):
    """Read x, y columns from an xmgrace/GROMACS .xvg file, skipping
    '#' comment and '@'/'&' metadata lines."""
    x, y = [], []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line[0] in "#@&":
                continue
            cols = line.split()
            x.append(float(cols[0]))
            y.append(float(cols[1]))
    return np.array(x), np.array(y)


def plot_lipid(ax, lipid):
    """Overlay the CHARMM/AMBER/SLipids PMF profiles for one lipid onto ``ax``.
    Returns True if any curve was drawn, False if the lipid has no profile data."""
    plotted = False
    for key, label, color in FORCEFIELDS:
        path = DATA / f"{key}_{lipid}_profile.xvg"
        if not path.exists():
            continue
        plotted = True

        # Partial profiles first, so they render behind the full curve.
        for part in range(1, N_PARTS + 1):
            part_path = DATA / f"{key}_{lipid}_part{part}_profile.xvg"
            if not part_path.exists():
                continue
            px, py = load_xvg(part_path)
            ax.plot(px, py - py.min(), color=color, lw=0.5, alpha=0.3)

        x, y = load_xvg(path)
        ax.plot(x, y - y.min(), color=color, label=label)   # min-zeroed

    ax.axhline(0.0, color="gray", lw=0.5, ls="--")
    ax.set_xlim(0.0, 1.0)
    ax.tick_params(axis="y", right=True)
    ax.set_xlabel(r"$\xi$")
    ax.set_title(lipid)
    if not plotted:
        print(f"note: {lipid} has no profile data (blank panel)")
    return plotted


def main():
    PLOTS.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(LIPIDS), figsize=(15, 4.5), sharey=True)
    for ax, lipid in zip(axes, LIPIDS):
        plot_lipid(ax, lipid)
    axes[0].set_ylabel(r"PMF (kJ mol$^{-1}$)")

    # One shared legend (the force-field colour key) above the panels.
    handles = [Line2D([0], [0], color=c, label=l) for _, l, c in FORCEFIELDS]
    fig.legend(handles=handles, loc="upper center", ncol=len(FORCEFIELDS),
               frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    outpath = PLOTS / "pmf_profiles.png"
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {outpath}")


if __name__ == "__main__":
    main()
