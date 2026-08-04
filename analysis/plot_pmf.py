#!/usr/bin/env python3
"""Plot PMF profiles for each lipid.  Per lipid, the three force fields
(CHARMM, AMBER, SLipids) are overlaid, each shifted so that its minimum energy
is zero."""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from palette import FORCEFIELDS

# Project layout: this script lives in analysis/, data in data/, output in plots/.
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
PLOTS = ROOT / "plots"

# Lipids to plot; profiles are named "{ff}_{lipid}_profile.xvg".  A lipid with
# no profile files (only pullx trajectories) is skipped.
LIPIDS = ["DOPC", "DMPC", "DMPG"]

# Each full profile is accompanied by five partial profiles computed from
# consecutive 20 ns windows (0-20, 20-40, 40-60, 60-80, 80-100 ns).  They
# share the full profile's basename, e.g. "charmm_DMPC_profile.xvg" ->
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


def plot_lipid(lipid):
    """Overlay the CHARMM/AMBER/SLipids PMF profiles for one lipid.  Returns
    True if a figure was produced, False if the lipid has no profile data."""
    fig = plt.figure(figsize=(6, 5))
    plotted = False

    for key, label, color in FORCEFIELDS:
        fname = f"{key}_{lipid}_profile.xvg"
        path = DATA / fname
        if not path.exists():
            continue
        plotted = True

        # Partial profiles first, so they render behind the full curve.
        for part in range(1, N_PARTS + 1):
            part_path = DATA / f"{key}_{lipid}_part{part}_profile.xvg"
            if not part_path.exists():
                continue
            px, py = load_xvg(part_path)
            plt.plot(px, py - py.min(), color=color, lw=0.5, alpha=0.3)

        x, y = load_xvg(path)
        # Shift so the curve's minimum sits at y == 0
        plt.plot(x, y - y.min(), color=color, label=label)

    if not plotted:
        plt.close(fig)
        print(f"skipped {lipid}: no profile data")
        return False

    ax = plt.gca()
    ax.axhline(0.0, color="gray", lw=0.5, ls="--")
    ax.set_xlim(0.0, 1.0)
    # Mirror the y-axis ticks onto the right spine.
    ax.tick_params(axis="y", right=True)
    ax.set_xlabel(r"$\xi$")
    ax.set_ylabel(r"PMF (kJ mol$^{-1}$)")
    ax.set_title(lipid)
    ax.legend()
    plt.tight_layout()
    outpath = PLOTS / f"pmf_profiles_{lipid}.png"
    plt.savefig(outpath, dpi=300)
    plt.close(fig)
    print(f"wrote {outpath}")
    return True


def main():
    for lipid in LIPIDS:
        plot_lipid(lipid)


if __name__ == "__main__":
    main()
