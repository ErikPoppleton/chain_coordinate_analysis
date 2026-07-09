#!/usr/bin/env python3
"""Plot PMF profiles from three xmgrace (.xvg) files, each shifted so that
its energy is zero at the collective-variable value x = 0.2."""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# Project layout: this script lives in analysis/, data in data/, output in plots/.
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
PLOTS = ROOT / "plots"

# (filename, label, color)
DATASETS = [
    ("charmm_profile.xvg", "CHARMM", "black"),
    ("amber_profile.xvg", "AMBER", "red"),
    ("slipids_profile.xvg", "SLipids", "green"),
]

SHIFT_X = 0.2  # x-value at which each curve is set to y = 0

# Each full profile is accompanied by five partial profiles computed from
# consecutive 20 ns windows (0-20, 20-40, 40-60, 60-80, 80-100 ns).  They
# share the full profile's basename, e.g. "charmm_profile.xvg" ->
# "charmm_part1_profile.xvg" ... "charmm_part5_profile.xvg".
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


def main():
    plt.figure(figsize=(6, 5))

    for fname, label, color in DATASETS:
        # Partial profiles first, so they render behind the full curve.
        prefix = fname[: -len("_profile.xvg")]
        for part in range(1, N_PARTS + 1):
            part_fname = f"{prefix}_part{part}_profile.xvg"
            px, py = load_xvg(DATA / part_fname)
            pidx = np.argmin(np.abs(px - SHIFT_X))
            plt.plot(px, py - py[pidx], color=color, lw=0.5, alpha=0.3)

        x, y = load_xvg(DATA / fname)
        # Shift so y == 0 at the point nearest to x = SHIFT_X
        idx = np.argmin(np.abs(x - SHIFT_X))
        y_shifted = y - y[idx]
        plt.plot(x, y_shifted, color=color, label=label)

    plt.axhline(0.0, color="gray", lw=0.5, ls="--")
    plt.xlabel(r"$\xi$")
    plt.ylabel(r"PMF (kJ mol$^{-1}$)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS / "pmf_profiles.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
