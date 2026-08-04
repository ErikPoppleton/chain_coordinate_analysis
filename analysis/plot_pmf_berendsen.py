#!/usr/bin/env python3
"""Compare the CHARMM DMPC PMF profile from the V-rescale + MD production run
against the Berendsen + SD variant.  Same styling as plot_pmf.py: x-axis fixed
to 0-1, each curve shifted so its minimum energy is zero, y-ticks mirrored onto
the right spine."""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# Project layout: this script lives in analysis/, data in data/, output in plots/.
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
PLOTS = ROOT / "plots"

# (filename stem, legend label, colour): the usual CHARMM blue for the
# V-rescale + MD run, a lighter blue for the Berendsen + SD variant.
PROFILES = [
    ("charmm_DMPC_profile", "V-rescale + MD", "#2171b5"),
    ("charmm_DMPC_profile_berendsen", "Berendsen + SD", "#6baed6"),
]


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
    fig = plt.figure(figsize=(6, 5))

    for stem, label, color in PROFILES:
        x, y = load_xvg(DATA / f"{stem}.xvg")
        # Shift so the curve's minimum sits at y == 0
        plt.plot(x, y - y.min(), color=color, label=label)

    ax = plt.gca()
    ax.axhline(0.0, color="gray", lw=0.5, ls="--")
    ax.set_xlim(0.0, 1.0)
    # Mirror the y-axis ticks onto the right spine.
    ax.tick_params(axis="y", right=True)
    ax.set_xlabel(r"$\xi$")
    ax.set_ylabel(r"PMF (kJ mol$^{-1}$)")
    ax.set_title("CHARMM DMPC")
    ax.legend()
    plt.tight_layout()
    outpath = PLOTS / "pmf_profiles_DMPC_berendsen.png"
    plt.savefig(outpath, dpi=300)
    plt.close(fig)
    print(f"wrote {outpath}")


if __name__ == "__main__":
    main()
