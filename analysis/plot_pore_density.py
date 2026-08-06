#!/usr/bin/env python3
"""Plot pore (radius, z) density maps as four 3x3 grids -- one figure per atom
selection (water, phosphorus, glycerol C2, midpoint C), lipids across columns
(DOPC/DMPC/DMPG), force fields down rows (CHARMM/AMBER/SLipids).

Reads the ``data/poredens_{ff}_{lipid}.npz`` files written by ``pore_density.py``
(a ``(z, r)`` number-density map per selection, plus shared ``r_edges`` /
``z_edges`` in nm).  Each panel is drawn as an ``imshow`` (viridis), **peak-
normalised to 0-1** so panels with very different absolute densities are
comparable; radius runs along x, z relative to the pore centre (z=0) along y.
Panels with no data are left blank.  → ``plots/poredens_{selection}.png``.

Requires only numpy + matplotlib.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from palette import FORCEFIELDS  # [(key, label, colour), ...]

# The chain-coordinate CV acts on a cylinder: radius 0.9 nm, height 2.7 nm,
# centred on the pore centre (z=0).  In the (r, z) maps that is the box
# r in [0, 0.9], z in [-1.35, 1.35].
CYL_R, CYL_H = 0.9, 2.7

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
PLOTS = ROOT / "plots"

LIPIDS = ["DOPC", "DMPC", "DMPG"]

# (npz key, output-file stem, panel-figure title).
SELECTIONS = [
    ("water", "water", "Water density around the pore"),
    ("phosphorus", "phosphorus", "Phosphate density around the pore"),
    ("glycerol_C2", "glycerol_c2", "Glycerol C2 density around the pore"),
    ("midpoint_C", "midpoint_c", "Mid-chain carbon density around the pore"),
]


def load(ff, lipid):
    path = DATA / f"poredens_{ff}_{lipid}.npz"
    return np.load(path) if path.exists() else None


def plot_selection(key, stem, title):
    fig, axes = plt.subplots(len(FORCEFIELDS), len(LIPIDS), figsize=(6, 13),
                             sharex=True, sharey=True)
    any_data = False
    last_im = None
    xlim = ylim = None
    for r, (ff, ff_label, _) in enumerate(FORCEFIELDS):
        for c, lipid in enumerate(LIPIDS):
            ax = axes[r][c]
            if r == 0:
                ax.set_title(lipid)
            if c == 0:
                ax.set_ylabel(f"{ff_label}\nz (nm)")
            npz = load(ff, lipid)
            if npz is None or key not in npz.files:
                continue
            dens = npz[key]
            peak = np.nanmax(dens)
            if peak > 0:
                dens = dens / peak
            r_edges, z_edges = npz["r_edges"], npz["z_edges"]
            extent = [r_edges[0], r_edges[-1], z_edges[0], z_edges[-1]]
            xlim, ylim = (extent[0], extent[1]), (extent[2], extent[3])
            # aspect="auto" here; equal scaling is enforced on *all* axes below
            # so empty panels match the data panels (shared axes otherwise let
            # the box get squeezed and 1 nm differs between x and y).
            last_im = ax.imshow(dens, origin="lower", aspect="auto",
                                extent=extent, cmap="viridis", vmin=0, vmax=1)
            ax.add_patch(Rectangle((0.0, -CYL_H / 2), CYL_R, CYL_H, fill=False,
                                   edgecolor="cyan", ls=":", lw=0.8))
            any_data = True
    if not any_data:
        plt.close(fig)
        print(f"skip {stem}: no data/poredens_*.npz with '{key}'")
        return
    # Equal data scaling (1 nm identical on x and y) on every panel.
    xticks = np.arange(np.ceil(xlim[0]), np.floor(xlim[1]) + 1)
    for row in axes:
        for ax in row:
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.set_xticks(xticks)   # narrow panels else auto-drop the r=3 tick
            ax.set_aspect("equal", adjustable="box")
    for ax in axes[-1]:
        ax.set_xlabel("r (nm)")
    fig.suptitle(title)
    cbar = fig.colorbar(last_im, ax=axes, fraction=0.046, pad=0.04)
    cbar.set_label("normalized density")
    out = PLOTS / f"poredens_{stem}.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def main():
    PLOTS.mkdir(parents=True, exist_ok=True)
    for key, stem, title in SELECTIONS:
        plot_selection(key, stem, title)


if __name__ == "__main__":
    main()
