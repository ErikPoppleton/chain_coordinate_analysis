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
from lipiddefs import LIPIDS

# The chain-coordinate CV acts on a cylinder: radius 0.9 nm, height 2.7 nm,
# centred on the pore centre (z=0).  In the (r, z) maps that is the box
# r in [0, 0.9], z in [-1.35, 1.35].
CYL_R, CYL_H = 0.9, 2.7

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
PLOTS = ROOT / "plots"


# Shared colour scale is set from a high percentile of the pooled bins rather
# than the raw max: the innermost radial shells (r->0) have tiny volumes, so
# dividing counts by them makes a handful of noise spikes that would otherwise
# grab the whole top of the colour range (max ~= 2x the bulk-water density).
PEAK_PCTL = 99.5

# Scale applied to the whole figure -- canvas, line widths, tick sizes and
# paddings -- so the layout keeps the unscaled design's proportions, just
# smaller.  At 1.0 the cropped PNG is ~5.8 x 11.1 in, too tall for a page; this
# brings it to 8.5 in tall.  Fonts do NOT follow it below MIN_FONT_PT: scaling
# them too would put the 10 pt body text at ~7.7 pt, so they are clamped and
# end up relatively larger than in the unscaled design.
SCALE = 0.765
MIN_FONT_PT = 8.0

# rcParams measured in points, scaled together with the canvas.
_POINT_PARAMS = ["font.size", "axes.titlepad", "axes.labelpad",
                 "axes.linewidth", "lines.linewidth", "patch.linewidth"] + [
    "%s.%s.%s" % (ax, which, attr) for ax in ("xtick", "ytick")
    for which in ("major", "minor") for attr in ("size", "width", "pad")]


def _scaled_rc(scale):
    rc = {k: plt.rcParamsDefault[k] * scale for k in _POINT_PARAMS}
    rc["font.size"] = max(plt.rcParamsDefault["font.size"] * scale, MIN_FONT_PT)
    rc["savefig.pad_inches"] = plt.rcParamsDefault["savefig.pad_inches"] * scale
    return rc


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


def plot_selection(key, stem, title, region=None):
    with plt.rc_context(_scaled_rc(SCALE)):
        _plot_selection(key, stem, title, region)


def _plot_selection(key, stem, title, region=None):
    """Draw the 3x3 (r, z) density grid for one selection.

    Normalisation is **shared across all nine panels** (one peak per atom-type
    figure) so colours are comparable between panels -- the colourbar is a common
    absolute scale, which is the point: it lets the water-in-pore density be
    compared between force fields.  The shared peak is a high percentile
    (``PEAK_PCTL``) of the pooled bins, not the raw max, to ignore the r->0
    tiny-shell noise spikes (see ``PEAK_PCTL``).  ``region=(r_hi, z_half)``
    restricts the view to the chain-coordinate cylinder (r in [0, r_hi], z in
    [-z_half, z_half]) and takes that shared peak *within the region* (a separate,
    region-specific scale, so the zoomed content still uses the full colour
    range); ``region=None`` shows the whole map (peak over the whole map, all
    panels) with the CV cylinder outlined."""
    # First pass: gather every available panel and the shared peak.  For the
    # zoom, the peak is taken within the region only (still shared across panels).
    panels = {}
    for ff, _, _ in FORCEFIELDS:
        for lipid in LIPIDS:
            npz = load(ff, lipid)
            if npz is None or key not in npz.files:
                continue
            r_edges, z_edges = npz["r_edges"], npz["z_edges"]
            panels[(ff, lipid)] = (npz[key], r_edges, z_edges)
    if not panels:
        print(f"skip {stem}: no data/poredens_*.npz with '{key}'")
        return

    if region is None:
        pooled = [d.ravel() for d, _, _ in panels.values()]
        r0 = next(iter(panels.values()))
        xlim = (r0[1][0], r0[1][-1])
        ylim = (r0[2][0], r0[2][-1])
    else:
        r_hi, z_half = region
        xlim, ylim = (0.0, r_hi), (-z_half, z_half)
        pooled = []
        for d, r_edges, z_edges in panels.values():
            rc = 0.5 * (r_edges[:-1] + r_edges[1:])
            zc = 0.5 * (z_edges[:-1] + z_edges[1:])
            pooled.append(d[np.ix_(np.abs(zc) <= z_half, rc <= r_hi)].ravel())
    peak = np.nanpercentile(np.concatenate(pooled), PEAK_PCTL)
    if peak <= 0:
        peak = 1.0

    # Second pass: draw, every panel divided by the one shared peak.
    fig, axes = plt.subplots(len(FORCEFIELDS), len(LIPIDS), figsize=(6 * SCALE, 13 * SCALE),
                             sharex=True, sharey=True)
    last_im = None
    for r, (ff, ff_label, _) in enumerate(FORCEFIELDS):
        for c, lipid in enumerate(LIPIDS):
            ax = axes[r][c]
            if r == 0:
                ax.set_title(lipid)
            if c == 0:
                ax.set_ylabel(f"{ff_label}\nz (nm)")
            if (ff, lipid) not in panels:
                continue
            dens, r_edges, z_edges = panels[(ff, lipid)]
            extent = [r_edges[0], r_edges[-1], z_edges[0], z_edges[-1]]
            # aspect="auto" here; equal scaling is enforced on *all* axes below
            # so empty panels match the data panels (shared axes otherwise let
            # the box get squeezed and 1 nm differs between x and y).
            last_im = ax.imshow(dens / peak, origin="lower", aspect="auto",
                                extent=extent, cmap="viridis", vmin=0, vmax=1)
            if region is None:  # outline the CV cylinder; in zoom it IS the frame
                ax.add_patch(Rectangle((0.0, -CYL_H / 2), CYL_R, CYL_H, fill=False,
                                       edgecolor="magenta", ls=":", lw=0.8 * SCALE))
    # Equal data scaling (1 nm identical on x and y) on every panel.
    if region is None:
        xticks = np.arange(np.ceil(xlim[0]), np.floor(xlim[1]) + 1)
    else:
        # 0 .. r_hi in 2 steps: at the 8 pt floor, four labels (0.0 0.3 0.6
        # 0.9) nearly touch inside a 0.9 nm-wide panel.
        xticks = np.linspace(xlim[0], xlim[1], 3)
    for row in axes:
        for ax in row:
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.set_xticks(xticks, ["%g" % x for x in xticks])  # explicit: narrow
            # panels otherwise auto-drop the edge tick; %g prints 0 / 0.45 / 0.9
            ax.set_aspect("equal", adjustable="box")
    for ax in axes[-1]:
        ax.set_xlabel("r (nm)")
    cbar = fig.colorbar(last_im, ax=axes, fraction=0.046, pad=0.04)
    cbar.set_label("normalized density")
    # The equal-aspect panels are far shorter than the 13 in canvas, so a
    # default suptitle (pinned to the canvas top) floats well above them.  Put
    # it just above the top row's column titles instead; the tight bbox then
    # crops the empty canvas above it.
    fig.canvas.draw()
    inv = fig.transFigure.inverted()
    top = max(inv.transform(ax.get_tightbbox().corners())[:, 1].max()
              for ax in axes[0])
    fig.suptitle(title, y=top + 0.006, va="bottom")
    out = PLOTS / f"poredens_{stem}.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def main():
    PLOTS.mkdir(parents=True, exist_ok=True)
    # Full-map figures for every selection.
    for key, stem, title in SELECTIONS:
        plot_selection(key, stem, title)
    # Zoomed figures restricted to the CV cylinder, water + phosphate only.
    for key, stem, title in SELECTIONS[:2]:
        plot_selection(key, f"zoom_{stem}", title.replace("around the pore", "in the CV cylinder"),
                       region=(CYL_R, CYL_H / 2))


if __name__ == "__main__":
    main()
