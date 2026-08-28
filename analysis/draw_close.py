#!/usr/bin/env python3
"""Panel painter for the main-text **pore-closure** figure (Figure 3).

``draw_close(ax, ff, lipid)`` overlays the ten ``close`` replicate trajectories
of one (force field, lipid) system -- the chain coordinate xi against time after
the umbrella restraint is released -- as a single-hue light->dark gradient
(``palette.CMAPS``), the same replicate colouring ``plot_close_pullx.py`` uses.

The traces are **strided while reading** (each ``*_close{n}_pullx.xvg`` holds
2.5 M points for a 5 ns run, and a 52 mm panel at 600 dpi is ~1200 px wide, so
even the default stride leaves the curve ~20x oversampled); parsing every float
would cost minutes for the 90 files.  They are drawn ``rasterized=True``, which
matplotlib inlines as base64 inside the SVG -- the figure stays monolithic and
small instead of carrying millions of vector segments.
"""

from pathlib import Path

import numpy as np

from palette import CMAPS

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

N_REP = 10           # close replicates per system
STRIDE = 100         # keep every Nth sample (2.5 M points is far more than a
                     # 1200 px panel can show)
LINEWIDTH = 0.25     # thinner than the house default so individual traces read
PS_TO_NS = 1e-3


def load_pullx(path, stride=STRIDE):
    """x (ns), y (xi) from a pullx .xvg, striding *while reading*."""
    xs, ys = [], []
    with open(path) as fh:
        for i, line in enumerate(fh):
            if not line or line[0] in "#@&\n":
                continue
            if i % stride:
                continue
            cols = line.split()
            xs.append(float(cols[0]))
            ys.append(float(cols[1]))
    return np.array(xs) * PS_TO_NS, np.array(ys)


def draw_close(ax, ff, lipid, stride=STRIDE, lw=LINEWIDTH, verbose=False):
    """The ten close trajectories of one system, dark (rep 1) -> light (rep 10)."""
    cmap = CMAPS[ff]
    shades = np.linspace(0.9, 0.3, N_REP)
    drawn = 0
    for i in range(N_REP):
        path = DATA / f"{ff}_{lipid}_close{i + 1}_pullx.xvg"
        if not path.exists():
            continue
        x, y = load_pullx(path, stride)
        ax.plot(x, y, color=cmap(shades[i]), lw=lw, rasterized=True)
        drawn += 1
    if verbose:
        print(f"  {lipid:5s} {ff:8s} {drawn}/{N_REP} replicates")
    elif drawn < N_REP:
        print(f"note: {ff}_{lipid} has {drawn}/{N_REP} close replicates")
    ax.set_xlim(0, None)
    return drawn
