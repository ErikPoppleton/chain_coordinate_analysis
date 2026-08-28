#!/usr/bin/env python3
"""Panel painters for the main-text **pore-occupancy** figure (Figure 4).

``count_rows(bases, suffix)``      per-row integer count ceiling + the shared
                                   time axis, read from the nine
                                   ``poredens_counts_{ff}_{lipid}.npz``.
``draw_counts_cell(axt, axh, ...)``one cell of the occupancy grid: the
                                   occupancy-vs-time trajectory (wide) and the
                                   marginal count histogram (narrow, to its
                                   right, sharing the count axis), the three
                                   force fields overlaid.
``draw_snapshot(ax, png)``         a rendered PyMOL snapshot as an image panel.

The counts are the raw per-frame number of atoms inside the chain-coordinate CV
cylinder; ``pore_density.py`` writes both the full cylinder and its **middle
third** (the ``_mid`` keys, |z| <= 0.45 nm -- the pore's narrowest waist), which
is what this figure uses.  Histogram bins are **1 atom wide** and centred on the
integers, because the quantity is a count.
"""

from pathlib import Path

import numpy as np
import matplotlib.image as mpimg
from matplotlib.ticker import MaxNLocator

from lipiddefs import LIPIDS
from palette import FORCEFIELDS

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
SNAPSHOTS = ROOT / "snapshots"

# Row key -> y-axis label.  ``glycerol_C2`` is also in the npz but is left out
# of the main-text figure.
COUNT_LABELS = {
    "water": "water O count",
    "phosphorus": "phosphate P count",
    "glycerol_C2": "glycerol C2 count",
}


def load_counts():
    """``{(ff, lipid): npz}`` for every system that has a counts file."""
    out = {}
    for ff, _, _ in FORCEFIELDS:
        for lipid in LIPIDS:
            path = DATA / f"poredens_counts_{ff}_{lipid}.npz"
            if path.exists():
                out[(ff, lipid)] = np.load(path)
    return out


def count_rows(panels, bases, suffix="_mid"):
    """``(keys, row_hi, tmax_ns)`` for the requested rows.

    ``row_hi`` is the integer count ceiling per row, shared across the lipids so
    a row's three panels (and their histogram bins) are on one scale.
    """
    keys = {base: base + suffix for base in bases}
    row_hi = {}
    for base in bases:
        k = keys[base]
        vals = [int(npz[k].max()) for npz in panels.values() if k in npz.files]
        row_hi[base] = max(vals) if vals else 1
    tmax = max((npz["time_ps"].max() for npz in panels.values()),
               default=1000.0) / 1000.0
    return keys, row_hi, tmax


def draw_counts_cell(axt, axh, panels, base, lipid, key, hi, tmax,
                     xticks=None, verbose=False):
    """One occupancy cell: trajectory (``axt``) + marginal histogram (``axh``).

    ``axh`` must already share ``axt``'s y axis.  Returns the line handles, so
    the caller can build one shared legend.
    """
    edges = np.arange(-0.5, hi + 1.5, 1.0)      # 1-atom bins, integer-centred
    # In the middle-third slab the phosphate counts sit at 0 much of the time,
    # so drop ymin below zero to keep that line off the axis spine.
    ymin = -0.5 if base != "water" else 0
    axt.set_ylim(ymin, hi * 1.08)
    axt.set_xlim(0, tmax)
    if xticks is not None:
        axt.set_xticks(xticks)
    # The histogram's shared y ticks move to its right edge, out of the narrow
    # gap between the two axes; its lowest x tick is pruned so it cannot collide
    # with the trajectory's last one.
    axh.yaxis.set_ticks_position("right")
    axh.tick_params(labelleft=False, labelright=False)
    axh.xaxis.set_major_locator(MaxNLocator(nbins=2, prune="lower"))

    handles = []
    for ff, ff_label, colour in FORCEFIELDS:
        npz = panels.get((ff, lipid))
        if npz is None or key not in npz.files:
            continue
        y = npz[key]
        (line,) = axt.plot(npz["time_ps"] / 1000.0, y, color=colour, lw=0.6,
                           label=ff_label, rasterized=True)
        axh.hist(y, bins=edges, orientation="horizontal", histtype="step",
                 color=colour, density=True, lw=0.8)
        handles.append(line)
        if verbose:
            print(f"  {lipid:5s} {ff:8s} {base:11s} mean {y.mean():6.2f}  "
                  f"max {int(y.max()):3d}")
    return handles


def draw_snapshot(ax, path, verbose=False):
    """A rendered snapshot PNG as a bare image panel (no axes, no frame)."""
    path = Path(path)
    ax.set_axis_off()
    if not path.exists():
        print(f"note: {path.name} missing (blank panel)")
        return False
    ax.imshow(mpimg.imread(str(path)), interpolation="antialiased")
    if verbose:
        print(f"  snapshot {path.name}")
    return True


def snapshot_path(ff, lipid, which="last"):
    return SNAPSHOTS / f"{ff}_{lipid}_{which}.png"
