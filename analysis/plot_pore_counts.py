#!/usr/bin/env python3
"""Plot the per-frame **pore-cylinder occupancy** as a trajectory over time --
the raw number of water oxygens, phosphates, and glycerol C2 carbons inside the
chain-coordinate CV cylinder, written by ``pore_density.py`` to
``data/poredens_counts_{ff}_{lipid}.npz``.

Two figure sets are produced, one per z-slab counted by ``pore_density.py``:
  * ``plots/poredens_counts.png``        -- the **full** CV cylinder (r = 0.9 nm,
    h = 2.7 nm, |z| <= 1.35 nm);
  * ``plots/poredens_counts_middle.png`` -- its **middle third** (|z| <= 0.45 nm,
    the central 9 A -- the pore's narrowest waist), from the ``{name}_mid`` keys.

Each is a 3x3 grid with **atom type down the rows** (water / phosphate /
glycerol C2) and **lipid across the columns** (DOPC/DMPC/DMPG).  Every cell is a
pair of axes sharing the count (y) axis: a wide occupancy-vs-time trajectory
(3/4 width) and, to its right, a narrow **vertical histogram** (1/4 width) that
aggregates that trajectory into the marginal distribution of the count -- one
outline per force field, with **1-atom (integer) bins**.  Within each panel the
three force fields overlay in the ``palette`` scheme (CHARMM blue / AMBER green
/ SLipids gold); each row (atom type) shares a y-scale.  Panels with no data are
left blank.

Requires only numpy + matplotlib.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.ticker import MaxNLocator

from palette import FORCEFIELDS  # [(key, label, colour), ...]
from lipiddefs import LIPIDS

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
PLOTS = ROOT / "plots"


# Rows: (base npz key, y-axis label).  One row per atom type.
ROWS = [
    ("water", "water O count"),
    ("phosphorus", "phosphate P count"),
    ("glycerol_C2", "glycerol C2 count"),
]


def load(ff, lipid):
    path = DATA / f"poredens_counts_{ff}_{lipid}.npz"
    return np.load(path) if path.exists() else None


def plot_counts(panels, suffix, title, out_stem):
    """Draw the atom-type x lipid grid for one z-slab (``suffix`` = '' full
    cylinder, '_mid' middle third).  Each cell is a trajectory + a vertical,
    1-atom-binned histogram of the same data, overlaying the force fields."""
    keys = {base: base + suffix for base, _ in ROWS}
    if not any(keys[b] in npz.files for b, _ in ROWS for npz in panels.values()):
        print(f"skip {out_stem}: no '{suffix or 'full'}' count arrays in the npz files")
        return

    # Per-row (atom-type) integer count max -> shared y-scale + histogram bins.
    row_hi = {}
    for base, _ in ROWS:
        k = keys[base]
        vals = [int(npz[k].max()) for npz in panels.values() if k in npz.files]
        row_hi[base] = max(vals) if vals else 1
    tmax = max((npz["time_ps"].max() for npz in panels.values()), default=1000.0) / 1000.0

    fig = plt.figure(figsize=(11, 8))
    outer = GridSpec(len(ROWS), len(LIPIDS), figure=fig, left=0.07, right=0.985,
                     top=0.87, bottom=0.08, wspace=0.34, hspace=0.18)
    xticks = np.arange(50, tmax + 1, 50)   # keep the 200 ns endpoint, drop 0;
    handles, labels = [], []               # the histogram's 0.00 tick is pruned

    for r, (base, ylabel) in enumerate(ROWS):
        k = keys[base]
        hi = row_hi[base]
        edges = np.arange(-0.5, hi + 1.5, 1.0)   # 1-atom bins centred on integers
        # In the middle-third figure the phosphate/glycerol counts often sit at 0,
        # so drop ymin below 0 there to keep the y=0 line off the axis spine.
        ymin = -0.5 if (suffix == "_mid" and base != "water") else 0
        ylim = (ymin, hi * 1.08)
        for c, lipid in enumerate(LIPIDS):
            inner = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[r, c],
                                            width_ratios=[3, 1], wspace=0.05)
            axt = fig.add_subplot(inner[0])
            axh = fig.add_subplot(inner[1], sharey=axt)
            axt.set_ylim(*ylim)
            axt.set_xlim(0, tmax)
            axt.set_xticks(xticks)
            # Move the histogram's shared-y ticks to its right edge so they don't
            # clutter the narrow gap between the trajectory and the histogram.
            axh.yaxis.set_ticks_position("right")
            axh.tick_params(labelleft=False, labelright=False)
            axh.xaxis.set_major_locator(MaxNLocator(nbins=2, prune="lower"))
            if r == 0:
                axt.set_title(lipid)
            if c == 0:
                axt.set_ylabel(ylabel)
            if r == len(ROWS) - 1:
                axt.set_xlabel("time (ns)")
                axh.set_xlabel("density")
            else:
                axt.tick_params(labelbottom=False)
                axh.tick_params(labelbottom=False)
            for ff, ff_label, colour in FORCEFIELDS:
                npz = panels.get((ff, lipid))
                if npz is None or k not in npz.files:
                    continue
                (line,) = axt.plot(npz["time_ps"] / 1000.0, npz[k],
                                   color=colour, lw=0.8, label=ff_label)
                axh.hist(npz[k], bins=edges, orientation="horizontal",
                         histtype="step", color=colour, density=True, lw=0.9)
                if ff_label not in labels:
                    handles.append(line)
                    labels.append(ff_label)

    fig.suptitle(title, y=0.99)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.955),
               frameon=False, ncol=3)
    out = PLOTS / f"{out_stem}.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"wrote {out}")


def main():
    PLOTS.mkdir(parents=True, exist_ok=True)
    panels = {(ff, lip): npz for ff, _, _ in FORCEFIELDS for lip in LIPIDS
              if (npz := load(ff, lip)) is not None}
    if not panels:
        print("skip: no data/poredens_counts_*.npz files found")
        return
    plot_counts(panels, "", "Pore-cylinder occupancy over time "
                "(CV cylinder: r = 0.9 nm, h = 2.7 nm)", "poredens_counts")
    plot_counts(panels, "_mid", "Pore-cylinder occupancy over time "
                "(middle third: r = 0.9 nm, |z| <= 0.45 nm)", "poredens_counts_middle")


if __name__ == "__main__":
    main()
