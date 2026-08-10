#!/usr/bin/env python3
"""Plot the per-frame **pore-cylinder occupancy** as a trajectory over time --
the raw number of water oxygens, phosphates, and glycerol C2 carbons inside the
chain-coordinate CV cylinder (radius 0.9 nm, height 2.7 nm, centred on the pore
centre), written by ``pore_density.py`` to
``data/poredens_counts_{ff}_{lipid}.npz``.

A 3x3 grid, lipids across columns (DOPC/DMPC/DMPG) and force fields down rows
(CHARMM/AMBER/SLipids); within each panel the three selections are overlaid as
lines vs time.  Water uses the left y-axis (there are far more oxygens than
lipid atoms); phosphate and C2 share a right y-axis.  Panels with no
``.npz`` are left blank.  ->  ``plots/poredens_counts.png``.

Requires only numpy + matplotlib.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from palette import FORCEFIELDS  # [(key, label, colour), ...]

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
PLOTS = ROOT / "plots"

LIPIDS = ["DOPC", "DMPC", "DMPG"]

# (npz key, legend label, colour, which y-axis: "left" for water else "right").
SERIES = [
    ("water", "water O", "#2171b5", "left"),
    ("phosphorus", "phosphate P", "#d4a017", "right"),
    ("glycerol_C2", "glycerol C2", "#0f7d3b", "right"),
]


def load(ff, lipid):
    path = DATA / f"poredens_counts_{ff}_{lipid}.npz"
    return np.load(path) if path.exists() else None


def main():
    PLOTS.mkdir(parents=True, exist_ok=True)

    # Gather panels and shared axis limits so rows/cols are comparable.
    panels = {}
    for ff, _, _ in FORCEFIELDS:
        for lipid in LIPIDS:
            npz = load(ff, lipid)
            if npz is not None:
                panels[(ff, lipid)] = npz
    if not panels:
        print("skip: no data/poredens_counts_*.npz files found")
        return

    def axis_max(axis):
        vals = [npz[k].max() for npz in panels.values()
                for k, _, _, ax in SERIES if ax == axis and k in npz.files]
        return max(vals) * 1.1 if vals else 1.0

    left_max, right_max = axis_max("left"), axis_max("right")
    tmax = max((npz["time_ps"].max() for npz in panels.values()), default=1.0) / 1000.0

    fig, axes = plt.subplots(len(FORCEFIELDS), len(LIPIDS), figsize=(11, 8),
                             sharex=True)
    handles, labels = [], []
    for r, (ff, ff_label, _) in enumerate(FORCEFIELDS):
        for c, lipid in enumerate(LIPIDS):
            ax = axes[r][c]
            axr = ax.twinx()
            ax.set_ylim(0, left_max)
            axr.set_ylim(0, right_max)
            ax.set_xlim(0, tmax)
            if r == 0:
                ax.set_title(lipid)
            if c == 0:
                ax.set_ylabel(f"{ff_label}\nwater O count")
            else:
                ax.tick_params(labelleft=False)
            if c == len(LIPIDS) - 1:
                axr.set_ylabel("phosphate / C2 count")
            else:
                axr.tick_params(labelright=False)
            if r == len(FORCEFIELDS) - 1:
                ax.set_xlabel("time (ns)")
            npz = panels.get((ff, lipid))
            if npz is None:
                axr.set_axis_off()
                continue
            t = npz["time_ps"] / 1000.0
            for key, label, colour, which in SERIES:
                if key not in npz.files:
                    continue
                target = ax if which == "left" else axr
                (line,) = target.plot(t, npz[key], color=colour, lw=0.8, label=label)
                if label not in labels:
                    handles.append(line)
                    labels.append(label)

    fig.suptitle("Pore-cylinder occupancy over time "
                 "(CV cylinder: r = 0.9 nm, h = 2.7 nm)")
    fig.legend(handles, labels, loc="upper right", frameon=False, ncol=3)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = PLOTS / "poredens_counts.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
