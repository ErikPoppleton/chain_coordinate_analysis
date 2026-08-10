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
glycerol C2) and **lipid across the columns** (DOPC/DMPC/DMPG).  Within each
panel the three force fields overlay as lines coloured by the ``palette``
force-field scheme (CHARMM blue / AMBER green / SLipids gold) -- so the top-left
panel is water occupancy for DOPC with one line per force field.  Each row (atom
type) shares a y-scale; panels with no data are left blank.

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
    cylinder, '_mid' middle third), overlaying the force fields per panel."""
    keys = {base: base + suffix for base, _ in ROWS}
    # Per-row (atom-type) shared y-max, and a shared x (time) range.
    row_max = {}
    for base, _ in ROWS:
        k = keys[base]
        vals = [npz[k].max() for npz in panels.values() if k in npz.files]
        row_max[base] = max(vals) * 1.1 if vals else 1.0
    if not any(keys[b] in npz.files for b, _ in ROWS for npz in panels.values()):
        print(f"skip {out_stem}: no '{suffix or 'full'}' count arrays in the npz files")
        return
    tmax = max((npz["time_ps"].max() for npz in panels.values()), default=1000.0) / 1000.0

    fig, axes = plt.subplots(len(ROWS), len(LIPIDS), figsize=(11, 8), sharex=True)
    handles, labels = [], []
    for r, (base, ylabel) in enumerate(ROWS):
        k = keys[base]
        for c, lipid in enumerate(LIPIDS):
            ax = axes[r][c]
            ax.set_ylim(0, row_max[base])
            ax.set_xlim(0, tmax)
            if r == 0:
                ax.set_title(lipid)
            if c == 0:
                ax.set_ylabel(ylabel)
            else:
                ax.tick_params(labelleft=False)
            if r == len(ROWS) - 1:
                ax.set_xlabel("time (ns)")
            for ff, ff_label, colour in FORCEFIELDS:
                npz = panels.get((ff, lipid))
                if npz is None or k not in npz.files:
                    continue
                (line,) = ax.plot(npz["time_ps"] / 1000.0, npz[k],
                                  color=colour, lw=0.8, label=ff_label)
                if ff_label not in labels:
                    handles.append(line)
                    labels.append(ff_label)

    fig.suptitle(title, y=0.99)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.955),
               frameon=False, ncol=3)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
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
