#!/usr/bin/env python3
"""Plot gmx density profiles as a 3x3 grid (lipids across columns DOPC/DMPC/DMPG,
force fields down rows CHARMM/AMBER/SLipids).

Reads ``data/{ff}_{lipid}_density.xvg`` -- **one replicate** per system, which is
all that is pulled down (9 groups from ``make_density_index.py``:
water, sodium, potassium, magnesium, calcium, phosphorus, glycerol_C2,
midpoint_C, terminal_C).  Each curve is **peak-normalised** (divided by its own maximum) so
the abundant water doesn't dwarf the sparse ions, and every panel's z-axis is
**shifted so the terminal_C peak sits at 0** (the bilayer centre, where the two
leaflets' tail ends meet).  → ``plots/density_profiles.png``.

Requires only numpy + matplotlib.
"""

import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from palette import FORCEFIELDS  # [(key, label, colour), ...]
from lipiddefs import LIPIDS

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
PLOTS = ROOT / "plots"


# Fixed group order + display colour.  Water blue, sodium purple, magnesium
# green, calcium teal, phosphorus orange; the three carbons in greys darkest
# (glycerol C2, nearest the head) -> lightest (terminal C, the tail end).
# Mg/Ca dropped: only 1-4 atoms, so peak-normalising just amplifies noise into
# an already-busy head-group region (Na, P, glycerol C2).
GROUPS = ["water", "sodium", "potassium", "phosphorus",
          "glycerol_C2", "midpoint_C", "terminal_C"]
COLORS = {
    "water": "#1f77b4",       # blue
    "sodium": "#9467bd",      # purple
    "potassium": "#d62728",   # red -- the DMPG counterion after the rebuild
    "magnesium": "#2ca02c",   # green
    "calcium": "#008080",     # teal
    "phosphorus": "#ff7f0e",  # orange
    "glycerol_C2": "0.0",     # black
    "midpoint_C": "0.45",     # mid grey
    "terminal_C": "0.7",      # light grey
}
LABELS = {
    "water": "water", "sodium": "Na$^+$", "potassium": "K$^+$",
    "magnesium": "Mg$^{2+}$",
    "calcium": "Ca$^{2+}$", "phosphorus": "P", "glycerol_C2": "glycerol C2",
    "midpoint_C": "midpoint C", "terminal_C": "terminal C",
}


def load_xvg(path):
    """Return (z_nm, {group_name: density}) from a gmx density .xvg, mapping
    columns to names via the ``@ sN legend "..."`` header lines."""
    names, rows = [], []
    for line in path.read_text().splitlines():
        m = re.match(r'@\s*s(\d+)\s+legend\s+"([^"]*)"', line)
        if m:
            names.append((int(m.group(1)), m.group(2)))
        elif line.startswith(("#", "@")) or not line.strip():
            continue
        else:
            rows.append([float(v) for v in line.split()])
    data = np.array(rows)
    ordered = [n for _, n in sorted(names)]
    cols = {name: data[:, i + 1] for i, name in enumerate(ordered)}
    return data[:, 0], cols


def main():
    PLOTS.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(FORCEFIELDS), len(LIPIDS), figsize=(13, 9),
                             sharex=True, sharey=True)
    any_data = False
    drawn = set()          # groups plotted in any panel, for the legend
    for r, (ff, ff_label, _) in enumerate(FORCEFIELDS):
        for c, lipid in enumerate(LIPIDS):
            ax = axes[r][c]
            if r == 0:
                ax.set_title(lipid)
            if c == 0:
                ax.set_ylabel(ff_label)
            path = DATA / f"{ff}_{lipid}_density.xvg"
            if not path.exists():
                continue
            z, cols = load_xvg(path)
            if "terminal_C" not in cols:
                print(f"skip {path.name}: no terminal_C column")
                continue
            any_data = True
            z0 = z - z[np.argmax(cols["terminal_C"])]  # centre on terminal_C peak
            for g in GROUPS:
                if g not in cols:
                    continue
                y = cols[g]
                peak = np.nanmax(y)
                y = y / peak if peak > 0 else y
                ax.plot(z0, y, color=COLORS[g], lw=1.2)
                drawn.add(g)
    if not any_data:
        plt.close(fig)
        print("skip density: no data/*_density.xvg files found")
        return
    for ax in axes[-1]:
        ax.set_xlabel("Distance from center (nm)")
    axes[0][0].set_ylim(0, 1.05)
    fig.supylabel("Normalized density")
    # Legend from every group actually drawn *anywhere*, in GROUPS order -- not
    # from one panel: the ion species differ by system (the DMPG systems carry
    # potassium and no sodium), so keying off the top-left panel silently
    # dropped K+ from the key while still plotting the curve.
    keys = [g for g in GROUPS if g in drawn]
    handles = [Line2D([], [], color=COLORS[g], lw=1.2) for g in keys]
    fig.legend(handles, [LABELS[g] for g in keys], loc="upper center",
               ncol=len(keys), bbox_to_anchor=(0.5, 0.99), fontsize=9)
    fig.tight_layout(rect=(0.02, 0, 1, 0.96))
    out = PLOTS / "density_profiles.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
