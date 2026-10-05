#!/usr/bin/env python3
"""Scatter two equilibrium membrane properties -- the hydrophobic-core thickness
d_HC and the Lipidator/ReSiS tilt modulus kappa_theta -- against the
**pore-nucleation energy barrier** (the min->max range of that system's PMF
profile), as a single 1x3 row: d_HC panel, kappa_theta panel, and a legend in the
third slot.

A trimmed companion to ``plot_property_vs_barrier.py`` (which shows all seven
properties); the property means and per-system barriers come from the same
helpers, so the two stay consistent.  Error bars: d_HC carries the hierarchical
95% CI (autocorrelation-aware) from ``collect()``, kappa_theta the SD across
replicates.  Points are **coloured by force field** (palette) and **shaped by
lipid**.

Requires numpy + matplotlib (+ the same data as the summary table + the PMFs).
->  ``plots/thick_kappa_vs_barrier.png``.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Manuscript sizing: 16.2 cm wide (journal figure), 14 pt text.  The two data
# panels are forced square (set_box_aspect below); the canvas is made taller so
# the square panels plus a two-row top legend fit at the fixed 16.2 cm width.
CM = 1.0 / 2.54
FIG_W_IN = 16.2 * CM
FIG_H_IN = 3.8
plt.rcParams.update({"font.size": 14})

from palette import FORCEFIELDS
from lipiddefs import LIPIDS
import make_equilibrium_table as met

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
PLOTS = ROOT / "plots"

LIPID_MARKERS = {"DOPC": "o", "DMPC": "s", "DMPG": "^"}

# Just the two requested properties (subset of plot_property_vs_barrier's PROPS).
PROPS = [
    ("dhc", "$d_\\mathrm{HC}$ (nm)"),
    ("kappa", "$\\kappa_\\theta$ (kJ mol$^{-1}$ nm$^{-2}$)"),
]


def load_xvg(path):
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


# barrier() lives in draw_pore_energy -- one definition, since it is subtler
# than it looks (the xi->0 edge can out-top the pore peak; see its docstring).
from draw_pore_energy import barrier  # noqa: E402


def main():
    rows, _ = met.collect()                     # {(ff,lipid): {prop: (mean, sd)}}
    barriers = {(ff, lip): barrier(ff, lip)
                for ff, _, _ in FORCEFIELDS for lip in LIPIDS}

    PLOTS.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(FIG_W_IN, FIG_H_IN))

    for ax, (key, plabel) in zip(axes, PROPS):
        ax.set_box_aspect(1)            # square plotting area
        for ff, ff_label, color in FORCEFIELDS:
            for lip in LIPIDS:
                b = barriers[(ff, lip)]
                cell = rows.get((ff, lip), {}).get(key)
                if b is None or cell is None:
                    continue
                mean, err = cell
                # barrier on y; property on x (d_HC: hierarchical 95% CI;
                # kappa_theta: replicate SD).
                ax.errorbar(mean, b, xerr=err, marker=LIPID_MARKERS[lip],
                            color=color, mec="black", mew=0.4, ms=7,
                            capsize=3, linestyle="none")
        ax.set_xlabel(plabel)
        ax.set_ylabel(r"$\Delta G$ (kJ/mol)")
        ax.grid(True, lw=0.3, alpha=0.4)

    # Combined legend on top: force fields (colour) then lipids (marker), two rows.
    ff_handles = [Line2D([0], [0], marker="o", color=c, mec="black", mew=0.4,
                         ls="none", ms=8, label=l) for _, l, c in FORCEFIELDS]
    lip_handles = [Line2D([0], [0], marker=LIPID_MARKERS[lip], color="0.4",
                          mec="black", mew=0.4, ls="none", ms=8, label=lip)
                   for lip in LIPIDS]
    # Two stacked rows: force-field colours on top, lipid shapes below (kept as
    # separate legends so the two categories don't interleave column-major).
    leg_kw = dict(ncol=3, loc="upper center", frameon=False,
                  columnspacing=1.2, handletextpad=0.4)
    l1 = fig.legend(handles=ff_handles, bbox_to_anchor=(0.5, 1.01), **leg_kw)
    fig.add_artist(l1)
    fig.legend(handles=lip_handles, bbox_to_anchor=(0.5, 0.93), **leg_kw)

    # Manual margins (not tight_layout, which fights set_box_aspect): leave the
    # top strip for the two-row legend; square panels centre in the region below.
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.15, top=0.82, wspace=0.35)
    out = PLOTS / "thick_kappa_vs_barrier.png"
    fig.savefig(out, dpi=300)   # no tight bbox: keep the width exactly 16.2 cm
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
