#!/usr/bin/env python3
"""Scatter every equilibrium membrane property against the **pore-nucleation
energy barrier** (the min->max range of that system's PMF profile), one panel
per property, in a 3-wide grid.

For each (force field, lipid) system the barrier is ``max(PMF) - min(PMF)`` read
from ``data/{ff}_{lipid}_profile.xvg`` (the full chain-coordinate PMF), and the
equilibrium properties (APL, thickness, D, S_CD, and the two tilt moduli) are
the per-system means reused from ``make_equilibrium_table.collect()`` with its
error bars: a **hierarchical 95% CI** (autocorrelation-aware) for the per-frame
observables APL / d_HC, and the SD across replicates for the fitted D / S_CD /
chi / kappa.  Points are **coloured by force field** (palette) and
**shaped by lipid**; a shared legend fills a leftover grid cell if one exists,
else it sits above the panels.

Requires numpy + matplotlib (+ the same data as the summary table + the PMFs).
->  ``plots/property_vs_barrier.png``.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from palette import FORCEFIELDS
from lipiddefs import LIPIDS
import make_equilibrium_table as met

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
PLOTS = ROOT / "plots"

LIPID_MARKERS = {"DOPC": "o", "DMPC": "s", "DMPG": "^"}

# Per-property y-axis labels (matplotlib mathtext; independent of the LaTeX
# table's units).  Order = panel order.
PROPS = [
    ("apl",   "APL (Å$^2$)"),
    ("dhc",   "$d_\\mathrm{HC}$ (nm)"),
    ("diff",  "$D$ (µm$^2$ s$^{-1}$)"),
    ("scd",   "$\\langle -S_\\mathrm{CD}\\rangle$"),
    ("chi",   "$\\chi$ (kJ mol$^{-1}$ rad$^{-2}$)"),
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


def barrier(ff, lipid):
    """Pore-nucleation barrier = max(PMF) - min(PMF) from the full profile."""
    path = DATA / f"{ff}_{lipid}_profile.xvg"
    if not path.exists():
        return None
    _, y = load_xvg(path)
    return float(y.max() - y.min())


def main():
    rows, _ = met.collect()                     # {(ff,lipid): {prop: (mean, sd)}}
    barriers = {(ff, lip): barrier(ff, lip)
                for ff, _, _ in FORCEFIELDS for lip in LIPIDS}

    print("pore-nucleation barriers (kJ/mol):")
    for ff, label, _ in FORCEFIELDS:
        for lip in LIPIDS:
            b = barriers[(ff, lip)]
            print(f"  {label:8s} {lip}: {'--' if b is None else '%.1f' % b}")

    PLOTS.mkdir(parents=True, exist_ok=True)
    ncol = 3
    nrow = -(-len(PROPS) // ncol)               # ceil
    fig, axes = plt.subplots(nrow, ncol, figsize=(13, 4.1 * nrow))
    axflat = axes.flatten()

    for ax, (key, plabel) in zip(axflat, PROPS):
        for ff, ff_label, color in FORCEFIELDS:
            for lip in LIPIDS:
                b = barriers[(ff, lip)]
                cell = rows.get((ff, lip), {}).get(key)
                if b is None or cell is None:
                    continue
                mean, err = cell
                # barrier on y; property on x with its error bar (hierarchical
                # 95% CI for APL/d_HC, replicate SD for the fitted quantities).
                ax.errorbar(mean, b, xerr=err, marker=LIPID_MARKERS[lip],
                            color=color, mec="black", mew=0.4, ms=7,
                            capsize=3, linestyle="none")
        ax.set_xlabel(plabel)
        ax.set_ylabel("pore-nucleation barrier (kJ mol$^{-1}$)")
        ax.grid(True, lw=0.3, alpha=0.4)

    # Leftover cells: put a combined legend in the first empty one, hide the rest.
    ff_handles = [Line2D([0], [0], marker="o", color=c, mec="black", mew=0.4,
                         ls="none", ms=8, label=l) for _, l, c in FORCEFIELDS]
    lip_handles = [Line2D([0], [0], marker=LIPID_MARKERS[lip], color="0.4",
                          mec="black", mew=0.4, ls="none", ms=8, label=lip)
                   for lip in LIPIDS]
    for ax in axflat[len(PROPS):]:
        ax.axis("off")
    if len(axflat) > len(PROPS):
        leg_ax = axflat[len(PROPS)]
        # Stack the two legends tight against the panel's centre line.
        l1 = leg_ax.legend(handles=ff_handles, title="Force field",
                           loc="lower center", bbox_to_anchor=(0.5, 0.52),
                           frameon=False)
        leg_ax.add_artist(l1)
        leg_ax.legend(handles=lip_handles, title="Lipid",
                      loc="upper center", bbox_to_anchor=(0.5, 0.48),
                      frameon=False)
        fig.suptitle("Equilibrium properties vs pore-nucleation energy barrier",
                     y=0.99)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
    else:
        # No leftover cell: title on top, legend as a strip just beneath it.
        fig.suptitle("Equilibrium properties vs pore-nucleation energy barrier",
                     y=1.0)
        fig.legend(handles=ff_handles + lip_handles, ncol=6,
                   loc="upper center", bbox_to_anchor=(0.5, 0.965), frameon=False)
        fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = PLOTS / "property_vs_barrier.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
