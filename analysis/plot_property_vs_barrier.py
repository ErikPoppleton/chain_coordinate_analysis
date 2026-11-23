#!/usr/bin/env python3
"""Scatter every equilibrium membrane property against the **pore-nucleation
energy barrier** (the climb from the PMF basin to its pore-side peak), one
panel per property, in a 3-wide grid.

For each (force field, lipid) system the barrier is
``draw_pore_energy.barrier()`` of ``data/{ff}_{lipid}_profile.xvg`` (the full
chain-coordinate PMF; *not* max - min, see its docstring), and the equilibrium
properties (APL, d_HC, D and the two tilt moduli) are the per-system means
reused from ``make_equilibrium_table.collect()`` with its error bars: a
**hierarchical 95% CI** (autocorrelation-aware) for APL / d_HC / chi / kappa,
and the SD across replicates for D.  The chain-averaged g_lomepro <-S_CD> is
not in the table, so it is read here (``scd_rep_scalars``) and carries the SD
across replicates too.  Points are **coloured by force field** (palette) and
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
from lipiddefs import LIPIDS, REPS
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


def scd_rep_scalars(ff, lipid):
    """Chain-averaged -S_CD per rep, pooling both tails (col 2 of each .dat)."""
    out = []
    for n in REPS:
        vals = []
        for sn in (1, 2):
            f = DATA / ("%s_%s_%d_scd_sn%d.dat" % (ff, lipid, n, sn))
            if f.exists():
                arr = np.loadtxt(f)
                if arr.ndim == 1:
                    arr = arr[None, :]
                vals.append(arr[:, 1])            # col 2 = mean -S_CD per carbon
        out.append(float(np.concatenate(vals).mean()) if vals else None)
    return out


# barrier() lives in draw_pore_energy -- one definition, since it is subtler
# than it looks (the xi->0 edge can out-top the pore peak; see its docstring).
from draw_pore_energy import barrier  # noqa: E402


def main():
    rows, _ = met.collect()                     # {(ff,lipid): {prop: (mean, err)}}
    for (ff, lip), cell in rows.items():
        cell["scd"] = met.summarise_reps(scd_rep_scalars(ff, lip))
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
                # 95% CI for APL/d_HC/chi/kappa, replicate SD for D/S_CD).
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
