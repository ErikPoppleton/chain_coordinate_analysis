#!/usr/bin/env python3
"""Plot the Lipidator ReSiS bilayer **tilt modulus** as a grouped bar chart --
lipids on x (DOPC/DMPC/DMPG), force fields as coloured bars in the ``palette``
scheme.  Reads ``data/lipidator_moduli.json`` (per-replicate) written by
``run_lipidator.py``; bars are the mean over replicates and error bars are the
**population SD across the per-replicate values** (project convention).

ReSiS reports the area-normalised modulus in **k_BT/nm^2**; the plot's primary
axis is **kJ/mol/nm^2** (= k_BT/nm^2 x RT at 310 K), with a secondary twin axis
keeping the k_BT/nm^2 scale.  This is the area-normalised analogue of the
per-lipid k_BT/rad^2 from the ost_pymodules pipeline.

Requires numpy + matplotlib.  ->  ``plots/lipidator_tilt_modulus.png``.
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from palette import FORCEFIELDS

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
PLOTS = ROOT / "plots"
LIPIDS = ["DOPC", "DMPC", "DMPG"]

# All simulations are at T = 310 K.  kappa is an energy per nm^2, so k_BT -> kJ/mol
# is a multiply by RT = R*T (R = 8.314e-3 kJ/mol/K).
TEMPERATURE_K = 310.0
RT_KJ_PER_MOL = 8.314462618e-3 * TEMPERATURE_K   # 2.577 kJ/mol at 310 K


def main():
    path = DATA / "lipidator_moduli.json"
    if not path.exists():
        raise SystemExit("no %s -- run run_lipidator.py first" % path)
    with open(path) as fh:
        results = json.load(fh)

    by = defaultdict(list)
    for r in results.values():
        by[(r["ff"], r["lipid"])].append(r["tilt_kt_per_nm2"])
    agg = {k: (float(np.mean(v)), float(np.std(v)) if len(v) > 1 else 0.0, len(v))
           for k, v in by.items()}

    # (re)write summary in both unit systems (conversion is x RT at 310 K)
    summary = {"%s_%s" % (ff, lip): {
                   "tilt_kt_per_nm2": m, "sd_kt_per_nm2": s, "n_rep": n,
                   "tilt_kJ_per_mol_nm2": m * RT_KJ_PER_MOL,
                   "sd_kJ_per_mol_nm2": s * RT_KJ_PER_MOL,
                   "T_K": TEMPERATURE_K}
               for (ff, lip), (m, s, n) in agg.items()}
    with open(DATA / "lipidator_moduli_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)

    PLOTS.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(LIPIDS))
    nff = len(FORCEFIELDS)
    w = 0.8 / nff
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ymax_kT = 0.0
    for i, (ff, label, colour) in enumerate(FORCEFIELDS):
        means_kT = [agg.get((ff, lip), (np.nan, 0.0, 0))[0] for lip in LIPIDS]
        sds_kT = [agg.get((ff, lip), (np.nan, 0.0, 0))[1] for lip in LIPIDS]
        ymax_kT = max(ymax_kT, *[m + s for m, s in zip(means_kT, sds_kT) if m == m])
        means = [m * RT_KJ_PER_MOL for m in means_kT]   # k_BT/nm^2 -> kJ/mol/nm^2
        sds = [s * RT_KJ_PER_MOL for s in sds_kT]
        ax.bar(x + (i - (nff - 1) / 2.0) * w, means, w, yerr=sds, capsize=3,
               color=colour, label=label, edgecolor="black", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(LIPIDS)
    ax.set_ylabel(r"tilt modulus  $\kappa_\theta$  (kJ mol$^{-1}\,\mathrm{nm}^{-2}$, $T=310$ K)")
    ax.set_title("Lipid tilt modulus by force field (Lipidator / ReSiS)")
    ax.set_ylim(0, ymax_kT * RT_KJ_PER_MOL * 1.15)
    ax.legend(frameon=False, loc="upper left")
    ax2 = ax.twinx()
    ax2.set_ylim(0, ymax_kT * 1.15)
    ax2.set_ylabel(r"$\kappa_\theta$  ($k_\mathrm{B}T\,\mathrm{nm}^{-2}$)")
    fig.tight_layout()
    out = PLOTS / "lipidator_tilt_modulus.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print("wrote %s" % out)


if __name__ == "__main__":
    main()
