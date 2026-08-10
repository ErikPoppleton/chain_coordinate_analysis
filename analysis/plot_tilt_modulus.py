#!/usr/bin/env python3
"""Plot the membrane lipid **tilt modulus** chi as a grouped bar chart -- lipids
along x (DOPC/DMPC/DMPG), the three force fields as coloured bars in the
``palette.py`` scheme (CHARMM blue / AMBER green / SLipids gold).

Reads ``data/tilt_moduli.json`` written by ``tilt_modulus.py`` (one entry per
``{ff}_{lipid}_{rep}`` with ``chi_kT``).  For each force field / lipid it takes
the mean chi over the replicates and, per the project convention, an error bar of
**+/- the population SD across the per-replicate chi values** (each replicate is
already a mean over lipids and frames, so the replicate spread is the honest
reproducibility measure; a single replicate gets no bar).  chi is in units of
k_B T / rad^2 (per-lipid tilt modulus, no area normalisation -- Khelashvili 2013).

Also writes ``data/tilt_moduli_summary.json`` (mean/sd/n per ff/lipid).

Requires only numpy + matplotlib.  ->  ``plots/tilt_modulus.png``.
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from palette import FORCEFIELDS  # [(key, label, colour), ...]

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
PLOTS = ROOT / "plots"

LIPIDS = ["DOPC", "DMPC", "DMPG"]

# All simulations are at T = 310 K.  chi is an energy per rad^2, so k_BT -> kJ/mol
# is a multiply by RT = R*T (R = 8.314e-3 kJ/mol/K).
TEMPERATURE_K = 310.0
RT_KJ_PER_MOL = 8.314462618e-3 * TEMPERATURE_K   # 2.577 kJ/mol at 310 K


def aggregate(results):
    """{(ff,lipid): [chi per rep]} -> {(ff,lipid): (mean, sd, n)}."""
    bykey = defaultdict(list)
    for r in results.values():
        bykey[(r["ff"], r["lipid"])].append(r["chi_kT"])
    agg = {}
    for k, chis in bykey.items():
        chis = np.array(chis, dtype=float)
        agg[k] = (float(chis.mean()),
                  float(chis.std(ddof=0)) if len(chis) > 1 else 0.0,
                  len(chis))
    return agg


def main():
    path = DATA / "tilt_moduli.json"
    if not path.exists():
        raise SystemExit("no %s -- run tilt_modulus.py first" % path)
    with open(path) as fh:
        results = json.load(fh)
    agg = aggregate(results)

    # summary file (both unit systems; conversion is x RT at 310 K)
    summary = {"%s_%s" % (ff, lip): {
                   "chi_kT": m, "sd_kT": s, "n_rep": n,
                   "chi_kJ_per_mol": m * RT_KJ_PER_MOL,
                   "sd_kJ_per_mol": s * RT_KJ_PER_MOL,
                   "T_K": TEMPERATURE_K}
               for (ff, lip), (m, s, n) in agg.items()}
    with open(DATA / "tilt_moduli_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)

    PLOTS.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(LIPIDS))
    nff = len(FORCEFIELDS)
    w = 0.8 / nff
    fig, ax = plt.subplots(figsize=(7, 4.5))
    any_data = False
    for i, (ff, label, colour) in enumerate(FORCEFIELDS):
        means, sds = [], []
        for lip in LIPIDS:
            m, s, n = agg.get((ff, lip), (np.nan, 0.0, 0))
            means.append(m)
            sds.append(s)
            if n:
                any_data = True
        offset = (i - (nff - 1) / 2.0) * w
        ax.bar(x + offset, means, w, yerr=sds, capsize=3, color=colour,
               label=label, edgecolor="black", linewidth=0.4)
    if not any_data:
        plt.close(fig)
        raise SystemExit("no tilt moduli to plot")
    ax.set_xticks(x)
    ax.set_xticklabels(LIPIDS)
    ax.set_ylabel(r"tilt modulus $\chi$  ($k_\mathrm{B}T\,\mathrm{rad}^{-2}$)")
    ax.set_title("Lipid tilt modulus by force field")
    ax.legend(frameon=False, loc="upper left")
    # Fixed y-limit with headroom, then a twin axis in kJ/mol/rad^2 (x RT, 310 K).
    ymax = max(m + s for (m, s, n) in agg.values() if n) * 1.15
    ax.set_ylim(0, ymax)
    ax2 = ax.twinx()
    ax2.set_ylim(0, ymax * RT_KJ_PER_MOL)
    ax2.set_ylabel(r"$\chi$  (kJ mol$^{-1}$ rad$^{-2}$, $T=310$ K)")
    fig.tight_layout()
    out = PLOTS / "tilt_modulus.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print("wrote %s" % out)
    print("wrote %s" % (DATA / "tilt_moduli_summary.json"))


if __name__ == "__main__":
    main()
