#!/usr/bin/env python3
"""Plot the membrane lipid **tilt modulus** chi as a grouped bar chart -- lipids
along x (DOPC/DMPC/DMPG), the three force fields as coloured bars in the
``palette.py`` scheme (CHARMM blue / AMBER green / SLipids gold).

Reads ``data/tilt_moduli.json`` written by ``tilt_modulus.py`` (one entry per
``{ff}_{lipid}_{rep}`` with ``chi_kT`` and a block-averaged within-run SEM
``chi_block_sem``).  For each force field / lipid the error bar is a
**hierarchical 95% CI** (``autocorr.hierarchical_ci_from_estimates``): it
combines each replicate's within-run block SEM with the between-replicate spread
(``max(between^2, within^2)/n``), so well-reproduced systems get a tight,
pooled CI and irreproducible ones (AMBER DMPG) an honestly wide one.  chi is in
k_B T / rad^2 (per-lipid tilt modulus, no area normalisation -- Khelashvili 2013).

Also writes ``data/tilt_moduli_summary.json`` (mean / SD / 95% CI / n per ff/lipid).

Requires numpy + matplotlib + autocorr.  ->  ``plots/tilt_modulus.png``.
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import autocorr
from palette import FORCEFIELDS  # [(key, label, colour), ...]
from lipiddefs import LIPIDS

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
PLOTS = ROOT / "plots"


# All simulations are at T = 310 K.  chi is an energy per rad^2, so k_BT -> kJ/mol
# is a multiply by RT = R*T (R = 8.314e-3 kJ/mol/K).
TEMPERATURE_K = 310.0
RT_KJ_PER_MOL = 8.314462618e-3 * TEMPERATURE_K   # 2.577 kJ/mol at 310 K


def aggregate(results):
    """{(ff,lipid): (mean, sd, ci_half, n)} in k_BT/rad^2 -- ci_half is the
    hierarchical 95% CI from each rep's chi + block SEM (falls back to the SD
    across reps if no block SEMs are present)."""
    bykey = defaultdict(list)
    for r in results.values():
        bykey[(r["ff"], r["lipid"])].append(r)
    agg = {}
    for k, recs in bykey.items():
        chis = np.array([r["chi_kT"] for r in recs], dtype=float)
        ci = autocorr.hierarchical_ci_from_estimates(
            [r["chi_kT"] for r in recs],
            [r.get("chi_block_sem", float("nan")) for r in recs],
            [r.get("n_blocks") for r in recs])
        agg[k] = (float(chis.mean()),
                  float(chis.std(ddof=0)) if len(chis) > 1 else 0.0,
                  float(ci["ci_half"]) if ci else 0.0,
                  len(chis))
    return agg


def main():
    path = DATA / "tilt_moduli.json"
    if not path.exists():
        raise SystemExit("no %s -- run tilt_modulus.py first" % path)
    with open(path) as fh:
        results = json.load(fh)
    agg = aggregate(results)

    # summary file (both unit systems; conversion is x RT at 310 K).  ci_k* is
    # the hierarchical 95% CI (block SEM + replicate spread); sd_* kept too.
    summary = {"%s_%s" % (ff, lip): {
                   "chi_kT": m, "sd_kT": s, "ci95_kT": c, "n_rep": n,
                   "chi_kJ_per_mol": m * RT_KJ_PER_MOL,
                   "sd_kJ_per_mol": s * RT_KJ_PER_MOL,
                   "ci95_kJ_per_mol": c * RT_KJ_PER_MOL,
                   "T_K": TEMPERATURE_K}
               for (ff, lip), (m, s, c, n) in agg.items()}
    with open(DATA / "tilt_moduli_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)

    PLOTS.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(LIPIDS))
    nff = len(FORCEFIELDS)
    w = 0.8 / nff
    fig, ax = plt.subplots(figsize=(7, 4.5))
    any_data = False
    for i, (ff, label, colour) in enumerate(FORCEFIELDS):
        means, cis = [], []
        for lip in LIPIDS:
            m, s, c, n = agg.get((ff, lip), (np.nan, 0.0, 0.0, 0))
            means.append(m * RT_KJ_PER_MOL)   # k_BT/rad^2 -> kJ/mol/rad^2
            cis.append(c * RT_KJ_PER_MOL)     # hierarchical 95% CI half-width
            if n:
                any_data = True
        offset = (i - (nff - 1) / 2.0) * w
        ax.bar(x + offset, means, w, yerr=cis, capsize=3, color=colour,
               label=label, edgecolor="black", linewidth=0.4)
    if not any_data:
        plt.close(fig)
        raise SystemExit("no tilt moduli to plot")
    ax.set_xticks(x)
    ax.set_xticklabels(LIPIDS)
    ax.set_ylabel(r"tilt modulus $\chi$  (kJ mol$^{-1}\,\mathrm{rad}^{-2}$, $T=310$ K)")
    ax.set_title("Lipid tilt modulus by force field")
    ax.legend(frameon=False, loc="upper left")
    # Primary axis in kJ/mol/rad^2 (= chi[k_BT/rad^2] x RT at 310 K); secondary
    # twin axis keeps the underlying k_BT/rad^2 scale for reference.
    ymax_kT = max(m + c for (m, s, c, n) in agg.values() if n) * 1.15
    ax.set_ylim(0, ymax_kT * RT_KJ_PER_MOL)
    ax2 = ax.twinx()
    ax2.set_ylim(0, ymax_kT)
    ax2.set_ylabel(r"$\chi$  ($k_\mathrm{B}T\,\mathrm{rad}^{-2}$)")
    fig.tight_layout()
    out = PLOTS / "tilt_modulus.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print("wrote %s" % out)
    print("wrote %s" % (DATA / "tilt_moduli_summary.json"))


if __name__ == "__main__":
    main()
