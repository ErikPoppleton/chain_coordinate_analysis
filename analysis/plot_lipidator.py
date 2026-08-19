#!/usr/bin/env python3
"""Plot the Lipidator ReSiS bilayer **tilt modulus** as a grouped bar chart --
lipids on x (DOPC/DMPC/DMPG), force fields as coloured bars in the ``palette``
scheme.  Error bars are a **hierarchical 95% CI**
(``autocorr.hierarchical_ci_from_estimates``): when ``data/lipidator_blocks.json``
is present (per-replicate kappa + block-averaged within-run SEM from
``run_lipidator_blocks.py``) it combines each run's block SEM with the
between-replicate spread; otherwise it falls back to the SD across replicates
from ``data/lipidator_moduli.json``.

ReSiS reports the area-normalised modulus in **k_BT/nm^2**; the plot's primary
axis is **kJ/mol/nm^2** (= k_BT/nm^2 x RT at 310 K), with a secondary twin axis
keeping the k_BT/nm^2 scale.  This is the area-normalised analogue of the
per-lipid k_BT/rad^2 from the ost_pymodules pipeline.

Requires numpy + matplotlib + autocorr.  ->  ``plots/lipidator_tilt_modulus.png``.
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import autocorr
from palette import FORCEFIELDS
from lipiddefs import LIPIDS

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
PLOTS = ROOT / "plots"

# All simulations are at T = 310 K.  kappa is an energy per nm^2, so k_BT -> kJ/mol
# is a multiply by RT = R*T (R = 8.314e-3 kJ/mol/K).
TEMPERATURE_K = 310.0
RT_KJ_PER_MOL = 8.314462618e-3 * TEMPERATURE_K   # 2.577 kJ/mol at 310 K


def aggregate():
    """{(ff,lipid): (mean, sd, ci_half, n)} in k_BT/nm^2.  Prefers the block
    JSON (hierarchical 95% CI from kappa + block SEM); else SD-only from the
    plain per-rep moduli JSON."""
    blocks = DATA / "lipidator_blocks.json"
    if blocks.exists():
        recs = json.load(open(blocks))
        by = defaultdict(list)
        for r in recs.values():
            by[(r["ff"], r["lipid"])].append(r)
        agg = {}
        for k, rs in by.items():
            vals = np.array([r["kappa_full"] for r in rs], dtype=float)
            ci = autocorr.hierarchical_ci_from_estimates(
                [r["kappa_full"] for r in rs],
                [r.get("kappa_block_sem", float("nan")) for r in rs],
                [r.get("n_blocks") for r in rs])
            agg[k] = (float(vals.mean()),
                      float(vals.std(ddof=0)) if len(vals) > 1 else 0.0,
                      float(ci["ci_half"]) if ci else 0.0, len(vals))
        return agg
    path = DATA / "lipidator_moduli.json"
    if not path.exists():
        raise SystemExit("no lipidator_blocks.json or lipidator_moduli.json -- "
                         "run run_lipidator.py (and _blocks) first")
    by = defaultdict(list)
    for r in json.load(open(path)).values():
        by[(r["ff"], r["lipid"])].append(r["tilt_kt_per_nm2"])
    return {k: (float(np.mean(v)), float(np.std(v)) if len(v) > 1 else 0.0,
                float(np.std(v)) if len(v) > 1 else 0.0, len(v))
            for k, v in by.items()}       # no block SEM -> CI falls back to SD


def main():
    agg = aggregate()

    # (re)write summary in both unit systems (conversion is x RT at 310 K).
    # ci95_* is the hierarchical 95% CI (block SEM + replicate spread).
    summary = {"%s_%s" % (ff, lip): {
                   "tilt_kt_per_nm2": m, "sd_kt_per_nm2": s, "ci95_kt_per_nm2": c,
                   "n_rep": n,
                   "tilt_kJ_per_mol_nm2": m * RT_KJ_PER_MOL,
                   "sd_kJ_per_mol_nm2": s * RT_KJ_PER_MOL,
                   "ci95_kJ_per_mol_nm2": c * RT_KJ_PER_MOL,
                   "T_K": TEMPERATURE_K}
               for (ff, lip), (m, s, c, n) in agg.items()}
    with open(DATA / "lipidator_moduli_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)

    PLOTS.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(LIPIDS))
    nff = len(FORCEFIELDS)
    w = 0.8 / nff
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ymax_kT = 0.0
    for i, (ff, label, colour) in enumerate(FORCEFIELDS):
        means_kT = [agg.get((ff, lip), (np.nan, 0.0, 0.0, 0))[0] for lip in LIPIDS]
        cis_kT = [agg.get((ff, lip), (np.nan, 0.0, 0.0, 0))[2] for lip in LIPIDS]
        ymax_kT = max(ymax_kT, *[m + c for m, c in zip(means_kT, cis_kT) if m == m])
        means = [m * RT_KJ_PER_MOL for m in means_kT]   # k_BT/nm^2 -> kJ/mol/nm^2
        cis = [c * RT_KJ_PER_MOL for c in cis_kT]        # hierarchical 95% CI
        ax.bar(x + (i - (nff - 1) / 2.0) * w, means, w, yerr=cis, capsize=3,
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
