#!/usr/bin/env python3
"""Plot the Lipidator ReSiS bilayer **tilt modulus** as a grouped bar chart --
lipids on x (DOPC/DMPC/DMPG), force fields as coloured bars in the ``palette``
scheme.  Reads ``data/lipidator_moduli.json`` (per-replicate) written by
``run_lipidator.py``; bars are the mean over replicates and error bars are the
**population SD across the per-replicate values** (project convention).

The modulus is area-normalised, **k_BT/nm^2** (as ReSiS reports it) -- distinct
from the per-lipid k_BT/rad^2 of the ost_pymodules pipeline.

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

    PLOTS.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(LIPIDS))
    nff = len(FORCEFIELDS)
    w = 0.8 / nff
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ymax = 0.0
    for i, (ff, label, colour) in enumerate(FORCEFIELDS):
        means = [agg.get((ff, lip), (np.nan, 0.0, 0))[0] for lip in LIPIDS]
        sds = [agg.get((ff, lip), (np.nan, 0.0, 0))[1] for lip in LIPIDS]
        ax.bar(x + (i - (nff - 1) / 2.0) * w, means, w, yerr=sds, capsize=3,
               color=colour, label=label, edgecolor="black", linewidth=0.4)
        ymax = max(ymax, *[m + s for m, s in zip(means, sds) if m == m])
    ax.set_xticks(x)
    ax.set_xticklabels(LIPIDS)
    ax.set_ylabel(r"tilt modulus  $\kappa_\theta$  (k$_\mathrm{B}T\,\mathrm{nm}^{-2}$)")
    ax.set_title("Lipid tilt modulus by force field (Lipidator / ReSiS)")
    ax.set_ylim(0, ymax * 1.15)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    out = PLOTS / "lipidator_tilt_modulus.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print("wrote %s" % out)


if __name__ == "__main__":
    main()
