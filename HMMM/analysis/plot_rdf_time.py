#!/usr/bin/env python3
"""Plot the solvent-RDF first-peak position against time (``rdf_time.py``).

2 x 3 grid: the two pairs (DCLE-DCLE, C6-DCLE) down the rows, the ensembles
across the columns, force fields overlaid in the ``palette`` colours.  Each
run's stability time (``stability.json``) is a dashed vertical line in its
force-field colour, so a peak that moves when the bilayer breaks up would show
as a step at that line.  The grey band in each panel is the spread
(min-max) of the posres runs' windows for that pair -- the window-to-window
noise of a run known to stay a bilayer, i.e. the yardstick for "moved".

    python3 HMMM/analysis/plot_rdf_time.py        # -> HMMM/plots/rdf_peak_time.png

Requires numpy + matplotlib only.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT.parent / "analysis"))

import figstyle
from palette import COLORS
from hmmm_defs import FORCEFIELDS, FF_LABELS, ENSEMBLES, ENS_LABELS, stem

DATA = ROOT / "data"
PLOTS = ROOT / "plots"
PAIRS = [("dcle_dcle", "DCLE–DCLE"), ("c6_dcle", "C6–DCLE")]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=DATA / "rdf_time.npz")
    ap.add_argument("--rep", type=int, default=1)
    ap.add_argument("--out", type=Path, default=PLOTS / "rdf_peak_time.png")
    args = ap.parse_args()

    d = np.load(args.data, allow_pickle=True)
    stab_path = DATA / "stability.json"
    stab = json.loads(stab_path.read_text()) if stab_path.exists() else {}

    figstyle.use()
    fig, axes = plt.subplots(len(PAIRS), len(ENSEMBLES), sharex=True,
                             sharey="row", figsize=(figstyle.mm(180),
                                                    figstyle.mm(100)),
                             layout="constrained")

    print("first-peak position (nm): mean +- SD over windows, "
          "stable part vs after the stability time")
    for row, (pair, plabel) in enumerate(PAIRS):
        # Noise yardstick: every posres window of this pair, all force fields.
        ref = np.concatenate([d["%s_%s_rpeak" % (stem(ff, "posre", args.rep), pair)]
                              for ff in FORCEFIELDS
                              if "%s_%s_rpeak" % (stem(ff, "posre", args.rep), pair) in d])
        ref = ref[np.isfinite(ref)] / 10.0
        for col, ens in enumerate(ENSEMBLES):
            ax = axes[row, col]
            if ref.size:
                ax.axhspan(ref.min(), ref.max(), color="0.85", lw=0, zorder=0)
            for ff in FORCEFIELDS:
                s = stem(ff, ens, args.rep)
                key = "%s_%s_rpeak" % (s, pair)
                if key not in d:
                    continue
                t = d["%s_t_ns" % s]
                r = d[key] / 10.0
                ax.plot(t, r, "-o", color=COLORS[ff], ms=2.5, lw=1,
                        label=FF_LABELS[ff])
                rec = stab.get(s)
                if rec and not rec.get("survived_full_run", True):
                    ts = rec["stability_ns"]
                    ax.axvline(ts, color=COLORS[ff], ls="--", lw=0.8)
                    pre, post = r[t < ts], r[t >= ts]
                    print("  %-6s %-9s %-14s stable %.3f +- %.3f (n=%d)   "
                          "after %.3f +- %.3f (n=%d)"
                          % (pair.split("_")[0], ens, s,
                             np.nanmean(pre) if pre.size else np.nan,
                             np.nanstd(pre) if pre.size else np.nan, pre.size,
                             np.nanmean(post), np.nanstd(post), post.size))
                else:
                    print("  %-6s %-9s %-14s full run %.3f +- %.3f (n=%d)"
                          % (pair.split("_")[0], ens, s, np.nanmean(r),
                             np.nanstd(r), r.size))
            if row == 0:
                ax.set_title(ENS_LABELS[ens])
            if col == 0:
                ax.set_ylabel("%s first peak (nm)" % plabel)
            if row == len(PAIRS) - 1:
                ax.set_xlabel("time (ns)")
            ax.set_xlim(0, 1000)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles.append(plt.Line2D([], [], color="0.3", ls="--", lw=0.8))
    labels.append("stability time")
    handles.append(plt.Rectangle((0, 0), 1, 1, color="0.85"))
    labels.append("posres window range")
    fig.legend(handles, labels, loc="outside upper center", ncol=len(labels),
               frameon=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=300)
    print("wrote %s" % args.out)


if __name__ == "__main__":
    main()
