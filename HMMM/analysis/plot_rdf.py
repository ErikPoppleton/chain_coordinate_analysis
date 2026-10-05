#!/usr/bin/env python3
"""Plot the HMMM solvent radial distribution functions computed by ``rdf.py``.

Two figures, each a 2 x 3 grid -- the pair down the rows, the ensemble across
the columns -- with the three force fields overlaid in the ``palette`` colours,
the line the mean over replicates and the band their SD:

``plots/rdf.png``      g(r) for DCLE-DCLE (solvent packing) and C6-DCLE
                       (solvent around the acyl-tail terminus).
``plots/rdf_cn.png``   the running coordination number, i.e. the mean number of
                       DCLE centres within r of the reference atom.  This is the
                       integral of the same data and is immune to the
                       normalisation caveat below, so it is the number to quote.

    PYTHONNOUSERSITE=1 conda run -n base python HMMM/analysis/plot_rdf.py

**The g(r) plateau is not 1**, and the dotted grey line in each panel marks
where it actually sits: MDAnalysis normalises against the box-average density,
and DCLE is confined to the bilayer core, so its local density is a few times
the box average.  Peak *positions* are unaffected; peak *heights* are scaled by
that factor, which differs between systems as the core thickness does.
``--scale-plateau`` divides each curve by its own plateau, which makes the
heights comparable at the cost of no longer being a textbook g(r).

Requires numpy + matplotlib only.
"""

import argparse
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

from palette import COLORS
from hmmm_defs import (FORCEFIELDS, FF_LABELS, ENSEMBLES, ENS_LABELS, REPS,
                       system as system_key)

DATA = ROOT / "data"
PLOTS = ROOT / "plots"

PAIRS = [("dcle_dcle", "DCLE–DCLE"),
         ("c6_dcle", "C6–DCLE")]


def curves(npz, ff, ens, pair, kind=""):
    """``(r_nm, mean, sd)`` over the replicates present, or ``None``."""
    sysname = system_key(ff, ens)
    bins = "%s_bins" % sysname
    if bins not in npz:
        return None
    suffix = "_cn" if kind == "cn" else ""
    ys = [npz["%s_%s%s_rep%d" % (sysname, pair, suffix, r)] for r in REPS
          if "%s_%s%s_rep%d" % (sysname, pair, suffix, r) in npz]
    if not ys:
        return None
    ys = np.vstack(ys)
    sd = ys.std(axis=0, ddof=1) if len(ys) > 1 else np.zeros(ys.shape[1])
    return np.asarray(npz[bins]) / 10.0, ys.mean(axis=0), sd


def plateau(npz, ff, ens, pair):
    key = "%s_%s_plateau" % (system_key(ff, ens), pair)
    return float(np.mean(npz[key])) if key in npz else None


def draw(npz, kind, scale_plateau=False):
    fig, axes = plt.subplots(len(PAIRS), len(ENSEMBLES), figsize=(11, 6.5),
                             sharex=True, sharey="row")
    axes = np.atleast_2d(axes)
    for pi, (pair, pair_label) in enumerate(PAIRS):
        for ei, ens in enumerate(ENSEMBLES):
            ax = axes[pi, ei]
            drew = False
            for ff in FORCEFIELDS:
                got = curves(npz, ff, ens, pair, kind)
                if got is None:
                    continue
                r, mean, sd = got
                norm = 1.0
                if kind != "cn" and scale_plateau:
                    norm = plateau(npz, ff, ens, pair) or 1.0
                ax.plot(r, mean / norm, color=COLORS[ff], lw=1.2,
                        label=FF_LABELS[ff])
                ax.fill_between(r, (mean - sd) / norm, (mean + sd) / norm,
                                color=COLORS[ff], alpha=0.20, lw=0)
                drew = True
                if kind != "cn" and not scale_plateau:
                    p = plateau(npz, ff, ens, pair)
                    if p is not None:
                        ax.axhline(p, color=COLORS[ff], lw=0.6, ls=":",
                                   alpha=0.7)
            if not drew:
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        transform=ax.transAxes, color="0.6")
            if pi == 0:
                ax.set_title(ENS_LABELS[ens])
            if pi == len(PAIRS) - 1:
                ax.set_xlabel("$r$ (nm)")
            if ei == 0:
                ax.set_ylabel(("$N(r)$, " if kind == "cn" else "$g(r)$, ")
                              + pair_label)
            ax.set_xlim(0, r[-1] if drew else 2.0)
            if kind != "cn":
                ax.set_ylim(bottom=0)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=len(labels),
                   frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=DATA / "rdf.npz")
    ap.add_argument("--out", type=Path, default=PLOTS)
    ap.add_argument("--scale-plateau", action="store_true",
                    help="divide each g(r) by its own large-r plateau")
    args = ap.parse_args()
    if not args.data.exists():
        raise SystemExit("%s missing -- run HMMM/analysis/rdf.py first"
                         % args.data)
    npz = np.load(args.data)
    args.out.mkdir(parents=True, exist_ok=True)

    for kind, name in (("", "rdf"), ("cn", "rdf_cn")):
        fig = draw(npz, kind, scale_plateau=args.scale_plateau)
        path = args.out / ("%s.png" % name)
        fig.savefig(path, dpi=300)
        plt.close(fig)
        print("wrote %s" % path)

    # A short numeric summary: where the first solvation shell sits and how many
    # neighbours it holds.  Peak positions are the normalisation-free part of
    # the measurement, so they are the safe thing to read off.
    print("\nfirst peak / coordination number at 0.8 nm")
    print("  %-16s %-12s %10s %10s %10s" % ("system", "pair", "r_peak(nm)",
                                            "g_peak", "N(0.8nm)"))
    for ens in ENSEMBLES:
        for ff in FORCEFIELDS:
            for pair, _ in PAIRS:
                got = curves(npz, ff, ens, pair)
                cn = curves(npz, ff, ens, pair, "cn")
                if got is None:
                    continue
                r, mean, _ = got
                lo = r > 0.2          # skip the empty-core noise
                i = np.argmax(mean * lo)
                n08 = np.interp(0.8, cn[0], cn[1]) if cn else float("nan")
                print("  %-16s %-12s %10.3f %10.2f %10.2f"
                      % (system_key(ff, ens), pair, r[i], mean[i], n08))


if __name__ == "__main__":
    main()
