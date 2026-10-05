#!/usr/bin/env python3
"""First-peak position of the solvent RDFs over time, over the WHOLE trajectory.

A diagnostic companion to ``rdf.py``.  ``rdf.py`` cuts every run at its
stability time on the grounds that the solvent structure of a broken-up
membrane is not the quantity being reported.  This script tests whether that
cut is actually needed for the RDFs: it bins the same two pairs (``dcle_dcle``,
``c6_dcle``, same DCLE centre and C6 selections, imported from ``rdf.py``) in
consecutive time windows over the **entire** run, with no stability truncation,
and tracks the first-peak position per window.  If the peak does not move when
the bilayer breaks up, the local solvation shell is insensitive to the
mesoscale breakup and the whole trajectory is fair game.

    PYTHONNOUSERSITE=1 conda run -n lipid64 python HMMM/analysis/rdf_time.py --nprocs 8

Writes ``HMMM/data/rdf_time.npz``; ``plot_rdf_time.py`` styles it.

Defaults to **replicate 1** of every (force field, ensemble) condition.

Why the cutoff is 1 nm here, not rdf.py's 2 nm
----------------------------------------------
The first peak sits at ~0.5 nm, so a 1 nm range covers it and the first
minimum with room to spare.  It also matters for correctness: minimum image
needs ``r_max <= L/2``, and the collapsed CHARMM NPT boxes shrink to ~2.2 nm
laterally -- a 2 nm cutoff would have to discard exactly the frames this
diagnostic exists to look at, while 1 nm keeps them.  Frames that still violate
the bound are skipped and counted, never silently used.

The peak position is the argmax of g(r) inside ``PEAK_WINDOW``, refined by a
three-point parabola through the maximum bin and its neighbours (the bins are
0.1 A, so the refinement is a small correction, not a fit).
"""

import argparse
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
from MDAnalysis.analysis.rdf import InterRDF

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from hmmm_defs import (FORCEFIELDS, ENSEMBLES, DCLE_SEL, DT_NS,
                       stem as run_stem)
from topology import build_universe
from rdf import dcle_centre, c6_selection, PAIRS

DATA = ROOT / "data"
TRAJ = ROOT / "traj"

R_MAX = 10.0              # A -- 1 nm, see the module docstring
NBINS = 100               # 0.1 A bins, as in rdf.py
PEAK_WINDOW = (3.5, 8.0)  # A -- where the first peak is looked for
WINDOW_NS = 25.0          # length of one time window


def first_peak(bins, g):
    """``(r_peak, g_peak)`` of the first solvation peak, parabola-refined."""
    lo, hi = PEAK_WINDOW
    idx = np.where((bins >= lo) & (bins <= hi))[0]
    if not len(idx) or not np.isfinite(g[idx]).any():
        return np.nan, np.nan
    i = idx[np.nanargmax(g[idx])]
    if 0 < i < len(g) - 1:
        y0, y1, y2 = g[i - 1], g[i], g[i + 1]
        denom = y0 - 2.0 * y1 + y2
        if denom < 0:
            shift = 0.5 * (y0 - y2) / denom      # in bins, within (-0.5, 0.5)
            dr = bins[1] - bins[0]
            return (bins[i] + shift * dr,
                    y1 - 0.25 * (y0 - y2) * shift)
    return bins[i], g[i]


def run_one(job):
    """Windowed RDFs for one run.  Top-level (picklable) for the process pool."""
    ff, ens, rep, traj_dir, window_frames, stride = job
    stem = run_stem(ff, ens, rep)
    print("rdf_time %s" % stem, flush=True)
    u = build_universe(stem, traj_dir=traj_dir, quiet=True)
    centre = dcle_centre(u, ff)
    dcle = u.select_atoms("%s and name %s" % (DCLE_SEL, centre))
    c6 = u.select_atoms(c6_selection(ff))

    ok = np.array([0.5 * ts.dimensions[:3].min() >= R_MAX
                   for ts in u.trajectory])
    n = len(ok)
    # Full windows only; a leftover shorter than half a window is folded into
    # the last one (1001 frames in 25-frame windows would otherwise end on a
    # single-frame window whose peak is pure noise).
    starts = list(range(0, n, window_frames))
    if len(starts) > 1 and n - starts[-1] < window_frames / 2.0:
        starts.pop()
    ends = starts[1:] + [n]
    out = {"t_ns": [], "n_frames": [], "n_dropped": []}
    for pair in PAIRS:
        out[pair + "_rpeak"] = []
        out[pair + "_gpeak"] = []
        out[pair + "_g"] = []
    bins = None
    for s, e in zip(starts, ends):
        win = np.arange(s, e)
        frames = win[ok[win]][::stride]
        out["t_ns"].append((win[0] + win[-1]) / 2.0 * DT_NS)
        out["n_frames"].append(len(frames))
        out["n_dropped"].append(int((~ok[win]).sum()))
        for pair, (g1, g2, self_pair) in (("dcle_dcle", (dcle, dcle, True)),
                                          ("c6_dcle", (c6, dcle, False))):
            if not len(frames):
                out[pair + "_rpeak"].append(np.nan)
                out[pair + "_gpeak"].append(np.nan)
                out[pair + "_g"].append(np.full(NBINS, np.nan))
                continue
            kw = dict(nbins=NBINS, range=(0.0, R_MAX))
            if self_pair:
                kw["exclusion_block"] = (1, 1)
            r = InterRDF(g1, g2, **kw)
            r.run(frames=frames)
            bins = np.asarray(r.results.bins)
            g = np.asarray(r.results.rdf)
            rp, gp = first_peak(bins, g)
            out[pair + "_rpeak"].append(rp)
            out[pair + "_gpeak"].append(gp)
            out[pair + "_g"].append(g)
    out = {k: np.asarray(v) for k, v in out.items()}
    out["bins"] = bins
    out["centre"] = centre
    print("  %-14s %d windows, %d/%d frames dropped for a too-small box, "
          "DCLE-DCLE peak %.3f -> %.3f nm, C6-DCLE %.3f -> %.3f nm"
          % (stem, len(starts), out["n_dropped"].sum(), n,
             out["dcle_dcle_rpeak"][0] / 10, out["dcle_dcle_rpeak"][-1] / 10,
             out["c6_dcle_rpeak"][0] / 10, out["c6_dcle_rpeak"][-1] / 10),
          flush=True)
    return stem, out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traj-dir", type=Path, default=TRAJ)
    ap.add_argument("--out", type=Path, default=DATA / "rdf_time.npz")
    ap.add_argument("--forcefields", nargs="+", default=FORCEFIELDS)
    ap.add_argument("--ensembles", nargs="+", default=ENSEMBLES)
    ap.add_argument("--reps", nargs="+", type=int, default=[1])
    ap.add_argument("--window-ns", type=float, default=WINDOW_NS)
    ap.add_argument("--stride", type=int, default=1,
                    help="use every n-th frame inside a window (default 1)")
    ap.add_argument("--nprocs", type=int, default=1)
    args = ap.parse_args()

    window_frames = max(1, int(round(args.window_ns / DT_NS)))
    jobs = []
    for ff in args.forcefields:
        for ens in args.ensembles:
            for rep in args.reps:
                stem = run_stem(ff, ens, rep)
                if not ((args.traj_dir / (stem + ".xtc")).exists()
                        and (args.traj_dir / (stem + ".gro")).exists()):
                    print("skip %s: .gro/.xtc not found" % stem)
                    continue
                jobs.append((ff, ens, rep, args.traj_dir, window_frames,
                             args.stride))
    if not jobs:
        print("no trajectories found; nothing to do")
        return

    nprocs = max(1, min(args.nprocs, len(jobs)))
    print("%d trajectories, %d worker(s), %g ns windows"
          % (len(jobs), nprocs, args.window_ns))
    if nprocs == 1:
        completed = [run_one(j) for j in jobs]
    else:
        import multiprocessing as mp
        with mp.Pool(nprocs) as pool:
            completed = list(pool.imap_unordered(run_one, jobs))

    # Update rather than rewrite, as the other compute steps do.
    out = dict(np.load(args.out, allow_pickle=True)) if args.out.exists() else {}
    for stem, res in completed:
        for k, v in res.items():
            out["%s_%s" % (stem, k)] = v
    out["window_ns"] = np.asarray(args.window_ns)
    out["r_max_A"] = np.asarray(R_MAX)
    np.savez_compressed(args.out, **out)
    print("wrote %s  (%d runs updated)" % (args.out, len(completed)))


if __name__ == "__main__":
    main()
