#!/usr/bin/env python3
"""Per-frame **lateral variation** of the HMMM bilayer thickness.

``lipid_properties.py`` stores one ``d_HH`` per frame -- the mean over the
membrane plane.  This script keeps the other half of that frame: the
**standard deviation of d_HH across the membrane plane**, which is the natural
stability signal.  An intact bilayer is flat, so its thickness barely varies
from place to place (~0.15 nm here); as a membrane buckles, opens or breaks up,
the leaflet-to-leaflet distance starts to depend strongly on where you measure
it, and the spatial SD climbs by an order of magnitude long before the *mean*
thickness looks wrong.

    PYTHONNOUSERSITE=1 conda run -n lipid64 python HMMM/analysis/thickness_variation.py --nprocs 6

Writes ``HMMM/data/thickness_map.npz`` with, per run, ``{sys}_rep{r}_mean`` and
``{sys}_rep{r}_sd`` (nm) plus ``{sys}_rep{r}_time`` (ns).

Method -- the same recipe lipyphilic's ``MembThickness`` uses, but keeping the
grid instead of averaging it away: leaflets are assigned on the phosphates,
each leaflet's phosphate z is binned onto a lateral grid (mean z per cell), and
the per-cell leaflet separation is the local thickness.

``--nbins`` is the number of cells **per side**, so the default 6 is a 6 x 6 =
36-cell grid *per leaflet*.  With 128 lipids in a leaflet that is 128/36 = 3.6
phosphates per cell on average (measured: min 0, max 7) -- enough for a stable
per-cell mean without smoothing away the very undulations being measured.  A
cell that happens to be empty in either leaflet has no thickness, so it is
dropped from that frame rather than counted as zero; at this resolution that is
about one cell in 36, and at ``--nbins 8`` (2.0 per cell) it would be closer to
one in nine, which is the practical limit on how fine the grid can go.
"""

import argparse
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
from scipy.stats import binned_statistic_2d
from lipyphilic import AssignLeaflets

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from hmmm_defs import (FORCEFIELDS, ENSEMBLES, REPS, PHOSPHATE_SEL,
                       stem as run_stem, system as system_key)
from topology import build_universe

DATA = ROOT / "data"
TRAJ = ROOT / "traj"
NBINS = 6


def thickness_series(u, nbins=NBINS, stride=1):
    """``(time_ns, mean_nm, sd_nm)`` -- per-frame lateral thickness statistics."""
    leaflets = AssignLeaflets(u, lipid_sel=PHOSPHATE_SEL)
    leaflets.run(step=stride)
    phos = u.select_atoms(PHOSPHATE_SEL)
    leaf = np.asarray(leaflets.leaflets)

    times, means, sds = [], [], []
    for i, ts in enumerate(u.trajectory[::stride]):
        frame_leaf = leaf[:, i] if leaf.ndim == 2 else leaf
        bx, by = float(ts.dimensions[0]), float(ts.dimensions[1])
        x_bins = np.linspace(0.0, bx, nbins + 1)
        y_bins = np.linspace(0.0, by, nbins + 1)
        surfaces = {}
        for sign in (1, -1):
            sel = frame_leaf == sign
            pos = phos.positions[sel]
            xy = pos[:, :2] % np.array([bx, by])
            surfaces[sign] = binned_statistic_2d(
                xy[:, 0], xy[:, 1], pos[:, 2], statistic="mean",
                bins=(x_bins, y_bins)).statistic
        local = (surfaces[1] - surfaces[-1]) / 10.0          # A -> nm
        local = local[np.isfinite(local)]
        times.append(ts.time / 1000.0)                       # ps -> ns
        means.append(float(local.mean()) if local.size else np.nan)
        sds.append(float(local.std()) if local.size else np.nan)
    return np.asarray(times), np.asarray(means), np.asarray(sds)


def run_one(job):
    ff, ens, rep, traj_dir, nbins, stride = job
    stem = run_stem(ff, ens, rep)
    print("thickness map %s" % stem, flush=True)
    u = build_universe(stem, traj_dir=traj_dir, quiet=True)
    t, m, s = thickness_series(u, nbins=nbins, stride=stride)
    print("  %-14s %4d frames  d_HH %.2f nm  lateral SD %.3f -> %.3f nm "
          "(max %.3f at %.0f ns)" % (stem, len(t), np.nanmean(m), s[0],
                                     np.nanmean(s[-10:]), np.nanmax(s),
                                     t[np.nanargmax(s)]), flush=True)
    return ff, ens, rep, t, m, s


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traj-dir", type=Path, default=TRAJ)
    ap.add_argument("--out", type=Path, default=DATA)
    ap.add_argument("--forcefields", nargs="+", default=FORCEFIELDS)
    ap.add_argument("--ensembles", nargs="+", default=ENSEMBLES)
    ap.add_argument("--reps", nargs="+", type=int, default=REPS)
    ap.add_argument("--nbins", type=int, default=NBINS)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--nprocs", type=int, default=1)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    jobs = []
    for ff in args.forcefields:
        for ens in args.ensembles:
            for rep in args.reps:
                stem = run_stem(ff, ens, rep)
                if not ((args.traj_dir / (stem + ".xtc")).exists()
                        and (args.traj_dir / (stem + ".gro")).exists()):
                    print("skip %s: .gro/.xtc not found" % stem)
                    continue
                jobs.append((ff, ens, rep, args.traj_dir, args.nbins,
                             args.stride))
    if not jobs:
        print("no trajectories found; nothing to do")
        return

    nprocs = max(1, min(args.nprocs, len(jobs)))
    print("%d trajectories, %d worker(s)" % (len(jobs), nprocs))
    if nprocs == 1:
        completed = [run_one(j) for j in jobs]
    else:
        import multiprocessing as mp
        with mp.Pool(nprocs) as pool:
            completed = list(pool.imap_unordered(run_one, jobs))

    # Update rather than rewrite, so a later SLipids-only run does not clobber
    # the CHARMM/AMBER results.
    path = args.out / "thickness_map.npz"
    out = dict(np.load(path)) if path.exists() else {}
    for ff, ens, rep, t, m, s in completed:
        key = "%s_rep%d" % (system_key(ff, ens), rep)
        out[key + "_time"] = t
        out[key + "_mean"] = m
        out[key + "_sd"] = s
    np.savez_compressed(path, **out)
    print("wrote %s  (%d runs updated)" % (path, len(completed)))


if __name__ == "__main__":
    main()
