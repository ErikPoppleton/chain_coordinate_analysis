#!/usr/bin/env python3
"""Compute equilibrium HMMM membrane properties with lipyphilic, for every
force field / ensemble / replicate, and save plot-ready arrays as ``.npz``.

The compute/plot split is the parent project's: this (slow) script writes
``HMMM/data/lipprops_*.npz`` once, and the styling lives in the separate
``plot_*``/``draw_*`` scripts.  Everything runs locally -- unlike the parent
project there is no g_lomepro step and nothing needs the cluster.

    PYTHONNOUSERSITE=1 conda run -n lipid64 python HMMM/analysis/lipid_properties.py

Input layout::

    HMMM/traj/{ff}_{ens}{rep}.gro     topology  (AMBER is renumbered on the fly)
    HMMM/traj/{ff}_{ens}{rep}.xtc     trajectory

Properties, one ``.npz`` per property in ``HMMM/data/``:

  * ``area_per_lipid``    -- AreaPerLipid, ``(n_frames, n_lipids)`` per run (A^2)
  * ``memb_thickness``    -- MembThickness (d_HH), per-frame samples (A)
  * ``lateral_diffusion`` -- MSD vs lag time + the fitted D per run (cm^2/s)

plus ``box.npz``, the per-frame box vectors of every run -- cheap to keep and
the record of which runs stayed stable (see below).

All three are keyed on the one-phosphorus-per-lipid selection ``name P P31``,
which drives leaflet assignment, the Voronoi tessellation, the leaflet surfaces
and the MSD reference -- exactly as in the parent project, so the numbers are
directly comparable with the full DOPC bilayers.

Two HMMM-specific notes
-----------------------
*No waters-stripped second copy.*  These trajectories carry no water and no
ions at all (the core solvent is DCLE), so the downloaded ``traj/`` files are
both the full system and the membrane-only set.

*The CHARMM NPT runs are not all 1 us.*  ``charmm_npt1`` (790 frames) and
``charmm_npt3`` (891 frames) end in a pressure-coupling failure: the box
contracts laterally and stretches along z from ~400 ns and runs away to a
2.3 x 2.3 x 270 nm collapse over the last ~100 frames.  **Every available frame
is analysed as-is** -- the blow-up is real output, not something this script
decides to hide -- but it does dominate those two runs' statistics, so the
script prints a per-run box summary and stores the full box trace in
``box.npz``.
"""

import argparse
import os
import sys
from pathlib import Path

# lipyphilic's analyses are serial-only, so pin BLAS to one thread per worker
# (before numpy is imported) and parallelise over trajectories instead.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

from MDAnalysis.transformations import NoJump
from lipyphilic import AssignLeaflets
from lipyphilic.analysis.area_per_lipid import AreaPerLipid
from lipyphilic.analysis.memb_thickness import MembThickness
from lipyphilic.analysis.lateral_diffusion import MSD

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hmmm_defs import (FORCEFIELDS, ENSEMBLES, REPS, PHOSPHATE_SEL,
                       stem as run_stem, system as system_key)
from topology import build_universe

ROOT = Path(__file__).resolve().parent.parent      # HMMM/
DATA = ROOT / "data"
TRAJ = ROOT / "traj"

SAMPLE_PROPS = ["area_per_lipid", "memb_thickness"]


def analyse(u):
    """Run the three lipyphilic analyses on one trajectory.

    Order matters: the leaflet/Voronoi/surface analyses need in-box
    coordinates, while MSD assumes no PBC jumps (a boundary crossing otherwise
    reads as a box-length displacement and inflates D by orders of magnitude),
    so the ``NoJump`` transformation is added only after them, with MSD last.
    """
    box = np.array([ts.dimensions.copy() for ts in u.trajectory])
    times = np.array([ts.time for ts in u.trajectory])

    leaflets = AssignLeaflets(u, lipid_sel=PHOSPHATE_SEL)
    leaflets.run()

    apl = AreaPerLipid(u, lipid_sel=PHOSPHATE_SEL, leaflets=leaflets.leaflets)
    apl.run()

    thick = MembThickness(u, lipid_sel=PHOSPHATE_SEL, leaflets=leaflets.leaflets)
    thick.run()

    u.trajectory.add_transformations(NoJump())
    msd = MSD(u, lipid_sel=PHOSPHATE_SEL, com_removal_sel=PHOSPHATE_SEL)
    msd.run()
    D, Dsem = msd.diffusion_coefficient()

    return {
        # Keep APL 2-D (n_frames, n_lipids): a histogram flattens it anyway,
        # but the per-frame axis is what the trajectory/stats steps need.
        "area_per_lipid": np.asarray(apl.areas).T,
        "memb_thickness": np.asarray(thick.memb_thickness).ravel(),
        "msd_lagtimes": np.asarray(msd.lagtimes),
        "msd_curve": np.nanmean(np.asarray(msd.msd), axis=0),   # mean over lipids
        "D": float(D),
        "Dsem": float(Dsem),
        "box": box[:, :3],
        "times": times,
    }


def run_one(job):
    """Analyse one run.  Top-level (picklable) so it can go to a process pool.

    The trajectory directory travels in the job rather than in a module global:
    macOS spawns pool workers, which re-import this module and would otherwise
    lose a ``--traj-dir`` set on the command line.
    """
    ff, ens, rep, traj_dir = job
    stem = run_stem(ff, ens, rep)
    print("analysing %s" % stem, flush=True)
    u = build_universe(stem, traj_dir=traj_dir)
    res = analyse(u)
    box = res["box"] / 10.0
    print("  %-14s %4d frames  box x %.2f -> %.2f nm, z %.2f -> %.2f nm  "
          "D = %.3g cm^2/s" % (stem, len(box), box[0, 0], box[-1, 0],
                               box[0, 2], box[-1, 2], res["D"]), flush=True)
    return ff, ens, rep, res


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traj-dir", type=Path, default=TRAJ)
    ap.add_argument("--out", type=Path, default=DATA)
    ap.add_argument("--forcefields", nargs="+", default=FORCEFIELDS)
    ap.add_argument("--ensembles", nargs="+", default=ENSEMBLES)
    ap.add_argument("--reps", nargs="+", type=int, default=REPS)
    ap.add_argument("--nprocs", type=int, default=1,
                    help="analyse this many trajectories in parallel (1 core each)")
    args = ap.parse_args()

    traj_dir = args.traj_dir
    args.out.mkdir(parents=True, exist_ok=True)

    jobs = []
    for ff in args.forcefields:
        for ens in args.ensembles:
            for rep in args.reps:
                stem = run_stem(ff, ens, rep)
                if not ((traj_dir / (stem + ".xtc")).exists()
                        and (traj_dir / (stem + ".gro")).exists()):
                    print("skip %s: .gro/.xtc not found in %s" % (stem, traj_dir))
                    continue
                jobs.append((ff, ens, rep, traj_dir))
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

    # Regroup by system, keyed by replicate, so out-of-order completion is fine.
    per_run = {}
    for ff, ens, rep, res in completed:
        per_run.setdefault(system_key(ff, ens), {})[rep] = res

    # ---- per-frame sample properties: pooled + per-replicate ----
    # Existing analyses **must not be clobbered** when the script is run on a
    # subset (e.g. SLipids alone, once those runs land), so each .npz is loaded
    # and updated rather than rewritten from scratch.
    for prop in SAMPLE_PROPS:
        path = args.out / ("lipprops_%s.npz" % prop)
        out = dict(np.load(path)) if path.exists() else {}
        for sysname, by_rep in per_run.items():
            reps = sorted(by_rep)
            out[sysname] = np.concatenate([by_rep[r][prop] for r in reps])
            for r in reps:
                out["%s_rep%d" % (sysname, r)] = by_rep[r][prop]
        np.savez_compressed(path, **out)
        print("wrote %s  (%d systems updated)" % (path, len(per_run)))

    # ---- lateral diffusion: MSD curve + fitted D per replicate ----
    path = args.out / "lipprops_lateral_diffusion.npz"
    out = dict(np.load(path)) if path.exists() else {}
    for sysname, by_rep in per_run.items():
        reps = sorted(by_rep)
        msds = [by_rep[r]["msd_curve"] for r in reps]
        nlag = min(len(m) for m in msds)      # runs may differ in length
        out["%s_lagtimes" % sysname] = by_rep[reps[0]]["msd_lagtimes"][:nlag]
        out["%s_msd" % sysname] = np.vstack([m[:nlag] for m in msds])
        out["%s_D" % sysname] = np.asarray([by_rep[r]["D"] for r in reps])
        out["%s_Dsem" % sysname] = np.asarray([by_rep[r]["Dsem"] for r in reps])
        out["%s_reps" % sysname] = np.asarray(reps)
    np.savez_compressed(path, **out)
    print("wrote %s  (%d systems updated)" % (path, len(per_run)))

    # ---- box vectors: the record of which runs stayed stable ----
    path = args.out / "box.npz"
    out = dict(np.load(path)) if path.exists() else {}
    for sysname, by_rep in per_run.items():
        for r in sorted(by_rep):
            out["%s_rep%d" % (sysname, r)] = by_rep[r]["box"] / 10.0    # nm
            out["%s_rep%d_time" % (sysname, r)] = by_rep[r]["times"] / 1000.0  # ns
    np.savez_compressed(path, **out)
    print("wrote %s" % path)


if __name__ == "__main__":
    main()
