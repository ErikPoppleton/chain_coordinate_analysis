#!/usr/bin/env python3
"""Lateral diffusion of the HMMM lipids **and** of the DCLE solvent, measured
only over the part of each run that was still a bilayer.

``lipid_properties.py`` fits the lipid diffusion coefficient over the whole
trajectory.  For the properties figure that is wrong for the runs that broke up:
once the membrane is no longer a membrane, the "lateral" displacement of a
headgroup is not lateral diffusion in a bilayer.  This script refits D over each
run's **stable window** (``stability.py``) and adds the solvent, which
``lipid_properties.py`` never looked at.

    PYTHONNOUSERSITE=1 conda run -n lipid64 python HMMM/analysis/diffusion.py --nprocs 6

Two species, both as **lateral (xy)** mean-squared displacement, so the two are
directly comparable and comparable with the parent project's lipid D:

``lipid``  one phosphate per lipid (``name P P31``) -- the same selection
           ``lipid_properties.py`` uses, so the difference between the two
           numbers is the truncation and nothing else.
``dcle``   the CHCl2 carbon of each solvent molecule.  3-D would be the obvious
           choice for a solvent, but DCLE is trapped in a ~3 nm slab, so its z
           displacement saturates at the core thickness within a few ns and a
           3-D MSD would report a plateau, not a diffusion coefficient.  The
           in-plane motion is unbounded and is the quantity of interest.

Both remove the membrane centre-of-mass drift (``com_removal_sel`` = the
phosphates) and both need PBC-unwrapped coordinates -- lipyphilic's ``MSD``
assumes no jumps, so an MDAnalysis ``NoJump`` transformation is added first.

Writes ``HMMM/data/diffusion.npz``: per system and species, ``_lagtimes``,
``_msd`` (reps x lag), ``_D`` (cm^2/s), ``_Dsem``, ``_reps``, ``_frames`` (how
many frames each replicate contributed after truncation).
"""

import argparse
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
from MDAnalysis.transformations import NoJump
from lipyphilic.analysis.lateral_diffusion import MSD

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from hmmm_defs import (FORCEFIELDS, ENSEMBLES, REPS, PHOSPHATE_SEL, DCLE_SEL,
                       stem as run_stem, system as system_key)
from topology import build_universe
from rdf import dcle_centre
import stability as stab

DATA = ROOT / "data"
TRAJ = ROOT / "traj"
SPECIES = ["lipid", "dcle"]


def run_one(job):
    ff, ens, rep, traj_dir, n_stable = job
    stem = run_stem(ff, ens, rep)
    print("diffusion %s" % stem, flush=True)
    u = build_universe(stem, traj_dir=traj_dir, quiet=True)
    total = u.trajectory.n_frames
    stop = total if n_stable is None else max(2, min(n_stable, total))

    centre = dcle_centre(u, ff)
    sels = {"lipid": PHOSPHATE_SEL,
            "dcle": "%s and name %s" % (DCLE_SEL, centre)}

    # MSD assumes no PBC jumps; a wrapped crossing would read as a box-length
    # displacement.  Nothing before this point needs in-box coordinates.
    u.trajectory.add_transformations(NoJump())

    out = {"frames": stop, "n_traj_frames": total}
    for name, sel in sels.items():
        msd = MSD(u, lipid_sel=sel, com_removal_sel=PHOSPHATE_SEL)
        msd.run(stop=stop)
        D, Dsem = msd.diffusion_coefficient()
        out[name] = {"lagtimes": np.asarray(msd.lagtimes),
                     "msd": np.nanmean(np.asarray(msd.msd), axis=0),
                     "D": float(D), "Dsem": float(Dsem)}
    print("  %-14s %4d/%d frames  D_lipid = %.3g  D_dcle = %.3g cm^2/s"
          % (stem, stop, total, out["lipid"]["D"], out["dcle"]["D"]), flush=True)
    return ff, ens, rep, out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traj-dir", type=Path, default=TRAJ)
    ap.add_argument("--out", type=Path, default=DATA)
    ap.add_argument("--forcefields", nargs="+", default=FORCEFIELDS)
    ap.add_argument("--ensembles", nargs="+", default=ENSEMBLES)
    ap.add_argument("--reps", nargs="+", type=int, default=REPS)
    ap.add_argument("--nprocs", type=int, default=1)
    ap.add_argument("--no-truncate", dest="truncate", action="store_false",
                    help="fit over the whole trajectory instead of the stable "
                         "window (the pre-stability behaviour)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    records = stab.load() if args.truncate else {}

    jobs = []
    for ff in args.forcefields:
        for ens in args.ensembles:
            for rep in args.reps:
                stem = run_stem(ff, ens, rep)
                if not ((args.traj_dir / (stem + ".xtc")).exists()
                        and (args.traj_dir / (stem + ".gro")).exists()):
                    print("skip %s: .gro/.xtc not found" % stem)
                    continue
                rec = records.get(stem)
                jobs.append((ff, ens, rep, args.traj_dir,
                             rec["stability_frames"] if rec else None))
    if not jobs:
        print("no trajectories found; nothing to do")
        return

    nprocs = max(1, min(args.nprocs, len(jobs)))
    print("%d trajectories, %d worker(s)%s"
          % (len(jobs), nprocs, "" if args.truncate else ", NO truncation"))
    if nprocs == 1:
        completed = [run_one(j) for j in jobs]
    else:
        import multiprocessing as mp
        with mp.Pool(nprocs) as pool:
            completed = list(pool.imap_unordered(run_one, jobs))

    per_run = {}
    for ff, ens, rep, res in completed:
        per_run.setdefault(system_key(ff, ens), {})[rep] = res

    path = args.out / "diffusion.npz"
    out = dict(np.load(path)) if path.exists() else {}
    for sysname, by_rep in per_run.items():
        reps = sorted(by_rep)
        out["%s_reps" % sysname] = np.asarray(reps)
        out["%s_frames" % sysname] = np.asarray([by_rep[r]["frames"]
                                                 for r in reps])
        for sp in SPECIES:
            curves = [by_rep[r][sp]["msd"] for r in reps]
            nlag = min(len(c) for c in curves)   # truncated runs differ in length
            out["%s_%s_lagtimes" % (sysname, sp)] = \
                by_rep[reps[0]][sp]["lagtimes"][:nlag]
            out["%s_%s_msd" % (sysname, sp)] = \
                np.vstack([c[:nlag] for c in curves])
            out["%s_%s_D" % (sysname, sp)] = \
                np.asarray([by_rep[r][sp]["D"] for r in reps])
            out["%s_%s_Dsem" % (sysname, sp)] = \
                np.asarray([by_rep[r][sp]["Dsem"] for r in reps])
    np.savez_compressed(path, **out)
    print("wrote %s  (%d systems updated)" % (path, len(per_run)))


if __name__ == "__main__":
    main()
