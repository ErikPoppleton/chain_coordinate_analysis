#!/usr/bin/env python3
"""Radial distribution functions for the HMMM organic solvent.

Two pairs, both keyed on the **CHCl2 carbon** of DCLE as the molecule's single
site:

``dcle_dcle``  solvent-solvent packing -- DCLE centre against DCLE centre, with
               the self-pair excluded.
``c6_dcle``    solvent around the tail terminus -- the last acyl carbon of each
               chain (chain position 6: ``C2T``/``C3T`` for CHARMM, ``C26``/``C36`` for SLipids,
               ``C16`` for AMBER's ``VL`` residues) against the DCLE centre.

    PYTHONNOUSERSITE=1 conda run -n lipid64 python HMMM/analysis/rdf.py --nprocs 6

Writes ``HMMM/data/rdf.npz``; ``plot_rdf.py`` styles it.

Which carbon is the centre
--------------------------
CHARMM and the AMBER/SLipids ``DCLE.itp`` **swap C1 and C2** (CHARMM's C1
carries the two chlorines, AMBER's C2 does).  Hard-coding the wrong one shifts
every peak by a C-C bond length and would never raise, so the centre is
identified **geometrically** in each file -- the candidate carbon closest to the
molecule's chlorines wins -- and checked against the expectation in
``hmmm_defs.DCLE_CENTRE``, which is reported as a mismatch rather than a crash.

Two things about the normalisation
----------------------------------
*The plateau is not 1.*  MDAnalysis normalises g(r) against the **box-average**
number density, and DCLE is confined to the bilayer core -- roughly a third of
the box -- so its local density is ~3x the box average and the curves level off
near 1.7 rather than 1.  Peak *positions* are unaffected; peak *heights* are
scaled by that factor, so compare heights between systems, not against 1.  The
large-r plateau is stored per curve (``*_plateau``) so a plot can divide it out,
and the raw histogram counts are stored so any other normalisation can be
rebuilt.

*Only the stable window is used.*  Each run is cut at its stability time
(``stability.py``) before anything is binned: the solvent structure of a
membrane that has broken up is not the quantity being reported.  ``--no-truncate``
restores the whole-trajectory behaviour.  On top of that, the minimum-image
convention needs ``r_max <= L/2``, which the collapsed frames of the failed
CHARMM NPT runs do not satisfy at all -- that cut is a correctness bound rather
than a judgement, and both counts are printed per run.
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

from hmmm_defs import (FORCEFIELDS, ENSEMBLES, REPS, SCHEME, DCLE_SEL,
                       DCLE_CARBONS, DCLE_CENTRE, TAIL_NC, chain_atom_names,
                       lipid_resnames, AMBER_TAIL,
                       stem as run_stem, system as system_key)
from topology import build_universe

DATA = ROOT / "data"
TRAJ = ROOT / "traj"

R_MAX = 20.0            # A -- 2 nm, well past the second solvation shell
NBINS = 200             # 0.1 A bins
PAIRS = ["dcle_dcle", "c6_dcle"]
PLATEAU_FROM = 18.0     # A -- the window the large-r plateau is averaged over


def dcle_centre(u, ff):
    """Name of DCLE's CHCl2 carbon in this file, found geometrically.

    Takes the first DCLE residue and picks whichever carbon sits closest to the
    molecule's chlorines.  Warns if it disagrees with ``hmmm_defs.DCLE_CENTRE``
    -- the two topologies swap C1 and C2, so the expectation is worth checking
    but not worth trusting blindly.
    """
    res = u.select_atoms(DCLE_SEL).residues[0].atoms
    chlorines = res.select_atoms("name CL*")
    if not chlorines.n_atoms:
        raise RuntimeError("no CL* atoms in the first DCLE residue")
    # Candidates are named explicitly: a `name C*` glob also matches CL11/CL3,
    # and a chlorine is of course the atom closest to the chlorines.
    candidates = res.select_atoms("name " + " ".join(sorted(DCLE_CARBONS)))
    if not candidates.n_atoms:
        raise RuntimeError("no %s atoms in the first DCLE residue"
                           % "/".join(sorted(DCLE_CARBONS)))
    best, best_d = None, np.inf
    for atom in candidates:
        d = np.linalg.norm(chlorines.positions - atom.position, axis=1).mean()
        if d < best_d:
            best, best_d = atom.name, d
    expected = DCLE_CENTRE.get(ff)
    if expected and best != expected:
        print("  ! %s: DCLE centre is %s (%.2f A to the chlorines), not the "
              "expected %s -- using the geometric answer"
              % (ff, best, best_d, expected), flush=True)
    return best


def c6_selection(ff):
    """Selection string for the terminal acyl carbon of both chains.

    Chain position ``TAIL_NC`` (6), which ``hmmm_defs`` spells ``C2T``/``C3T``
    (CHARMM) or ``C26``/``C36`` (SLipids) and ``C16`` for AMBER's ``VL`` tails.
    """
    scheme = SCHEME[ff]
    names = "name " + " ".join(sorted(chain_atom_names(scheme, TAIL_NC)))
    if scheme == "amber":
        return "resname %s and %s" % (AMBER_TAIL, names)
    return "resname %s and %s" % (" ".join(sorted(lipid_resnames(scheme))), names)


def usable_frames(u, r_max, n_stable=None):
    """``(indices, n_dropped)`` -- frames this RDF may legitimately use.

    Two independent cuts.  ``n_stable`` (from ``stability.py``) keeps only the
    part of the run that was still a bilayer, because the solvent structure of a
    membrane that has broken up is not the quantity being reported.  On top of
    that, minimum image needs ``r_max <= L/2``, which the collapsed frames of
    the failed CHARMM NPT runs do not satisfy at all -- that one is a
    correctness bound rather than a judgement about the data.
    """
    limit = u.trajectory.n_frames if n_stable is None else min(n_stable,
                                                               u.trajectory.n_frames)
    keep = []
    for i, ts in enumerate(u.trajectory):
        if i >= limit:
            break
        if 0.5 * ts.dimensions[:3].min() >= r_max:
            keep.append(i)
    return np.asarray(keep, dtype=int), limit - len(keep)


def _rdf(g1, g2, frames, self_pair):
    """Run InterRDF over ``frames`` and return ``(bins, g, counts, n_frames)``."""
    kw = dict(nbins=NBINS, range=(0.0, R_MAX))
    if self_pair:
        # One centre atom per molecule, so a (1, 1) exclusion block removes
        # exactly the self-pairs.
        kw["exclusion_block"] = (1, 1)
    r = InterRDF(g1, g2, **kw)
    r.run(frames=frames)
    return (np.asarray(r.results.bins), np.asarray(r.results.rdf),
            np.asarray(r.results.count), int(r.n_frames))


def run_one(job):
    """Both RDFs for one run.  Top-level (picklable) for the process pool."""
    ff, ens, rep, traj_dir, stride, r_max, n_stable = job
    stem = run_stem(ff, ens, rep)
    print("rdf %s" % stem, flush=True)
    u = build_universe(stem, traj_dir=traj_dir, quiet=True)

    centre = dcle_centre(u, ff)
    dcle = u.select_atoms("%s and name %s" % (DCLE_SEL, centre))
    c6 = u.select_atoms(c6_selection(ff))
    if not dcle.n_atoms or not c6.n_atoms:
        raise RuntimeError("%s: empty selection (DCLE %d, C6 %d)"
                           % (stem, dcle.n_atoms, c6.n_atoms))

    frames, dropped = usable_frames(u, r_max, n_stable)
    frames = frames[::stride]
    if not len(frames):
        raise RuntimeError("%s: no frame has a box wide enough for r_max=%.1f A"
                           % (stem, r_max))

    out = {"n_dcle": dcle.n_atoms, "n_c6": c6.n_atoms, "centre": centre,
           "dropped": dropped, "n_traj_frames": u.trajectory.n_frames,
           "n_stable": n_stable or u.trajectory.n_frames}
    for pair, (g1, g2, self_pair) in (("dcle_dcle", (dcle, dcle, True)),
                                      ("c6_dcle", (c6, dcle, False))):
        bins, g, counts, nfr = _rdf(g1, g2, frames, self_pair)
        out["bins"] = bins
        out[pair] = g
        # Running coordination number: pairs per reference atom within r.
        out[pair + "_cn"] = np.cumsum(counts) / float(nfr * g1.n_atoms)
        out[pair + "_plateau"] = float(g[bins >= PLATEAU_FROM].mean())
        out["n_frames"] = nfr
    print("  %-14s %4d frames from the stable %d of %d (%d dropped for a "
          "too-small box), centre %s, plateau %.2f / %.2f"
          % (stem, out["n_frames"], out["n_stable"], u.trajectory.n_frames,
             dropped, centre, out["dcle_dcle_plateau"],
             out["c6_dcle_plateau"]), flush=True)
    return ff, ens, rep, out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traj-dir", type=Path, default=TRAJ)
    ap.add_argument("--out", type=Path, default=DATA)
    ap.add_argument("--forcefields", nargs="+", default=FORCEFIELDS)
    ap.add_argument("--ensembles", nargs="+", default=ENSEMBLES)
    ap.add_argument("--reps", nargs="+", type=int, default=REPS)
    ap.add_argument("--stride", type=int, default=5,
                    help="use every n-th usable frame (default 5; g(r) is a "
                         "smooth average and converges long before every frame)")
    ap.add_argument("--r-max", type=float, default=R_MAX)
    ap.add_argument("--nprocs", type=int, default=1)
    ap.add_argument("--no-truncate", dest="truncate", action="store_false",
                    help="use the whole trajectory instead of each run's "
                         "stable window")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    import stability as stab
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
                jobs.append((ff, ens, rep, args.traj_dir, args.stride,
                             args.r_max,
                             rec["stability_frames"] if rec else None))
    if not jobs:
        print("no trajectories found; nothing to do")
        return

    nprocs = max(1, min(args.nprocs, len(jobs)))
    print("%d trajectories, %d worker(s), stride %d" % (len(jobs), nprocs,
                                                        args.stride))
    if nprocs == 1:
        completed = [run_one(j) for j in jobs]
    else:
        import multiprocessing as mp
        with mp.Pool(nprocs) as pool:
            completed = list(pool.imap_unordered(run_one, jobs))

    per_run = {}
    for ff, ens, rep, res in completed:
        per_run.setdefault(system_key(ff, ens), {})[rep] = res

    # Update rather than rewrite, so a later SLipids-only run does not clobber
    # the CHARMM/AMBER results.
    path = args.out / "rdf.npz"
    out = dict(np.load(path, allow_pickle=True)) if path.exists() else {}
    for sysname, by_rep in per_run.items():
        reps = sorted(by_rep)
        out["%s_bins" % sysname] = by_rep[reps[0]]["bins"]
        out["%s_reps" % sysname] = np.asarray(reps)
        for pair in PAIRS:
            for r in reps:
                out["%s_%s_rep%d" % (sysname, pair, r)] = by_rep[r][pair]
                out["%s_%s_cn_rep%d" % (sysname, pair, r)] = by_rep[r][pair + "_cn"]
            out["%s_%s_plateau" % (sysname, pair)] = np.asarray(
                [by_rep[r][pair + "_plateau"] for r in reps])
        out["%s_nframes" % sysname] = np.asarray(
            [by_rep[r]["n_frames"] for r in reps])
        out["%s_dropped" % sysname] = np.asarray(
            [by_rep[r]["dropped"] for r in reps])
        out["%s_stable_frames" % sysname] = np.asarray(
            [by_rep[r]["n_stable"] for r in reps])
    np.savez_compressed(path, **out)
    print("wrote %s  (%d systems updated)" % (path, len(per_run)))


if __name__ == "__main__":
    main()
