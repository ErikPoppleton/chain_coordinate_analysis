#!/usr/bin/env python3
"""Compute equilibrium lipid properties with lipyphilic, for every
force-field / lipid / replicate, and save plot-ready arrays as .npz.

This is the *analysis* half of the pipeline; styling lives in the separate
``plot_lipid_properties.py`` so the (slow) analysis need only run once.

Run cluster-side in the ``lipid`` conda environment, which has lipyphilic +
MDAnalysis installed::

    conda run -n lipid64 python analysis/lipid_properties.py

Input layout (identical on the local box and the cluster)::

    ~/Simulations/{ff}_{lipid}/eq{1,2,3}/md.tpr        topology
    ~/Simulations/{ff}_{lipid}/eq{1,2,3}/md.xtc        trajectory (--traj to override)
    ~/Simulations/{ff}_{lipid}/eq{1,2,3}/md.gro        last-frame topology fallback

The GROMACS 2026.3 ``.tpr`` files are too new for MDAnalysis's TPR parser, so
if MDAnalysis can't read ``md.tpr`` directly the script falls back to the
last-configuration ``md.gro`` that every run directory contains.

Properties (all via lipyphilic), one .npz written to ``data/`` per property:
  * area_per_lipid   -- AreaPerLipid, per-lipid-per-frame samples (A^2)
  * memb_thickness   -- MembThickness, per-frame samples (A)
  * registration     -- Registration, per-frame samples (correlation coeff.)
  * order_parameter  -- SCC, per-acyl-chain-per-frame samples (S_CC)
  * lateral_diffusion-- MSD curve (vs lag time) + fitted diffusion coefficient

The first four are histogram-ready pooled sample arrays; lateral_diffusion
stores the MSD-vs-lagtime curve and the fitted D per replicate instead.
"""

import argparse
import os
import sys
from pathlib import Path

# The lipyphilic analyses are numpy-light and only run serially, so pin BLAS to
# one thread (before numpy imports) to keep --nprocs workers from oversubscribing
# cores.  Override by exporting OMP_NUM_THREADS etc. yourself.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import MDAnalysis as mda
from MDAnalysis.transformations import NoJump
from lipyphilic import AssignLeaflets
from lipyphilic.analysis.area_per_lipid import AreaPerLipid
from lipyphilic.analysis.registration import Registration
from lipyphilic.analysis.order_parameter import SCC
from lipyphilic.analysis.memb_thickness import MembThickness
from lipyphilic.analysis.lateral_diffusion import MSD

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

FORCEFIELDS = ["charmm", "amber", "slipids"]
LIPIDS = ["DOPC", "DMPC", "DMPG"]
REPLICATES = [1, 2, 3]

# One phosphorus per lipid in every force field: CHARMM/SLipids name it "P",
# AMBER (Lipid17) names it "P31".  This single selection drives leaflet
# assignment, area-per-lipid, thickness, registration and the MSD reference.
PHOSPHATE = "name P P31"

# Carbons per acyl tail: DOPC is oleoyl (18:1), DMPC/DMPG are myristoyl (14:0).
TAIL_NC = {"DOPC": 18, "DMPC": 14, "DMPG": 14}
# AMBER splits each lipid into a head residue + two tail residues; the tail
# resname is the acyl species.
AMBER_TAIL_RESN = {"DOPC": "OL", "DMPC": "MY", "DMPG": "MY"}


def tail_selections(ff, lipid):
    """Selection string(s) for the acyl-chain carbons, one entry per sn chain.

    CHARMM/SLipids keep both tails in one residue, so the sn-1 (C2x) and sn-2
    (C3x) chains are separated by atom name.  AMBER puts each tail in its own
    residue, so a single ``resname`` selection yields both chains (lipyphilic's
    SCC then treats each residue as an independent chain)."""
    n = TAIL_NC[lipid]
    if ff in ("charmm", "slipids"):
        sn1 = f"resname {lipid} and name " + " ".join(f"C2{i}" for i in range(2, n + 1))
        sn2 = f"resname {lipid} and name " + " ".join(f"C3{i}" for i in range(2, n + 1))
        return [sn1, sn2]
    # amber
    return [f"resname {AMBER_TAIL_RESN[lipid]} and name C1*"]


def build_universe(run_dir, traj):
    """Universe for ``run_dir``, coping with too-new .tpr files.

    Tries ``md.tpr`` directly; on failure falls back to the last-configuration
    ``md.gro``, which every run directory always contains and which MDAnalysis
    can read regardless of GROMACS version."""
    tpr = run_dir / "md.tpr"
    if tpr.exists():
        try:
            return mda.Universe(str(tpr), str(traj))
        except Exception as exc:  # tpx version too new for MDAnalysis, etc.
            print(f"    tpr unreadable by MDAnalysis ({str(exc)[:50]}...); using md.gro")
    return mda.Universe(str(run_dir / "md.gro"), str(traj))


def analyse(u, ff, lipid):
    """Run all five lipyphilic analyses on one trajectory; return a dict of
    plot-ready arrays."""
    leaflets = AssignLeaflets(u, lipid_sel=PHOSPHATE)
    leaflets.run()

    apl = AreaPerLipid(u, lipid_sel=PHOSPHATE, leaflets=leaflets.leaflets)
    apl.run()

    thick = MembThickness(u, lipid_sel=PHOSPHATE, leaflets=leaflets.leaflets)
    thick.run()

    reg = Registration(u, upper_sel=PHOSPHATE, lower_sel=PHOSPHATE,
                       leaflets=leaflets.leaflets)
    reg.run()

    # Order parameter: pool every acyl chain (both tails) into one sample set.
    # lipyphilic's SCC groups tail atoms by residue and needs a uniform carbon
    # count per residue.  If a topology gives a lipid's two tails the same resid
    # (as GROMACS does for AMBER Lipid17), MDAnalysis merges them into one lumpy
    # residue and SCC dies with an opaque reshape error -- so check up front.
    n_expected = TAIL_NC[lipid] - 1  # carbons per chain feeding into S_CC
    scc_samples = []
    for sel in tail_selections(ff, lipid):
        tail = u.select_atoms(sel)
        _, counts = np.unique(tail.resindices, return_counts=True)
        if not np.all(counts == n_expected):
            raise ValueError(
                f"{ff} {lipid}: acyl-chain selection yields "
                f"{sorted(set(int(c) for c in counts))} carbons per residue, "
                f"expected {n_expected}. A lipid's two tails "
                f"are sharing a residue id (GROMACS numbers them identically), so "
                f"MDAnalysis merges them and SCC can't run.  Renumber md.gro so "
                f"every tail is its own residue first."
            )
        chain = SCC(u, tail_sel=sel)
        chain.run()
        scc_samples.append(np.asarray(chain.SCC).ravel())

    # MSD needs unwrapped coordinates -- lipyphilic assumes no PBC jumps, and a
    # wrapped trajectory turns every boundary crossing into a ~box-length jump
    # that massively inflates the MSD.  Add NoJump now, after the four analyses
    # above (which need in-box coordinates for their PBC-aware leaflet/Voronoi
    # logic); MSD is the last analysis so they are unaffected.
    u.trajectory.add_transformations(NoJump())
    msd = MSD(u, lipid_sel=PHOSPHATE, com_removal_sel=PHOSPHATE)
    msd.run()
    D, Dsem = msd.diffusion_coefficient()

    return {
        "area_per_lipid": np.asarray(apl.areas).ravel(),
        "memb_thickness": np.asarray(thick.memb_thickness).ravel(),
        "registration": np.asarray(reg.registration).ravel(),
        "order_parameter": np.concatenate(scc_samples),
        "msd_lagtimes": np.asarray(msd.lagtimes),
        "msd_curve": np.nanmean(np.asarray(msd.msd), axis=0),  # mean over lipids
        "D": float(D),
        "Dsem": float(Dsem),
    }


# Properties saved as pooled 1-D sample arrays (histograms downstream).
SAMPLE_PROPS = ["area_per_lipid", "memb_thickness", "registration", "order_parameter"]


def run_one(job):
    """Analyse a single trajectory.  Top-level (picklable) so it can be farmed
    out to a process pool.  Returns ``(ff, lipid, rep, result_dict)``."""
    ff, lipid, rep, run_dir, traj = job
    print(f"analysing {ff}_{lipid} eq{rep}  ({traj.name})", flush=True)
    u = build_universe(run_dir, traj)
    return ff, lipid, rep, analyse(u, ff, lipid)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sims-root", type=Path, default=Path.home() / "Simulations")
    ap.add_argument("--forcefields", nargs="+", default=FORCEFIELDS)
    ap.add_argument("--lipids", nargs="+", default=LIPIDS)
    ap.add_argument("--reps", nargs="+", type=int, default=REPLICATES)
    ap.add_argument("--traj", default="md.xtc",
                    help="trajectory filename inside each eq dir (default: md.xtc)")
    ap.add_argument("--nprocs", type=int, default=1,
                    help="analyse this many trajectories in parallel, one core "
                         "each (the analyses themselves are serial-only). "
                         "Capped at the number of trajectories found.")
    ap.add_argument("--out", type=Path, default=DATA)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    # Build the (existence-checked) list of independent trajectory jobs.
    jobs = []
    for ff in args.forcefields:
        for lipid in args.lipids:
            for rep in args.reps:
                run_dir = args.sims_root / f"{ff}_{lipid}" / f"eq{rep}"
                if not run_dir.is_dir():
                    print(f"skip {ff}_{lipid} eq{rep}: {run_dir} not found")
                    continue
                traj = run_dir / args.traj
                if not traj.exists():
                    print(f"skip {ff}_{lipid} eq{rep}: {traj.name} not found")
                    continue
                jobs.append((ff, lipid, rep, run_dir, traj))

    if not jobs:
        print("no trajectories found; nothing to do")
        return

    # Each job is single-core and independent, so parallelise over trajectories.
    nprocs = max(1, min(args.nprocs, len(jobs)))
    print(f"{len(jobs)} trajectories, {nprocs} worker(s)")
    if nprocs == 1:
        completed = [run_one(j) for j in jobs]
    else:
        import multiprocessing as mp
        with mp.Pool(nprocs) as pool:
            completed = list(pool.imap_unordered(run_one, jobs))

    # Regroup by system, keyed by replicate so out-of-order completion is fine.
    results = {p: {} for p in SAMPLE_PROPS}  # prop -> system -> {rep: array}
    diffusion = {}                           # system -> {rep: result_dict}
    for ff, lipid, rep, res in completed:
        system = f"{ff}_{lipid}"
        for p in SAMPLE_PROPS:
            results[p].setdefault(system, {})[rep] = res[p]
        diffusion.setdefault(system, {})[rep] = res

    # ---- write sample-based properties: pooled + per-replicate ----
    for p in SAMPLE_PROPS:
        out = {}
        for system, by_rep in results[p].items():
            reps = sorted(by_rep)
            out[system] = np.concatenate([by_rep[r] for r in reps])
            for r in reps:
                out[f"{system}_eq{r}"] = by_rep[r]
        path = args.out / f"lipprops_{p}.npz"
        np.savez_compressed(path, **out)
        print(f"wrote {path}  ({len(results[p])} systems)")

    # ---- write lateral diffusion: MSD curve + fitted D per replicate ----
    out = {}
    for system, by_rep in diffusion.items():
        reps = sorted(by_rep)
        msds = [by_rep[r]["msd_curve"] for r in reps]
        # Replicates share dt but may differ in length; truncate to the common min.
        nlag = min(len(m) for m in msds)
        out[f"{system}_lagtimes"] = by_rep[reps[0]]["msd_lagtimes"][:nlag]
        out[f"{system}_msd"] = np.vstack([m[:nlag] for m in msds])
        out[f"{system}_D"] = np.asarray([by_rep[r]["D"] for r in reps])
        out[f"{system}_Dsem"] = np.asarray([by_rep[r]["Dsem"] for r in reps])
    path = args.out / "lipprops_lateral_diffusion.npz"
    np.savez_compressed(path, **out)
    print(f"wrote {path}  ({len(diffusion)} systems)")


if __name__ == "__main__":
    sys.exit(main())
