#!/usr/bin/env python3
"""Compute equilibrium lipid properties with lipyphilic, for every
force-field / lipid / replicate, and save plot-ready arrays as .npz.

This is the *analysis* half of the pipeline; styling lives in the separate
``plot_lipid_properties.py`` so the (slow) analysis need only run once.

Run cluster-side in the ``lipid64`` conda environment, which has lipyphilic +
MDAnalysis installed::

    conda run -n lipid64 python analysis/lipid_properties.py

Input layout (identical on the local box and the cluster)::

    ~/Simulations/{ff}_{lipid}/eq{1,2,3}/md.tpr        topology
    ~/Simulations/{ff}_{lipid}/eq{1,2,3}/*.xtc         trajectory

The GROMACS 2026.3 ``.tpr`` files are too new for MDAnalysis's TPR parser, so
if MDAnalysis can't read ``md.tpr`` directly the script converts it to a
``.gro`` once via ``gmx editconf`` (set ``$GMX`` if the binary isn't ``gmx``).
That step needs GROMACS >= the version that wrote the tpr on PATH.

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
import subprocess
import sys
from pathlib import Path

import numpy as np
import MDAnalysis as mda
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

# Acyl tails carry twice as many carbons in DOPC (18:1) as in DMPC/DMPG (14:0).
TAIL_NC = {"DOPC": 18, "DMPC": 14, "DMPG": 14}
# AMBER splits each lipid into a head residue + two tail residues; the tail
# resname is the acyl species.
AMBER_TAIL_RESN = {"DOPC": "OL", "DMPC": "MY", "DMPG": "MY"}

GMX = os.environ.get("GMX", "gmx")


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


def pick_trajectory(run_dir, explicit=None):
    """Choose the production trajectory in ``run_dir``.  With ``explicit`` set,
    use that filename; otherwise pick the ``.xtc`` with the most frames (so the
    50-frame test trajectory and a 1200-frame production run both work)."""
    if explicit:
        p = run_dir / explicit
        return p if p.exists() else None
    xtcs = sorted(run_dir.glob("*.xtc"))
    if not xtcs:
        return None
    if len(xtcs) == 1:
        return xtcs[0]
    best, best_n = None, -1
    for x in xtcs:
        try:
            n = len(mda.coordinates.XTC.XTCReader(str(x)))
        except Exception:
            n = -1
        if n > best_n:
            best, best_n = x, n
    print(f"    multiple .xtc found, using {best.name} ({best_n} frames)")
    return best


def build_universe(run_dir, traj):
    """Universe for ``run_dir``, coping with too-new .tpr files.

    Tries ``md.tpr`` directly; on failure reuses/creates a ``.gro`` (via
    ``gmx editconf``) as a topology MDAnalysis can read."""
    tpr = run_dir / "md.tpr"
    if tpr.exists():
        try:
            return mda.Universe(str(tpr), str(traj))
        except Exception as exc:  # tpx version too new for MDAnalysis, etc.
            print(f"    tpr unreadable by MDAnalysis ({str(exc)[:50]}...); using .gro")
    # Reuse an existing structure file if present, else generate one from the tpr.
    gro = run_dir / "mda_structure.gro"
    if not gro.exists():
        existing = sorted(run_dir.glob("*.gro"))
        if existing:
            gro = existing[0]
        else:
            print(f"    running {GMX} editconf to make {gro.name}")
            subprocess.run(
                [GMX, "editconf", "-f", str(tpr), "-o", str(gro)],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
    return mda.Universe(str(gro), str(traj))


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
    scc_samples = []
    for sel in tail_selections(ff, lipid):
        n_expected = TAIL_NC[lipid] - 1  # carbons per chain feeding into S_CC
        chain = SCC(u, tail_sel=sel)
        chain.run()
        per_chain_carbons = u.select_atoms(sel).n_atoms / chain.SCC.shape[0]
        if abs(per_chain_carbons - n_expected) > 1.5:
            print(f"    WARNING: {ff} {lipid} SCC sees {per_chain_carbons:.1f} "
                  f"carbons/chain, expected ~{n_expected} -- tails may be merged "
                  f"into one residue; check topology resids.")
        scc_samples.append(np.asarray(chain.SCC).ravel())

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


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sims-root", type=Path, default=Path.home() / "Simulations")
    ap.add_argument("--forcefields", nargs="+", default=FORCEFIELDS)
    ap.add_argument("--lipids", nargs="+", default=LIPIDS)
    ap.add_argument("--reps", nargs="+", type=int, default=REPLICATES)
    ap.add_argument("--traj", default=None,
                    help="explicit trajectory filename inside each eq dir "
                         "(default: the .xtc with the most frames)")
    ap.add_argument("--out", type=Path, default=DATA)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    # Accumulate: results[prop][f"{ff}_{lipid}"] -> list of per-replicate arrays.
    results = {p: {} for p in SAMPLE_PROPS}
    diffusion = {}  # f"{ff}_{lipid}" -> dict of lists

    for ff in args.forcefields:
        for lipid in args.lipids:
            system = f"{ff}_{lipid}"
            for rep in args.reps:
                run_dir = args.sims_root / system / f"eq{rep}"
                if not run_dir.is_dir():
                    print(f"skip {system} eq{rep}: {run_dir} not found")
                    continue
                traj = pick_trajectory(run_dir, args.traj)
                if traj is None:
                    print(f"skip {system} eq{rep}: no trajectory")
                    continue
                print(f"analysing {system} eq{rep}  ({traj.name})")
                u = build_universe(run_dir, traj)
                res = analyse(u, ff, lipid)

                for p in SAMPLE_PROPS:
                    results[p].setdefault(system, []).append(res[p])
                d = diffusion.setdefault(
                    system, {"lagtimes": [], "msd": [], "D": [], "Dsem": []})
                d["lagtimes"].append(res["msd_lagtimes"])
                d["msd"].append(res["msd_curve"])
                d["D"].append(res["D"])
                d["Dsem"].append(res["Dsem"])

    # ---- write sample-based properties: pooled + per-replicate ----
    for p in SAMPLE_PROPS:
        out = {}
        for system, reps in results[p].items():
            out[system] = np.concatenate(reps)
            for i, arr in enumerate(reps, start=1):
                out[f"{system}_eq{i}"] = arr
        path = args.out / f"lipprops_{p}.npz"
        np.savez_compressed(path, **out)
        print(f"wrote {path}  ({len(results[p])} systems)")

    # ---- write lateral diffusion: MSD curve + fitted D per replicate ----
    out = {}
    for system, d in diffusion.items():
        # Replicates share dt but may differ in length; truncate to the common min.
        nlag = min(len(x) for x in d["msd"])
        out[f"{system}_lagtimes"] = d["lagtimes"][0][:nlag]
        out[f"{system}_msd"] = np.vstack([m[:nlag] for m in d["msd"]])
        out[f"{system}_D"] = np.asarray(d["D"])
        out[f"{system}_Dsem"] = np.asarray(d["Dsem"])
    path = args.out / "lipprops_lateral_diffusion.npz"
    np.savez_compressed(path, **out)
    print(f"wrote {path}  ({len(diffusion)} systems)")


if __name__ == "__main__":
    sys.exit(main())
