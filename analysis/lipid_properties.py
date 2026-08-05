#!/usr/bin/env python3
"""Compute equilibrium lipid properties with lipyphilic, for every
force-field / lipid / replicate, and save plot-ready arrays as .npz.

This is the *analysis* half of the pipeline; styling lives in the separate
``plot_lipid_properties.py`` so the (slow) analysis need only run once.

Run cluster-side in the ``lipid`` conda environment, which has lipyphilic +
MDAnalysis installed::

    conda run -n lipid64 python analysis/lipid_properties.py

Input layout (identical on the local box and the cluster)::

    ~/Simulations/{ff}_{lipid}/eq{1,2,3}/memb.gro      topology (waters-stripped)
    ~/Simulations/{ff}_{lipid}/eq{1,2,3}/inbox.xtc     trajectory (waters-stripped; --traj to override)

Both ``memb.gro`` and ``inbox.xtc`` have waters stripped (to speed up universe
setup); their atom sets must match.  ``memb.gro`` is used as the topology for
both the MDAnalysis universes and g_lomepro's ``-s``, which also sidesteps
MDAnalysis's inability to read the too-new GROMACS 2026.3 ``md.tpr``.

Properties, one .npz written to ``data/`` per property:
  * area_per_lipid   -- AreaPerLipid, (n_frames, n_lipids) per replicate (A^2)
  * memb_thickness   -- MembThickness, per-frame samples (A)
  * registration     -- Registration, per-frame samples (correlation coeff.)
  * order_parameter  -- full per-carbon S_CC for every chain and frame
  * lateral_diffusion-- MSD curve (vs lag time) + fitted diffusion coefficient

The first three are histogram-ready pooled sample arrays.  order_parameter
stores the full per-carbon S_CC array per replicate (large -- ~100 MB -- with
the per-carbon averaging left to the plotting script), and lateral_diffusion
stores the MSD-vs-lagtime curve and the fitted D per replicate.

For replicate 1 it also renders two AreaPerLipid.project_area() heatmaps per
system directly to ``plots/lipprop_aplproj_{ff}_{lipid}_{avg,final}.png`` (APL
projected onto the membrane plane: mean over the trajectory, and the final
frame).

S_CD (deuterium) order parameter via g_lomepro
----------------------------------------------
By default the script *also* computes the NMR-comparable S_CD order parameter
with the external g_lomepro binary (a second step, alongside the lipyphilic npz
analysis above).  For each system it builds the shared index
``{ff}_{lipid}/lomepro.ndx`` (``make_order_index.generate``) and runs g_lomepro
``-order`` in every ``eq{rep}/lomepro/``, then copies the resulting per-chain
profiles into ``data/{ff}_{lipid}_{rep}_scd_sn{1,2}.dat`` for ``plot_scd``.

``--skip-lomepro`` / ``--skip-lipyphilic`` run just one half; the g_lomepro half
needs only numpy + the binary (not lipyphilic), so on a bare node use::

    python analysis/lipid_properties.py --skip-lipyphilic [--lomepro-bin PATH]
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
# Force a headless matplotlib backend before lipyphilic imports pyplot, so the
# AreaPerLipid.project_area() figures render in worker processes without a display.
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np

# MDAnalysis + lipyphilic are only needed for the default (S_CC / npz) analysis.
# The --lomepro path just shells out to the g_lomepro binary, so keep these
# imports optional and error only if the lipyphilic path is actually taken.
try:
    import MDAnalysis as mda
    from MDAnalysis.transformations import NoJump
    from lipyphilic import AssignLeaflets
    from lipyphilic.analysis.area_per_lipid import AreaPerLipid
    from lipyphilic.analysis.registration import Registration
    from lipyphilic.analysis.memb_thickness import MembThickness
    from lipyphilic.analysis.lateral_diffusion import MSD
    _HAVE_LIPYPHILIC = True
except ImportError as _exc:
    _HAVE_LIPYPHILIC = False
    _LIPYPHILIC_ERR = _exc

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
PLOTS = ROOT / "plots"

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
    residue, so a single ``resname`` selection yields both chains (scc_profile
    then treats each residue as an independent chain)."""
    n = TAIL_NC[lipid]
    if ff in ("charmm", "slipids"):
        sn1 = f"resname {lipid} and name " + " ".join(f"C2{i}" for i in range(2, n + 1))
        sn2 = f"resname {lipid} and name " + " ".join(f"C3{i}" for i in range(2, n + 1))
        return [sn1, sn2]
    # amber
    return [f"resname {AMBER_TAIL_RESN[lipid]} and name C1*"]


def scc_samples(u, ff, lipid):
    """Per-carbon S_CC for every acyl chain at every frame of this trajectory.

    Returns ``(carbons, data)`` where ``carbons`` are the acyl carbon numbers and
    ``data`` has shape ``(n_chain_frames, n_carbons)`` -- the full distribution,
    pooled over both tails (rows are one chain at one frame).  S_CC of each C-C
    bond is ½⟨3cos²θ−1⟩ with the bilayer normal along z (lipyphilic's default);
    the plotting script averages over the sample axis to get the profile, and
    averaging that profile over carbons reproduces lipyphilic's per-chain SCC.

    The full array is large (this is the ~100 MB file), but keeping every sample
    lets the styling stage choose how to reduce it."""
    n = TAIL_NC[lipid]
    n_carbons = n - 1                    # tail carbons in the selection (acyl 2..n)
    carbons = np.arange(2, n)            # acyl number of each bond's first carbon
    blocks = []
    for sel in tail_selections(ff, lipid):
        tail = u.select_atoms(sel)
        _, counts = np.unique(tail.resindices, return_counts=True)
        if not np.all(counts == n_carbons):
            raise ValueError(
                f"{ff} {lipid}: acyl-chain selection yields "
                f"{sorted(set(int(c) for c in counts))} carbons per residue, "
                f"expected {n_carbons}. A lipid's two tails are sharing a residue "
                f"id (GROMACS numbers them identically), so MDAnalysis merges "
                f"them.  Renumber memb.gro so every tail is its own residue first."
            )
        m = tail.n_residues
        # Order carbons within each residue by acyl number so CHARMM (ascending in
        # the topology) and AMBER (descending) give the same head->tail profile.
        acyl = np.array([int(name[2:]) for name in tail.names])
        order = np.lexsort((acyl, tail.resindices))
        for ts in u.trajectory:
            pos = tail.positions[order].reshape(m, n_carbons, 3)
            cc = np.diff(pos, axis=1)                 # C-C bond vectors
            box = ts.dimensions[:3]
            cc -= box * np.round(cc / box)            # minimum image (intramolecular)
            cos = cc[:, :, 2] / np.linalg.norm(cc, axis=2)
            blocks.append(0.5 * (3 * cos**2 - 1))     # (m chains, n_carbons)
    return carbons, np.vstack(blocks)                 # (n_chain_frames, n_carbons)


def build_universe(run_dir, traj):
    """Universe for ``run_dir`` from ``memb.gro`` + the trajectory.

    ``memb.gro`` is the waters-stripped membrane structure whose atom set matches
    the (also waters-stripped) ``inbox.xtc``.  Using it both keeps the topology
    consistent with the trajectory and sidesteps MDAnalysis's inability to read
    the too-new GROMACS 2026.3 ``md.tpr``."""
    return mda.Universe(str(run_dir / "memb.gro"), str(traj))


def save_area_projections(apl, u, ff, lipid):
    """Write AreaPerLipid.project_area() heatmaps for this system to ``plots/``:
    the whole-trajectory mean and the final frame.  ``plots/lipprop_aplproj_
    {ff}_{lipid}_{avg,final}.png``.

    Must be called *before* the NoJump transform is added -- project_area reads
    lipid COMs from the trajectory and wraps them into the unit cell, which only
    works while coordinates are still in-box.  A failure here is downgraded to a
    warning so it never costs the (already-computed) numerical results."""
    import matplotlib.pyplot as plt
    PLOTS.mkdir(parents=True, exist_ok=True)
    # project_area() takes each lipid's COM with unwrap=True, which needs bonds;
    # a .gro topology (memb.gro) has none, so guess them on the lipid atoms first.
    try:
        apl.membrane.residues.atoms.guess_bonds()
    except Exception as exc:
        print(f"    WARNING: bond guessing failed for {ff}_{lipid}; skipping APL "
              f"projections: {str(exc)[:80]}", flush=True)
        return
    n = u.trajectory.n_frames
    # (tag, project_area kwargs, title): default kwargs average over all frames;
    # start=n-1 selects only the last frame.
    for tag, kw, sub in [("avg", {}, "mean over trajectory"),
                         ("final", {"start": n - 1}, "final frame")]:
        try:
            proj = apl.project_area(cmap="viridis", **kw)
            proj.ax.set_title(f"{ff} {lipid} — APL, {sub} (rep 1)",
                              loc="left", weight="bold", fontsize=9)
            if proj.cbar is not None:
                proj.cbar.set_label(r"APL ($\mathrm{\AA}^2$)")
            out = PLOTS / f"lipprop_aplproj_{ff}_{lipid}_{tag}.png"
            proj.fig.savefig(out, dpi=300, bbox_inches="tight")
            plt.close(proj.fig)
            print(f"    wrote {out.name}", flush=True)
        except Exception as exc:  # visualisation only -- keep the numbers
            print(f"    WARNING: APL {tag} projection failed for {ff}_{lipid}: "
                  f"{str(exc)[:80]}", flush=True)


def analyse(u, ff, lipid, rep):
    """Run all five lipyphilic analyses on one trajectory; return a dict of
    plot-ready arrays.  For replicate 1 it also writes the APL projection maps."""
    leaflets = AssignLeaflets(u, lipid_sel=PHOSPHATE)
    leaflets.run()

    apl = AreaPerLipid(u, lipid_sel=PHOSPHATE, leaflets=leaflets.leaflets)
    apl.run()

    # APL projection maps (rep 1 only) -- before NoJump, while coords are in-box.
    if rep == 1:
        save_area_projections(apl, u, ff, lipid)

    thick = MembThickness(u, lipid_sel=PHOSPHATE, leaflets=leaflets.leaflets)
    thick.run()

    reg = Registration(u, upper_sel=PHOSPHATE, lower_sel=PHOSPHATE,
                       leaflets=leaflets.leaflets)
    reg.run()

    # Order parameter: keep the full per-carbon S_CC for every chain and frame
    # (the plotting stage averages to a profile).  scc_samples() also guards
    # against a topology that merges a lipid's two tails (see its docstring).
    scc_carbons, scc_data = scc_samples(u, ff, lipid)

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
        # Keep APL 2-D (n_frames, n_lipids) rather than ravelling: the histogram
        # flattens it anyway, but the per-frame trajectory plot needs the frame
        # axis to show per-frame mean +/- SD over lipids.
        "area_per_lipid": np.asarray(apl.areas).T,
        "memb_thickness": np.asarray(thick.memb_thickness).ravel(),
        "registration": np.asarray(reg.registration).ravel(),
        "scc_carbons": scc_carbons,
        "scc_data": scc_data,
        "msd_lagtimes": np.asarray(msd.lagtimes),
        "msd_curve": np.nanmean(np.asarray(msd.msd), axis=0),  # mean over lipids
        "D": float(D),
        "Dsem": float(Dsem),
    }


# Properties saved as pooled 1-D sample arrays (histograms downstream).  Order
# parameter and lateral diffusion are stored differently (see main()).
SAMPLE_PROPS = ["area_per_lipid", "memb_thickness", "registration"]


def run_one(job):
    """Analyse a single trajectory.  Top-level (picklable) so it can be farmed
    out to a process pool.  Returns ``(ff, lipid, rep, result_dict)``."""
    ff, lipid, rep, run_dir, traj = job
    print(f"analysing {ff}_{lipid} eq{rep}  ({traj.name})", flush=True)
    u = build_universe(run_dir, traj)
    return ff, lipid, rep, analyse(u, ff, lipid, rep)


def run_lomepro(args):
    """Run g_lomepro's ``-order`` (S_CD deuterium order parameter) analysis in
    every ``eq{rep}/lomepro/`` directory.

    For each ``{ff}_{lipid}`` system this (re)generates the shared order index
    ``{ff}_{lipid}/lomepro.ndx`` from a replicate's ``memb.gro`` (via
    ``make_order_index.generate``), then per replicate creates a fresh
    ``eq{rep}/lomepro/`` (overwriting any previous one) and runs, from the
    ``eq{rep}`` directory::

        echo "<groups>" | g_lomepro_static -f inbox.xtc -n ../lomepro.ndx \\
            -s memb.gro -lip_num <N> [-unsat 2] -order lomepro/scd.out

    The ``sn{1,2}_double_bond`` groups and ``-unsat 2`` are DOPC-specific; the
    myristoyl lipids (DMPC/DMPG) are saturated, so they select only
    ``Membrane sn1 sn2`` and omit ``-unsat``.  g_lomepro writes
    ``lomepro/scd.out_AVG_sn{1,2}.dat`` (columns: carbon, mean -S_CD, sd).

    g_lomepro needs ``GMXLIB`` set to GROMACS's data directory; if it is unset,
    it is derived from ``gmx`` on ``PATH`` (``<prefix>/bin/gmx`` ->
    ``<prefix>/share/gromacs/top``)."""
    import shutil
    import subprocess
    from make_order_index import generate

    binary = Path(args.lomepro_bin)
    if not binary.exists():
        sys.exit(f"g_lomepro binary not found: {binary} (set --lomepro-bin)")
    DATA.mkdir(parents=True, exist_ok=True)  # destination for the copied S_CD data

    # g_lomepro (a GROMACS tool) needs GMXLIB pointing at GROMACS's data dir.
    # Respect an existing GMXLIB; otherwise derive it from `gmx` on PATH, whose
    # bin/ sits a fixed two levels above share/gromacs/top
    # (<prefix>/bin/gmx -> <prefix>/share/gromacs/top).
    env = dict(os.environ)
    if "GMXLIB" not in env:
        gmx = shutil.which("gmx")
        if gmx:
            gmxlib = Path(gmx).parent.parent / "share" / "gromacs" / "top"
            env["GMXLIB"] = str(gmxlib)
            print(f"GMXLIB={gmxlib}"
                  + ("" if gmxlib.is_dir() else "  (WARNING: does not exist)"))
        else:
            print("WARNING: GMXLIB unset and `gmx` not on PATH; g_lomepro may fail")

    ran = 0
    for ff in args.forcefields:
        for lipid in args.lipids:
            system_dir = args.sims_root / f"{ff}_{lipid}"
            if not system_dir.is_dir():
                print(f"skip {ff}_{lipid}: {system_dir} not found")
                continue
            # The index is shared across replicates (identical atom order), so
            # build it once from the first replicate that has a memb.gro.  It must
            # come from memb.gro (not md.gro) so the serials match the
            # waters-stripped inbox.xtc that g_lomepro reads.
            gro = next((system_dir / f"eq{r}" / "memb.gro" for r in args.reps
                        if (system_dir / f"eq{r}" / "memb.gro").exists()), None)
            if gro is None:
                print(f"skip {ff}_{lipid}: no eq*/memb.gro to build the index from")
                continue
            ndx = system_dir / "lomepro.ndx"
            info = generate(str(gro), out=str(ndx), ff=ff, lipid=lipid)
            groups = "Membrane sn1 sn2"
            if info["unsat"]:
                groups += " sn1_double_bond sn2_double_bond"
            print(f"{ff}_{lipid}: wrote {ndx.name} "
                  f"({info['nlip']} lipids, -unsat {info['unsat']})")

            for rep in args.reps:
                run_dir = system_dir / f"eq{rep}"
                traj, gro_r = run_dir / args.traj, run_dir / "memb.gro"
                if not run_dir.is_dir():
                    continue
                if not (traj.exists() and gro_r.exists()):
                    print(f"  skip eq{rep}: {args.traj} or memb.gro missing")
                    continue
                lome = run_dir / "lomepro"
                if lome.exists():
                    shutil.rmtree(lome)          # overwrite a previous run
                lome.mkdir()
                cmd = [str(binary), "-f", args.traj, "-n", "../lomepro.ndx",
                       "-s", "memb.gro", "-lip_num", str(info["nlip"]),
                       "-order", "lomepro/scd.out"]
                if info["unsat"]:
                    cmd += ["-unsat", str(info["unsat"])]
                print(f"  eq{rep}: running g_lomepro ...", flush=True)
                proc = subprocess.run(cmd, input=groups + "\n", text=True,
                                      cwd=str(run_dir), capture_output=True, env=env)
                if proc.returncode != 0:
                    print(f"  eq{rep}: FAILED (rc={proc.returncode}); last output:")
                    tail = (proc.stderr or proc.stdout).strip().splitlines()[-15:]
                    print("    " + "\n    ".join(tail))
                else:
                    ran += 1
                    print(f"  eq{rep}: wrote {lome}/scd.out_AVG_sn*.dat")
                    # Copy the per-chain S_CD profiles into data/ for plot_scd(),
                    # keyed by replicate so each run is kept (plot_scd globs all
                    # {ff}_{lipid}_*scd_sn*.dat and treats each as one sample).
                    for sn in ("sn1", "sn2"):
                        src = lome / f"scd.out_AVG_{sn}.dat"
                        dst = DATA / f"{ff}_{lipid}_{rep}_scd_{sn}.dat"
                        if src.exists():
                            shutil.copyfile(src, dst)
                            print(f"         copied -> {dst}")
                        else:
                            print(f"         WARNING: {src.name} not produced")
    print(f"g_lomepro finished: {ran} replicate(s) run")


def run_lipyphilic(args):
    """Compute the lipyphilic S_CC/APL/thickness/registration/diffusion properties
    for every trajectory and write the ``lipprops_*.npz`` files."""
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
    per_run = {}                             # system -> {rep: result_dict}
    for ff, lipid, rep, res in completed:
        system = f"{ff}_{lipid}"
        for p in SAMPLE_PROPS:
            results[p].setdefault(system, {})[rep] = res[p]
        per_run.setdefault(system, {})[rep] = res

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

    # ---- write order parameter: full per-carbon S_CC, one array per replicate --
    # (per-rep only; together they are the full data, so no pooled duplicate).
    out = {}
    for system, by_rep in per_run.items():
        reps = sorted(by_rep)
        out[f"{system}_carbons"] = by_rep[reps[0]]["scc_carbons"]
        for r in reps:
            out[f"{system}_eq{r}"] = by_rep[r]["scc_data"]  # (n_chain_frames, n_carbons)
    path = args.out / "lipprops_order_parameter.npz"
    np.savez_compressed(path, **out)
    print(f"wrote {path}  ({len(per_run)} systems)")

    # ---- write lateral diffusion: MSD curve + fitted D per replicate ----
    out = {}
    for system, by_rep in per_run.items():
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
    print(f"wrote {path}  ({len(per_run)} systems)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sims-root", type=Path, default=Path.home() / "Simulations")
    ap.add_argument("--forcefields", nargs="+", default=FORCEFIELDS)
    ap.add_argument("--lipids", nargs="+", default=LIPIDS)
    ap.add_argument("--reps", nargs="+", type=int, default=REPLICATES)
    ap.add_argument("--traj", default="inbox.xtc",
                    help="trajectory filename inside each eq dir (default: inbox.xtc)")
    ap.add_argument("--nprocs", type=int, default=1,
                    help="analyse this many trajectories in parallel, one core "
                         "each (the analyses themselves are serial-only). "
                         "Capped at the number of trajectories found.")
    ap.add_argument("--out", type=Path, default=DATA)
    # By default both analyses run (the whole pipeline is cheap to re-run); use
    # the skip flags to do just one -- e.g. --skip-lipyphilic to run g_lomepro on
    # a node without the lipid64 conda env.
    ap.add_argument("--skip-lipyphilic", action="store_true",
                    help="skip the lipyphilic S_CC/APL/thickness/registration/"
                         "diffusion analysis (the lipprops_*.npz files)")
    ap.add_argument("--skip-lomepro", action="store_true",
                    help="skip the g_lomepro S_CD analysis")
    ap.add_argument("--lomepro-bin",
                    default=str(Path.home() / "software" / "g_lomepro" / "g_lomepro_static"),
                    help="path to the g_lomepro static binary")
    args = ap.parse_args()

    if not args.skip_lipyphilic:
        if not _HAVE_LIPYPHILIC:
            sys.exit(f"MDAnalysis/lipyphilic unavailable ({_LIPYPHILIC_ERR}); "
                     "install them (e.g. the lipid64 env) or pass --skip-lipyphilic.")
        run_lipyphilic(args)
    if not args.skip_lomepro:
        run_lomepro(args)


if __name__ == "__main__":
    sys.exit(main())
