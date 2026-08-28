#!/usr/bin/env python3
"""Within-run (autocorrelation-aware) error for the **Lipidator/ReSiS** tilt
modulus kappa_theta, by block-averaging -- the direct analogue of the block
refit that ``tilt_modulus.py`` does for the ost_pymodules chi.

Why a separate script (and env).  ``run_lipidator.py`` is stdlib-only and runs
``resis`` on the whole xtc, reporting one kappa per replicate; the project error
bar is then the SD across the 3 replicate means.  To go to the pseudoreplication
limit we also need each run's *within-run* sampling error.  ``resis`` has no
frame-range flag, so the block analogue is to run it on contiguous
sub-trajectory **xtc slices** -- which needs MDAnalysis to write them.  Hence
this lives apart from the stdlib ``run_lipidator.py`` and runs in ``lipid64``
(with ``PYTHONNOUSERSITE=1`` so a stray x86_64 scipy in ``~/.local`` can't
shadow the env's arm64 build -- the same guard ``run_tilt.sh`` uses)::

    PYTHONNOUSERSITE=1 conda run -n lipid64 python analysis/run_lipidator_blocks.py
    PYTHONNOUSERSITE=1 conda run -n lipid64 python analysis/run_lipidator_blocks.py --forcefields charmm --lipids DMPC

Production span.  ``resis`` reported kappa over the **whole** xtc, burn-in
included, while every other equilibrium observable (APL, d_HH, d_HC and now the
ost_pymodules chi, see ``refit_tilt_t0.py``) is measured on the pymbar
production span ``[t0:]``.  Same fix, same mechanism: the frames before ``t0``
are dropped when the sub-trajectory xtc is written, so both the reported
``kappa_full`` and the blocks are production-span quantities.  ``t0`` (frames)
is the ``tilt`` observable's cutoff in ``data/equilibrium_timeseries_stats.json``
(the per-frame tilt spread <delta theta^2>, the same series that sizes the
blocks) -- so **run stats.py first**; ``--no-t0`` restores the old whole-run
behaviour.  The whole-run value is still reported as ``kappa_whole_run`` for
comparison.

Block sizing.  Blocks must be at least as long as the tilt decorrelation time so
successive block kappas are ~independent.  That timescale is a property of the
bilayer, not of the exact director, so we reuse ``g_var`` (frames) from the
ost_pymodules side (``data/tilt_moduli.json``, written by ``tilt_modulus.py``)
rather than recomputing a per-frame director tilt here -- run the tilt pipeline
first.  If a system's ``g_var`` is missing we fall back to ``--default-g`` and
flag it.  ``autocorr.choose_n_blocks`` then picks the block count exactly as the
tilt pipeline does, and ``autocorr.block_sem`` is the SEM over block kappas.

Writes ``data/lipidator_blocks.json`` keyed ``{ff}_{lipid}_{rep}``:
``kappa_full`` (production-span resis, i.e. frames ``[t0:]``), ``kappa_whole_run``
(the same over every frame), ``t0_frames``/``t0_ns``, ``kappa_blocks`` (per block),
``kappa_block_mean``/``kappa_block_sem``, ``n_blocks``/``block_size``,
``g_used``/``g_source`` and ``block_size_lt_g`` (True => blocks shorter than the
decorrelation time, so the SEM is optimistic).
"""

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import MDAnalysis as mda

import autocorr
from run_lipidator import (RESIS, DIRECTORS, DATA, TRAJ, TILT_RE,
                           FORCEFIELDS, LIPIDS, REPS)

TILT_JSON = DATA / "tilt_moduli.json"
STATS_JSON = DATA / "equilibrium_timeseries_stats.json"
NS_PER_FRAME = 0.2


def load_t0():
    """{(ff, lipid, rep): t0_frames} for the per-frame tilt spread (stats.py)."""
    if not STATS_JSON.exists():
        return {}
    stats = json.loads(STATS_JSON.read_text())["per_replicate"]
    out = {}
    for sysname, recs in stats.get("tilt", {}).items():
        ff, lipid = sysname.rsplit("_", 1)
        for r in recs:
            out[(ff, lipid, str(r["rep"]))] = int(r["t0"])
    return out


def load_gvar():
    """{(ff,lipid,rep): g_var_frames} from the ost_pymodules tilt output."""
    if not TILT_JSON.exists():
        return {}
    d = json.loads(TILT_JSON.read_text())
    out = {}
    for key, rec in d.items():
        g = rec.get("g_var_frames")
        if g is not None and rec.get("ff") and rec.get("lipid") and rec.get("rep"):
            out[(rec["ff"], rec["lipid"], str(rec["rep"]))] = float(g)
    return out


def run_resis(xtc, ldx, ndx):
    """Return kappa (kt/nm^2) from a resis run, or None on failure/no match."""
    proc = subprocess.run([str(RESIS), str(xtc), str(ldx), str(ndx)],
                          capture_output=True, text=True)
    m = TILT_RE.search(proc.stdout)
    if proc.returncode != 0 or not m:
        return None
    return float(m.group(1))


def write_block_xtc(u, start, stop, path):
    """Write frames [start, stop) of universe ``u`` to ``path`` (all atoms, so
    the atom order still matches the lipidator .ndx)."""
    with mda.Writer(str(path), n_atoms=u.atoms.n_atoms) as W:
        for _ in u.trajectory[start:stop]:
            W.write(u.atoms)


def analyse_one(ff, lipid, rep, gvar, t0s, default_g, tmp_root):
    xtc = TRAJ / ("%s_%s%s.xtc" % (ff, lipid, rep))
    gro = TRAJ / ("%s_%s.gro" % (ff, lipid))
    ldx = DIRECTORS / ("%s.ldx" % ff)
    ndx = DATA / ("lipidator_%s_%s.ndx" % (ff, lipid))
    for f in (xtc, gro, ldx, ndx):
        if not f.exists():
            print("  skip %s_%s_%s: missing %s" % (ff, lipid, rep, f.name))
            return None

    kappa_whole = run_resis(xtc, ldx, ndx)
    if kappa_whole is None:
        print("  FAIL %s_%s_%s: full-trajectory resis produced no tilt line"
              % (ff, lipid, rep))
        return None

    u = mda.Universe(str(gro), str(xtc))
    n_frames_full = u.trajectory.n_frames
    t0 = t0s.get((ff, lipid, str(rep)), 0)
    t0 = 0 if t0 >= n_frames_full - 1 else t0
    n_frames = n_frames_full - t0
    g = gvar.get((ff, lipid, str(rep)))
    g_source = "tilt_moduli.json"
    if g is None:
        g, g_source = default_g, "default"
    n_blocks, block_size = autocorr.choose_n_blocks(n_frames, g)
    block_short = block_size < g

    kappa_blocks = []
    tmp = Path(tempfile.mkdtemp(dir=str(tmp_root)))
    try:
        # Production span [t0:] -- the canonical kappa, matching the convention
        # every other equilibrium observable uses.
        if t0 > 0:
            ppath = tmp / "prod.xtc"
            write_block_xtc(u, t0, n_frames_full, ppath)
            kappa_full = run_resis(ppath, ldx, ndx)
            ppath.unlink(missing_ok=True)
            if kappa_full is None:
                print("    production-span resis failed; falling back to whole run")
                kappa_full = kappa_whole
        else:
            kappa_full = kappa_whole
        for bi, (s, e) in enumerate(autocorr.block_ranges(n_frames, block_size, n_blocks)):
            bpath = tmp / ("block_%02d.xtc" % bi)
            write_block_xtc(u, t0 + s, t0 + e, bpath)
            kb = run_resis(bpath, ldx, ndx)
            bpath.unlink(missing_ok=True)
            if kb is not None and np.isfinite(kb):
                kappa_blocks.append(kb)
            else:
                print("    block %d resis failed" % bi)
    finally:
        shutil.rmtree(str(tmp), ignore_errors=True)

    sem = autocorr.block_sem(kappa_blocks)
    mean = float(np.mean(kappa_blocks)) if kappa_blocks else None
    print("  %s_%s_%s: kappa = %.2f (prod) / %.2f (whole run), t0 = %.0f ns | "
          "g=%.1f fr (%s) -> %d blocks x %d fr%s | "
          "kappa_block = %.2f +/- %.2f kt/nm^2"
          % (ff, lipid, rep, kappa_full, kappa_whole, t0 * NS_PER_FRAME,
             g, g_source, n_blocks, block_size,
             " [block<g: optimistic]" if block_short else "",
             mean if mean is not None else float("nan"), sem))
    return {"ff": ff, "lipid": lipid, "rep": int(rep),
            "kappa_full": kappa_full, "kappa_whole_run": kappa_whole,
            "t0_frames": int(t0), "t0_ns": t0 * NS_PER_FRAME,
            "n_frames": int(n_frames), "n_frames_full": int(n_frames_full),
            "g_used": float(g), "g_source": g_source,
            "n_blocks": int(n_blocks), "block_size": int(block_size),
            "block_size_lt_g": bool(block_short),
            "kappa_blocks": kappa_blocks, "kappa_block_mean": mean,
            "kappa_block_sem": sem}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--forcefields", nargs="+", default=FORCEFIELDS)
    ap.add_argument("--lipids", nargs="+", default=LIPIDS)
    ap.add_argument("--reps", nargs="+", default=[str(r) for r in REPS])
    ap.add_argument("--no-t0", action="store_true",
                    help="analyse the whole trajectory (old behaviour) instead "
                         "of the pymbar production span [t0:]")
    ap.add_argument("--default-g", type=float, default=30.0,
                    help="fallback decorrelation time (frames) when tilt g_var "
                         "is unavailable for a system (default 30)")
    args = ap.parse_args()

    if not RESIS.exists():
        raise SystemExit("resis binary not found at %s" % RESIS)
    gvar = load_gvar()
    t0s = {} if args.no_t0 else load_t0()
    if not args.no_t0 and not t0s:
        print("note: no tilt t0 found in %s; analysing whole trajectories "
              "(run stats.py first)" % STATS_JSON.name)
    if not gvar:
        print("note: no g_var found in %s; using --default-g=%.0f for all "
              "(run tilt_modulus.py first for per-system block sizing)"
              % (TILT_JSON.name, args.default_g))

    tmp_root = Path(tempfile.gettempdir())
    results = {}
    for ff in args.forcefields:
        for lipid in args.lipids:
            for rep in args.reps:
                res = analyse_one(ff, lipid, str(rep), gvar, t0s,
                                  args.default_g, tmp_root)
                if res:
                    results["%s_%s_%s" % (ff, lipid, rep)] = res

    out = DATA / "lipidator_blocks.json"
    out.write_text(json.dumps(results, indent=2, sort_keys=True))
    print("\nwrote %s (%d systems)" % (out, len(results)))


if __name__ == "__main__":
    main()
