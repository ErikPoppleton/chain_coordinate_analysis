#!/usr/bin/env python2
"""Refit the ost_pymodules tilt modulus **chi on the pymbar production span**
``[t0:]``, so chi follows the same equilibration convention as every other
equilibrium observable (APL, d_HH, d_HC are all sliced to ``[t0:]`` before the
mean and the hierarchical CI are formed).

This is cheap: ``tilt_modulus.py`` already stores the raw per-frame per-lipid
tilt array in ``data/tilt_series_{ff}_{lipid}_{rep}.npz``, so the refit needs
only ``FitTiltDistribution`` -- **not** another OST density/normal pass (hours).
It still has to run under the OpenStructure Python 2.7, because
``ost_pymodules.lipid_analysis`` does ``from ost import *`` at import::

    ~/miniconda3/envs/openstructure/bin/python analysis/refit_tilt_t0.py

``t0`` (frames) comes from ``data/equilibrium_timeseries_stats.json``
(``stats.py``, observable ``tilt`` = the per-frame tilt spread <delta theta^2>,
the coordinate chi is fitted from) -- so **run stats.py first**.

Rewrites ``data/tilt_moduli.json`` in place, per replicate:
  * ``chi_kT`` / ``dchi_kT`` / ``chi_block_sem`` / ``g_var_frames`` /
    ``n_blocks`` / ``block_size`` / ``n_frames``  -> the **post-t0** values
    (these are the keys the table and plot scripts read, so they now carry the
    production-span fit);
  * ``t0_frames`` / ``t0_ns``                     -> what was discarded;
  * ``*_full``                                    -> the previous whole-run
    values, kept for comparison (written once, never overwritten, and the refit
    always starts from the raw npz, so re-running is idempotent).
"""

import os
import sys
import json
import argparse

import numpy as np

OST_PYMODULES = os.path.expanduser("~/software/ost_pymodules")
sys.path.append(OST_PYMODULES)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import lipid_analysis
import autocorr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
PLOTS = os.path.join(ROOT, "plots", "tilt")

NS_PER_FRAME = 0.2
FULL_KEYS = ["chi_kT", "dchi_kT", "chi_block_sem", "chi_block_mean",
             "g_var_frames", "g_mean_frames", "n_blocks", "block_size",
             "n_frames", "neff_var", "block_size_lt_g"]


def load_t0():
    """{(ff, lipid, rep): t0_frames} for the 'tilt' observable."""
    p = os.path.join(DATA, "equilibrium_timeseries_stats.json")
    if not os.path.isfile(p):
        raise SystemExit("missing %s -- run 'conda run -n base python "
                         "analysis/stats.py' first" % os.path.basename(p))
    with open(p) as fh:
        stats = json.load(fh)["per_replicate"]
    if "tilt" not in stats:
        raise SystemExit("no 'tilt' observable in the stats JSON -- rerun "
                         "stats.py (it reads data/tilt_series_*.npz)")
    out = {}
    for sysname, recs in stats["tilt"].items():
        # Split from the *left*: the force field is always one token, while the
        # lipid label may carry a condition suffix (DMPG_JC).  rsplit would
        # read "charmm_DMPG_JC" as ff="charmm_DMPG", lipid="JC" and silently
        # match no tilt entry.
        ff, lipid = sysname.split("_", 1)
        for r in recs:
            out[(ff, lipid, int(r["rep"]))] = int(r["t0"])
    return out


def refit(T, nbins, outdir, basis):
    """(chi, dchi, chi_block_sem, ...) from an (n_lipids, n_frames) tilt array."""
    n_frames = T.shape[1]
    pooled = [x for row in T for x in row]
    chi, dchi, chi_list = lipid_analysis.FitTiltDistribution(
        pooled, nbins=nbins, outdir=outdir, filename_basis=basis)

    theta_var_t = T.var(axis=0)
    theta_mean_t = T.mean(axis=0)
    g_var = float(autocorr.statistical_inefficiency(theta_var_t))
    g_mean = float(autocorr.statistical_inefficiency(theta_mean_t))
    n_blocks, block_size = autocorr.choose_n_blocks(n_frames, g_var)
    chi_blocks = []
    for s, e in autocorr.block_ranges(n_frames, block_size, n_blocks):
        sub = [x for row in T[:, s:e] for x in row]
        try:
            cb, _, _ = lipid_analysis.FitTiltDistribution(sub, nbins=nbins)
            if np.isfinite(cb):
                chi_blocks.append(float(cb))
        except Exception as exc:
            print("    block fit failed: %s" % exc)
    return {"chi_kT": float(chi), "dchi_kT": float(dchi),
            "chi_list": [float(c) for c in chi_list],
            "g_var_frames": g_var, "g_mean_frames": g_mean,
            "neff_var": float(n_frames / g_var),
            "n_blocks": int(n_blocks), "block_size": int(block_size),
            "block_size_lt_g": bool(block_size < g_var),
            "n_frames": int(n_frames),
            "chi_blocks": chi_blocks,
            "chi_block_mean": float(np.mean(chi_blocks)) if chi_blocks else None,
            "chi_block_sem": autocorr.block_sem(chi_blocks)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nbins", type=int, default=90)
    ap.add_argument("--keys", nargs="+", default=None,
                    help="only refit these {ff}_{lipid}_{rep} keys")
    ap.add_argument("--no-plots", action="store_true",
                    help="skip the regenerated tilt-distribution/PMF PNGs")
    args = ap.parse_args()

    t0s = load_t0()
    jpath = os.path.join(DATA, "tilt_moduli.json")
    if not os.path.isfile(jpath):
        raise SystemExit("missing %s -- run the tilt pipeline first" % jpath)
    with open(jpath) as fh:
        moduli = json.load(fh)

    print("%-22s %8s %10s %10s %10s" % ("system", "t0(ns)", "chi_full",
                                        "chi_prod", "block SEM"))
    for key in sorted(moduli):
        if args.keys and key not in args.keys:
            continue
        rec = moduli[key]
        ff, lipid, rep = rec["ff"], rec["lipid"], int(rec["rep"])
        t0 = t0s.get((ff, lipid, rep))
        if t0 is None:
            print("  skip %s: no tilt t0 in the stats JSON" % key)
            continue
        npz = os.path.join(DATA, "tilt_series_%s.npz" % key)
        if not os.path.isfile(npz):
            print("  skip %s: missing %s" % (key, os.path.basename(npz)))
            continue

        # The npz holds the RAW full-run tilts, so the refit is idempotent.
        T = np.load(npz)["tilts"]
        if t0 >= T.shape[1] - 1:
            print("  skip %s: t0=%d >= n_frames=%d" % (key, t0, T.shape[1]))
            continue

        # Stash the whole-run numbers once, before the first overwrite.
        for k in FULL_KEYS:
            fk = k + "_full"
            if fk not in rec and k in rec:
                rec[fk] = rec[k]
        chi_full = rec.get("chi_kT_full", rec.get("chi_kT"))

        outdir = "" if args.no_plots else os.path.join(PLOTS, "%s_%s" % (ff, lipid))
        if outdir and not os.path.isdir(outdir):
            os.makedirs(outdir)
        res = refit(T[:, t0:], args.nbins, outdir, key)
        rec.update(res)
        rec["t0_frames"] = int(t0)
        rec["t0_ns"] = t0 * NS_PER_FRAME
        rec["n_frames_full"] = int(T.shape[1])
        moduli[key] = rec
        print("%-22s %8.1f %10.3f %10.3f %10.3f"
              % (key, t0 * NS_PER_FRAME, chi_full, res["chi_kT"],
                 res["chi_block_sem"]))

    with open(jpath, "w") as fh:
        json.dump(moduli, fh, indent=2, sort_keys=True)
    print("\nwrote %s" % jpath)


if __name__ == "__main__":
    main()
