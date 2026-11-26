#!/usr/bin/env python3
"""Lateral lipid diffusion from the full-system equilibrium trajectories.

Replaces the lipyphilic ``MSD`` route in ``lipid_properties.py`` (which fitted
lipyphilic's default 20-80 % of the 0-500 ns lag range, i.e. lags of
100-400 ns, where a 500 ns run has only one to four independent time origins).

Two stages, cached so the slow one runs once:

1. **extract** (MDAnalysis, ``lipid64``): per replicate, the in-plane path of
   every phosphate (``name P P31``, one per lipid), unwrapped across the
   periodic boundary *per step with that frame's box* (the "toroidal"
   unwrapping that stays correct under NPT box fluctuations; von Buelow et al.
   2020), plus each lipid's leaflet (phosphate z against the bilayer mid-plane
   in frame 0) and the per-frame box.  ->  ``data/diffusion/P_{ff}_{lipid}_{rep}.npz``
2. **analyse** (numpy): MSD over *all* time origins (FFT, per lipid), with the
   centre-of-mass drift removed either over the whole bilayer (the convention
   the periodic Saffman-Delbrueck correction assumes; Venable 2017, Voegele
   2018) or per leaflet (**diagnostic only, never a result**: it also subtracts
   part of each lipid's own motion -- Klauda, Brooks & Pastor 2006 -- but a
   large gap flags collective leaflet sliding in this small box).  ``D`` = slope/4 of the lipid-averaged MSD over Venable et
   al.'s window ``10 ns < t < (T - 10 ns)/2``.  ->  ``data/diffusion.json``

    PYTHONNOUSERSITE=1 conda run -n lipid64 python analysis/diffusion.py --nprocs 8
    python3 analysis/diffusion.py --analyse-only          # re-fit from the cache
"""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np                                          # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lipiddefs import FORCEFIELDS, LIPIDS, REPS, PHOSPHATE_SEL   # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TRAJ = ROOT / "traj"
CACHE = ROOT / "data" / "diffusion"
OUT = ROOT / "data" / "diffusion.json"

FIT_T_MIN_NS = 10.0          # Venable 2017: drop the sub-diffusive regime ...
FIT_T_END_NS = 10.0          # ... and fit up to (T - 10 ns)/2
OLD_WINDOW = (0.2, 0.8)      # lipyphilic's default fraction of the lag range


# ----------------------------------------------------------------- extract
def extract(job):
    ff, lipid, rep = job
    out = CACHE / f"P_{ff}_{lipid}_{rep}.npz"
    if out.exists():
        return f"{out.name} cached"
    import MDAnalysis as mda
    gro = TRAJ / f"{ff}_{lipid}.gro"
    xtc = TRAJ / f"{ff}_{lipid}{rep}.xtc"
    if not (gro.exists() and xtc.exists()):
        return f"{ff}_{lipid}_{rep}: missing {gro.name} / {xtc.name}"
    u = mda.Universe(str(gro), str(xtc))
    p = u.select_atoms(PHOSPHATE_SEL)
    n = len(u.trajectory)
    xy = np.empty((n, p.n_atoms, 2))
    z0 = None
    box = np.empty((n, 3))
    times = np.empty(n)
    prev = None
    for i, ts in enumerate(u.trajectory):
        pos = p.positions.astype(float)
        L = ts.dimensions[:2].astype(float)
        if prev is None:
            xy[i] = pos[:, :2]
            z0 = pos[:, 2].copy()
        else:
            d = pos[:, :2] - prev
            d -= L * np.round(d / L)              # minimum image, this frame's box
            xy[i] = xy[i - 1] + d
        prev = pos[:, :2]
        box[i] = ts.dimensions[:3]
        times[i] = ts.time
    # leaflet from frame 0: the trajectories are centred on the membrane, so
    # the mid-plane is the mean phosphate z; flip-flop is negligible in 500 ns
    leaflet = np.where(z0 > z0.mean(), 1, -1)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, xy=xy / 10.0, box=box / 10.0,          # nm
                        time_ns=(times - times[0]) / 1000.0, leaflet=leaflet)
    return f"{out.name}: {n} frames, {p.n_atoms} P, " \
           f"{(leaflet > 0).sum()}/{(leaflet < 0).sum()} upper/lower"


# ----------------------------------------------------------------- analyse
def msd_fft(x):
    """MSD over all time origins for (n_frames, ...) positions; returns
    (n_frames, ...) with the lag on axis 0 (Calandrini et al. 2011 algorithm)."""
    n = x.shape[0]
    d2 = (x ** 2).sum(axis=-1)                               # (n, m)
    zero = np.zeros((1,) + d2.shape[1:])
    d2 = np.concatenate([d2, zero])
    q = 2 * d2.sum(axis=0)
    s1 = np.empty((n,) + d2.shape[1:])
    for m in range(n):
        q = q - d2[m - 1] - d2[n - m]
        s1[m] = q / (n - m)
    f = np.fft.rfft(x, n=2 * n, axis=0)
    ac = np.fft.irfft(f * f.conj(), axis=0)[:n].sum(axis=-1)
    s2 = ac / (n - np.arange(n)).reshape((-1,) + (1,) * (ac.ndim - 1))
    return s1 - 2 * s2


def remove_com(xy, leaflet, mode):
    if mode == "bilayer":
        return xy - xy.mean(axis=1, keepdims=True)
    out = xy.copy()
    for s in (-1, 1):
        m = leaflet == s
        out[:, m] -= xy[:, m].mean(axis=1, keepdims=True)
    return out


def fit_D(t, msd, lo, hi):
    """D (um^2/s) = slope/4 of msd (nm^2) vs t (ns) over lo <= t <= hi."""
    m = (t >= lo) & (t <= hi)
    slope = np.polyfit(t[m], msd[m], 1)[0]                   # nm^2/ns
    return slope / 4 * 1e3                                    # nm^2/ns -> um^2/s


def running_slope(t, msd, width_ns=20.0):
    """Local D(t) = d(MSD)/dt / 4 by a sliding linear fit, for the flatness check."""
    dt = t[1] - t[0]
    w = max(3, int(round(width_ns / dt)))
    centres, ds = [], []
    for i in range(0, len(t) - w, max(1, w // 2)):
        sl = slice(i, i + w)
        centres.append(t[sl].mean())
        ds.append(np.polyfit(t[sl], msd[sl], 1)[0] / 4 * 1e3)
    return np.array(centres), np.array(ds)


def analyse_rep(path):
    d = np.load(path)
    t, xy, leaflet = d["time_ns"], d["xy"], d["leaflet"]
    T = t[-1]
    win = (FIT_T_MIN_NS, (T - FIT_T_END_NS) / 2)
    old = (OLD_WINDOW[0] * T, OLD_WINDOW[1] * T)
    res = {"T_ns": float(T), "window_ns": win, "old_window_ns": old,
           "L_nm": float(d["box"][:, 0].mean()),
           "Lz_nm": float(d["box"][:, 2].mean())}
    curves = {}
    for mode in ("bilayer", "leaflet"):
        msd = msd_fft(remove_com(xy, leaflet, mode)).mean(axis=1)   # over lipids
        curves[mode] = msd
        res[f"D_{mode}"] = fit_D(t, msd, *win)
        res[f"D_{mode}_oldwindow"] = fit_D(t, msd, *old)
    # flatness of the local slope over the fit window (bilayer COM)
    tc, ds = running_slope(t, curves["bilayer"])
    m = (tc >= win[0]) & (tc <= win[1])
    res["local_D_in_window"] = [float(ds[m].min()), float(ds[m].max())]
    return res, t, curves


def analyse():
    out, curves_out = {}, {}
    for ff in FORCEFIELDS:
        for lipid in LIPIDS:
            reps = {}
            for rep in REPS:
                p = CACHE / f"P_{ff}_{lipid}_{rep}.npz"
                if not p.exists():
                    print(f"  missing {p.name}")
                    continue
                r, t, curves = analyse_rep(p)
                reps[rep] = r
                for mode, c in curves.items():
                    curves_out[f"{ff}_{lipid}_{rep}_{mode}"] = c
                curves_out[f"{ff}_{lipid}_{rep}_t"] = t
            if not reps:
                continue
            sysr = {"reps": reps}
            for key in ("D_bilayer", "D_leaflet", "D_bilayer_oldwindow",
                        "D_leaflet_oldwindow"):
                v = np.array([r[key] for r in reps.values()])
                sysr[key] = {"mean": float(v.mean()),
                             "sd": float(v.std(ddof=1)) if v.size > 1 else np.nan,
                             "sem": float(v.std(ddof=1) / np.sqrt(v.size))
                             if v.size > 1 else np.nan,
                             "n": int(v.size)}
            sysr["L_nm"] = float(np.mean([r["L_nm"] for r in reps.values()]))
            sysr["Lz_nm"] = float(np.mean([r["Lz_nm"] for r in reps.values()]))
            out[f"{ff}_{lipid}"] = sysr
            b, lf, o = (sysr["D_bilayer"], sysr["D_leaflet"],
                        sysr["D_bilayer_oldwindow"])
            print(f"{ff:8s} {lipid}  D (um^2/s)  bilayer-COM "
                  f"{b['mean']:5.2f} +/- {b['sd']:4.2f}   leaflet-COM "
                  f"{lf['mean']:5.2f} +/- {lf['sd']:4.2f}   old window "
                  f"{o['mean']:5.2f} +/- {o['sd']:4.2f}   reps "
                  + " ".join(f"{r['D_bilayer']:.1f}" for r in reps.values()))
    OUT.write_text(json.dumps(out, indent=1))
    np.savez_compressed(CACHE / "msd_curves.npz", **curves_out)
    print(f"wrote {OUT.relative_to(ROOT)} and data/diffusion/msd_curves.npz")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nprocs", type=int, default=1)
    ap.add_argument("--analyse-only", action="store_true")
    ap.add_argument("--forcefields", nargs="+", default=FORCEFIELDS)
    ap.add_argument("--lipids", nargs="+", default=LIPIDS)
    args = ap.parse_args()
    if not args.analyse_only:
        jobs = [(ff, lp, r) for ff in args.forcefields for lp in args.lipids
                for r in REPS]
        if args.nprocs > 1:
            from multiprocessing import Pool
            with Pool(min(args.nprocs, len(jobs))) as pool:
                for msg in pool.imap_unordered(extract, jobs):
                    print(msg, flush=True)
        else:
            for j in jobs:
                print(extract(j), flush=True)
    analyse()


if __name__ == "__main__":
    main()
