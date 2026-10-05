#!/usr/bin/env python3
"""How long each HMMM run stayed a bilayer.

The **stability time** of a run is the first moment its bilayer stops being a
bilayer, read off the per-frame **lateral** variation of the thickness computed
by ``thickness_variation.py``: a flat membrane has a nearly uniform d_HH
(~0.17 nm SD across the plane here), and buckling, opening or breaking up makes
the leaflet separation depend strongly on *where* it is measured long before
the mean thickness looks wrong.

The rule, fixed after looking at all six CHARMM/AMBER conditions:

    smooth the lateral SD with a 25 ns rolling mean, and take the first frame
    at which it exceeds 1.0 nm; a run that never does is stable for its whole
    trajectory.

That threshold sits in the gap between the runs that hold together (plateaux at
0.17 / 0.35 / 0.55-0.75 nm, no step) and the two that do not (clean steps out of
their plateau to 1.0-1.6 and to 20 nm), and it reproduces the by-eye
classification of every run.  It is an *absolute* threshold, so it is tied to
the ``--nbins`` grid ``thickness_variation.py`` used; the relative alternative
(a multiple of each run's own early plateau) is printed alongside as a check,
since it judges the quietest runs hardest and the two should not be confused.

    PYTHONNOUSERSITE=1 conda run -n base python HMMM/analysis/stability.py

Writes ``HMMM/data/stability.json``: per run the stability time in ns and in
frames, the trajectory length, and whether the run survived its full µs.  The
figure scripts import :func:`load` / :func:`stable_frames` rather than
re-deriving any of it.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from hmmm_defs import (FORCEFIELDS, ENSEMBLES, REPS, DT_NS,
                       stem as run_stem, system as system_key)

DATA = ROOT / "data"

SMOOTH_NS = 25.0        # rolling mean applied before thresholding
THRESHOLD_NM = 1.0      # lateral SD above which the membrane is not a bilayer
RELATIVE_FACTOR = 2.0   # cross-check: multiple of the run's own early plateau
BASELINE_NS = 100.0     # window the early plateau is measured over
FULL_NS = 1000.0        # nominal run length; shorter means the run also crashed


def rolling_mean(y, window):
    """Centred rolling mean; edges shortened rather than padded."""
    if window < 2:
        return np.asarray(y, dtype=float)
    y = np.asarray(y, dtype=float)
    out = np.convolve(y, np.ones(window) / window, mode="same")
    half = window // 2
    for i in list(range(half)) + list(range(len(y) - half, len(y))):
        lo, hi = max(0, i - half), min(len(y), i + half + 1)
        out[i] = y[lo:hi].mean()
    return out


def _first_crossing(t, y, threshold):
    """``(time, index)`` of the first sample above ``threshold``, or ``(None, None)``."""
    above = np.nonzero(y > threshold)[0]
    if not above.size:
        return None, None
    i = int(above[0])
    return float(t[i]), i


def analyse_run(t, sd, smooth_ns=SMOOTH_NS, threshold=THRESHOLD_NM):
    """Stability record for one run from its ``(time_ns, lateral_sd_nm)`` series."""
    dt = float(np.median(np.diff(t))) if len(t) > 1 else DT_NS
    smoothed = rolling_mean(sd, max(1, int(round(smooth_ns / dt))))
    t_abs, i_abs = _first_crossing(t, smoothed, threshold)
    baseline = float(smoothed[t <= BASELINE_NS].mean())
    t_rel, _ = _first_crossing(t, smoothed, RELATIVE_FACTOR * baseline)

    end = float(t[-1]) + dt          # the run's own length
    # A run is "stable" only if it neither crossed the threshold nor stopped
    # early: charmm_npt1 does both, and a run that simply died at 790 ns did not
    # demonstrate a microsecond of stability either.
    stability_ns = end if t_abs is None else t_abs
    frames = len(t) if i_abs is None else i_abs
    return {
        "stability_ns": stability_ns,
        "stability_frames": int(frames),
        "n_frames": int(len(t)),
        "traj_ns": end,
        "crossed": t_abs is not None,
        "survived_full_run": t_abs is None and end >= FULL_NS,
        "baseline_sd_nm": baseline,
        "max_smoothed_sd_nm": float(np.nanmax(smoothed)),
        "relative_stability_ns": end if t_rel is None else t_rel,
    }


def build(npz, smooth_ns=SMOOTH_NS, threshold=THRESHOLD_NM):
    """``{run_stem: record}`` for every run present in ``thickness_map.npz``."""
    out = {}
    for ff in FORCEFIELDS:
        for ens in ENSEMBLES:
            for rep in REPS:
                key = "%s_rep%d" % (system_key(ff, ens), rep)
                if key + "_sd" not in npz:
                    continue
                rec = analyse_run(npz[key + "_time"], npz[key + "_sd"],
                                  smooth_ns=smooth_ns, threshold=threshold)
                rec.update(ff=ff, ensemble=ens, rep=rep)
                out[run_stem(ff, ens, rep)] = rec
    return out


def load(path=None):
    """Read ``stability.json``; raises with an actionable message if absent."""
    path = Path(path or DATA / "stability.json")
    if not path.exists():
        raise SystemExit("%s missing -- run HMMM/analysis/thickness_variation.py "
                         "then HMMM/analysis/stability.py" % path)
    with open(path) as fh:
        return json.load(fh)


def stable_frames(records, ff, ens, rep, n_available=None):
    """Number of leading frames of this run that are still a bilayer.

    ``n_available`` clamps the answer to the length of whatever series is being
    truncated, which may be sampled differently from the thickness series.
    """
    rec = records.get(run_stem(ff, ens, rep))
    if rec is None:
        return n_available
    n = rec["stability_frames"]
    if n_available is not None:
        # The stability series is one frame per ns; scale to the target series.
        n = int(round(n * n_available / float(rec["n_frames"])))
        n = max(1, min(n, n_available))
    return n


def any_unstable(records, ff, ens):
    """True if any replicate of this system failed to survive its full run.

    This is what the red ``//\\\\`` hatching marks in the properties figure."""
    recs = [records.get(run_stem(ff, ens, r)) for r in REPS]
    recs = [r for r in recs if r]
    return bool(recs) and not all(r["survived_full_run"] for r in recs)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=DATA / "thickness_map.npz")
    ap.add_argument("--out", type=Path, default=DATA / "stability.json")
    ap.add_argument("--smooth-ns", type=float, default=SMOOTH_NS)
    ap.add_argument("--threshold", type=float, default=THRESHOLD_NM)
    args = ap.parse_args()
    if not args.data.exists():
        raise SystemExit("%s missing -- run thickness_variation.py first"
                         % args.data)

    records = build(np.load(args.data), smooth_ns=args.smooth_ns,
                    threshold=args.threshold)
    if not records:
        raise SystemExit("no runs found in %s" % args.data)

    print("stability: smoothed (%.0f ns) lateral SD of d_HH first above %.2f nm"
          % (args.smooth_ns, args.threshold))
    print("  %-16s %10s %9s %10s %12s  %s"
          % ("run", "stable(ns)", "traj(ns)", "baseline", "rel. rule", "verdict"))
    for ff in FORCEFIELDS:
        for ens in ENSEMBLES:
            for rep in REPS:
                rec = records.get(run_stem(ff, ens, rep))
                if rec is None:
                    continue
                verdict = ("full run" if rec["survived_full_run"] else
                           ("broke at %.0f ns" % rec["stability_ns"]
                            if rec["crossed"] else
                            "run ended at %.0f ns" % rec["traj_ns"]))
                print("  %-16s %10.0f %9.0f %10.3f %12.0f  %s"
                      % (run_stem(ff, ens, rep), rec["stability_ns"],
                         rec["traj_ns"], rec["baseline_sd_nm"],
                         rec["relative_stability_ns"], verdict))
    print("\nsystems hatched as unstable in the properties figure:")
    flagged = ["%s_%s" % (ff, ens) for ff in FORCEFIELDS for ens in ENSEMBLES
               if any_unstable(records, ff, ens)]
    print("  " + (", ".join(flagged) if flagged else "none"))

    args.out.write_text(json.dumps(records, indent=2, sort_keys=True))
    print("\nwrote %s" % args.out)


if __name__ == "__main__":
    main()
