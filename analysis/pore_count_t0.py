#!/usr/bin/env python3
"""pymbar equilibration time of the pore-waist occupancy, per umbrella window.

For every ``{ff}_{lipid}`` system and umbrella window 0-23 this runs
``pymbar.timeseries.detect_equilibration`` (Chodera 2016, ``t0 = argmax Neff``)
on the whole per-frame count series of water oxygens and phosphates in the
middle third of the CV cylinder (``{water,phosphorus}_mid`` in
``data/poredens_windows/poredens_counts_{ff}_{lipid}_wNN.npz``).

A series with no variation (e.g. no phosphate ever reaches the waist, as in
every window below xi ~ 0.7) has no defined statistical inefficiency; it is
recorded with ``t0 = 0`` and ``constant = true``.

pymbar lives in the base env:

    conda run -n base python analysis/pore_count_t0.py
    -> data/pore_count_t0.json
    conda run -n base python analysis/pore_count_t0.py --full
    -> data/pore_count_t0_full.json   (whole CV cylinder, not the middle third)
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lipiddefs import FORCEFIELDS, LIPIDS                  # noqa: E402

DATA = Path(__file__).resolve().parent.parent / "data"
WINDOWS = DATA / "poredens_windows"
OUT = DATA / "pore_count_t0.json"
N_WINDOWS = 24
MID_KEYS = ("water_mid", "phosphorus_mid")
FULL_KEYS = ("water", "phosphorus")


def equilibration(series):
    """``(t0_frame, g, Neff_max, constant)`` for one series."""
    from pymbar import timeseries
    a = np.asarray(series, dtype=float)
    if np.ptp(a) == 0:
        return 0, float("nan"), float("nan"), True
    t0, g, neff = timeseries.detect_equilibration(a)
    return int(t0), float(g), float(neff), False


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true",
                    help="whole CV cylinder instead of its middle third")
    args = ap.parse_args()
    KEYS = FULL_KEYS if args.full else MID_KEYS
    out_path = OUT.with_name(OUT.stem + "_full.json") if args.full else OUT
    out = {}
    for ff in FORCEFIELDS:
        for lipid in LIPIDS:
            for w in range(N_WINDOWS):
                p = WINDOWS / f"poredens_counts_{ff}_{lipid}_w{w:02d}.npz"
                if not p.exists():
                    print(f"missing {p.name}")
                    continue
                d = np.load(p)
                t = (d["time_ps"] - d["time_ps"][0]) / 1000.0     # ns
                dt = float(t[1] - t[0])
                rec = {"n_frames": int(len(t)), "dt_ns": dt,
                       "run_ns": float(t[-1])}
                if "xi" in d.files:
                    rec["xi_mean"] = float(np.mean(d["xi"]))
                for key in KEYS:
                    t0, g, neff, const = equilibration(d[key])
                    rec[key] = {"t0_frames": t0, "t0_ns": t0 * dt,
                                "g": g, "Neff_max": neff, "constant": const,
                                "mean": float(np.mean(d[key]))}
                out[f"{ff}_{lipid}_w{w:02d}"] = rec
                print(f"{ff:8s} {lipid} w{w:02d}  "
                      + "  ".join(f"{k}: t0={rec[k]['t0_ns']:6.1f} ns"
                                  + (" (const)" if rec[k]["constant"] else "")
                                  for k in KEYS))
    out_path.write_text(json.dumps(out, indent=1))
    print(f"wrote {out_path.relative_to(DATA.parent)}")


if __name__ == "__main__":
    main()
