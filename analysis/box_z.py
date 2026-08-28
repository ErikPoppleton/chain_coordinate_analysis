#!/usr/bin/env python3
"""Per-frame box **z** dimension of the equilibrium trajectories, straight from
the ``traj/`` xtc box vectors.

This exists as a fallback/cross-check for the cluster-side ``gmx energy``
Box-Z ``.xvg`` files: it needs no ``.edr``, and it cannot be extracted with the
wrong energy term (``data/z{1,2,3}.xvg`` were exported as *Potential* rather
than *Box-Z*, which ``draw_box_z.py`` detects and routes around).  The xtc
carries the box in every frame, so this is the same quantity GROMACS would
report.

    PYTHONNOUSERSITE=1 conda run -n lipid64 python analysis/box_z.py
    PYTHONNOUSERSITE=1 conda run -n lipid64 python analysis/box_z.py --forcefields amber --lipids DMPG

Writes ``data/box_z.npz`` with, per (ff, lipid, rep), ``{ff}_{lipid}{rep}_t``
(ns) and ``{ff}_{lipid}{rep}_z`` (nm).
"""

import argparse
from pathlib import Path

import numpy as np
import MDAnalysis as mda

from lipiddefs import FORCEFIELDS, LIPIDS, REPS

ROOT = Path(__file__).resolve().parent.parent
TRAJ = ROOT / "traj"
DATA = ROOT / "data"


def box_z(gro, xtc):
    """(time in ns, box z in nm) per frame."""
    u = mda.Universe(str(gro), str(xtc))
    t, z = [], []
    for ts in u.trajectory:
        t.append(ts.time / 1000.0)
        z.append(ts.dimensions[2] / 10.0)
    return np.array(t), np.array(z)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--forcefields", nargs="+", default=FORCEFIELDS)
    ap.add_argument("--lipids", nargs="+", default=LIPIDS)
    ap.add_argument("--reps", nargs="+", type=int, default=REPS)
    args = ap.parse_args()

    out = {}
    p = DATA / "box_z.npz"
    if p.exists():                        # keep systems this run does not touch
        with np.load(p) as old:
            out.update({k: old[k] for k in old.files})
    for ff in args.forcefields:
        for lipid in args.lipids:
            gro = TRAJ / f"{ff}_{lipid}.gro"
            for rep in args.reps:
                xtc = TRAJ / f"{ff}_{lipid}{rep}.xtc"
                if not (gro.exists() and xtc.exists()):
                    continue
                t, z = box_z(gro, xtc)
                key = f"{ff}_{lipid}{rep}"
                out[f"{key}_t"], out[f"{key}_z"] = t, z
                print(f"  {key}: {z.size} frames, z {z[0]:.2f} -> {z[-1]:.2f} nm")
    if not out:
        raise SystemExit("no trajectories found under traj/")
    np.savez(p, **out)
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
