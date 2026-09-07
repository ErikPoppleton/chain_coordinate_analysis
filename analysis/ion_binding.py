#!/usr/bin/env python3
"""Counterion coordination around the lipid phosphate, N(r).

The mechanism behind the DMPG ion comparison (see the SI-figure section in
CLAUDE.md): the AMBER Lipid21 paper attributes the over-ordering and thickening
of anionic lipids to **Joung-Cheetham** ions binding the headgroup too
strongly.  This measures that directly -- the cumulative number of counterions
within a distance ``r`` of a lipid phosphorus, averaged over phosphates and
frames:

    N(r) = <number of counterions within r of a P> per phosphorus

so N rises from 0 and its first plateau is the first-shell occupancy.  An
over-binding ion gives a steep early rise and a high plateau; a weakly bound
one stays flat until the diffuse layer.

Counterions are the **cations** present (``POT``/``K+`` for the potassium
builds, ``SOD``/``Na+``/``NA`` for the Joung-Cheetham ones), so the same script
covers both conditions without being told which is which.  Distances are
minimum-image (the ions are wrapped into the box, and the diffuse layer wraps
across the z faces).

Runs locally on the full-system copy-B trajectories, ``traj/{ff}_{lipid}.gro`` +
``traj/{ff}_{lipid}{rep}.xtc`` -- pass ``--lipids DMPG DMPG_JC`` for the two ion
conditions (``DMPG_JC`` is the condition label; see ``lipiddefs.base_lipid``).

    conda run -n lipid64 python analysis/ion_binding.py --lipids DMPG DMPG_JC

Writes ``data/ion_binding.npz``: ``r_A`` (bin right edges, shared by every
system), per replicate the cumulative ``{ff}_{lipid}_nr_rep{n}`` and the pair
distribution ``{ff}_{lipid}_gr_rep{n}``, and ``{ff}_{lipid}_ion`` recording
which cation resname was found.  Merges into any existing file, so a
subset run does not drop the systems it skipped.
"""

import argparse
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import MDAnalysis as mda
from MDAnalysis.lib.distances import capped_distance

from lipiddefs import (FORCEFIELDS, LIPIDS, REPS, PHOSPHATE_NAMES,
                       SODIUM, POTASSIUM)

ROOT = Path(__file__).resolve().parent.parent
TRAJ = ROOT / "traj"
DATA = ROOT / "data"

RMAX = 10.0             # A -- past the first shell and into the diffuse layer
DR = 0.1                # A
EDGES = np.arange(0.0, RMAX + DR, DR)
R_A = EDGES[1:]         # right edges: N(r) is cumulative, so r is the upper limit
# A -- first-shell boundary for the reported occupancy.  This is a P...cation
# distance (not cation-oxygen), so the first peak sits near 3.7 A and N(r)
# plateaus just past 4 A; 3.5 A would quote a number off the steep rise.
FIRST_SHELL = 4.5
STRIDE = 10             # ions decorrelate fast; every 10th frame is plenty

CATIONS = sorted(set(SODIUM) | set(POTASSIUM))


def coordination(u, stride=STRIDE):
    """(N(r), g(r), cation resname) for one universe, or Nones if no cation.

    ``N(r)`` is the cumulative count per phosphorus; ``g(r)`` is the pair
    distribution function of the same P...cation pairs, each shell's count
    divided by what an ideal gas at the mean ion number density would put in
    that shell.  **Caveat on g(r)**: the ions are confined to the water slab,
    not spread through the box, so the box-average density used for the ideal
    reference is an underestimate and g(r) does not tend to 1 at large r.  The
    first-shell *peak* is still directly comparable between systems (same box,
    same ion count), which is what the figure uses it for.
    """
    phos = u.select_atoms("name " + " ".join(sorted(PHOSPHATE_NAMES)))
    ions = u.select_atoms("resname " + " ".join(CATIONS))
    if phos.n_atoms == 0:
        raise SystemExit("no phosphorus atoms (names %s)"
                         % sorted(PHOSPHATE_NAMES))
    if ions.n_atoms == 0:
        return None, None
    ion_resn = sorted(set(ions.resnames))

    counts = np.zeros(len(R_A))
    nframes = 0
    volume = 0.0
    for ts in u.trajectory[::stride]:
        # Minimum-image pair distances out to RMAX; capped_distance returns only
        # the pairs inside the cutoff, so the cost does not scale with the box.
        _, dists = capped_distance(phos.positions, ions.positions,
                                   max_cutoff=RMAX, box=ts.dimensions,
                                   return_distances=True)
        if dists.size:
            counts += np.histogram(dists, bins=EDGES)[0]
        volume += float(np.prod(ts.dimensions[:3]))
        nframes += 1
    if not nframes:
        return None, None, None
    # per phosphorus, per frame, cumulative in r
    norm = nframes * phos.n_atoms
    nr = np.cumsum(counts) / norm
    # g(r): shell count vs the ideal-gas expectation at the mean ion density
    shell_vol = (4.0 / 3.0) * np.pi * (EDGES[1:] ** 3 - EDGES[:-1] ** 3)
    rho = ions.n_atoms / (volume / nframes)
    gr = (counts / norm) / (shell_vol * rho)
    return nr, gr, "/".join(ion_resn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--forcefields", nargs="+", default=FORCEFIELDS)
    ap.add_argument("--lipids", nargs="+", default=LIPIDS)
    ap.add_argument("--reps", nargs="+", type=int, default=REPS)
    ap.add_argument("--stride", type=int, default=STRIDE)
    args = ap.parse_args()

    DATA.mkdir(parents=True, exist_ok=True)
    out = {}
    path = DATA / "ion_binding.npz"
    if path.exists():                     # keep systems this run does not touch
        with np.load(path) as old:
            out.update({k: old[k] for k in old.files})
    out["r_A"] = R_A

    shell = int(round(FIRST_SHELL / DR)) - 1
    for ff in args.forcefields:
        for lipid in args.lipids:
            gro = TRAJ / f"{ff}_{lipid}.gro"
            if not gro.exists():
                continue
            means, peaks, ion = [], [], None
            for rep in args.reps:
                xtc = TRAJ / f"{ff}_{lipid}{rep}.xtc"
                if not xtc.exists():
                    print(f"  skip {ff}_{lipid} rep{rep}: no {xtc.name}")
                    continue
                u = mda.Universe(str(gro), str(xtc))
                nr, gr, ion = coordination(u, args.stride)
                if nr is None:
                    print(f"  skip {ff}_{lipid} rep{rep}: no cations "
                          f"(looked for {CATIONS})")
                    continue
                out[f"{ff}_{lipid}_nr_rep{rep}"] = nr
                out[f"{ff}_{lipid}_gr_rep{rep}"] = gr
                means.append(nr[shell])
                peaks.append(float(np.nanmax(gr)))
            if not means:
                continue
            out[f"{ff}_{lipid}_ion"] = np.array(ion)
            print(f"{ff}_{lipid}: {ion:8s} N({FIRST_SHELL:.1f} A) = "
                  f"{np.mean(means):.3f} +- {np.std(means):.3f} per phosphate "
                  f"| peak g(r) = {np.mean(peaks):.1f} ({len(means)} reps)")

    np.savez(path, **out)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
