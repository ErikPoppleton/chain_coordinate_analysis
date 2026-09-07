#!/usr/bin/env python3
"""Hydrophobic-core thickness d_HC of each ``traj/{ff}_{lipid}{rep}`` trajectory,
by **two metrics**, for comparison to the head-to-head thickness d_HH:

  1. **lipyphilic** -- ``MembThickness`` keyed on the first aliphatic tail carbon
     of each chain (``C22``/``C32`` for CHARMM/SLipids, ``C12`` of each ``OL``/
     ``MY`` tail residue for AMBER): the leaflet-to-leaflet distance of the
     polar/apolar boundary plane.  Computed over the **full trajectory** (the
     250 ns runs already have equilibration dropped).
  2. **tail mass-density width @ 500 kg/m^3** -- the reference method (over the
     **last 50 ns**, matching the published method): the mass density profile of
     the hydrophobic tail atoms (the ``tails`` set of ``make_tail_index.py``) is
     a slab centred on the bilayer mid-plane, and d_HC is its full width where it
     crosses 500 kg/m^3.  Computed here directly in MDAnalysis (mass-weighted
     z-histogram, per frame centred on the tail COM) so it runs locally on
     ``traj/`` and needs no ``.tpr``; it reproduces ``gmx density -dens mass`` on
     the same ``tails`` group (the cluster route), which ``plot_core_thickness.py``
     cross-checks against these numbers.

Also computes d_HH per frame (lipyphilic ``MembThickness`` on the phosphate),
**frame-aligned** with the lipyphilic d_HC over the same full trajectory, so a
genuine per-frame headgroup H(t) = d_HH(t) - d_HC(t) can be formed downstream
(``decompose_thickness.py``) without the tail<->head anti-correlation washing out
in a mean-of-means.

Writes ``data/core_thickness.npz``:
  * ``{ff}_{lipid}_lipy_rep{r}``  -- per-frame lipyphilic d_HC (nm), full traj
  * ``{ff}_{lipid}_dhh_rep{r}``   -- per-frame lipyphilic d_HH (nm), full traj,
                                    frame-aligned with ``_lipy_rep{r}``
  * ``{ff}_{lipid}_denswidth``    -- per-replicate density width @ 500 (nm), 1-D
  * ``{ff}_{lipid}_densprof_z`` / ``_densprof`` -- replicate-mean density profile
                                    (relative z in nm, density in kg/m^3), for the
                                    supplementary profile figure.

    conda run -n lipid64 python analysis/core_thickness.py
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import MDAnalysis as mda
from lipyphilic import AssignLeaflets
from lipyphilic.analysis.memb_thickness import MembThickness

from make_density_index import detect
from make_tail_index import tail_atoms
from lipiddefs import FORCEFIELDS, LIPIDS, REPS, AMBER_TAIL, PHOSPHATE_SEL

ROOT = Path(__file__).resolve().parent.parent
TRAJ = ROOT / "traj"
DATA = ROOT / "data"

LAST_NS = 50.0                 # analyse only the final 50 ns
DENS_THRESHOLD = 500.0         # kg/m^3, hydrophobic-core density cutoff
BIN_A = 0.5                    # density z-bin width (A) ~ gmx -sl 200 over a ~10 nm box
AMU_A3_TO_KGM3 = 1660.539      # 1 amu/A^3 -> kg/m^3
DENS_EDGES = np.arange(-35.0, 35.0 + BIN_A, BIN_A)   # relative z (A)
DENS_Z_NM = 0.5 * (DENS_EDGES[:-1] + DENS_EDGES[1:]) / 10.0


def tail_carbon_sel(scheme, lipid):
    """lipyphilic selection: first aliphatic tail carbon of each chain."""
    if scheme == "amber":
        return f"resname {AMBER_TAIL[lipid]} and name C12"
    return f"resname {lipid} and name C22 C32"


def start_frame(u):
    n = u.trajectory.n_frames
    return max(0, n - int(round(LAST_NS * 1000.0 / u.trajectory.dt)))


def _membthickness(u, sel, start):
    """Per-frame lipyphilic leaflet-to-leaflet thickness (nm) for a selection."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lf = AssignLeaflets(u, lipid_sel=sel)
        lf.run(start=start)
        th = MembThickness(u, lipid_sel=sel, leaflets=lf.leaflets)
        th.run(start=start)
    return np.asarray(th.memb_thickness).ravel() / 10.0


def lipy_thickness(u, scheme, lipid, start):
    """Per-frame lipyphilic core thickness d_HC (nm), first tail carbon plane."""
    return _membthickness(u, tail_carbon_sel(scheme, lipid), start)


def dhh_thickness(u, start):
    """Per-frame lipyphilic head-head thickness d_HH (nm), phosphate plane.
    Frame-aligned with ``lipy_thickness`` (same universe, same start) so a
    per-frame headgroup H(t) = d_HH(t) - d_HC(t) can be formed."""
    return _membthickness(u, PHOSPHATE_SEL, start)


def tail_density(u, scheme, lipid, start):
    """(width_nm, density_kg_m3) -- tail mass-density profile (relative z centred
    on the tail COM each frame) and its full width at DENS_THRESHOLD."""
    tail = tail_atoms(u, scheme, lipid)
    mass = tail.masses
    acc = np.zeros(len(DENS_Z_NM))
    area_sum = 0.0
    nfr = 0
    for ts in u.trajectory[start:]:
        z = tail.positions[:, 2] - tail.center_of_mass()[2]
        acc += np.histogram(z, bins=DENS_EDGES, weights=mass)[0]
        area_sum += ts.dimensions[0] * ts.dimensions[1]
        nfr += 1
    dens = acc / nfr / (area_sum / nfr * BIN_A) * AMU_A3_TO_KGM3
    return width_at(DENS_Z_NM, dens), dens


def width_at(z, d, thr=DENS_THRESHOLD):
    """Full width of profile ``d(z)`` at level ``thr`` (outermost crossings, nm)."""
    above = np.where(d >= thr)[0]
    if above.size == 0:
        return float("nan")
    lo, hi = above[0], above[-1]
    zl = (z[0] if lo == 0
          else np.interp(thr, [d[lo - 1], d[lo]], [z[lo - 1], z[lo]]))
    zr = (z[-1] if hi == len(z) - 1
          else np.interp(thr, [d[hi + 1], d[hi]], [z[hi + 1], z[hi]]))
    return float(zr - zl)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--forcefields", nargs="+", default=FORCEFIELDS)
    ap.add_argument("--lipids", nargs="+", default=LIPIDS)
    ap.add_argument("--reps", nargs="+", type=int, default=REPS)
    args = ap.parse_args()

    DATA.mkdir(parents=True, exist_ok=True)
    out = {}
    for ff in args.forcefields:
        for lipid in args.lipids:
            gro = TRAJ / f"{ff}_{lipid}.gro"
            if not gro.exists():
                continue
            widths, profs, lipy_means = [], [], []
            for r in args.reps:
                xtc = TRAJ / f"{ff}_{lipid}{r}.xtc"
                if not xtc.exists():
                    print(f"  skip {ff}_{lipid} rep{r}: no {xtc.name}")
                    continue
                u = mda.Universe(str(gro), str(xtc))
                _, scheme, lip = detect(u.atoms.resnames)
                start = start_frame(u)          # last-50 ns window (density metric)

                # lipyphilic d_HC and d_HH over the FULL trajectory (the 250 ns
                # runs already have equilibration dropped), frame-aligned so
                # H(t) = d_HH(t) - d_HC(t) is a genuine per-frame quantity.
                t = lipy_thickness(u, scheme, lip, 0)
                out[f"{ff}_{lipid}_lipy_rep{r}"] = t
                lipy_means.append(float(np.nanmean(t)))
                out[f"{ff}_{lipid}_dhh_rep{r}"] = dhh_thickness(u, 0)

                # density metric keeps the published last-50 ns window.
                w, dens = tail_density(u, scheme, lip, start)
                widths.append(w)
                profs.append(dens)
            if not widths:
                continue
            out[f"{ff}_{lipid}_denswidth"] = np.array(widths)
            out[f"{ff}_{lipid}_densprof_z"] = DENS_Z_NM
            out[f"{ff}_{lipid}_densprof"] = np.mean(profs, axis=0)
            print(f"{ff}_{lipid}: d_HC lipyphilic = "
                  f"{np.mean(lipy_means):.3f}±{np.std(lipy_means):.3f} nm | "
                  f"density@500 = {np.mean(widths):.3f}±{np.std(widths):.3f} nm "
                  f"({len(widths)} reps, last {LAST_NS:.0f} ns)")
    if not out:
        raise SystemExit("no trajectories found under traj/")
    # Merge into any existing file (as box_z.py does): a subset run
    # (--lipids/--forcefields/--reps) must not drop the systems it skipped.
    path = DATA / "core_thickness.npz"
    if path.exists():
        with np.load(path) as old_npz:
            merged = {k: old_npz[k] for k in old_npz.files}
        merged.update(out)
        out = merged
    np.savez(path, **out)
    print(f"\nwrote {DATA / 'core_thickness.npz'}")


if __name__ == "__main__":
    main()
