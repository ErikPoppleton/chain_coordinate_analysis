#!/usr/bin/env python3
"""Compute cylindrically-averaged (radius, z) density maps of selected atoms
around a **pore centre** for the out-of-equilibrium pore simulations.

Each pore run directory holds:
  * ``md.xtc`` / ``md.gro`` -- the (full, water+ion) trajectory + topology,
  * ``rerunerr.log``        -- the chain-coordinate rerun log, which prints the
                               pore-cylinder centre once per frame on lines
                               ``Cylinder center at (X/Y/Z):  cx  cy  cz`` (nm).

For every frame we take the in-plane distance ``r`` of each atom from that
frame's pore centre (minimum-image in x/y) and its ``z`` **relative to that
frame's pore-centre z** (so z=0 is the pore centre), and bin into a 2-D
``(z, r)`` histogram.  Counts are divided by the annular shell volume
``2*pi*r*dr*dz`` (so bulk, uniformly-filled regions read *flat* rather than
brightening with r) and by the frame count -> a number density in atoms/nm^3.

Four atom selections are mapped, reusing ``make_density_index`` so the
per-force-field/-lipid water & lipid atom names are handled automatically:
  * ``water``       -- water oxygens,
  * ``phosphorus``  -- lipid phosphate P,
  * ``glycerol_C2`` -- glycerol sn-2 carbon,
  * ``midpoint_C``  -- mid-chain carbon (DOPC upper C=C C9 / DMPx C7), both tails.

Writes one ``data/poredens_{ff}_{lipid}.npz`` per run (fixed bin edges, so every
map is directly comparable):
  ``r_edges``, ``z_edges`` (nm) and one 2-D array per selection name.

Also writes ``data/poredens_counts_{ff}_{lipid}.npz`` -- the **raw per-frame
count** of atoms inside the chain-coordinate CV cylinder (radius 0.9 nm, height
2.7 nm, centred on the pore centre) for water oxygens, phosphates, and glycerol
C2 carbons -- so pore occupancy can be plotted as a trajectory over time
(``plot_pore_counts.py``).  Two z-slabs are counted per selection: the **full**
cylinder (|z| <= 1.35 nm) and its **middle third** (|z| <= 0.45 nm, the central
9 A -- the pore's narrowest waist).  Keys: ``frame``, ``time_ps``, ``cyl_r_nm``,
``cyl_h_nm``, ``cyl_mid_h_nm``, ``xi`` (the per-frame chain coordinate, the
``xiChain =`` value the rerun log prints for each frame), one 1-D count array
per selection (``water``, ``phosphorus``, ``glycerol_C2``) for the full
cylinder, and a ``{name}_mid`` array for the middle third.

**Every umbrella window** (``--windows all``, or a list such as ``0 5 23``):
discovery then visits ``{ff}_{lipid}/umbrella{w}/`` for each window and writes
``data/poredens_windows/poredens{,_counts}_{ff}_{lipid}_w{w:02d}.npz`` instead
(the umbrella23 files above are left alone).  Each window needs its own
``rerunerr.log`` -- ``setup_files/submit_pore_windows.sh`` makes them and then
runs this script.  ``--nprocs N`` farms the runs over a process pool.

Run in the MDAnalysis env.  With no arguments it discovers all nine systems the
same way ``lipid_properties.py`` does -- iterating force field x lipid under
``--sims-root`` (default ``~/Simulations``), looking in
``{ff}_{lipid}/{--run-subdir}/`` (default ``umbrella23``) for the ``md.{xtc,gro}``
+ ``rerunerr.log`` triple.  Positional run directories override discovery (each
must hold that triple), e.g. for a local test against ``tmp/``::

    conda run -n lipid64 python analysis/pore_density.py                 # all 9
    conda run -n lipid64 python analysis/pore_density.py --lipids DMPC    # subset
    conda run -n lipid64 python analysis/pore_density.py tmp/             # one dir
    conda run -n lipid64 python analysis/pore_density.py --windows all --nprocs 24

then plot with ``plot_pore_density.py``.
"""

import argparse
import re
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import MDAnalysis as mda

import make_density_index as mdi
from lipiddefs import FORCEFIELDS, LIPIDS, WATER_OXYGEN

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

# Selections to map (subset of make_density_index's groups).
SELECTIONS = ["water", "phosphorus", "glycerol_C2", "midpoint_C"]

# Fixed bin edges (nm) shared by every run so the 3x3 grids line up.
R_MAX, DR = 3.0, 0.05    # in-plane radius from the pore centre
Z_HALF, DZ = 4.5, 0.05   # z relative to the pore centre (boxes here are ~8.2 nm)

# The chain-coordinate CV acts on a cylinder (nm): radius 0.9, height 2.7, centred
# on the pore centre -> r <= 0.9, |z| <= 1.35.  Alongside the (r, z) density maps
# we also count the raw number of atoms inside this cylinder per frame, so the
# occupancy can be plotted as a trajectory over time (plot_pore_counts.py).
CYL_R, CYL_H = 0.9, 2.7
COUNT_SELECTIONS = ["water", "phosphorus", "glycerol_C2"]

_CENTER_RE = re.compile(
    r"Cylinder center at \(X/Y/Z\):\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)")


_XI_RE = re.compile(r"xiChain =\s*([-+\d.eE]+)")
N_WINDOWS = 24                   # umbrella0 .. umbrella23
WINDOW_DIR = "poredens_windows"  # data/ subdirectory for the per-window output


def parse_log(log):
    """``(centres, xi)`` from a chain-coordinate rerun log, in file order.

    ``centres`` is an (n_frame, 3) array of pore centres (nm); ``xi`` the
    per-frame chain coordinate (``xiChain =``), or None if its count does not
    match the centres (an older log format).
    """
    centers, xi = [], []
    for line in log.read_text().splitlines():
        m = _CENTER_RE.search(line)
        if m:
            centers.append(tuple(float(x) for x in m.groups()))
            continue
        m = _XI_RE.search(line)
        if m:
            xi.append(float(m.group(1)))
    if not centers:
        raise RuntimeError(f"no 'Cylinder center at' lines found in {log}")
    if len(xi) != len(centers):
        print(f"  WARNING {log}: {len(xi)} xiChain values vs {len(centers)} "
              f"centres; not storing xi")
        xi = None
    return np.array(centers), (np.array(xi) if xi is not None else None)


def water_oxygen(group):
    """Reduce a whole-water atom group to just its oxygens."""
    return group[np.isin(group.names, list(WATER_OXYGEN))]


def process(run_dir, out_dir, suffix=""):
    """Map + count one run dir; ``suffix`` is appended to the output stems."""
    gro, xtc, log = run_dir / "md.gro", run_dir / "md.xtc", run_dir / "rerunerr.log"
    for f in (gro, xtc, log):
        if not f.exists():
            print(f"skip {run_dir}: missing {f.name}")
            return

    centers, xi = parse_log(log)
    u = mda.Universe(str(gro), str(xtc))
    ff, scheme, lipid = mdi.detect(u.atoms.resnames)
    nframes = len(u.trajectory)
    if len(centers) < nframes:
        # A log written with the fork's default SLICE_COORD_NST_OUTPUT (1000)
        # has a centre for frame 0 only; zipping would silently keep 1 frame.
        raise RuntimeError(f"{log}: {len(centers)} centres for {nframes} frames "
                           f"-- rerun with SLICE_COORD_NST_OUTPUT=1")
    if len(centers) != nframes:
        print(f"  WARNING {ff} {lipid}: {len(centers)} centres vs {nframes} "
              f"frames; using min and zipping in order")
    n = min(len(centers), nframes)

    groups = dict(mdi.build_groups(u, scheme, lipid))
    sel = {}
    for name in SELECTIONS:
        g = groups.get(name)
        if g is None or g.n_atoms == 0:
            print(f"  WARNING: selection '{name}' empty; skipping")
            continue
        sel[name] = water_oxygen(g) if name == "water" else g

    r_edges = np.arange(0.0, R_MAX + DR, DR)
    z_edges = np.arange(-Z_HALF, Z_HALF + DZ, DZ)
    hists = {name: np.zeros((len(z_edges) - 1, len(r_edges) - 1)) for name in sel}

    # Per-frame raw in-cylinder counts (occupancy trajectory): once for the full
    # CV cylinder (|z| <= CYL_H/2) and once for its middle third (|z| <= CYL_H/6,
    # i.e. the central 9 A of the 27 A cylinder -> the pore's narrowest waist).
    count_names = [name for name in COUNT_SELECTIONS if name in sel]
    counts = {name: np.zeros(n, dtype=int) for name in count_names}
    counts_mid = {name: np.zeros(n, dtype=int) for name in count_names}
    times = np.zeros(n)          # ps, from the trajectory timestamps
    z_half_cv = CYL_H / 2.0
    z_mid_half = CYL_H / 6.0     # middle third

    for i, ts in enumerate(u.trajectory):
        if i >= n:
            break
        times[i] = ts.time
        cx, cy, cz = centers[i] * 10.0             # nm -> A (mda units)
        box = ts.dimensions[:3]
        for name, g in sel.items():
            pos = g.positions
            dx = pos[:, 0] - cx
            dy = pos[:, 1] - cy
            dx -= box[0] * np.round(dx / box[0])   # minimum image in x/y
            dy -= box[1] * np.round(dy / box[1])
            r = np.sqrt(dx * dx + dy * dy) / 10.0  # A -> nm
            z = (pos[:, 2] - cz) / 10.0            # z relative to the pore centre
            h, _, _ = np.histogram2d(z, r, bins=(z_edges, r_edges))
            hists[name] += h
            if name in counts:                     # atoms inside the CV cylinder
                inside_r = r <= CYL_R
                absz = np.abs(z)
                counts[name][i] = np.count_nonzero(inside_r & (absz <= z_half_cv))
                counts_mid[name][i] = np.count_nonzero(inside_r & (absz <= z_mid_half))

    # Counts -> number density: divide by annular shell volume and frame count.
    shell_area = np.pi * (r_edges[1:] ** 2 - r_edges[:-1] ** 2)   # nm^2, per r-bin
    shell_vol = shell_area[None, :] * DZ                          # nm^3
    out = {"r_edges": r_edges, "z_edges": z_edges}
    for name, h in hists.items():
        out[name] = h / (shell_vol * n)

    dest = out_dir / f"poredens_{ff}_{lipid}{suffix}.npz"
    np.savez_compressed(dest, **out)
    print(f"wrote {dest}  ({ff} {lipid}; {n} frames)  "
          f"selections: {', '.join(hists)}")

    # Per-frame in-cylinder occupancy -> its own npz (frame + time axes; one
    # raw-count array per selection for the full cylinder plus a ``{name}_mid``
    # array for the middle-third slab; CV geometry stored for reference).
    cdest = out_dir / f"poredens_counts_{ff}_{lipid}{suffix}.npz"
    cout = {"frame": np.arange(n), "time_ps": times,
            "cyl_r_nm": np.array(CYL_R), "cyl_h_nm": np.array(CYL_H),
            "cyl_mid_h_nm": np.array(CYL_H / 3.0)}
    if xi is not None:
        cout["xi"] = xi[:n]
    cout.update(counts)
    cout.update({f"{name}_mid": c for name, c in counts_mid.items()})
    np.savez_compressed(cdest, **cout)
    print(f"wrote {cdest}  (per-frame counts, full + middle-third: "
          f"{', '.join(count_names)})")


def _process_safe(run_dir, out_dir, suffix=""):
    """``process`` for the pool: one bad run is reported, not fatal to the rest."""
    try:
        process(run_dir, out_dir, suffix)
    except Exception as exc:                     # noqa: BLE001
        print(f"FAILED {run_dir}: {type(exc).__name__}: {exc}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*", type=Path,
                    help="explicit pore run directories (each with md.xtc/md.gro/"
                         "rerunerr.log); overrides discovery when given")
    ap.add_argument("--sims-root", type=Path, default=Path.home() / "Simulations")
    ap.add_argument("--forcefields", nargs="+", default=FORCEFIELDS)
    ap.add_argument("--lipids", nargs="+", default=LIPIDS)
    ap.add_argument("--run-subdir", default="umbrella23",
                    help="pore run subdir inside each {ff}_{lipid} (default: umbrella23)")
    ap.add_argument("--windows", nargs="+", default=None,
                    help="process umbrella windows instead of --run-subdir: 'all' "
                         f"(0..{N_WINDOWS - 1}) or a list of indices; output goes "
                         f"to data/{WINDOW_DIR}/ with a _wNN suffix")
    ap.add_argument("--nprocs", type=int, default=1,
                    help="parallel worker processes (one run each)")
    args = ap.parse_args()

    if args.windows:
        windows = (range(N_WINDOWS) if args.windows == ["all"]
                   else [int(w) for w in args.windows])
        out_dir = DATA / WINDOW_DIR
        jobs = [(args.sims_root / f"{ff}_{lipid}" / f"umbrella{w}", out_dir,
                 f"_w{w:02d}")
                for ff in args.forcefields for lipid in args.lipids
                for w in windows]
    else:
        out_dir = DATA
        runs = args.runs or [args.sims_root / f"{ff}_{lipid}" / args.run_subdir
                             for ff in args.forcefields for lipid in args.lipids]
        jobs = [(run, out_dir, "") for run in runs]

    out_dir.mkdir(parents=True, exist_ok=True)
    nprocs = max(1, min(args.nprocs, len(jobs)))
    if nprocs == 1:
        for job in jobs:
            _process_safe(*job)
    else:
        with Pool(nprocs) as pool:
            pool.starmap(_process_safe, jobs, chunksize=1)


if __name__ == "__main__":
    main()
