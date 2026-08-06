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

Run in the MDAnalysis env.  With no arguments it discovers all nine systems the
same way ``lipid_properties.py`` does -- iterating force field x lipid under
``--sims-root`` (default ``~/Simulations``), looking in
``{ff}_{lipid}/{--run-subdir}/`` (default ``umbrella23``) for the ``md.{xtc,gro}``
+ ``rerunerr.log`` triple.  Positional run directories override discovery (each
must hold that triple), e.g. for a local test against ``tmp/``::

    conda run -n lipid64 python analysis/pore_density.py                 # all 9
    conda run -n lipid64 python analysis/pore_density.py --lipids DMPC    # subset
    conda run -n lipid64 python analysis/pore_density.py tmp/             # one dir

then plot with ``plot_pore_density.py``.
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import MDAnalysis as mda

import make_density_index as mdi

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

FORCEFIELDS = ["charmm", "amber", "slipids"]
LIPIDS = ["DOPC", "DMPC", "DMPG"]

# Selections to map (subset of make_density_index's groups).
SELECTIONS = ["water", "phosphorus", "glycerol_C2", "midpoint_C"]
WATER_OXYGEN = {"O", "OW", "OH2", "OW1"}  # water O across TP3 / TIP3 / SOL

# Fixed bin edges (nm) shared by every run so the 3x3 grids line up.
R_MAX, DR = 3.0, 0.05    # in-plane radius from the pore centre
Z_HALF, DZ = 4.5, 0.05   # z relative to the pore centre (boxes here are ~8.2 nm)

_CENTER_RE = re.compile(
    r"Cylinder center at \(X/Y/Z\):\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)")


def parse_centers(log):
    """Return an (n_frame, 3) array of pore centres (nm) in file order."""
    centers = [tuple(float(x) for x in m.groups())
               for m in (_CENTER_RE.search(l) for l in log.read_text().splitlines())
               if m]
    if not centers:
        sys.exit(f"no 'Cylinder center at' lines found in {log}")
    return np.array(centers)


def water_oxygen(group):
    """Reduce a whole-water atom group to just its oxygens."""
    return group[np.isin(group.names, list(WATER_OXYGEN))]


def process(run_dir, out_dir):
    gro, xtc, log = run_dir / "md.gro", run_dir / "md.xtc", run_dir / "rerunerr.log"
    for f in (gro, xtc, log):
        if not f.exists():
            print(f"skip {run_dir}: missing {f.name}")
            return

    centers = parse_centers(log)
    u = mda.Universe(str(gro), str(xtc))
    ff, scheme, lipid = mdi.detect(u.atoms.resnames)
    nframes = len(u.trajectory)
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

    for i, ts in enumerate(u.trajectory):
        if i >= n:
            break
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

    # Counts -> number density: divide by annular shell volume and frame count.
    shell_area = np.pi * (r_edges[1:] ** 2 - r_edges[:-1] ** 2)   # nm^2, per r-bin
    shell_vol = shell_area[None, :] * DZ                          # nm^3
    out = {"r_edges": r_edges, "z_edges": z_edges}
    for name, h in hists.items():
        out[name] = h / (shell_vol * n)

    dest = out_dir / f"poredens_{ff}_{lipid}.npz"
    np.savez_compressed(dest, **out)
    print(f"wrote {dest}  ({ff} {lipid}; {n} frames)  "
          f"selections: {', '.join(hists)}")


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
    args = ap.parse_args()

    if args.runs:
        runs = args.runs
    else:
        runs = [args.sims_root / f"{ff}_{lipid}" / args.run_subdir
                for ff in args.forcefields for lipid in args.lipids]

    DATA.mkdir(parents=True, exist_ok=True)
    for run in runs:
        process(run, DATA)


if __name__ == "__main__":
    main()
