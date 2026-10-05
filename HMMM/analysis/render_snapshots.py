#!/usr/bin/env python3
"""Render headless PyMOL snapshots of the HMMM systems.

The HMMM counterpart of ``analysis/render_snapshots.py``: the whole ``.pml`` --
load, style, camera, ``png``, ``quit`` -- is generated from Python and handed to
``pymol -cq``, so nothing has to be driven by hand and the styling is
reproducible.  Unlike the parent renderer it is **not** AMBER-only: every
selection is built from ``hmmm_defs``, so CHARMM/SLipids (one residue per lipid,
chains told apart by atom name) and AMBER (head ``PC`` + two ``VL`` tail
residues) both work, and there is no water, ion or chain-coordinate cylinder to
handle.

Style
-----
Lipids as spheres coloured by region -- acyl-tail carbons ``gray30``, headgroup
carbons ``gray80``, phosphorus ``orange`` -- and DCLE as pale-green **sticks**.
Cheap: about a second a frame, so a whole set is seconds rather than minutes.

A translucent DCLE *surface* (``show surface`` on an ``extract``ed object with
``solvent_radius 3.2`` and ``vdw 2``) was tried and **rejected**: it is prettier
for an intact bilayer, but the whole point of these panels is to show the runs
where the membrane breaks up, and there the surface degenerates into disjoint
shards that read as a rendering artifact rather than as the physics.

Camera
------
As for the full membranes: ``turn x, 90`` from the loaded orientation, so the
bilayer is edge-on, and then a square window the width of the membrane (see
:func:`camera`).  The two runs that ended in a pressure-coupling failure get
their own treatment -- turned 60 deg under a wide-angle camera, so the collapsed
column recedes into the distance instead of lying across the frame as a
hairline.  The near/far clipping planes are widened by the system's own extent
either way: ``zoom`` leaves a slab that slices the foreground spheres open, the
same artifact the parent renderer works around.

Input is always ``traj/{ff}_{ens}{rep}.gro``, i.e. **the run's final
configuration** -- deliberately, and with no way to ask for another frame.  Some
of these runs end with the membrane broken up, and that is the point of the
panel; hunting through a trajectory for a frame that looks better would be
choosing the picture rather than reporting the run.

It is first **wrapped** (whole molecules back into the box) into a cached copy
under ``data/frames/`` -- the cluster's ``-pbc mol`` output keeps molecules
whole but lets them sit outside the cell, which renders as lipid patches
floating clear of the membrane.  ``--no-wrap`` renders the downloaded file
as-is.  AMBER's uniformly-``1`` resids are handled by
``topology.topology_for``; the PyMOL selections themselves never use a resid.

Needs MDAnalysis for the structure preparation and PyMOL for the render, which
live in different conda envs here, so run it under ``lipid64`` and let
``find_pymol`` reach into the conda root for the launcher::

    PYTHONNOUSERSITE=1 conda run -n lipid64 python HMMM/analysis/render_snapshots.py
    ... --ens npat --rep 1 --ff charmm
    ... --dry-run                                  # write the .pml only

Output: ``HMMM/plots/snapshots/{ff}_{ens}{rep}.png`` (+ the generated ``.pml``
beside it, a build artifact).
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                       # HMMM/
sys.path.insert(0, str(HERE))

from hmmm_defs import (FORCEFIELDS, ENSEMBLES, REPS, SCHEME, DCLE,
                       AMBER_HEAD, AMBER_TAIL, LIPID_RESNAMES, lipid_resnames,
                       tail_atom_names, head_carbon_names, stem as run_stem)

TRAJ = ROOT / "traj"
OUT = ROOT / "plots" / "snapshots"
FRAMES = ROOT / "data" / "frames"

PX = 1358               # 57.5 mm at 600 dpi -- the parent project's panel size
DPI = 600
# `zoom` fits atom *centres*, so a buffer smaller than a van der Waals radius
# clips the outermost spheres against the image border.  8 A clears them with a
# little air to spare.
ZOOM_BUFFER = 8.0       # A of padding around the system

DCLE_COLOUR = "palegreen"
LIPID_COLOUR = "gray60"          # anything not caught by the two below
TAIL_COLOUR = "gray30"           # acyl chains
HEAD_COLOUR = "gray80"           # glycerol / choline / carbonyls
PHOSPHORUS_COLOUR = "orange"


def find_pymol():
    """Path to the ``pymol`` launcher.

    The structure preparation needs MDAnalysis (the ``lipid64`` env) while
    PyMOL is installed in the conda **base** env, so the script is typically run
    under ``lipid64``, where ``pymol`` is not on ``PATH``.  Fall back to the
    conda root's ``bin`` (``<env>/../../bin/pymol``) before giving up.
    """
    found = shutil.which("pymol")
    if found:
        return found
    candidate = Path(sys.prefix).parent.parent / "bin" / "pymol"
    if candidate.exists():
        return str(candidate)
    return "pymol"          # let subprocess report it


def _resn_sel(names):
    """``resn A+B+C`` for a set of residue names, in a stable order."""
    return "resn " + "+".join(sorted(names))


def _name_sel(names):
    return "name " + "+".join(sorted(names))


def selections(ff):
    """Selection strings for one force field.

    Returns ``(dcle, lipids, tail_carbons, head_carbons)``.  For AMBER the two
    carbon selections are whole residues (``VL`` / ``PC``); for the
    single-residue force fields they are atom-name sets built from
    ``hmmm_defs``.
    """
    scheme = SCHEME[ff]
    dcle = _resn_sel(DCLE)
    lipids = _resn_sel(lipid_resnames(scheme))
    if scheme == "amber":
        tail = "name C* and resn %s" % AMBER_TAIL
        head = "name C* and resn %s" % AMBER_HEAD
    else:
        # One residue holds the whole lipid, so the acyl chains (C22..C25/C2T,
        # C32..C35/C3T) and the glycerol/choline/carbonyl carbons have to be
        # separated by name.  The terminal methyls are C2T/C3T in CHARMM but
        # C26/C36 in SLipids; tail_atom_names covers both.
        tail = "(%s) and %s" % (lipids, _name_sel(tail_atom_names(scheme)))
        head = "(%s) and %s" % (lipids, _name_sel(head_carbon_names(scheme)))
    return dcle, lipids, tail, head


Z_SLICES = 100          # z-histogram resolution for the cut-plane search

TURN_DEG = 90.0         # edge-on for an intact bilayer
COLLAPSED_TURN_DEG = 60.0   # see `collapsed()`
COLLAPSED_ASPECT = 3.0      # box_z / in-plane, above which a run counts as collapsed
COLLAPSED_FOV = 50.0        # deg -- wide angle, so the receding column reads as depth
# How many in-plane widths of the collapsed column to frame.  1 (the intact-run
# value) fills the frame with a handful of lipids and shows no depth at all;
# this pulls back far enough that the column visibly runs away from the camera.
COLLAPSED_WINDOW_MULT = 6.0


def box_from_gro(path):
    """The three box lengths (A) from a ``.gro``'s last line -- stdlib only."""
    with open(path) as fh:
        last = fh.readlines()[-1]
    return [10.0 * float(v) for v in last.split()[:3]]


def collapsed(gro):
    """True when this run ended with a runaway, non-membrane-shaped box.

    The two failed CHARMM NPT runs finish around 2.3 x 2.3 x 275 nm -- a box
    over a hundred times taller than it is wide, holding a filament rather than
    a bilayer.  Everything else is a normal slab (box_z below the in-plane
    size), so a factor of 3 separates them with room to spare.
    """
    try:
        bx, by, bz = box_from_gro(gro)
    except (OSError, ValueError, IndexError):
        return False
    return bz > COLLAPSED_ASPECT * max(bx, by)


def _prepare(stem, out):
    """Write the render-ready ``.gro``: molecules wrapped, cut where it is emptiest.

    The trajectories come off the cluster with ``-pbc mol``, which keeps every
    molecule whole but lets it sit outside the box, and does not guarantee the
    bilayer is centred along z.  Rendered as-is that shows up as lipid patches
    floating clear of the membrane and a slab riding high or low in frame --
    visible in ``charmm_npat``, where a couple of lipids had walked past the y
    face.

    So the system is translated along z and then wrapped by whole molecules (one
    residue per molecule, which is exactly what ``topology.topology_for``
    guarantees, including for AMBER).  **Where to put the periodic cut** is
    chosen from the data rather than assumed: the lipid-atom z histogram's
    emptiest slice is moved to the box edge.  For an intact bilayer that is the
    gap between periodic images, so the membrane ends up centred -- the same
    answer as "centre on the mean phosphate z", but it does not fall apart for a
    run whose membrane has broken up or reoriented, where the mean phosphate z
    is meaningless and could put the cut straight through the structure.  These
    panels exist to show those runs, so the rule has to survive them.

    Needs MDAnalysis, so it is imported here rather than at module scope: the
    result is cached and the rest of the script is stdlib + PyMOL only.
    """
    import numpy as np
    import MDAnalysis as mda
    sys.path.insert(0, str(HERE))
    from topology import topology_for

    # The .gro *is* the final configuration, so it is loaded on its own -- no
    # trajectory.  (Attaching the .xtc and forgetting to seek would silently
    # render frame 0, which is the starting structure for every run.)
    gro = TRAJ / (stem + ".gro")
    u = mda.Universe(str(topology_for(gro, quiet=True)))
    box = np.asarray(u.dimensions[:3], dtype=float)

    # A run that ended in a pressure-coupling failure has a box that no longer
    # describes its own contents (charmm_npt1 finishes at 2.3 x 2.3 x 275 nm
    # around a system still ~100 nm across).  Wrapping into that box would mince
    # the configuration, so leave those coordinates exactly as GROMACS wrote
    # them and say so.
    extent = u.atoms.positions.max(axis=0) - u.atoms.positions.min(axis=0)
    if np.any(extent > 2.0 * box):
        print("  ! %s: box %s nm does not contain its own extent %s nm "
              "(the run blew up) -- rendering unwrapped"
              % (stem, np.round(box / 10, 1), np.round(extent / 10, 1)))
    else:
        # Put the periodic cut where there is least to cut: the emptiest slice
        # of the lipid-atom z histogram.  For an intact bilayer that is the gap
        # between periodic images, so the membrane comes out centred; unlike
        # "centre on the mean phosphate z" it also survives a membrane that has
        # broken up or reoriented, which is what these panels exist to show.
        lz = float(box[2])
        # MDAnalysis syntax here, not PyMOL's: `resname A B`, not `resn A+B`.
        lipids = u.select_atoms("resname " + " ".join(sorted(LIPID_RESNAMES)))
        counts, edges = np.histogram(lipids.positions[:, 2] % lz, bins=Z_SLICES,
                                     range=(0.0, lz))
        cut = 0.5 * (edges[counts.argmin()] + edges[counts.argmin() + 1])
        u.atoms.translate([0.0, 0.0, -cut])
        u.atoms.wrap(compound="residues", inplace=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    u.atoms.write(str(out))
    return out


def structure_path(ff, ens, rep, wrap=True):
    """The ``.gro`` to render: the run's final configuration, wrapped.

    Without ``wrap`` this is the downloaded ``.gro`` used directly; otherwise a
    prepared copy is cached in ``data/frames/``.
    """
    stem = run_stem(ff, ens, rep)
    gro = TRAJ / (stem + ".gro")
    if not wrap:
        return gro
    out = FRAMES / ("%s_wrapped.gro" % stem)
    if out.exists() and gro.exists() and out.stat().st_mtime >= gro.stat().st_mtime:
        return out
    if not gro.exists():
        return gro                      # let the caller report the missing file
    try:
        _prepare(stem, out)
    except ImportError as exc:
        print("  ! %s: cannot prepare the structure (%s); rendering the raw "
              ".gro" % (stem, exc))
        return gro
    print("  prepared %s" % out.name)
    return out


def camera(lipids, buffer=ZOOM_BUFFER, is_collapsed=False,
           collapsed_turn=COLLAPSED_TURN_DEG,
           collapsed_window=COLLAPSED_WINDOW_MULT):
    """Frame the membrane, with a different treatment for a collapsed run.

    **Intact run** -- turned 90 deg to edge-on and framed in a **square window**
    whose side is the in-plane extent, centred on the median lipid z.  That is
    the same framing ``zoom all`` would give (the membrane is wider than it is
    thick), but it cannot be hijacked by a few stray atoms far up the box.

    **Collapsed run** -- the two failed CHARMM NPT runs end as a ~4 x 4 x 270 nm
    filament.  Fitting all 270 nm of it gives a hairline whichever way it is
    turned, so the framing stays the same square window and only the *view*
    changes: turned 60 deg rather than 90, under a wide-angle perspective
    camera, the column runs into the screen and the far end shrinks and fogs
    out instead of lying flat across the frame.  The near end is rendered at
    the same sort of scale as an intact membrane; the rest of the box is simply
    outside the frame, receding.
    """
    turn = collapsed_turn if is_collapsed else TURN_DEG
    L = ["",
         "# Orientation and framing -- see render_snapshots.camera()."]
    if is_collapsed:
        # Wide-angle perspective, or the tilt buys nothing: PyMOL's default
        # 20 deg field of view is near-orthographic, so a receding column just
        # reads as a shorter column rather than as distance.
        L += ["set orthoscopic, 0",
              "set field_of_view, %.0f" % COLLAPSED_FOV]
    L += ["turn x, %.0f" % turn,
          "python",
          "from pymol import cmd",
          "import numpy as np",
          "xyz = cmd.get_coords('all')",
          "lipid_z = cmd.get_coords('%s')[:, 2]" % lipids,
          "ext = xyz.max(0) - xyz.min(0)",
          "half = 0.5 * float(max(ext[0], ext[1])) * %.1f"
          % (collapsed_window if is_collapsed else 1.0),
          "zc = float(np.median(lipid_z))",
          "sel = 'all and z > %.3f and z < %.3f' % (zc - half, zc + half)",
          "cmd.zoom(sel if cmd.count_atoms(sel) else 'all', %.1f)" % buffer,
          "python end"]
    return L + [
    "",
    "# `zoom` leaves a near/far slab that slices the foreground spheres open.",
    "# Widening both planes by the system's own extent clears every atom",
    "# without touching the scale or the centring.",
    "python",
    "from pymol import cmd",
    "v = list(cmd.get_view())",
    "d = -v[11]                              # camera distance",
    "ext = cmd.get_extent('all')",
    "span = max(hi - lo for lo, hi in zip(*ext))",
    "v[15] = max(1.0, d - span)              # near",
    "v[16] = d + span                        # far",
    "cmd.set_view(v)",
    "python end",
    ]


def build_pml(ff, ens, rep, png, bg, hydrogens=False, px=PX, wrap=True,
              buffer=ZOOM_BUFFER, collapsed_turn=COLLAPSED_TURN_DEG,
              collapsed_window=COLLAPSED_WINDOW_MULT):
    """The complete PyMOL script for one panel."""
    dcle, lipids, tail, head = selections(ff)
    gro = structure_path(ff, ens, rep, wrap=wrap)
    if not gro.exists():
        raise FileNotFoundError(gro)

    L = ["# generated by HMMM/analysis/render_snapshots.py -- do not edit",
         "load %s, sys" % gro,
         "",
         "# Start from nothing: PyMOL's automatic styling on load depends on",
         "# the residue names it recognises, which differ between the three",
         "# force fields.",
         "hide everything",
         "",
         "# Lipids as spheres, coloured by region: dark tails, light heads,",
         "# orange phosphorus.  LIPID_COLOUR is the fallback for anything the",
         "# two carbon selections miss (oxygens, nitrogen).",
         "show spheres, %s" % lipids,
         "color %s, %s" % (LIPID_COLOUR, lipids),
         "color %s, %s" % (TAIL_COLOUR, tail),
         "color %s, %s" % (HEAD_COLOUR, head),
         "color %s, (%s) and name P*" % (PHOSPHORUS_COLOUR, lipids),
         "",
         "# The solvent as pale-green sticks: legible as individual molecules",
         "# filling the core, and cheap to ray-trace.",
         "show sticks, %s" % dcle,
         "color %s, %s" % (DCLE_COLOUR, dcle)]

    if not hydrogens:
        L += ["",
              "# Hydrogens off: at sphere scale they only add fuzz, and the",
              "# parent project's panels are drawn the same way.",
              "hide everything, hydro"]

    L += camera(lipids, buffer, is_collapsed=collapsed(gro),
                collapsed_turn=collapsed_turn,
                collapsed_window=collapsed_window)

    if bg == "transparent":
        L.append("set ray_opaque_background, 0")
    else:
        L += ["bg_color %s" % bg, "set ray_opaque_background, 1"]

    L += ["png %s, width=%d, height=%d, dpi=%d, ray=1" % (png, px, px, DPI),
          "quit"]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ff", nargs="*", default=FORCEFIELDS, choices=FORCEFIELDS)
    ap.add_argument("--ens", nargs="*", default=ENSEMBLES, choices=ENSEMBLES)
    ap.add_argument("--rep", nargs="*", type=int, default=REPS)
    ap.add_argument("--collapsed-turn", type=float, default=COLLAPSED_TURN_DEG,
                    help="view angle for the collapsed runs (default %.0f deg; "
                         "a smaller angle points the long box more directly "
                         "away from the camera)" % COLLAPSED_TURN_DEG)
    ap.add_argument("--collapsed-window", type=float,
                    default=COLLAPSED_WINDOW_MULT,
                    help="how many in-plane widths of a collapsed column to "
                         "frame (default %.0f)" % COLLAPSED_WINDOW_MULT)
    ap.add_argument("--buffer", type=float, default=ZOOM_BUFFER,
                    help="A of padding around the system (default %.0f)"
                         % ZOOM_BUFFER)
    ap.add_argument("--px", type=int, default=PX, help="image size in pixels")
    ap.add_argument("--bg", default="transparent",
                    help="PyMOL background colour (default: transparent)")
    ap.add_argument("--hydrogens", action="store_true",
                    help="keep hydrogens visible")
    ap.add_argument("--no-wrap", dest="wrap", action="store_false",
                    help="render the downloaded .gro as-is, without centring "
                         "the bilayer in z and wrapping molecules into the box")
    ap.add_argument("--pymol", default=None,
                    help="pymol launcher (default: PATH, then the conda root)")
    ap.add_argument("--dry-run", action="store_true",
                    help="write the .pml files but do not run PyMOL")
    args = ap.parse_args()
    pymol = args.pymol or find_pymol()

    OUT.mkdir(parents=True, exist_ok=True)
    for ff in args.ff:
        for ens in args.ens:
            for rep in args.rep:
                stem = run_stem(ff, ens, rep)
                png = OUT / ("%s.png" % stem)
                pml = OUT / ("%s_render.pml" % stem)
                try:
                    pml.write_text(build_pml(ff, ens, rep, png, args.bg,
                                             hydrogens=args.hydrogens,
                                             px=args.px, wrap=args.wrap,
                                             buffer=args.buffer,
                                             collapsed_turn=args.collapsed_turn,
                                             collapsed_window=args.collapsed_window))
                except FileNotFoundError as exc:
                    print("  ! %s: missing %s" % (stem, exc))
                    continue
                if args.dry_run:
                    print("wrote %s" % pml)
                    continue
                print("rendering %s ..." % stem, flush=True)
                subprocess.run([pymol, "-cq", str(pml)], check=True)
                print("wrote %s" % png)


if __name__ == "__main__":
    main()
