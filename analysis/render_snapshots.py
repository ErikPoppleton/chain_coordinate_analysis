#!/usr/bin/env python3
"""Render the PyMOL snapshot panels used by the main-text figures.

This replaces the hand-driven ``snapshots/*_style.pml`` workflow: those files
carry styling only (no ``load``, no ``png``), so they had to be ``@``-sourced
onto an already-loaded structure and rayed by hand, and the throwaway driver
scripts that did it were never committed.  Here the whole ``.pml`` -- load,
style, camera, ``png``, ``quit`` -- is generated from Python and handed to a
headless ``pymol -cq``, the same pattern as :mod:`make_box_snapshots`.

Two sets, both 1358 x 1358 px (57.5 mm at 600 dpi -- the panel size
``make_figures.py`` places at native scale):

    last   snapshots/{ff}_{lipid}_last_conf.gro  -> {ff}_{lipid}_last.png
           the end-of-umbrella pore, with the chain-coordinate cylinder drawn.
    close  snapshots/{ff}_{lipid}_close.gro      -> {ff}_{lipid}_close.png
           a closed-pore frame, zoomed on the membrane, no cylinder.

The cylinder geometry cannot be recomputed locally -- it comes from the run's
``slurm.*.err`` via ``analysis/cylinder_pml.sh`` -- so the ``cyl`` line is read
back out of the committed ``snapshots/{ff}_{lipid}_cyl_style.pml``, which stays
the source of truth for it.  ``cyl`` itself is the ``~/.pymol/draw_cylinder.py``
plugin loaded from ``.pymolrc`` (so do *not* run pymol with ``-k``).

Usage:
    python3 analysis/render_snapshots.py                    # all 9 x both sets
    python3 analysis/render_snapshots.py --ff charmm --lipid DMPC --set last
    python3 analysis/render_snapshots.py --dry-run          # write .pml only
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lipiddefs import (FORCEFIELDS, LIPIDS, SCHEME, AMBER_HEAD, AMBER_TAIL,
                       WATER_NAME)

ROOT = Path(__file__).resolve().parent.parent
SNAP = ROOT / "snapshots"

PX = 1358               # 57.5 mm at 600 dpi -- the make_figures panel size
DPI = 600
ZOOM_BUFFER = 2.0       # A of padding around the membrane in the "close" view

SETS = ["last", "close"]

# Ion residue names per force field, and the PyMOL colour each should get.
# PyMOL guesses the element from the atom name (a .gro carries no element
# column) and gets several of these wrong, differently per force field:
# CHARMM SOD comes out sulfur-yellow and CAL green, AMBER Na+ comes out white.
IONS = {
    "charmm":  [("SOD", "sodium"), ("CLA", "chlorine"),
                ("MG", "magnesium"), ("CAL", "calcium")],
    "amber":   [("Na+", "sodium"), ("Cl-", "chlorine"),
                ("MG", "magnesium"), ("CA", "calcium")],
    "slipids": [("NA", "sodium"), ("CL", "chlorine"),
                ("MG", "magnesium"), ("CA", "calcium")],
}

LIPID_COLOUR = "gray30"
PHOSPHORUS_COLOUR = "orange"


def esc(resn):
    """PyMOL selection-safe resname ('Na+' -> 'Na\\+'; quoting does not work)."""
    return resn.replace("+", r"\+").replace("-", r"\-")


def gro_path(ff, lipid, which):
    """Input structure, preferring a residue-renumbered copy where one exists.

    Raw Lipid17 gros number every residue 1, which PyMOL reads as one giant
    residue; the ``*_renumbered.gro`` copies fix that where they were made."""
    stem = f"{ff}_{lipid}_" + ("last_conf" if which == "last" else "close")
    renumbered = SNAP / f"{stem}_renumbered.gro"
    return renumbered if renumbered.exists() else SNAP / f"{stem}.gro"


def cylinder_line(ff, lipid):
    """The ``cyl x,y,z0,z1,u=nm`` line from the committed cyl_style.pml."""
    pml = SNAP / f"{ff}_{lipid}_cyl_style.pml"
    if not pml.exists():
        return None
    for line in pml.read_text().splitlines():
        if line.startswith("cyl "):
            return line.strip()
    return None


def selections(ff, lipid):
    """(lipid, tail, water, ions) selection strings for one system."""
    scheme = SCHEME[ff]
    water = f"resn {WATER_NAME[ff]}"
    ions = " or ".join(f"resn {esc(r)}" for r, _ in IONS[ff])
    if scheme == "amber":
        # Lipid17 splits each lipid into head + two identical tail residues,
        # so the tails are a residue selection.
        lipids = f"resn {AMBER_HEAD[lipid]} or resn {AMBER_TAIL[lipid]}"
        tail = f"resn {AMBER_TAIL[lipid]}"
    else:
        # CHARMM/SLipids put the whole lipid in one residue, so the acyl
        # chains have to be picked out by name: the C2*/C3* carbons minus the
        # glycerol backbone (C2, C3) and the ester carbonyls (C21, C31) --
        # which live in AMBER's *head* residue and so stay visible there too.
        lipids = f"resn {lipid}"
        tail = (f"resn {lipid} and name C2*+C3* "
                f"and not name C2+C3+C21+C31")
    return lipids, tail, water, ions


def build_pml(ff, lipid, which, png, bg):
    """The complete PyMOL script for one panel."""
    lipids, tail, water, ions = selections(ff, lipid)
    gro = gro_path(ff, lipid, which)
    if not gro.exists():
        raise FileNotFoundError(gro)

    L = [f"# generated by analysis/render_snapshots.py -- do not edit",
         f"load {gro}, sys",
         "set transparency_mode, 3",
         "",
         "# Start from nothing: PyMOL's automatic styling on load is not the",
         "# same for every water name (TP3 comes up as spheres, TIP3/SOL as",
         "# lines), and nb_spheres is no good here -- it only draws atoms with",
         "# no bonds, and PyMOL infers the O-H bonds when it reads the .gro.",
         "hide everything",
         "",
         "# Lipids as spheres, uniformly grey, with the phosphorus picked out.",
         f"show spheres, {lipids}",
         f"color {LIPID_COLOUR}, {lipids}",
         f"color {PHOSPHORUS_COLOUR}, ({lipids}) and name P+P31",
         "",
         "# Water and ions.",
         f"show spheres, {water}",
         f"show spheres, {ions}"]
    for resn, colour in IONS[ff]:
        L.append(f"color {colour}, resn {esc(resn)}")
    L += ["",
          "# Hide the lipid tails so the pore is visible through the membrane.",
          f"hide everything, {tail}",
          "",
          "# Hydrogens off everywhere: water reads as bare oxygen spheres and",
          "# the lipid heads keep their shape without the H fuzz.",
          "hide everything, hydro",
          "",
          "turn x, 90"]

    if which == "last":
        cyl = cylinder_line(ff, lipid)
        if cyl:
            L += ["", "# Chain-coordinate cylinder (geometry from the run's",
                  "# slurm log, via analysis/cylinder_pml.sh).", cyl]
        else:
            print(f"  ! no cyl line for {ff}_{lipid}; rendering without it")
    else:
        L += ["",
              "# Framing: fit the lipid, so the water above/below may be",
              "# cropped but the whole membrane is always in view.  Fog is off",
              "# because, without the cyl marker to orient us, the pore could",
              "# be anywhere in the box.",
              f"zoom not ({water} or {ions}), {ZOOM_BUFFER}",
              "set depth_cue, 0",
              "set ray_trace_fog, 0"]

    # Both framings leave a tight near/far slab that slices the foreground
    # spheres open: "last" because PyMOL auto-zooms onto the small `cyl` CGO
    # (whose slab is ~2x the cylinder, far shallower than the box), "close"
    # because `zoom` fits the lipid selection.  Widening both planes by the
    # system's own extent clears every atom without touching the scale or the
    # centring -- and hides nothing, since the tails are already off, so the
    # foreground is the same headgroup surface either way.
    L += ["",
          "python",
          "from pymol import cmd",
          "v = list(cmd.get_view())",
          "d = -v[11]                              # camera distance",
          'ext = cmd.get_extent("sys")',
          "span = max(hi - lo for lo, hi in zip(*ext))",
          "v[15] = max(1.0, d - span)              # near",
          "v[16] = d + span                        # far",
          "cmd.set_view(v)",
          "python end"]

    if bg == "transparent":
        L.append("set ray_opaque_background, 0")
    else:
        L += [f"bg_color {bg}", "set ray_opaque_background, 1"]

    L += [f"png {png}, width={PX}, height={PX}, dpi={DPI}, ray=1", "quit"]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ff", nargs="*", default=FORCEFIELDS, choices=FORCEFIELDS)
    ap.add_argument("--lipid", nargs="*", default=LIPIDS, choices=LIPIDS)
    ap.add_argument("--set", nargs="*", dest="sets", default=SETS, choices=SETS)
    ap.add_argument("--bg", default=None,
                    help="PyMOL background colour "
                         "(default: transparent, as the shipped panels are)")
    ap.add_argument("--pymol", default="pymol")
    ap.add_argument("--dry-run", action="store_true",
                    help="write the .pml files but do not run PyMOL")
    args = ap.parse_args()

    for which in args.sets:
        # Both sets ship on a transparent ground: the figures place the panels
        # on the manuscript's own background.
        bg = args.bg or "transparent"
        for lipid in args.lipid:
            for ff in args.ff:
                png = SNAP / f"{ff}_{lipid}_{which}.png"
                pml = SNAP / f"{ff}_{lipid}_{which}_render.pml"
                try:
                    pml.write_text(build_pml(ff, lipid, which, png, bg))
                except FileNotFoundError as exc:
                    print(f"  ! {ff}_{lipid}_{which}: missing {exc}")
                    continue
                if args.dry_run:
                    print(f"wrote {pml}")
                    continue
                subprocess.run([args.pymol, "-cq", str(pml)], check=True)
                print(f"wrote {png}")


if __name__ == "__main__":
    main()
