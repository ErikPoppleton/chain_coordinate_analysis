#!/usr/bin/env python3
"""Assemble the rendered snapshots into 1x3 figure panels, one per lipid.

Columns are the three force fields in the usual order (CHARMM, AMBER, SLipids),
each labelled above its panel.  One figure per lipid per snapshot set:

    figure_{lipid}_{set}.svg / .png      set = "last" | "close"

Layout (all dimensions in mm, so the SVG user unit is 1 mm):

    figure width  180.0  = 4 gaps of 1.875 + 3 panels of 57.5
    figure height  62.5  = 5.0 label band + 57.5 panel

The panels are the 57.5 mm / 600 dpi PNGs from snapshots/, placed at their
native size, so exporting at 600 dpi maps them essentially 1:1 with no
resampling.

The SVG *links* the PNGs by relative path (../snapshots/...), so keep it where
it is written.  Pass --embed to inline them as base64 instead, which makes each
SVG self-contained but much larger.

Usage:
    python3 analysis/make_figures.py              # write SVGs, export PNGs
    python3 analysis/make_figures.py --no-export  # SVGs only
    python3 analysis/make_figures.py --embed      # self-contained SVGs
"""

import argparse
import base64
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SNAPSHOTS = ROOT / "snapshots"
PLOTS = ROOT / "plots"

# Inkscape 1.x, headless.  Not on PATH in the usual macOS install.
INKSCAPE_CANDIDATES = [
    "/Applications/Inkscape.app/Contents/MacOS/inkscape",
    "inkscape",
]

LIPIDS = ["DOPC", "DMPC", "DMPG"]
SETS = ["last", "close"]

# (filename key, label) -- same force-field order as the plotting scripts.
FORCEFIELDS = [
    ("charmm", "CHARMM"),
    ("amber", "AMBER"),
    ("slipids", "SLipids"),
]

FIG_W = 180.0        # final figure width (mm)
PANEL = 57.5         # panel edge (mm) -- matches the rendered PNGs
NCOL = len(FORCEFIELDS)
GAP = (FIG_W - NCOL * PANEL) / (NCOL + 1)   # 1.875 mm: 2 outer + 2 inner gaps
LABEL_H = 5.0        # label band above the panels (mm)
FIG_H = LABEL_H + PANEL
FONT_PT = 9.0
FONT_MM = FONT_PT * 25.4 / 72.0             # 3.175 mm
# Baseline that puts the cap-height roughly mid-band (more portable across
# renderers than dominant-baseline).
LABEL_BASELINE = LABEL_H / 2 + FONT_MM * 0.35

DPI = 600


def panel_x(col):
    """Left edge of column `col` (mm)."""
    return GAP + col * (PANEL + GAP)


def image_href(png, embed):
    if embed:
        b64 = base64.b64encode(png.read_bytes()).decode("ascii")
        return f"data:image/png;base64,{b64}"
    # relative to the SVG's location in plots/
    return f"../snapshots/{png.name}"


def build_svg(lipid, which, embed):
    """Return the SVG source for one lipid/set, or None if a panel is missing."""
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" '
        'xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'width="{FIG_W}mm" height="{FIG_H}mm" '
        f'viewBox="0 0 {FIG_W} {FIG_H}">',
        f'  <title>{lipid} ({which})</title>',
    ]
    for col, (ff, label) in enumerate(FORCEFIELDS):
        png = SNAPSHOTS / f"{ff}_{lipid}_{which}.png"
        if not png.exists():
            print(f"  missing {png.name} -- skipping {lipid}/{which}")
            return None
        x = panel_x(col)
        parts.append(
            f'  <text x="{x + PANEL / 2:.4f}" y="{LABEL_BASELINE:.4f}" '
            f'font-family="Helvetica, Arial, sans-serif" '
            f'font-size="{FONT_MM:.4f}" text-anchor="middle" '
            f'fill="#000000">{label}</text>')
        parts.append(
            f'  <image x="{x:.4f}" y="{LABEL_H:.4f}" '
            f'width="{PANEL}" height="{PANEL}" '
            f'image-rendering="optimizeQuality" '
            f'xlink:href="{image_href(png, embed)}"/>')
    parts.append('</svg>')
    return "\n".join(parts) + "\n"


def find_inkscape():
    for cand in INKSCAPE_CANDIDATES:
        try:
            subprocess.run([cand, "--version"], capture_output=True, check=True)
            return cand
        except (OSError, subprocess.CalledProcessError):
            continue
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-export", action="store_true",
                    help="write the SVGs but do not run Inkscape")
    ap.add_argument("--embed", action="store_true",
                    help="embed the PNGs as base64 instead of linking them")
    args = ap.parse_args()

    PLOTS.mkdir(exist_ok=True)
    svgs = []
    for which in SETS:
        for lipid in LIPIDS:
            svg = build_svg(lipid, which, args.embed)
            if svg is None:
                continue
            out = PLOTS / f"figure_{lipid}_{which}.svg"
            out.write_text(svg)
            svgs.append(out)
            print(f"wrote {out.relative_to(ROOT)}")

    if args.no_export or not svgs:
        return

    ink = find_inkscape()
    if ink is None:
        print("\nInkscape not found -- SVGs written, PNG export skipped.",
              file=sys.stderr)
        return

    print()
    for svg in svgs:
        png = svg.with_suffix(".png")
        subprocess.run(
            [ink, str(svg), "--export-type=png",
             f"--export-filename={png}", f"--export-dpi={DPI}"],
            check=True, capture_output=True)
        print(f"exported {png.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
