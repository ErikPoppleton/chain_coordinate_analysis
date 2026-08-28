#!/usr/bin/env python3
"""Assemble the rendered snapshots into 1x3 figure panels, one per lipid.

Columns are the three force fields in the usual order (CHARMM, AMBER, SLipids),
each labelled above its panel and lettered a/b/c.  One figure per lipid per
snapshot set, written to ``plots/final/``:

    figure_{lipid}_{set}.svg / .png / _text.svg      set = "last" | "close"

Layout (millimetres, via :mod:`figcompose` -- the SVG user unit is 1 mm):

    figure width  180.0  = 4 gaps of 1.875 + 3 panels of 57.5
    figure height  62.5  = 5.0 label band + 57.5 panel

The panels are the 57.5 mm / 600 dpi PNGs from ``snapshots/``, placed at their
native size, so the 600 dpi export maps them essentially 1:1 with no
resampling.  They are inlined as base64, so the SVG is self-contained.

Usage:
    python3 analysis/make_figures.py
"""

import argparse
from pathlib import Path

import figstyle as fs
from figcompose import Mosaic
from lipiddefs import LIPIDS, FF_LABELLED as FORCEFIELDS

ROOT = Path(__file__).resolve().parent.parent
SNAPSHOTS = ROOT / "snapshots"

SETS = ["last", "close"]

PANEL = 57.5         # panel edge (mm) -- matches the rendered PNGs
GAP = 1.875          # 2 outer + 2 inner gaps across the 180 mm width
LABEL_H = 5.0        # label band above the panels (mm)
FIG_H = LABEL_H + PANEL


def build(lipid, which, letters=True):
    """A Mosaic for one lipid/set, or None if a panel PNG is missing."""
    m = Mosaic(height_mm=FIG_H)
    for col, (ff, label) in enumerate(FORCEFIELDS):
        png = SNAPSHOTS / f"{ff}_{lipid}_{which}.png"
        if not png.exists():
            print(f"  missing {png.name} -- skipping {lipid}/{which}")
            return None
        x, y, w, _ = m.grid_cell(col, len(FORCEFIELDS), aspect=1.0,
                                 gap_mm=GAP, top_mm=LABEL_H)
        m.place(png, x, y, w, PANEL, label="auto" if letters else None)
        # Force-field heading, centred over its panel, on the same baseline as
        # the panel letter.
        m.text(label, x + w / 2, y - 1.0, ha="middle")
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-letters", action="store_true",
                    help="omit the a/b/c panel letters")
    ap.add_argument("--sets", nargs="+", default=SETS, choices=SETS)
    ap.add_argument("--lipids", nargs="+", default=LIPIDS)
    args = ap.parse_args()

    for which in args.sets:
        for lipid in args.lipids:
            m = build(lipid, which, letters=not args.no_letters)
            if m is not None:
                m.save(f"figure_{lipid}_{which}")


if __name__ == "__main__":
    main()
