#!/usr/bin/env python3
"""Compose a production figure out of panels that matplotlib cannot draw.

Most main-text figures are one matplotlib ``Figure`` assembled by a
``figures/figNN_*.py`` script (see :mod:`figstyle`).  This module covers the
rest: panels that arrive as **finished files** -- the PyMOL snapshot renders in
``snapshots/``, an SVG written by another plotting script, anything from a
foreign tool -- laid out on the same 180 mm canvas and shipped in the same form.

    from figcompose import Mosaic

    m = Mosaic()                                   # 180 mm wide
    for col, (ff, label) in enumerate(FF_LABELLED):
        x, y, w = m.grid_cell(col, ncol=3, aspect=1.0)
        m.place(SNAPSHOTS / f"{ff}_{lipid}_last.png", x, y, w)
        m.text(label, x + w / 2, y - 1.0, ha="middle")
    m.save(f"figure_{lipid}_last")

Geometry is in **millimetres** throughout: the canvas is written with
``viewBox="0 0 180 H"``, so one SVG user unit is one millimetre and every
coordinate in a composition script reads as a physical size.

Everything placed is **inlined**: PNGs as base64 ``data:`` URIs, SVGs spliced in
as ``<g transform=...>`` subtrees.  A spliced SVG has every ``id`` it defines
(and every ``url(#...)`` / ``href="#..."`` referring to one) rewritten with a
per-panel prefix -- without that, the ``#pXXXXXXX`` glyph and clip-path ids
matplotlib generates collide between panels and one of them silently blanks.

Output goes through :func:`figstyle.finish_svg`, so a mosaic ships as the same
outlined-SVG + 600 dpi PNG + editable-text-SVG triple as a matplotlib figure.
"""

import base64
import re
import struct
import xml.etree.ElementTree as ET
from pathlib import Path

import figstyle as fs

SVG_NS = "http://www.w3.org/2000/svg"
XLINK_NS = "http://www.w3.org/1999/xlink"
ET.register_namespace("", SVG_NS)
ET.register_namespace("xlink", XLINK_NS)

# Panel-letter geometry, matched to figstyle so mixed figures label identically.
LABEL_BAND_MM = fs.LABEL_BAND_MM
FONT_MM = fs.LABEL_PT * 25.4 / 72.0
FONT_STACK = "Helvetica, Arial, sans-serif"
GAP_MM = 1.875              # inter-panel gap of the historical 1x3 layout

_LEN_RE = re.compile(r"^\s*([0-9.eE+-]+)\s*(mm|cm|in|pt|px|)\s*$")
_UNIT_MM = {"mm": 1.0, "cm": 10.0, "in": 25.4, "pt": 25.4 / 72,
            "px": 25.4 / 96, "": 25.4 / 96}
_URL_RE = re.compile(r"url\(#([^)]+)\)")


# ------------------------------------------------------------------ measuring
def length_mm(value):
    """An SVG length (``"180mm"``, ``"510.24pt"``, ``"96"``) in millimetres."""
    m = _LEN_RE.match(str(value))
    if not m:
        raise ValueError(f"cannot parse SVG length {value!r}")
    return float(m.group(1)) * _UNIT_MM[m.group(2)]


def svg_size_mm(text):
    """``(width, height)`` in mm from an SVG's root tag, or None.

    Starts at the ``<svg>`` tag itself: files open with an ``<?xml ...?>``
    declaration, and Inkscape's plain-SVG output spreads the root attributes
    over several lines.
    """
    i = text.find("<svg")
    if i < 0:
        return None
    root = text[i:text.index(">", i) + 1]
    out = []
    for attr in ("width", "height"):
        m = re.search(r'\b%s="([^"]+)"' % attr, root)
        if not m:
            return None
        out.append(length_mm(m.group(1)))
    return tuple(out)


def png_size_px(path):
    """``(width, height)`` in pixels, straight out of the PNG IHDR chunk."""
    with open(path, "rb") as fh:
        head = fh.read(24)
    return struct.unpack(">II", head[16:24])


def item_aspect(path):
    """height / width of a placeable file, for deriving one side from the other."""
    path = Path(path)
    if path.suffix.lower() == ".png":
        w, h = png_size_px(path)
    else:
        size = svg_size_mm(path.read_text())
        if size is None:
            raise ValueError(f"{path.name}: no width/height on the root <svg>")
        w, h = size
    return h / float(w)


# ----------------------------------------------------------------- id hygiene
def _prefix_ids(elem, prefix):
    """Rewrite every id this subtree defines or references, in place."""
    for node in elem.iter():
        new = {}
        for key, value in node.attrib.items():
            if key == "id":
                value = prefix + value
            elif key in ("href", f"{{{XLINK_NS}}}href") and value.startswith("#"):
                value = "#" + prefix + value[1:]
            elif "url(#" in value:
                value = _URL_RE.sub(lambda m: f"url(#{prefix}{m.group(1)})",
                                    value)
            new[key] = value
        node.attrib = new


# --------------------------------------------------------------------- mosaic
class Mosaic:
    """A 180 mm-wide canvas of placed panels, in millimetres.

    ``height_mm`` may be left None, in which case the canvas is sized at save
    time to the content plus ``pad_mm``.
    """

    def __init__(self, height_mm=None, width_mm=fs.FIG_W_MM, pad_mm=0.0):
        self.width = float(width_mm)
        self.height = height_mm
        self.pad = pad_mm
        self._nodes = []        # serialisable ET elements, in draw order
        self._extent = 0.0      # lowest content edge (mm), for auto height
        self._letters = 0

    # -- geometry helpers --------------------------------------------------
    def grid_cell(self, col, ncol, aspect=1.0, row=0, gap_mm=GAP_MM,
                  top_mm=LABEL_BAND_MM, row_gap_mm=None):
        """``(x, y, w, h)`` of one cell of a regular grid on this canvas.

        Columns share ``gap_mm`` outer and inner gaps (the 180 mm = 4 gaps +
        3 panels arithmetic of the original snapshot figure); ``aspect`` is
        height/width of the panel content; ``top_mm`` is the band left clear
        above the first row for panel letters or column headings.
        """
        w = (self.width - (ncol + 1) * gap_mm) / ncol
        h = w * aspect
        row_gap = gap_mm if row_gap_mm is None else row_gap_mm
        x = gap_mm + col * (w + gap_mm)
        y = top_mm + row * (h + row_gap + top_mm)
        return x, y, w, h

    # -- content -----------------------------------------------------------
    def place(self, path, x_mm, y_mm, w_mm=None, h_mm=None, label=None):
        """Inline a PNG or SVG panel at ``(x, y)`` with size ``w x h`` (mm).

        Give ``w_mm`` or ``h_mm`` and the other is derived from the file's own
        aspect ratio.  ``label`` draws a bold panel letter in the band above the
        panel; pass ``"auto"`` for the next letter in placement order.
        """
        path = Path(path)
        if w_mm is None and h_mm is None:
            raise ValueError("give at least one of w_mm / h_mm")
        aspect = item_aspect(path)
        if w_mm is None:
            w_mm = h_mm / aspect
        if h_mm is None:
            h_mm = w_mm * aspect

        if path.suffix.lower() == ".png":
            self._nodes.append(self._png_node(path, x_mm, y_mm, w_mm, h_mm))
        else:
            self._nodes.append(self._svg_node(path, x_mm, y_mm, w_mm, h_mm))

        if label is not None:
            self.panel_letter(x_mm, y_mm, None if label == "auto" else label)
        self._extent = max(self._extent, y_mm + h_mm)
        return x_mm, y_mm, w_mm, h_mm

    def _png_node(self, path, x, y, w, h):
        b64 = base64.b64encode(path.read_bytes()).decode("ascii")
        node = ET.Element(f"{{{SVG_NS}}}image", {
            "x": f"{x:.4f}", "y": f"{y:.4f}",
            "width": f"{w:.4f}", "height": f"{h:.4f}",
            "preserveAspectRatio": "none",
            "image-rendering": "optimizeQuality",
            f"{{{XLINK_NS}}}href": f"data:image/png;base64,{b64}",
        })
        return node

    def _svg_node(self, path, x, y, w, h):
        """Splice an SVG file in as a scaled, id-namespaced ``<g>``."""
        child = ET.fromstring(path.read_text())
        _prefix_ids(child, f"m{len(self._nodes)}_")

        vb = child.get("viewBox")
        if vb:
            vx, vy, vw, vh = (float(v) for v in vb.replace(",", " ").split())
        else:
            size = svg_size_mm(path.read_text())
            if size is None:
                raise ValueError(f"{path.name}: no viewBox and no width/height")
            vx, vy, (vw, vh) = 0.0, 0.0, size
        sx, sy = w / vw, h / vh

        g = ET.Element(f"{{{SVG_NS}}}g", {
            "transform": (f"translate({x:.4f},{y:.4f}) scale({sx:.6f},{sy:.6f}) "
                          f"translate({-vx:.4f},{-vy:.4f})"),
        })
        for node in list(child):
            g.append(node)
        return g

    def text(self, s, x_mm, y_mm, ha="start", pt=fs.LABEL_PT, bold=False,
             colour="#000000"):
        """A text label with its **baseline** at ``y_mm``.

        ``ha`` is ``start`` / ``middle`` / ``end`` (SVG's ``text-anchor``); an
        explicit baseline is used rather than ``dominant-baseline``, which
        renderers disagree about.
        """
        self._nodes.append(ET.Element(f"{{{SVG_NS}}}text", {
            "x": f"{x_mm:.4f}", "y": f"{y_mm:.4f}",
            "font-family": FONT_STACK,
            "font-size": f"{pt * 25.4 / 72.0:.4f}",
            "font-weight": "bold" if bold else "normal",
            "text-anchor": ha, "fill": colour,
        }))
        self._nodes[-1].text = s
        self._extent = max(self._extent, y_mm)
        return self._nodes[-1]

    def panel_letter(self, x_mm, y_mm, letter=None, dx_mm=0.0, dy_mm=-1.0):
        """Bold lower-case panel letter above-left of a panel's top-left corner.

        Letters run a, b, c... in call order unless given explicitly.
        """
        if letter is None:
            letter = chr(ord("a") + self._letters)
            self._letters += 1
        return self.text(letter, x_mm + dx_mm, y_mm + dy_mm, ha="start",
                         pt=fs.LABEL_PT, bold=True)

    # -- output ------------------------------------------------------------
    def to_svg(self):
        height = self.height if self.height is not None else self._extent + self.pad
        root = ET.Element(f"{{{SVG_NS}}}svg", {
            "width": f"{self.width}mm", "height": f"{height}mm",
            "viewBox": f"0 0 {self.width} {height}",
            "version": "1.1",
        })
        # An explicit white ground: a transparent SVG turns black in some
        # viewers and in PDF placement.
        root.append(ET.Element(f"{{{SVG_NS}}}rect", {
            "x": "0", "y": "0", "width": f"{self.width}",
            "height": f"{height}", "fill": "#ffffff"}))
        for node in self._nodes:
            root.append(node)
        return ('<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n'
                + ET.tostring(root, encoding="unicode") + "\n")

    def save(self, name, editable=True, outdir=fs.FINAL):
        """Ship ``{name}.svg`` (outlined) + ``{name}.png`` (600 dpi) +
        ``{name}_text.svg``, exactly as :func:`figstyle.save` does."""
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        text_svg = outdir / f"{name}_text.svg"
        text_svg.write_text(self.to_svg())
        written = fs.finish_svg(text_svg, name, editable=editable,
                                outdir=outdir, quiet=True)
        if written is None:
            print("Inkscape not found -- shipping the editable-text SVG only; "
                  "no outlined SVG and no PNG.")
            written = [text_svg]
        for p in written:
            print(f"wrote {p.relative_to(fs.ROOT)}")
        height = self.height if self.height is not None else self._extent + self.pad
        print(f"  {self.width:.2f} x {height:.2f} mm")
        return written
