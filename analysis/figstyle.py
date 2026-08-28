#!/usr/bin/env python3
"""Shared style + output machinery for the **main-text (production) figures**.

Every final figure is 180 mm wide, at most 10 inches tall, Helvetica everywhere
except math (STIX Two Sans via ``mathtext.fontset="stixsans"``), 9 pt for axis
labels/titles, 7 pt for tick labels, default linewidth 1, ticks *inside* the
axes and mirrored on all four spines with labels only on the left and bottom.
Panels are lettered with bold lower-case 9 pt ``a``, ``b``, ``c``.

Usage from a ``figures/figNN_*.py`` production script::

    import figstyle as fs

    fs.use()
    fig, axes = fs.figure_grid(2, 3, height_mm=110)
    ...                                   # draw_*(ax) from the analysis scripts
    fs.panel_labels(axes)
    fs.save(fig, "fig01_equilibrium")     # -> plots/final/fig01_equilibrium.{svg,png}

The shipped SVG has its text converted to paths by Inkscape, so it is
standalone and monolithic -- no font names, no linked images (rasterised
artists are inlined as base64).  An editable-text copy is written alongside as
``{name}_text.svg``, and the 600 dpi PNG is rendered *from the shipped SVG* so
the two deliverables are the same drawing.  See :func:`save`.

Layout is **constrained**, never ``bbox_inches="tight"``: the 180 mm width has
to survive to disk exactly, and a tight bounding box would shrink-wrap it.
"""

import subprocess
import sys
import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

ROOT = Path(__file__).resolve().parent.parent
FINAL = ROOT / "plots" / "final"

# ---------------------------------------------------------------- dimensions
FIG_W_MM = 180.0                       # journal double-column width
FIG_W_IN = FIG_W_MM / 25.4             # 7.0866 in
MAX_H_IN = 10.0                        # hard ceiling on figure height
DPI = 600
PNG_W_PX = int(round(FIG_W_IN * DPI))  # 4252 px -- pinned exactly on export

LABEL_PT = 9.0                         # axis labels, titles, panel letters
TICK_PT = 7.0                          # tick labels, legend text
LINEWIDTH = 1.0

# Inkscape 1.x, headless.  Not on PATH in the usual macOS install.
INKSCAPE_CANDIDATES = [
    "/Applications/Inkscape.app/Contents/MacOS/inkscape",
    "inkscape",
]


def mm(x):
    """Millimetres -> inches (matplotlib's unit for figure sizes)."""
    return x / 25.4


# -------------------------------------------------------------------- rcParams
RC = {
    # Helvetica everywhere; Arial/DejaVu only as fallbacks on other machines.
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    # ...except inside $...$, where STIX Two Sans supplies proper italic
    # variables and Greek without clashing with the Helvetica body text.
    "mathtext.fontset": "stixsans",

    "font.size": LABEL_PT,
    "axes.labelsize": LABEL_PT,
    "axes.titlesize": LABEL_PT,
    "figure.titlesize": LABEL_PT,
    "xtick.labelsize": TICK_PT,
    "ytick.labelsize": TICK_PT,
    "legend.fontsize": TICK_PT,
    "legend.title_fontsize": TICK_PT,

    "lines.linewidth": LINEWIDTH,
    "lines.markersize": 3.0,
    "axes.linewidth": LINEWIDTH,
    "patch.linewidth": LINEWIDTH,
    "hatch.linewidth": 0.6,
    "grid.linewidth": 0.5,

    # Ticks: inside the axes, mirrored on all four spines, labelled only on
    # the left and bottom.  ``mirror_ticks`` handles the twin-axis exception.
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.labeltop": False,
    "ytick.labelright": False,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "xtick.major.width": LINEWIDTH,
    "ytick.major.width": LINEWIDTH,
    "xtick.minor.width": 0.6,
    "ytick.minor.width": 0.6,
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "xtick.minor.size": 1.8,
    "ytick.minor.size": 1.8,
    "xtick.major.pad": 2.0,
    "ytick.major.pad": 2.0,

    "axes.labelpad": 2.0,
    "axes.titlepad": 3.0,
    "axes.formatter.use_mathtext": True,
    "axes.axisbelow": True,

    "legend.frameon": False,
    "legend.handlelength": 1.6,
    "legend.handletextpad": 0.5,
    "legend.columnspacing": 1.2,
    "legend.labelspacing": 0.3,
    "legend.borderaxespad": 0.3,

    # 127 dpi = exactly 5 px/mm, so the on-screen canvas' integer pixel size
    # does not quantise the 180 mm width away (at 100 dpi it lands on 179.83).
    "figure.dpi": 127,
    "savefig.dpi": DPI,
    "savefig.bbox": None,          # never shrink-wrap: the width is load-bearing
    "savefig.transparent": False,
    "savefig.facecolor": "white",

    # Monolithic SVG: text stays text (no external font *file* is referenced),
    # images/rasterised artists are inlined as base64.
    "svg.fonttype": "none",
    "svg.image_inline": True,
    "svg.hashsalt": "chain-coordinate-analysis",   # reproducible element ids
}


def use(**overrides):
    """Apply the production style globally (plus any ad-hoc overrides)."""
    mpl.rcParams.update(RC)
    if overrides:
        mpl.rcParams.update(overrides)


# --------------------------------------------------------------------- figures
LABEL_BAND_MM = 3.5    # strip reserved above the top row for the panel letters


def figure(height_mm, label_band_mm=LABEL_BAND_MM, **kwargs):
    """A blank 180 mm-wide figure ``height_mm`` tall, constrained-layout.

    ``label_band_mm`` is kept clear at the top of the canvas for the panel
    letters of the first row (which sit *outside* their axes and take no part
    in the layout, so without the reservation they are clipped by the canvas
    edge).  Lower rows need no reservation -- the row above them supplies the
    gap via its own x-axis label.  Pass 0 for a figure with no letters.
    """
    h_in = mm(height_mm)
    if h_in > MAX_H_IN + 1e-9:
        raise ValueError(f"height {height_mm:.1f} mm exceeds the {MAX_H_IN} in "
                         f"({MAX_H_IN * 25.4:.0f} mm) ceiling")
    kwargs.setdefault("layout", "constrained")
    fig = plt.figure(figsize=(FIG_W_IN, h_in), **kwargs)
    if label_band_mm and kwargs["layout"] == "constrained":
        fig.get_layout_engine().set(rect=(0, 0, 1, 1 - mm(label_band_mm) / h_in))
    return fig


def figure_grid(nrows, ncols, height_mm, **kwargs):
    """``(fig, axes)`` for a regular ``nrows x ncols`` panel grid.

    ``axes`` is always a flat list in reading order, which is the order
    :func:`panel_labels` letters them in.
    """
    fig_kw = ("layout", "facecolor", "label_band_mm")
    fig = figure(height_mm, **{k: v for k, v in kwargs.items() if k in fig_kw})
    subplot_kw = {k: v for k, v in kwargs.items() if k not in fig_kw}
    axes = fig.subplots(nrows, ncols, **subplot_kw)
    try:
        axes = list(axes.ravel())
    except AttributeError:
        axes = [axes]
    return fig, axes


# ---------------------------------------------------------------- panel letters
def panel_labels(axes, labels=None, dx_mm=None, dy_mm=None, inside=False,
                 **text_kw):
    """Bold lower-case 9 pt panel letters, one per axes, in reading order.

    Placed in **figure** coordinates offset from each axes' top-left corner, so
    the letters stay aligned across panels of unequal size (a per-axes
    ``transAxes`` offset drifts when panel heights differ).  ``labels`` may be
    given explicitly for merged or skipped panels.

    The default puts each letter *outside* the axes, up and to the left, in the
    space the y tick labels occupy.  On a grid with **shared** y axes that space
    is not there -- the panels abut, and the letter would land on its left-hand
    neighbour's frame -- so pass ``inside=True`` to tuck the letters just inside
    each panel's top-left corner instead.
    """
    axes = list(axes)
    if labels is None:
        labels = [chr(ord("a") + i) for i in range(len(axes))]
    fig = axes[0].get_figure()
    # Resolve the constrained layout so ``get_position`` is final.  The letters
    # are figure-level text and take no part in the layout, so adding them
    # afterwards cannot move the panels.
    fig.draw_without_rendering()
    w_in, h_in = fig.get_size_inches()
    if dx_mm is None:
        dx_mm = 1.5 if inside else -5.5
    if dy_mm is None:
        dy_mm = -1.0 if inside else 0.5
    dx, dy = mm(dx_mm) / w_in, mm(dy_mm) / h_in
    out = []
    for ax, label in zip(axes, labels):
        if label is None:
            continue
        box = ax.get_position()
        kw = dict(fontsize=LABEL_PT, fontweight="bold", ha="left",
                  va="top" if inside else "bottom")
        if inside:
            # Inside letters can land on data; a soft white pad keeps them
            # legible without hiding much of the trace underneath.
            kw["bbox"] = dict(facecolor="white", edgecolor="none",
                              alpha=0.75, pad=0.8)
        kw.update(text_kw)
        out.append(fig.text(box.x0 + dx, box.y1 + dy, label, **kw))
    return out


# ---------------------------------------------------------------- categorical
def categorical_x(ax):
    """Strip the x-axis ticks that only make sense on a continuous scale.

    For a categorical x (lipid names, force fields) the minor ticks between
    categories mark nothing, and the mirrored top ticks are equally
    meaningless, so both are removed; the labelled major ticks under each
    category stay.
    """
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.tick_params(axis="x", top=False, which="both")
    return ax


# ------------------------------------------------------------------ twin axes
def mirror_ticks(ax, right_data=False):
    """Configure the right-hand spine of ``ax``.

    By default the style mirrors unlabelled ticks on the right.  When the right
    axis carries a *separate dataset* (the k_BT twin of the tilt-modulus bar
    charts), call ``mirror_ticks(ax, right_data=True)``: the primary axes stop
    drawing right-hand ticks and the returned twin owns that spine, with labels.
    """
    if not right_data:
        ax.tick_params(axis="y", right=True, labelright=False, direction="in")
        return None
    ax.tick_params(axis="y", right=False)
    twin = ax.twinx()
    twin.tick_params(axis="y", left=False, right=True, labelright=True,
                     direction="in", labelsize=TICK_PT)
    twin.tick_params(axis="x", top=False, bottom=False)
    return twin


# --------------------------------------------------------------------- output
def find_inkscape():
    for cand in INKSCAPE_CANDIDATES:
        try:
            subprocess.run([cand, "--version"], capture_output=True, check=True)
            return cand
        except (OSError, subprocess.CalledProcessError):
            continue
    return None


def export_png(svg, png=None, width_px=PNG_W_PX):
    """Rasterise ``svg`` with Inkscape at exactly ``width_px`` pixels.

    ``--export-width`` rather than ``--export-dpi``: 180 mm at 600 dpi is
    4251.97 px, so the dpi flag would leave the final pixel width to Inkscape's
    rounding.  Returns the PNG path, or None if Inkscape is unavailable.
    """
    svg = Path(svg)
    png = Path(png) if png else svg.with_suffix(".png")
    ink = find_inkscape()
    if ink is None:
        return None
    subprocess.run([ink, str(svg), "--export-type=png",
                    f"--export-filename={png}", f"--export-width={width_px}"],
                   check=True, capture_output=True)
    return png


def export_outlines(svg, out=None):
    """Text-to-paths copy of ``svg`` (Inkscape), for journals that reject a
    font dependency.  The working SVG keeps editable text."""
    svg = Path(svg)
    out = Path(out) if out else svg.with_name(svg.stem + "_outlines.svg")
    ink = find_inkscape()
    if ink is None:
        return None
    subprocess.run([ink, str(svg), "--export-type=svg", "--export-text-to-path",
                    "--export-plain-svg", f"--export-filename={out}"],
                   check=True, capture_output=True)
    return out


def png_width_px(svg, default=PNG_W_PX):
    """600 dpi pixel width for ``svg``, from its own declared mm width."""
    from figcompose import svg_size_mm            # local: avoids a cycle
    try:
        size = svg_size_mm(Path(svg).read_text())
    except OSError:
        size = None
    if not size:
        return default
    return int(round(size[0] / 25.4 * DPI))


def finish_svg(text_svg, name, editable=True, outdir=FINAL, quiet=False):
    """Turn an editable-text SVG into the shipped pair.

    ``text_svg`` (which may already live in ``outdir``) is outlined by Inkscape
    into ``{name}.svg`` and rasterised into ``{name}.png``; the editable copy is
    kept as ``{name}_text.svg`` unless ``editable`` is False.  Returns the list
    of files written, or None if Inkscape is unavailable (the caller then has to
    supply its own fallback -- :func:`save` renders one with matplotlib).

    Shared by :func:`save` and ``figcompose.Mosaic.save``, so a matplotlib
    figure and a hand-composed mosaic ship in exactly the same form.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    text_svg = Path(text_svg)
    keep = outdir / f"{name}_text.svg"
    if text_svg != keep:
        keep.write_bytes(text_svg.read_bytes())
    svg, png = outdir / f"{name}.svg", outdir / f"{name}.png"
    if find_inkscape() is None:
        return None
    export_outlines(keep, svg)
    # 600 dpi at the canvas' *own* width: 180 mm figures come out at the pinned
    # 4252 px, and a deliberately narrower canvas (an SI strip sized to its
    # panels) still ships at 600 dpi rather than being upscaled to 180 mm.
    export_png(svg, png, width_px=png_width_px(svg))
    written = [svg, png] + ([keep] if editable else [])
    if not editable:
        keep.unlink()
    if not quiet:
        for p in written:
            print(f"wrote {p.relative_to(ROOT)}")
    return written


def save(fig, name, editable=True, outdir=FINAL):
    """Write the deliverables for ``name`` into ``plots/final/``:

      * ``{name}.svg`` -- **the deliverable**: standalone and monolithic, with
        the text converted to paths by Inkscape.  Matplotlib's own SVG keeps
        text as text, but mathtext then names STIX fonts (``STIXGeneral``,
        ``STIXNonUnicode``) that only exist inside matplotlib's package data on
        most machines, so a reader without them silently gets substituted
        glyphs.  Outlining removes every font dependency; rasterised artists
        are already inlined as base64, so nothing external remains.
      * ``{name}_text.svg`` -- the same figure with editable text, kept for
        last-minute tweaks in Inkscape (suppress with ``editable=False``).
      * ``{name}.png`` -- 600 dpi, 4252 px wide, rendered *from the deliverable
        SVG*, so the two are guaranteed to be the same drawing.

    Without Inkscape, matplotlib's own ``svg.fonttype="path"`` is used for the
    deliverable and the PNG is rasterised directly (warned).
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    w_in, h_in = fig.get_size_inches()
    if abs(w_in - FIG_W_IN) * 25.4 > 0.05:
        warnings.warn(f"{name}: figure is {w_in * 25.4:.2f} mm wide, not "
                      f"{FIG_W_MM:.0f} mm")
    if h_in > MAX_H_IN + 1e-9:
        warnings.warn(f"{name}: figure is {h_in:.2f} in tall, over the "
                      f"{MAX_H_IN} in ceiling")

    svg = outdir / f"{name}.svg"
    text_svg = outdir / f"{name}_text.svg"
    png = outdir / f"{name}.png"
    written = []

    ink = find_inkscape()
    if ink is None:
        print("Inkscape not found -- outlining via matplotlib, PNG rasterised "
              "directly.", file=sys.stderr)
        with mpl.rc_context({"svg.fonttype": "path"}):
            fig.savefig(svg, bbox_inches=None)
        fig.savefig(png, dpi=DPI, bbox_inches=None)
        written += [svg, png]
        if editable:
            fig.savefig(text_svg, bbox_inches=None)
            written.append(text_svg)
    else:
        fig.savefig(text_svg, bbox_inches=None)
        written = finish_svg(text_svg, name, editable=editable, outdir=outdir,
                             quiet=True)

    for p in written:
        print(f"wrote {p.relative_to(ROOT)}")
    print(f"  {w_in * 25.4:.2f} x {h_in * 25.4:.2f} mm  ({h_in:.2f} in tall)")
    return written
