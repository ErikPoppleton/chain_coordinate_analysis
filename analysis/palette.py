"""Shared colour scheme for the analysis plots.

Colour-blind-friendly blue / green / gold set, replacing the old xmgrace-era
black / red / green:

    CHARMM  -> blue    (#2171b5)
    AMBER   -> green   (#0f7d3b)
    SLipids -> gold    (#d4a017)  ("dark yellow")

The categorical trio (used as solid colours in the PMF and range plots) was
checked with the dataviz colour-blindness validator: worst adjacent-pair CVD
separation is ΔE 16.9 (protan), comfortably clear of the ΔE 8 target.  A naive
bright yellow fails here because it collapses onto green under red/green CVD,
which is why the yellow is a dark gold.

Each force field also gets a matching light->dark sequential colormap for the
10-replicate close-trajectory gradients (sampled from ~0.9 dark to ~0.3 light).
The gold ramp's light end is kept a solid gold rather than near-white, so the
lightest replicates don't wash out.
"""

import colorsys

from matplotlib.colors import LinearSegmentedColormap, ListedColormap

# (key as it appears in the filename prefix, display label, solid colour).
# One shared order used by every plot.
FORCEFIELDS = [
    ("charmm", "CHARMM", "#2171b5"),    # blue
    ("amber", "AMBER", "#0f7d3b"),      # green
    ("slipids", "SLipids", "#d4a017"),  # gold / dark yellow
]

# Light -> dark ramps for the per-replicate gradients, keyed by force field.
# Ordered light-first so cmap(0.9) is dark and cmap(0.3) is light, matching the
# convention of matplotlib's built-in sequential colormaps.
_RAMPS = {
    "charmm": ["#c6dbef", "#08306b"],   # light blue  -> navy
    "amber": ["#c7e9c0", "#00441b"],    # light green -> dark green
    "slipids": ["#e6cf72", "#5c4400"],  # solid gold  -> dark amber (never white)
}

# Matplotlib colormap per force field, built from the ramps above.
CMAPS = {key: LinearSegmentedColormap.from_list(key, stops)
         for key, stops in _RAMPS.items()}


def evenly_spaced_colors(n, lightness=0.5, saturation=0.6):
    """`n` RGB colours with hues evenly spaced around the wheel at *constant*
    HLS lightness and saturation.

    Intended for distinguishing many traces of the same class (e.g. the 64
    upper-leaflet lipid trajectories) where matplotlib's 10-colour default runs
    out and a light->dark sequential ramp would wrongly imply an ordering.  The
    fixed lightness keeps every colour equally prominent; only the hue changes.
    """
    return [colorsys.hls_to_rgb(i / n, lightness, saturation) for i in range(n)]


def evenly_spaced_cmap(n, lightness=0.5, saturation=0.6, name="iso"):
    """A ListedColormap of ``n`` evenly hue-spaced, constant-lightness colours
    (see :func:`evenly_spaced_colors`)."""
    return ListedColormap(evenly_spaced_colors(n, lightness, saturation),
                          name=f"{name}{n}")
