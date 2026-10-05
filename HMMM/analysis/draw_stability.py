#!/usr/bin/env python3
"""Panel painters for the HMMM **stability** figure.

Pure ``draw_*(ax)`` functions -- they create no figure and save nothing.

``draw_snapshot(ax, stem)``  a PyMOL render from ``plots/snapshots`` as an image
                             panel (the four illustrative configurations).
``draw_stability_time(ax)``  bars of how long each system stayed a bilayer.
``draw_sd_traces(ax)``       the per-frame lateral SD of d_HH that the stability
                             time is read from, one line per (force field,
                             ensemble) for replicate 1.

Colour convention, shared with the properties figure: the **force field** picks
the light-to-dark ramp (``palette.CMAPS``, as the pore project's closing
trajectories use) and the **ensemble** picks the shade within it -- every third
colour of that ten-step ramp, so NPT is darkest and posres lightest.  That way
one colour carries both axes and neither needs its own legend entry per
combination.

Reads ``data/thickness_map.npz`` and ``data/stability.json``, so run
``thickness_variation.py`` and ``stability.py`` first.  numpy + matplotlib only.
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PARENT = ROOT.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PARENT / "analysis"))

import figstyle
from palette import CMAPS
import stability as stab
from hmmm_defs import (FORCEFIELDS, FF_LABELS, ENSEMBLES, ENS_LABELS, REPS,
                       stem as run_stem, system as system_key)

DATA = ROOT / "data"
SNAPSHOTS = ROOT / "plots" / "snapshots"

# Every third colour of the ten-step close-trajectory ramp: dark -> light.
SHADES = np.linspace(0.9, 0.3, 10)[::3][:len(ENSEMBLES)]
FULL_NS = stab.FULL_NS
THRESHOLD_NM = stab.THRESHOLD_NM
SMOOTH_NS = stab.SMOOTH_NS

BAR_W = 0.26
OFFSETS = np.array([-0.28, 0.0, 0.28])

# The properties figure flags an unstable system the same way.  Matplotlib sets
# hatch density by *repeating* the pattern character, so HATCH_DENSITY is the
# number of repeats: the boxes being marked can be only a few mm tall at
# 180 mm, and at the one- or two-repeat default a cross-hatch puts barely a
# single X inside one.  `figstyle` already pins `hatch.linewidth` to 0.6, which
# keeps the dense pattern from filling in solid.
# Denser than this closes up into a solid red block and swamps the force-field
# fill colour underneath; sparser leaves barely one X inside a short box.
HATCH_DENSITY = 5
HATCH = "/" * HATCH_DENSITY + "\\" * HATCH_DENSITY
HATCH_COLOUR = "#d62728"


def colour(ff, ens):
    """The (force field, ensemble) colour -- ramp by force field, shade by ensemble."""
    return CMAPS[ff](SHADES[ENSEMBLES.index(ens)])


def draw_snapshot(ax, stem, title=None):
    """One PyMOL render as an image panel."""
    import matplotlib.image as mpimg
    path = SNAPSHOTS / ("%s.png" % stem)
    if not path.exists():
        ax.text(0.5, 0.5, "%s.png\nmissing" % stem, ha="center", va="center",
                transform=ax.transAxes, fontsize=7, color="0.6")
        ax.set_axis_off()
        return
    ax.imshow(mpimg.imread(str(path)))
    ax.set_axis_off()
    if title:
        ax.set_title(title)


def draw_stability_time(ax, verbose=False):
    """Bars of the time each system stayed a bilayer, ensembles along x.

    The bar is the mean over replicates and the dots are the individual runs --
    which matters here, because the systems that fail do not fail in all three
    (CHARMM NPT breaks in two runs of three, CHARMM NPAT in one).  A system with
    any replicate short of the full microsecond is overdrawn with the red
    ``//\\\\`` hatch used throughout the properties figure.
    """
    records = stab.load()
    for fi, ff in enumerate(FORCEFIELDS):
        xs, ys, hatched, points = [], [], [], []
        for ei, ens in enumerate(ENSEMBLES):
            times = [records[run_stem(ff, ens, r)]["stability_ns"]
                     for r in REPS if run_stem(ff, ens, r) in records]
            if not times:
                continue
            x = ei + OFFSETS[fi]
            xs.append(x)
            ys.append(float(np.mean(times)))
            hatched.append(stab.any_unstable(records, ff, ens))
            points.append((x, times))
            if verbose:
                print("  %-8s %-6s stability %s ns" % (ff, ens,
                      ", ".join("%.0f" % t for t in times)))
        if not xs:
            continue
        ax.bar(xs, ys, BAR_W, color=[colour(ff, e) for e in ENSEMBLES[:len(xs)]],
               edgecolor="black", linewidth=0.7,
               label=FF_LABELS[ff] if fi == 0 else None)
        # Red hatching is drawn as a second, face-less patch so the bar keeps
        # its black outline: matplotlib ties hatch colour to the edge colour.
        flag_x = [x for x, h in zip(xs, hatched) if h]
        flag_y = [y for y, h in zip(ys, hatched) if h]
        if flag_x:
            ax.bar(flag_x, flag_y, BAR_W, facecolor="none",
                   edgecolor=HATCH_COLOUR, linewidth=0.0, hatch=HATCH)
        for x, times in points:
            ax.plot([x] * len(times), times, ls="none", marker="o",
                    markersize=2.0, markerfacecolor="white",
                    markeredgecolor="black", markeredgewidth=0.5, zorder=4)

    ax.axhline(FULL_NS, color="0.4", lw=0.7, ls=(0, (3, 2)))
    ax.set_xticks(range(len(ENSEMBLES)), [ENS_LABELS[e] for e in ENSEMBLES])
    ax.set_xlim(-0.5, len(ENSEMBLES) - 0.5)
    ax.set_ylabel("stable as a bilayer (ns)")
    ax.set_ylim(0, FULL_NS * 1.12)
    figstyle.categorical_x(ax)


def _smooth(y, window):
    return stab.rolling_mean(y, window)


def draw_sd_traces(ax, rep=1, smooth_ns=SMOOTH_NS, show_raw=True, verbose=False):
    """Lateral SD of d_HH against time, one line per (force field, ensemble).

    This is the series the stability time is read off: the raw trace is drawn
    faintly behind a rolling mean, and the dashed line is the threshold.  The
    y axis is logarithmic because the six runs span 0.14 nm (a flat restrained
    bilayer) to 30 nm (a membrane that has become a filament) -- on a linear
    axis every stable run would sit on the baseline.
    """
    npz = np.load(DATA / "thickness_map.npz")
    for ff in FORCEFIELDS:
        for ens in ENSEMBLES:
            key = "%s_rep%d" % (system_key(ff, ens), rep)
            if key + "_sd" not in npz:
                continue
            t, sd = npz[key + "_time"], npz[key + "_sd"]
            dt = float(np.median(np.diff(t))) if len(t) > 1 else 1.0
            col = colour(ff, ens)
            if show_raw:
                ax.plot(t, sd, color=col, lw=0.3, alpha=0.18)
            ax.plot(t, _smooth(sd, max(1, int(round(smooth_ns / dt)))),
                    color=col, lw=1.0,
                    label="%s %s" % (FF_LABELS[ff], ENS_LABELS[ens]))
            if verbose:
                print("  %-8s %-6s SD %.3f -> %.3f nm" % (ff, ens, sd[0],
                                                          np.nanmean(sd[-10:])))
    ax.axhline(THRESHOLD_NM, color="0.4", lw=0.7, ls=(0, (3, 2)))
    ax.set_yscale("log")
    ax.set_xlabel("time (ns)")
    ax.set_ylabel(r"lateral SD of $d_\mathrm{HH}$ (nm)")
    ax.set_xlim(0, FULL_NS)


def legend_handles():
    """One entry per (force field, ensemble) in the shared colour scheme."""
    from matplotlib.lines import Line2D
    handles, labels = [], []
    for ff in FORCEFIELDS:
        for ens in ENSEMBLES:
            handles.append(Line2D([0], [0], color=colour(ff, ens), lw=1.4))
            labels.append("%s %s" % (FF_LABELS[ff], ENS_LABELS[ens]))
    return handles, labels
