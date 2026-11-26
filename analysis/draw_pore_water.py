#!/usr/bin/env python3
"""Panel painters comparing **pore-waist water** with the **sub-window PMFs**.

Figure 2 draws five partial PMFs per system: consecutive slices of the umbrella
production run, set by the ``-b``/``-e`` of each ``gmx wham`` run (10 ns slices
of 50 ns for DOPC/DMPG, 20 ns slices of 100 ns for DMPC).  Figure 4 counts the
water oxygens in the middle third of the CV cylinder (|z| <= 0.45 nm) in the
xi = 1 window (umbrella23, 200 ns).  This module puts the two on the same time
slices.

``subwindows(ff, lipid)``          the (b, e) ps range of each partial profile,
                                   read from its wham header.
``draw_water_vs_barrier(ax, part)`` one sub-window: mean umbrella23 water count
                                   over that slice (x) against that slice's
                                   partial-PMF barrier (y); colour = force
                                   field, shape = lipid.
``draw_subwindow_boxes(ax, lipid, window)``
                                   per-frame count distribution in one umbrella
                                   window, sub-window on x, one box per force
                                   field (the Figure 1b-c box style).
``draw_water_vs_xi(ax, lipid)``    water count against xi over all 24 windows,
                                   one curve per (force field, sub-window), the
                                   sub-windows as the force field's light->dark
                                   ``CMAPS`` gradient.  No grand mean.  Needs the
                                   per-window counts in ``data/poredens_windows/``
                                   (``pore_density.py --windows all``).

Every window is sliced with the **same** (b, e) as the wham run, so the umbrella23
slice in the scatter is exactly the slice that went into that partial PMF.
"""

import re
from pathlib import Path

import numpy as np

from lipiddefs import LIPIDS
from palette import FORCEFIELDS, CMAPS
from draw_pore_energy import LIPID_MARKERS, N_PARTS, load_xvg

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
WINDOWS = DATA / "poredens_windows"
N_WINDOWS = 24
COUNT_KEY = "water_mid"          # middle-third slab, as in Figure 4

# Sub-window k (0-based) -> position on the force field's light->dark ramp.
# Same span the close-replicate gradients use (draw_close: 0.3..0.9), early
# slices light, late slices dark.
SHADES = np.linspace(0.3, 0.9, N_PARTS)

COUNT_LABELS = {"water_mid": "water O in pore waist",
                "water": "water O in CV cylinder",
                "phosphorus_mid": "phosphate P in pore waist",
                "phosphorus": "phosphate P in CV cylinder"}

# Box geometry shared with Figure 1b-c (draw_equilibrium).
WHIS = (5, 95)                   # whisker percentiles
BOX_W = 0.22
OFFSETS = np.array([-0.26, 0.0, 0.26])
XI08_WINDOW = 15                 # restraint centre xi = 0.80 (select_umbrella_frames)
XI1_WINDOW = 23                  # xi = 1


def subwindows(ff, lipid, data=None):
    """``[(b_ps, e_ps) or None, ...]`` for parts 1..5, from the wham headers."""
    out = []
    for part in range(1, N_PARTS + 1):
        path = (data or DATA) / f"{ff}_{lipid}_part{part}_profile.xvg"
        if not path.exists():
            out.append(None)
            continue
        header = "".join(l for l in path.open() if l.startswith("#"))
        b = re.search(r"\s-b\s+([\d.]+)", header)
        e = re.search(r"\s-e\s+([\d.]+)", header)
        if not (b and e):
            raise ValueError(f"{path}: no -b/-e in the gmx wham command line")
        out.append((float(b.group(1)), float(e.group(1))))
    return out


def part_barrier(ff, lipid, part, data=None):
    """Basin-to-pore-side-peak barrier of partial profile ``part`` (1-based),
    the definition of ``draw_pore_energy.barrier``; None if missing."""
    path = (data or DATA) / f"{ff}_{lipid}_part{part}_profile.xvg"
    if not path.exists():
        return None
    _, y = load_xvg(path)
    y = y - y.min()
    return float(y[int(np.argmin(y)):].max())


def slice_mean(npz, b, e, key=COUNT_KEY):
    """``(mean count, mean xi or nan, n frames)`` over b <= t <= e (ps)."""
    t = npz["time_ps"]
    m = (t >= b) & (t <= e)
    n = int(m.sum())
    if n == 0:
        return np.nan, np.nan, 0
    xi = float(npz["xi"][m].mean()) if "xi" in npz.files else np.nan
    return float(npz[key][m].mean()), xi, n


def w23_counts(ff, lipid, data=None):
    path = (data or DATA) / f"poredens_counts_{ff}_{lipid}.npz"
    return np.load(path) if path.exists() else None


def draw_water_vs_barrier(ax, part, key=COUNT_KEY, data=None, verbose=False):
    """Sub-window ``part`` (1-based): umbrella23 water count vs partial barrier."""
    for ff, label, colour in FORCEFIELDS:
        for lipid in LIPIDS:
            npz = w23_counts(ff, lipid, data)
            sw = subwindows(ff, lipid, data)[part - 1]
            g = part_barrier(ff, lipid, part, data)
            if npz is None or sw is None or g is None:
                continue
            w, _, n = slice_mean(npz, *sw, key=key)
            if n == 0:
                print(f"  ! {ff} {lipid} part {part}: no umbrella23 frames in "
                      f"{sw[0]/1000:.0f}-{sw[1]/1000:.0f} ns")
                continue
            ax.plot(w, g, ls="none", marker=LIPID_MARKERS[lipid], color=colour,
                    mec="black", mew=0.4, ms=5)
            if verbose:
                print(f"  part {part} {lipid:5s} {ff:8s} "
                      f"{sw[0]/1000:4.0f}-{sw[1]/1000:4.0f} ns  n={n:3d}  "
                      f"water {w:6.2f}  barrier {g:6.1f} kJ/mol")
    ax.set_xlabel(COUNT_LABELS.get(key, key))


def subwindow_title(part, data=None):
    """'0-10 ns (DMPC 0-20)' from the CHARMM headers (same slicing for all ff)."""
    spans = {}
    for lipid in LIPIDS:
        for ff, _, _ in FORCEFIELDS:
            sw = subwindows(ff, lipid, data)[part - 1]
            if sw is not None:
                spans[lipid] = "%g–%g" % (sw[0] / 1000, sw[1] / 1000)
                break
    common = spans.get("DOPC") or next(iter(spans.values()), "")
    odd = {l: s for l, s in spans.items() if s != common}
    title = f"{common} ns"
    if odd:
        title += " (" + ", ".join(f"{l} {s}" for l, s in odd.items()) + ")"
    return title


def window_counts(ff, lipid, windows_dir=None):
    """``{window: npz}`` of the per-window counts present for one system."""
    d = windows_dir or WINDOWS
    out = {}
    for w in range(N_WINDOWS):
        p = d / f"poredens_counts_{ff}_{lipid}_w{w:02d}.npz"
        if p.exists():
            out[w] = np.load(p)
    return out


def draw_water_vs_xi(ax, lipid, key=COUNT_KEY, data=None, windows_dir=None,
                     ms=2.0, lw=0.8, verbose=False):
    """Water count vs xi across the 24 windows, one curve per (ff, sub-window).

    Each point is one window's mean (xi, count) over the sub-window's frames;
    xi is the rerun's per-frame chain coordinate, so it is the sampled xi, not
    the restraint centre.  Curves join the windows in window order.
    """
    drawn = False
    for ff, _, _ in FORCEFIELDS:
        wins = window_counts(ff, lipid, windows_dir)
        if not wins:
            continue
        missing = sorted(set(range(N_WINDOWS)) - set(wins))
        if missing:
            print(f"  ! {ff} {lipid}: windows missing: {missing}")
        for k, sw in enumerate(subwindows(ff, lipid, data)):
            if sw is None:
                continue
            pts = [slice_mean(wins[w], *sw, key=key) for w in sorted(wins)]
            pts = np.array([(xi, c) for c, xi, n in pts if n > 0])
            if len(pts) == 0 or np.isnan(pts[:, 0]).all():
                print(f"  ! {ff} {lipid} part {k + 1}: no frames/xi in range")
                continue
            ax.plot(pts[:, 0], pts[:, 1], color=CMAPS[ff](SHADES[k]), lw=lw,
                    marker="o", ms=ms, mew=0)
            drawn = True
            if verbose:
                print(f"  {lipid:5s} {ff:8s} part {k + 1}: {len(pts)} windows, "
                      f"count at max xi {pts[np.argmax(pts[:, 0]), 1]:.1f}")
    ax.set_xlim(0.0, 1.05)
    ax.set_xlabel(r"$\xi$")
    ax.set_title(lipid)
    if not drawn:
        print(f"note: {lipid} has no per-window counts (blank panel)")
    return drawn


def subwindow_handles(ms=2.0):
    """Neutral grey key for the sub-window gradient (light = early)."""
    from matplotlib import colormaps
    from matplotlib.lines import Line2D
    grey = colormaps["Greys"]
    return [Line2D([0], [0], color=grey(0.25 + 0.7 * s), marker="o", ms=ms,
                   mew=0, label=f"sub-window {k + 1}")
            for k, s in enumerate(SHADES)]


def draw_subwindow_boxes(ax, lipid, window, key=COUNT_KEY, windows_dir=None,
                         verbose=False):
    """Per-frame ``key`` counts in umbrella ``window``, one group per sub-window.

    x is the sub-window number (1-5; its time span differs for DMPC, see
    :func:`subwindows`), with a box per force field in the palette colours:
    quartile box, whiskers at the 5th/95th percentiles, no fliers -- the box
    style of Figure 1b-c.  No significance brackets: consecutive frames are
    correlated, and this is a picture of the distribution, not a test.
    """
    drawn = False
    for fi, (ff, _, colour) in enumerate(FORCEFIELDS):
        p = (windows_dir or WINDOWS) / f"poredens_counts_{ff}_{lipid}_w{window:02d}.npz"
        if not p.exists():
            print(f"  ! {ff} {lipid}: no umbrella{window} counts")
            continue
        npz = np.load(p)
        t = npz["time_ps"]
        for k, sw in enumerate(subwindows(ff, lipid)):
            if sw is None:
                continue
            m = (t >= sw[0]) & (t <= sw[1])
            vals = npz[key][m]
            if vals.size == 0:
                continue
            bp = ax.boxplot([vals], positions=[k + 1 + OFFSETS[fi]],
                            widths=BOX_W, whis=WHIS, showfliers=False,
                            patch_artist=True, manage_ticks=False)
            for patch in bp["boxes"]:
                patch.set(facecolor=colour, edgecolor="black", linewidth=0.7)
            for part in ("whiskers", "caps"):
                for line in bp[part]:
                    line.set(color="black", linewidth=0.7)
            for line in bp["medians"]:
                line.set(color="black", linewidth=0.9)
            drawn = True
            if verbose:
                xi = (f"  <xi> {npz['xi'][m].mean():.3f}" if "xi" in npz.files
                      else "")
                print(f"  w{window} {lipid:5s} {ff:8s} part {k + 1}: "
                      f"n={vals.size:3d}  median {np.median(vals):5.1f}  "
                      f"mean {vals.mean():6.2f}{xi}")
    ax.set_xticks(range(1, N_PARTS + 1))
    ax.set_xlim(0.5, N_PARTS + 0.5)
    ax.set_xlabel("sub-window")
    ax.set_title(lipid)
    return drawn


def draw_water_vs_barrier_all(ax, key=COUNT_KEY, connect=True, ms=7,
                              data=None, verbose=False):
    """All five sub-windows on one axes: umbrella23 count vs partial barrier.

    Colour = the force field's light->dark ramp by sub-window (early light),
    marker = lipid.  ``connect`` joins each system's five points in time order
    with a thin line in the force field's base colour, so the drift of one
    system reads as a path.
    """
    for ff, _, colour in FORCEFIELDS:
        for lipid in LIPIDS:
            npz = w23_counts(ff, lipid, data)
            if npz is None:
                continue
            pts = []
            for k, sw in enumerate(subwindows(ff, lipid, data)):
                g = part_barrier(ff, lipid, k + 1, data)
                if sw is None or g is None:
                    continue
                w, _, n = slice_mean(npz, *sw, key=key)
                if n:
                    pts.append((k, w, g))
            if not pts:
                continue
            pts = np.array(pts)
            if connect:
                ax.plot(pts[:, 1], pts[:, 2], color=colour, lw=0.6, alpha=0.5,
                        zorder=1)
            for k, w, g in pts:
                ax.plot(w, g, ls="none", marker=LIPID_MARKERS[lipid],
                        color=CMAPS[ff](SHADES[int(k)]), mec="black", mew=0.4,
                        ms=ms, zorder=2)
            if verbose:
                print(f"  {lipid:5s} {ff:8s} count {pts[:, 1].min():5.1f}-"
                      f"{pts[:, 1].max():5.1f}  barrier {pts[:, 2].min():5.1f}-"
                      f"{pts[:, 2].max():5.1f}")
    ax.set_xlabel(COUNT_LABELS.get(key, key))
    ax.set_ylabel(r"pore barrier (kJ mol$^{-1}$)")
