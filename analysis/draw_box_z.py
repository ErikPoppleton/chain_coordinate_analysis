#!/usr/bin/env python3
r"""Painter for the SI box-z figure: :math:`\Delta Z(t)` of the AMBER DMPG
equilibrium runs and of the two double-box control simulations.

The AMBER DMPG bilayer keeps thickening for hundreds of ns (see the d_HC
trajectory in Figure 1), and the box follows it along z.  The two controls test
whether that is a finite-size/periodic artifact: one with the box doubled along
z, one doubled with fewer counterions.

``draw_box_z(ax)`` creates no figure and saves nothing, so the same painter
serves an exploratory plot and the composed SI figure.  Each curve is shifted to
:math:`\Delta Z = 0` at its own first frame and truncated to the first
``WINDOW_NS``.

**Data sources, in order of preference:**

1. ``data/{stem}.xvg`` -- a ``gmx energy`` export, *validated by its
   ``@ s0 legend``*: only a ``Box-Z`` file is used.  ``data/z{1,2,3}.xvg`` are
   currently exports of **Potential** (kJ/mol, ~-363000), not Box-Z, so they are
   rejected with a warning rather than plotted as if they were nm.
2. ``data/box_z.npz`` (``analysis/box_z.py``) -- the box z read straight from
   the ``traj/`` xtc box vectors, which is the same quantity and needs no
   ``.edr``.  This is what the three replicates currently plot.
"""

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

WINDOW_NS = 50.0                 # plot the first 50 ns of every run
BOX_Z_LEGEND = "box-z"           # the gmx energy term a usable .xvg must hold

# (xvg stem, legend label, npz fallback key or None)
# Colours come from matplotlib's default cycle, NOT ``palette``: every curve
# here is the same force field and the same lipid, so the project's
# force-field hues would say something untrue.
SERIES = [
    ("z1", "run1", "amber_DMPG1"),
    ("z2", "run2", "amber_DMPG2"),
    ("z3", "run3", "amber_DMPG3"),
    ("z_doublebox", r"2$\times$ box", None),
    ("z_doublebox_lessions", r"2$\times$ box + less ions", None),
]


def load_xvg(path):
    """(legend, t_ns, y) from a gmx .xvg; legend is '' if the file names none."""
    legend, t, y = "", [], []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith("@") and " legend " in line and line.startswith("@ s"):
                legend = line.split('"')[1] if '"' in line else ""
                continue
            if not line or line[0] in "#@&":
                continue
            cols = line.split()
            t.append(float(cols[0]) / 1000.0)          # ps -> ns
            y.append(float(cols[1]))
    return legend, np.array(t), np.array(y)


def _from_npz(key):
    p = DATA / "box_z.npz"
    if not p.exists():
        return None
    with np.load(p) as d:
        if f"{key}_t" not in d.files:
            return None
        return np.asarray(d[f"{key}_t"]), np.asarray(d[f"{key}_z"])


def series(stem, npz_key=None, verbose=False):
    """(t_ns, dZ_nm, source) for one series, or None if no usable data.

    A ``.xvg`` is only trusted when its own legend says ``Box-Z`` -- the guard
    that keeps the mis-exported Potential files out of a nm-scaled axis.
    """
    src, data = None, None
    p = DATA / f"{stem}.xvg"
    if p.exists():
        legend, t, y = load_xvg(p)
        if legend.strip().lower() == BOX_Z_LEGEND:
            data, src = (t, y), f"{p.name}"
        elif verbose:
            print(f"    note: {p.name} holds '{legend}', not Box-Z -- ignored")
    if data is None and npz_key:
        alt = _from_npz(npz_key)
        if alt is not None:
            data, src = alt, f"box_z.npz[{npz_key}]"
    if data is None:
        return None
    t, y = data
    m = t <= WINDOW_NS + 1e-9
    return t[m], y[m] - y[m][0], src


def draw_box_z(ax, verbose=False):
    """Delta-Z against time for the three AMBER DMPG runs + the two controls."""
    for stem, label, npz_key in SERIES:
        got = series(stem, npz_key, verbose=verbose)
        if got is None:
            print(f"    note: no usable data for {stem} (curve absent)")
            continue
        t, dz, src = got
        ax.plot(t, dz, label=label)
        if verbose:
            print(f"    {label:22s} {t.size:4d} pts, dZ -> {dz[-1]:+.2f} nm "
                  f"at {t[-1]:.0f} ns   [{src}]")
    ax.set_xlim(0.0, WINDOW_NS)
    ax.set_xlabel("time (ns)")
    ax.set_ylabel(r"$\Delta Z$ (nm)")
    # Fully opaque (but unframed) legend: the curves climb through the
    # upper-left corner, so a transparent one let them cross the label text and
    # a translucent one left them ghosting through it.
    ax.legend(loc="upper left", frameon=True, facecolor="white",
              edgecolor="none", framealpha=1.0, handlelength=1.2,
              borderaxespad=0.2, labelspacing=0.25)
