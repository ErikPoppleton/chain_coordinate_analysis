"""Painters for the SI lateral-diffusion figure (figures/figSI_diffusion.py).

``draw_local_D(ax, lipid)``  local diffusion coefficient D(t) = (1/4) dMSD/dt
                             against lag time, every replicate of each force
                             field (thin) and their mean (thick), with
                             Venable et al.'s fit window 10 ns < t < (T-10)/2
                             shaded and each force field's fitted D drawn
                             across it (dashed).  From data/diffusion/
                             msd_curves.npz + data/diffusion.json
                             (analysis/diffusion.py).
``draw_D_vs_L(ax, lipid)``   the periodic Saffman-Delbrueck prediction of the
                             simulated D against box edge L (water layer H
                             held at the simulated value), median and 95 %
                             band over the posterior draws in
                             data/diffusion_psd_posterior.npz
                             (analysis/diffusion_psd.py); the simulated point
                             (+/- SEM) at its own L, and D_inf as a dashed
                             asymptote line.

Pure painters: no figure is created or saved here.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from palette import FORCEFIELDS                             # noqa: E402
import diffusion as dfn                                     # noqa: E402
import diffusion_psd as psd                                 # noqa: E402

DATA = Path(__file__).resolve().parent.parent / "data"
D_LABEL = r"$D$ ($\mathrm{\mu m^2\,s^{-1}}$)"
T_MAX_NS = 450.0              # the local slope is pure noise beyond this lag
L_GRID_NM = np.geomspace(4.0, 300.0, 22)
L_TICKS_NM = [5, 10, 20, 50, 100, 200]


def _json(name):
    return json.loads((DATA / name).read_text())


def draw_local_D(ax, lipid, width_ns=20.0, verbose=False):
    curves = np.load(DATA / "diffusion" / "msd_curves.npz")
    res = _json("diffusion.json")
    win = None
    for ff, label, colour in FORCEFIELDS:
        key = f"{ff}_{lipid}"
        if key not in res:
            continue
        locs = []
        for rep in res[key]["reps"]:
            t = curves[f"{key}_{rep}_t"]
            msd = curves[f"{key}_{rep}_bilayer"]
            tc, ds = dfn.running_slope(t, msd, width_ns)
            m = tc <= T_MAX_NS
            ax.plot(tc[m], ds[m], color=colour, lw=0.4, alpha=0.45)
            locs.append(ds[m])
            win = res[key]["reps"][rep]["window_ns"]
        ax.plot(tc[m], np.mean(locs, axis=0), color=colour, lw=1.0, label=label)
        D = res[key]["D_bilayer"]["mean"]
        ax.plot(win, [D, D], color=colour, lw=0.8, ls="--")
        if verbose:
            print(f"  {lipid} {ff:8s} fitted D = {D:.2f} um^2/s over "
                  f"{win[0]:.0f}-{win[1]:.0f} ns")
    if win is not None:
        ax.axvspan(*win, color="0.92", zorder=0, lw=0)
    ax.set_xlim(0, T_MAX_NS)
    ax.set_xlabel("lag time (ns)")
    ax.set_ylabel(r"local $D(t)$ ($\mathrm{\mu m^2\,s^{-1}}$)")


def draw_D_vs_L(ax, lipid, verbose=False):
    res = _json("diffusion_psd.json")
    post = np.load(DATA / "diffusion_psd_posterior.npz")
    for ff, label, colour in FORCEFIELDS:
        key = f"{ff}_{lipid}"
        if key not in res:
            continue
        r = res[key]
        th = [post[f"{key}_{n}"] for n in ("eta_m", "eta_f", "R", "b")]
        H = r["H_nm"] * 1e-9
        D = np.array([psd.d_pbc(*[x[:200] for x in th], L * 1e-9, H) * 1e12
                      for L in L_GRID_NM])
        lo, med, hi = np.percentile(D, [2.5, 50, 97.5], axis=1)
        ax.fill_between(L_GRID_NM, lo, hi, color=colour, alpha=0.18, lw=0)
        ax.plot(L_GRID_NM, med, color=colour, lw=1.0, label=label)
        ax.axhline(r["D_inf"]["median"], color=colour, lw=0.8, ls="--")
        ax.errorbar([r["L_nm"]], [r["D_sim"]], yerr=[r["D_sim_sem"]], fmt="o",
                    ms=3.5, color=colour, mec="black", mew=0.5, ecolor="black",
                    elinewidth=0.7, capsize=1.5, zorder=5)
        if verbose:
            print(f"  {lipid} {ff:8s} L = {r['L_nm']:.2f} nm  D_sim "
                  f"{r['D_sim']:.1f}  D(20 nm) {np.interp(20, L_GRID_NM, med):.1f}"
                  f"  D(100 nm) {np.interp(100, L_GRID_NM, med):.1f}  D(300 nm) {med[-1]:.1f}  D_inf {r['D_inf']['median']:.1f}")
    from matplotlib.ticker import FixedLocator, NullFormatter, ScalarFormatter
    ax.set_xscale("log")
    ax.set_xlim(L_GRID_NM[0], L_GRID_NM[-1])
    ax.xaxis.set_major_locator(FixedLocator(L_TICKS_NM))
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())
    # no set_ylim here: the caller sets a shared range once every panel is drawn
    ax.set_xlabel(r"box edge $L$ (nm)")
    ax.set_ylabel(D_LABEL)
