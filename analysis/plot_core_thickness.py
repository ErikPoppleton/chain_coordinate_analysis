#!/usr/bin/env python3
"""Compare the two hydrophobic-core-thickness (d_HC) metrics per force field /
lipid, with the head-to-head thickness d_HH for reference, and draw the tail
mass-density profiles that define the density metric.

Metrics (both over the last 50 ns; from ``core_thickness.py`` ->
``data/core_thickness.npz``):
  * **lipyphilic** -- ``MembThickness`` on the first aliphatic tail carbon;
  * **density @ 500 kg/m^3** -- full width of the tail mass-density profile at
    500 kg/m^3.  Preferred source is the cluster ``gmx density`` output
    ``data/{ff}_{lipid}_{rep}_taildens.xvg`` (the ``tails`` column); the local
    MDAnalysis density in the npz is the fallback and cross-check.

d_HH (P--P) is read from ``data/lipprops_memb_thickness.npz`` (A -> nm).

Bands / error bars are the **SD across the 3 replicates** (project convention).
Writes ``plots/core_thickness.png`` (grouped bars + d_HH) and
``plots/core_thickness_profiles.png`` (the tail density profiles), and prints a
table incl. the gmx-vs-MDAnalysis agreement.

    python3 analysis/plot_core_thickness.py
"""

import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from palette import FORCEFIELDS
from lipiddefs import LIPIDS, REPS

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
PLOTS = ROOT / "plots"

DENS_THRESHOLD = 500.0


def load_taildens(path):
    """(z, tail-density) from a gmx density .xvg, picking the ``tails`` column by
    its ``@ sN legend`` (falling back to the first density column)."""
    legends, z, cols = {}, [], []
    with open(path) as fh:
        for line in fh:
            s = line.strip()
            m = re.match(r'@\s*s(\d+)\s+legend\s+"(.*)"', s)
            if m:
                legends[int(m.group(1))] = m.group(2)
                continue
            if not s or s[0] in "#@&":
                continue
            p = s.split()
            z.append(float(p[0]))
            cols.append([float(x) for x in p[1:]])
    z, cols = np.array(z), np.array(cols)
    ti = next((i for i, n in legends.items() if n.lower() == "tails"), 0)
    return z, cols[:, ti]


def width_at(z, d, thr=DENS_THRESHOLD):
    """Full width of profile ``d(z)`` at level ``thr`` (outermost crossings, nm)."""
    above = np.where(d >= thr)[0]
    if above.size == 0:
        return np.nan
    lo, hi = above[0], above[-1]
    zl = (z[0] if lo == 0
          else np.interp(thr, [d[lo - 1], d[lo]], [z[lo - 1], z[lo]]))
    zr = (z[-1] if hi == len(z) - 1
          else np.interp(thr, [d[hi + 1], d[hi]], [z[hi + 1], z[hi]]))
    return float(zr - zl)


def summary(vals):
    v = [x for x in vals if x is not None and np.isfinite(x)]
    return (float(np.mean(v)), float(np.std(v))) if v else None


def collect():
    ct = np.load(DATA / "core_thickness.npz") if (
        DATA / "core_thickness.npz").exists() else None
    hh = np.load(DATA / "lipprops_memb_thickness.npz") if (
        DATA / "lipprops_memb_thickness.npz").exists() else None

    rows, profiles = {}, {}
    for ff, _, _ in FORCEFIELDS:
        for lipid in LIPIDS:
            cell = {"lipy": None, "dens_gmx": None, "dens_mda": None, "hh": None}
            if ct is not None:
                cell["lipy"] = summary(
                    [float(ct[k].mean()) for r in REPS
                     if (k := f"{ff}_{lipid}_lipy_rep{r}") in ct.files])
                dw = f"{ff}_{lipid}_denswidth"
                if dw in ct.files:
                    cell["dens_mda"] = summary(list(ct[dw]))
                pz, pf = f"{ff}_{lipid}_densprof_z", f"{ff}_{lipid}_densprof"
                if pz in ct.files:
                    profiles[(ff, lipid)] = (ct[pz], ct[pf])
            gmx = []
            for r in REPS:
                x = DATA / f"{ff}_{lipid}_{r}_taildens.xvg"
                if x.exists():
                    gmx.append(width_at(*load_taildens(x)))
            cell["dens_gmx"] = summary(gmx)
            if hh is not None:
                cell["hh"] = summary(
                    [float(hh[k].mean()) / 10.0 for r in REPS
                     if (k := f"{ff}_{lipid}_eq{r}") in hh.files])
            rows[(ff, lipid)] = cell
    return rows, profiles


def fmt(c):
    return "--" if c is None else f"{c[0]:.3f}±{c[1]:.3f}"


def plot_bars(rows):
    """Grouped bars: lipyphilic vs density d_HC per force field, d_HH dash."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3), sharey=True)
    w = 0.38
    used_gmx = False
    for ax, lipid in zip(axes, LIPIDS):
        for i, (ff, label, color) in enumerate(FORCEFIELDS):
            c = rows[(ff, lipid)]
            if c["lipy"] is not None:
                m, sd = c["lipy"]
                ax.bar(i - w / 2, m, w, yerr=sd, capsize=3, color=color,
                       edgecolor="black", lw=0.5)
            # density metric: prefer gmx, fall back to local MDAnalysis.
            dens = c["dens_gmx"] or c["dens_mda"]
            used_gmx = used_gmx or c["dens_gmx"] is not None
            if dens is not None:
                m, sd = dens
                ax.bar(i + w / 2, m, w, yerr=sd, capsize=3, color=color,
                       edgecolor="black", lw=0.5, hatch="///", alpha=0.55)
            if c["hh"] is not None:
                ax.plot([i - w, i + w], [c["hh"][0]] * 2, color="black", lw=1.6,
                        solid_capstyle="butt", zorder=5)
        ax.set_title(lipid)
        ax.set_xticks(range(len(FORCEFIELDS)))
        ax.set_xticklabels([l for _, l, _ in FORCEFIELDS])
        ax.grid(axis="y", lw=0.3, alpha=0.4)
        ax.set_ylim(0, None)
    axes[0].set_ylabel("thickness (nm)")
    dens_lbl = ("$d_{HC}$ — density @ 500 kg m$^{-3}$"
                + ("" if used_gmx else " (MDAnalysis)"))
    handles = [
        Patch(facecolor="0.6", edgecolor="black", label="$d_{HC}$ — lipyphilic (1st tail C)"),
        Patch(facecolor="0.6", edgecolor="black", hatch="///", alpha=0.55, label=dens_lbl),
        Line2D([0], [0], color="black", lw=1.6, label="$d_{HH}$ (P–P)"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("Hydrophobic-core thickness $d_{HC}$: two metrics", y=1.07)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out = PLOTS / "core_thickness.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def plot_profiles(profiles):
    """Tail mass-density profiles per lipid, force fields overlaid, 500 line."""
    if not profiles:
        return
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0), sharey=True, sharex=True)
    for ax, lipid in zip(axes, LIPIDS):
        for ff, label, color in FORCEFIELDS:
            if (ff, lipid) not in profiles:
                continue
            z, d = profiles[(ff, lipid)]
            ax.plot(z, d, color=color, lw=1.5, label=label)
        ax.axhline(DENS_THRESHOLD, color="0.4", ls=":", lw=1)
        ax.set_title(lipid)
        ax.set_xlabel("z from bilayer centre (nm)")
        ax.grid(lw=0.3, alpha=0.4)
    axes[0].set_ylabel(r"tail mass density (kg m$^{-3}$)")
    axes[-1].legend(frameon=False, fontsize=8)
    fig.suptitle("Hydrophobic-tail mass-density profiles (d_HC = width @ 500 kg m$^{-3}$)")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = PLOTS / "core_thickness_profiles.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"wrote {out}")


def main():
    rows, profiles = collect()
    print(f"{'system':16s} {'lipy':>12} {'dens gmx':>12} {'dens MDA':>12} {'d_HH':>12}")
    for ff, label, _ in FORCEFIELDS:
        for lipid in LIPIDS:
            c = rows[(ff, lipid)]
            print(f"{label + ' ' + lipid:16s} {fmt(c['lipy']):>12} "
                  f"{fmt(c['dens_gmx']):>12} {fmt(c['dens_mda']):>12} {fmt(c['hh']):>12}")
    # gmx-vs-MDAnalysis agreement on the density metric.
    diffs = [c["dens_gmx"][0] - c["dens_mda"][0]
             for c in rows.values() if c["dens_gmx"] and c["dens_mda"]]
    if diffs:
        print(f"\ndensity metric gmx vs MDAnalysis: mean |Δ| = "
              f"{np.mean(np.abs(diffs)):.3f} nm, max |Δ| = {np.max(np.abs(diffs)):.3f} nm")

    PLOTS.mkdir(parents=True, exist_ok=True)
    plot_bars(rows)
    plot_profiles(profiles)


if __name__ == "__main__":
    main()
