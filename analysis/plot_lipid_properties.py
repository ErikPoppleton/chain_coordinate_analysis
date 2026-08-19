#!/usr/bin/env python3
"""Plot the equilibrium lipid properties computed by ``lipid_properties.py``.

Reads the ``lipprops_*.npz`` files from ``data/`` and writes one figure per
property to ``plots/``.  Each figure is a 1x3 row of panels (DOPC, DMPC, DMPG);
within a panel the three force fields (CHARMM, AMBER, SLipids) are overlaid in
the shared colour scheme from ``palette.py``.

  * area_per_lipid, memb_thickness, registration
        -> overlaid histograms (density-normalised).
  * S_CD order parameter (g_lomepro)
        -> per-carbon profile, carbon number down an inverted y-axis.
  * lateral_diffusion
        -> MSD vs lag time (mean over replicates, shaded spread) plus an inset
           bar graph of the fitted diffusion coefficient.

Requires only numpy + matplotlib; run it anywhere (paths derive from __file__).
"""

import json
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from palette import FORCEFIELDS
from lipiddefs import LIPIDS

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
PLOTS = ROOT / "plots"


# (npz stem, figure title, axis label, scale, per_lipid).  The scale rescales
# samples for display.  per_lipid marks a per-lipid property (area): its npz
# arrays are 2-D (n_frames, n_lipids), so the trajectory plot shows a per-frame
# mean +/- SD over lipids; the whole-membrane properties are one value per frame.
HIST_PROPS = [
    ("area_per_lipid", "Area per lipid", r"APL ($\mathrm{\AA}^2$)", 1.0, True),
    ("memb_thickness", "Membrane thickness", r"$d_\mathrm{HH}$ (nm)", 0.1, False),
    ("registration", "Interleaflet registration", "registration coefficient", 1.0, False),
]


def load(stem):
    path = DATA / f"lipprops_{stem}.npz"
    if not path.exists():
        return None
    return np.load(path)


def plot_histograms(stem, title, xlabel, scale):
    data = load(stem)
    if data is None:
        print(f"skip {stem}: {DATA / f'lipprops_{stem}.npz'} not found")
        return

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharex=True)
    any_data = False
    for ax, lipid in zip(axes, LIPIDS):
        # Common bins per panel so the three force fields are comparable.
        pooled = [np.ravel(data[f"{ff}_{lipid}"]) * scale
                  for ff, _, _ in FORCEFIELDS if f"{ff}_{lipid}" in data.files]
        if pooled:
            allv = np.concatenate(pooled)
            bins = np.linspace(np.nanmin(allv), np.nanmax(allv), 41)
        for ff, label, color in FORCEFIELDS:
            key = f"{ff}_{lipid}"
            if key not in data.files:
                continue
            any_data = True
            vals = np.ravel(data[key]) * scale
            ax.hist(vals, bins=bins, density=True, color=color,
                    histtype="stepfilled", alpha=0.35, label=label)
            ax.hist(vals, bins=bins, density=True, color=color,
                    histtype="step", lw=1.5)
            # Dashed line at this force field's mean.
            ax.axvline(np.nanmean(vals), color=color, ls="--", lw=1.3)
        ax.set_title(lipid)
        ax.set_xlabel(xlabel)
        ax.tick_params(axis="y", right=True)
    axes[0].set_ylabel("density")
    if not any_data:
        plt.close(fig)
        print(f"skip {stem}: no systems present in npz")
        return
    # One legend for the row (force fields are identical across panels).
    handles, labels = axes[0].get_legend_handles_labels()
    if not handles:
        for ax in axes:
            h, l = ax.get_legend_handles_labels()
            if h:
                handles, labels = h, l
                break
    fig.legend(handles, labels, loc="upper right", ncol=len(FORCEFIELDS))
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = PLOTS / f"lipprop_{stem}.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"wrote {out}")


def plot_core_thickness_histograms():
    """d_HC companion to the d_HH (memb_thickness) histogram.

    Reads the per-frame **lipyphilic** hydrophobic-core thickness from
    ``core_thickness.npz`` (the only d_HC metric stored per frame -- the density
    metric is 3 scalars/system and can't form a distribution), pools the 3
    replicates (250 frames each, last 50 ns) and draws the same 1x3 overlaid
    density histogram as ``plot_histograms`` -> ``plots/lipprop_core_thickness.png``.
    Units are nm (the standard core-thickness unit; d_HH is in A)."""
    path = DATA / "core_thickness.npz"
    if not path.exists():
        print(f"skip core_thickness: {path} not found")
        return
    data = np.load(path)
    reps = (1, 2, 3)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharex=True)
    any_data = False
    for ax, lipid in zip(axes, LIPIDS):
        pooled = {}
        for ff, _, _ in FORCEFIELDS:
            arrs = [data[k] for r in reps
                    if (k := f"{ff}_{lipid}_lipy_rep{r}") in data.files]
            if arrs:
                pooled[ff] = np.concatenate([np.ravel(a) for a in arrs])
        if pooled:
            allv = np.concatenate(list(pooled.values()))
            # 27 bins = 40/1.5, i.e. 1.5x wider bins than the shared 40-bin
            # default, to smooth the otherwise-jagged core-thickness histograms.
            bins = np.linspace(np.nanmin(allv), np.nanmax(allv), 28)
        for ff, label, color in FORCEFIELDS:
            if ff not in pooled:
                continue
            any_data = True
            vals = pooled[ff]
            ax.hist(vals, bins=bins, density=True, color=color,
                    histtype="stepfilled", alpha=0.35, label=label)
            ax.hist(vals, bins=bins, density=True, color=color,
                    histtype="step", lw=1.5)
            ax.axvline(np.nanmean(vals), color=color, ls="--", lw=1.3)
        ax.set_title(lipid)
        ax.set_xlabel("$d_\\mathrm{HC}$ (nm)")
        ax.tick_params(axis="y", right=True)
    axes[0].set_ylabel("density")
    if not any_data:
        plt.close(fig)
        print("skip core_thickness: no systems present in npz")
        return
    handles, labels = axes[0].get_legend_handles_labels()
    if not handles:
        for ax in axes:
            h, l = ax.get_legend_handles_labels()
            if h:
                handles, labels = h, l
                break
    fig.legend(handles, labels, loc="upper right", ncol=len(FORCEFIELDS))
    fig.suptitle("Hydrophobic-core thickness $d_\\mathrm{HC}$ (lipyphilic)")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = PLOTS / "lipprop_core_thickness.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"wrote {out}")


# Candidate inset corners as [x0, y0, w, h] in axes fractions.  The top-left box
# keeps its bottom at 0.57 (so its overlap with the curves -- hence the chosen
# corner -- is unchanged) but stops at 0.86 rather than ~0.95, leaving a fixed
# gap for its title below the panel's top spine, decoupled from subplots_adjust.
_INSET_TL = [0.09, 0.57, 0.40, 0.29]  # top-left  (top at 0.86)
_INSET_BR = [0.57, 0.10, 0.40, 0.38]  # bottom-right


def _corner_overlap(curves, xlim, ylim, rect):
    """Number of MSD-curve points that fall inside ``rect`` (axes fractions),
    projecting the curves into log-log fraction coordinates to match the axes.
    Used to pick the inset corner the data intrudes into least."""
    x0, y0, w, h = rect
    (lx0, lx1), (ly0, ly1) = np.log10(xlim), np.log10(ylim)
    n = 0
    for x, y in curves:
        fx = (np.log10(x) - lx0) / (lx1 - lx0)
        fy = (np.log10(y) - ly0) / (ly1 - ly0)
        n += int(np.sum((fx >= x0) & (fx <= x0 + w) & (fy >= y0) & (fy <= y0 + h)))
    return n


def plot_lateral_diffusion():
    data = load("lateral_diffusion")
    if data is None:
        print("skip lateral_diffusion: npz not found")
        return

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharex=True, sharey=True)
    any_data = False
    panels = {}  # lipid -> (list of (lag, mean) curves, bar-graph arrays)
    for ax, lipid in zip(axes, LIPIDS):
        # Fitted D (µm^2/s), one bar per force field, drawn as an inset.
        curves = []
        bar_x, bar_h, bar_e, bar_c = [], [], [], []
        for i, (ff, label, color) in enumerate(FORCEFIELDS):
            lag_key = f"{ff}_{lipid}_lagtimes"
            msd_key = f"{ff}_{lipid}_msd"
            if lag_key not in data.files:
                continue
            any_data = True
            lag = data[lag_key]
            msd = data[msd_key]  # (n_reps, n_lag)
            mean = msd.mean(axis=0)
            # Drop the lag = 0 / MSD = 0 point, undefined on a log-log plot.
            nz = lag > 0
            lag, mean, msd = lag[nz], mean[nz], msd[:, nz]
            ax.plot(lag, mean, color=color, lw=1.8, label=label)
            curves.append((lag, mean))
            if msd.shape[0] > 1:
                sd = msd.std(axis=0)
                # Clip the lower band to stay positive for the log y-axis.
                lo = np.clip(mean - sd, 1e-6, None)
                ax.fill_between(lag, lo, mean + sd, color=color, alpha=0.2, lw=0)

            D = data[f"{ff}_{lipid}_D"] * 1e8  # cm^2/s -> µm^2/s
            bar_x.append(i)
            bar_h.append(np.mean(D))
            bar_e.append(np.std(D) if D.size > 1 else 0.0)
            bar_c.append(color)
        ax.set_title(lipid)
        ax.set_xlabel("lag time (ns)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.tick_params(axis="y", right=True)
        panels[lipid] = (curves, (bar_x, bar_h, bar_e, bar_c))
    axes[0].set_ylabel(r"MSD ($\mathrm{nm}^2$)")
    if not any_data:
        plt.close(fig)
        print("skip lateral_diffusion: no systems present in npz")
        return

    # Now that the (shared) log axes have autoscaled, place each panel's inset in
    # whichever corner -- top-left or bottom-right -- the curve overlaps least.
    xlim, ylim = axes[0].get_xlim(), axes[0].get_ylim()
    for ax, lipid in zip(axes, LIPIDS):
        curves, (bar_x, bar_h, bar_e, bar_c) = panels[lipid]
        if not bar_x:
            continue
        rect = (_INSET_TL
                if _corner_overlap(curves, xlim, ylim, _INSET_TL)
                <= _corner_overlap(curves, xlim, ylim, _INSET_BR)
                else _INSET_BR)
        # Transparent background, no top/right spines, so it doesn't box off data.
        inset = ax.inset_axes(rect)
        inset.patch.set_alpha(0.0)
        inset.spines[["top", "right"]].set_visible(False)
        inset.bar(bar_x, bar_h, yerr=bar_e, color=bar_c, capsize=2, width=0.7)
        inset.set_xticks(bar_x)
        inset.set_xticklabels([FORCEFIELDS[i][0].upper()[:1] for i in bar_x],
                              fontsize=7)
        inset.tick_params(labelsize=7)
        inset.set_title(r"$D$ ($\mathrm{\mu m^2/s}$)", fontsize=7)
    handles, labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        if h:
            handles, labels = h, l
            break
    fig.legend(handles, labels, loc="upper right", ncol=len(FORCEFIELDS))
    fig.suptitle("Lateral diffusion")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    # Grow the panels upward into the empty band below the suptitle: the inset's
    # title clears the top spine by a margin that scales with the panel height, so
    # taller panels tuck it under the spine instead of overlapping it.
    fig.subplots_adjust(top=0.86)
    out = PLOTS / "lipprop_lateral_diffusion.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"wrote {out}")


def plot_scd():
    """S_CD (deuterium) order-parameter profiles from g_lomepro's ``-order`` output.

    Carbon number runs down an inverted y-axis (C2/headgroup end at top, tail end
    at bottom), order parameter on x, the three force fields overlaid per lipid
    panel.

    Reads ``data/{ff}_{lipid}_{rep}_scd_sn{1,2}.dat`` -- g_lomepro's
    ``_AVG_sn{1,2}.dat`` output copied per replicate, columns
    ``carbon_number, mean(-S_CD), sd``.  To match the per-replicate error scheme
    used everywhere else, the two tails are
    averaged *within* each replicate to give one profile per replicate; the line
    is the mean over replicates and the band the **SD across replicates**.
    (Files with no replicate token are treated as a single replicate.)"""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharex=True)
    any_data = False
    for ax, lipid in zip(axes, LIPIDS):
        cmax = 0
        for ff, label, color in FORCEFIELDS:
            # Group chain files by replicate: {rep: {sn: profile}}.  The rep token
            # is optional so pre-per-rep files (…_scd_sn1.dat) count as one rep.
            pat = re.compile(rf"^{re.escape(ff)}_{re.escape(lipid)}_"
                             rf"(?:(\d+)_)?scd_(sn[12])\.dat$")
            by_rep, carbons = {}, None
            for f in sorted(DATA.glob(f"{ff}_{lipid}_*scd_sn*.dat")):
                m = pat.match(f.name)
                if not m:
                    continue
                arr = np.loadtxt(f, comments=("#", "@"))
                if arr.ndim != 2 or arr.shape[0] == 0:
                    continue
                carbons = arr[:, 0].astype(int)
                rep = int(m.group(1)) if m.group(1) else 0
                by_rep.setdefault(rep, {})[m.group(2)] = arr[:, 1]  # col 1 = mean -S_CD
            if not by_rep:
                continue
            any_data = True
            # Per-replicate profile = mean over that rep's tails, then aggregate reps.
            profiles = np.array([np.mean(list(tails.values()), axis=0)
                                 for _, tails in sorted(by_rep.items())])
            mean = profiles.mean(axis=0)
            cmax = max(cmax, int(carbons.max()))
            ax.plot(mean, carbons, color=color, marker="o", ms=3, lw=1.5, label=label)
            if len(profiles) > 1:
                sd = profiles.std(axis=0)
                ax.fill_betweenx(carbons, mean - sd, mean + sd, color=color,
                                 alpha=0.2, lw=0)
        ax.set_title(lipid)
        ax.set_xlabel(r"$-S_\mathrm{CD}$")
        if cmax:
            ax.set_ylim(cmax + 1, 0.5)  # carbon 2 at top, terminal carbon at bottom
        ax.tick_params(axis="y", right=True)
    axes[0].set_ylabel("tail carbon number")
    if not any_data:
        plt.close(fig)
        print("skip scd: no data/*_scd_sn*.dat files found")
        return
    handles, labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        if h:
            handles, labels = h, l
            break
    fig.legend(handles, labels, loc="upper right", ncol=len(FORCEFIELDS))
    fig.suptitle(r"Acyl chain order parameter ($S_\mathrm{CD}$)")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = PLOTS / "lipprop_scd.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"wrote {out}")


# npz stem -> stats.py observable key, for looking up the pymbar t0.
STEM_TO_STATS_OBS = {"area_per_lipid": "apl", "memb_thickness": "thickness",
                     "registration": "registration"}
NS_PER_FRAME = 0.2                     # 200 ps/frame


def _t0_rep1(obs):
    """{system: t0 (frames) for replicate 1} from stats.py's equilibrium-
    timeseries JSON (pymbar detect_equilibration on the full 500 ns), or {} if
    the JSON / observable is absent."""
    if obs is None:
        return {}
    p = DATA / "equilibrium_timeseries_stats.json"
    if not p.exists():
        return {}
    recs_by_sys = json.loads(p.read_text())["per_replicate"].get(obs, {})
    out = {}
    for sysn, recs in recs_by_sys.items():
        for r in recs:
            if int(r["rep"]) == 1:
                out[sysn] = int(r["t0"])
    return out


def plot_trajectories(stem, title, ylabel, scale, per_lipid):
    """Per-frame trajectory of ``stem`` for replicate 1, as a 3x3 grid: lipids
    across columns (DOPC/DMPC/DMPG), force fields down rows (CHARMM/AMBER/
    SLipids).  A per-lipid property (area) shows the per-frame mean +/- SD over
    lipids; a whole-membrane property (thickness, registration) shows its single
    per-frame value.  All nine panels share the y-axis, so the parameter is
    directly comparable across force field and lipid.

    Each panel marks the pymbar equilibration time ``t0`` for that (force field,
    lipid) replicate-1 run (dashed line + shaded discarded burn-in), read from
    ``stats.py``'s JSON -- so the plot shows exactly what was dropped before the
    reported means/CIs."""
    data = load(stem)
    if data is None:
        print(f"skip {stem} trajectory: {DATA / f'lipprops_{stem}.npz'} not found")
        return
    t0map = _t0_rep1(STEM_TO_STATS_OBS.get(stem))

    fig, axes = plt.subplots(len(FORCEFIELDS), len(LIPIDS), figsize=(13, 9),
                             sharex=True, sharey=True)
    any_data = False
    for r, (ff, ff_label, color) in enumerate(FORCEFIELDS):
        for c, lipid in enumerate(LIPIDS):
            ax = axes[r][c]
            if r == 0:
                ax.set_title(lipid)
            if c == 0:
                ax.set_ylabel(ff_label)
            key = f"{ff}_{lipid}_eq1"
            if key not in data.files:
                continue
            arr = data[key] * scale
            if per_lipid:
                if arr.ndim != 2:
                    continue  # old ravelled data; needs a fresh analysis run
                m, sd = arr.mean(axis=1), arr.std(axis=1)
                x = np.arange(arr.shape[0])
                ax.plot(x, m, color=color, lw=1.0)
                ax.fill_between(x, m - sd, m + sd, color=color, alpha=0.25, lw=0)
            else:
                y = np.ravel(arr)
                ax.plot(np.arange(y.shape[0]), y, color=color, lw=1.0)
            any_data = True
            # mark the pymbar equilibration cutoff for this rep-1 run.
            t0 = t0map.get(f"{ff}_{lipid}")
            if t0 is not None:
                nfr = arr.shape[0]
                if t0 > 0:
                    ax.axvspan(0, t0, color="0.5", alpha=0.15, lw=0)
                ax.axvline(t0, color="0.25", ls="--", lw=0.9)
                right = t0 > 0.6 * nfr
                ax.text(t0, 0.97, (f"$t_0$={t0 * NS_PER_FRAME:.0f} ns "
                                   if right else f" $t_0$={t0 * NS_PER_FRAME:.0f} ns"),
                        transform=ax.get_xaxis_transform(),
                        ha="right" if right else "left", va="top",
                        fontsize=7, color="0.25")
    if not any_data:
        plt.close(fig)
        extra = (" (per-lipid arrays must be 2-D -- rerun lipid_properties.py)"
                 if per_lipid else "")
        print(f"skip {stem} trajectory: no replicate-1 data{extra}")
        return
    for ax in axes[-1]:
        ax.set_xlabel("frame")
    fig.supylabel(ylabel)
    fig.suptitle(f"{title} — per-frame trajectory (replicate 1; "
                 f"dashed = pymbar $t_0$, shaded = discarded burn-in)")
    fig.tight_layout(rect=(0.02, 0, 1, 0.97))
    out = PLOTS / f"lipprop_traj_{stem}.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"wrote {out}")


def main():
    PLOTS.mkdir(parents=True, exist_ok=True)
    for stem, title, xlabel, scale, per_lipid in HIST_PROPS:
        plot_histograms(stem, title, xlabel, scale)
        plot_trajectories(stem, title, xlabel, scale, per_lipid)
    plot_core_thickness_histograms()
    plot_scd()
    plot_lateral_diffusion()


if __name__ == "__main__":
    main()
