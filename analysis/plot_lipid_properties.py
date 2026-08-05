#!/usr/bin/env python3
"""Plot the equilibrium lipid properties computed by ``lipid_properties.py``.

Reads the ``lipprops_*.npz`` files from ``data/`` and writes one figure per
property to ``plots/``.  Each figure is a 1x3 row of panels (DOPC, DMPC, DMPG);
within a panel the three force fields (CHARMM, AMBER, SLipids) are overlaid in
the shared colour scheme from ``palette.py``.

  * area_per_lipid, memb_thickness, registration, order_parameter
        -> overlaid histograms (density-normalised).
  * lateral_diffusion
        -> MSD vs lag time (mean over replicates, shaded spread) plus an inset
           bar graph of the fitted diffusion coefficient.

Requires only numpy + matplotlib; run it anywhere (paths derive from __file__).
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from palette import FORCEFIELDS

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
PLOTS = ROOT / "plots"

LIPIDS = ["DOPC", "DMPC", "DMPG"]

# (npz stem, figure title, x-axis label, sample scale factor).  The scale
# rescales samples for display (e.g. diffusion coefficients ~1e-7 cm^2/s).
HIST_PROPS = [
    ("area_per_lipid", "Area per lipid", r"APL ($\mathrm{\AA}^2$)", 1.0),
    ("memb_thickness", "Membrane thickness", r"thickness ($\mathrm{\AA}$)", 1.0),
    ("registration", "Interleaflet registration", "registration coefficient", 1.0),
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
        pooled = [data[f"{ff}_{lipid}"] * scale
                  for ff, _, _ in FORCEFIELDS if f"{ff}_{lipid}" in data.files]
        if pooled:
            allv = np.concatenate(pooled)
            bins = np.linspace(np.nanmin(allv), np.nanmax(allv), 41)
        for ff, label, color in FORCEFIELDS:
            key = f"{ff}_{lipid}"
            if key not in data.files:
                continue
            any_data = True
            ax.hist(data[key] * scale, bins=bins, density=True, color=color,
                    histtype="stepfilled", alpha=0.35, label=label)
            ax.hist(data[key] * scale, bins=bins, density=True, color=color,
                    histtype="step", lw=1.5)
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


# Candidate inset corners as [x0, y0, w, h] in axes fractions.
_INSET_TL = [0.09, 0.57, 0.40, 0.38]  # top-left
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
        # Fitted D (10^-7 cm^2/s), one bar per force field, drawn as an inset.
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

            D = data[f"{ff}_{lipid}_D"] * 1e7  # -> 10^-7 cm^2/s
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
        inset.set_title(r"$D$ ($10^{-7}\,\mathrm{cm^2/s}$)", fontsize=7)
    handles, labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        if h:
            handles, labels = h, l
            break
    fig.legend(handles, labels, loc="upper right", ncol=len(FORCEFIELDS))
    fig.suptitle("Lateral diffusion")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = PLOTS / "lipprop_lateral_diffusion.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"wrote {out}")


def plot_order_parameter():
    """S_CC-vs-carbon profiles, drawn the way lipids are conventionally shown:
    carbon number down the y-axis (C1/headgroup at top, tail end at bottom) and
    the order parameter along the x-axis.

    The npz holds the full per-carbon S_CC for every chain and frame, one array
    ``{ff}_{lipid}_eq{r}`` of shape (n_chain_frames, n_carbons) per replicate;
    the profile is the mean over that sample axis, and the shaded band is the
    spread across the per-replicate profiles."""
    data = load("order_parameter")
    if data is None:
        print("skip order_parameter: npz not found")
        return

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharex=True)
    any_data = False
    for ax, lipid in zip(axes, LIPIDS):
        cmax = 0
        for ff, label, color in FORCEFIELDS:
            ck = f"{ff}_{lipid}_carbons"
            if ck not in data.files:
                continue
            reps = sorted(int(k.rsplit("_eq", 1)[1])
                          for k in data.files if k.startswith(f"{ff}_{lipid}_eq"))
            if not reps:
                continue
            any_data = True
            carbons = data[ck]
            # Mean over all chain-frame samples in each replicate -> per-rep profile.
            profiles = np.array([data[f"{ff}_{lipid}_eq{r}"].mean(axis=0) for r in reps])
            mean = profiles.mean(axis=0)
            cmax = max(cmax, int(carbons.max()))
            ax.plot(mean, carbons, color=color, marker="o", ms=3, lw=1.5, label=label)
            if len(profiles) > 1:
                sd = profiles.std(axis=0)
                ax.fill_betweenx(carbons, mean - sd, mean + sd, color=color,
                                 alpha=0.2, lw=0)
        ax.set_title(lipid)
        ax.set_xlabel(r"$S_\mathrm{CC}$")
        # Carbon 1 at the top, terminal carbon at the bottom.
        if cmax:
            ax.set_ylim(cmax + 1, 0.5)
        ax.tick_params(axis="y", right=True)
    axes[0].set_ylabel("tail carbon number")
    if not any_data:
        plt.close(fig)
        print("skip order_parameter: no systems present in npz")
        return
    handles, labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        if h:
            handles, labels = h, l
            break
    fig.legend(handles, labels, loc="upper right", ncol=len(FORCEFIELDS))
    fig.suptitle("Acyl chain order parameter")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = PLOTS / "lipprop_order_parameter.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"wrote {out}")


def main():
    PLOTS.mkdir(parents=True, exist_ok=True)
    for stem, title, xlabel, scale in HIST_PROPS:
        plot_histograms(stem, title, xlabel, scale)
    plot_order_parameter()
    plot_lateral_diffusion()


if __name__ == "__main__":
    main()
