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
    ("order_parameter", "Acyl chain order parameter", r"$S_\mathrm{CC}$", 1.0),
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


def plot_lateral_diffusion():
    data = load("lateral_diffusion")
    if data is None:
        print("skip lateral_diffusion: npz not found")
        return

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharex=True, sharey=True)
    any_data = False
    for ax, lipid in zip(axes, LIPIDS):
        # Fitted D (10^-7 cm^2/s), one bar per force field, drawn as an inset.
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
        if bar_x:
            # Only draw the inset for panels that actually have data.
            inset = ax.inset_axes([0.12, 0.55, 0.42, 0.4])
            inset.bar(bar_x, bar_h, yerr=bar_e, color=bar_c, capsize=2, width=0.7)
            inset.set_xticks(bar_x)
            inset.set_xticklabels([FORCEFIELDS[i][0].upper()[:1] for i in bar_x],
                                  fontsize=7)
            inset.tick_params(labelsize=7)
            inset.set_title(r"$D$ ($10^{-7}\,\mathrm{cm^2/s}$)", fontsize=7)
        ax.set_title(lipid)
        ax.set_xlabel("lag time (ns)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.tick_params(axis="y", right=True)
    axes[0].set_ylabel(r"MSD ($\mathrm{nm}^2$)")
    if not any_data:
        plt.close(fig)
        print("skip lateral_diffusion: no systems present in npz")
        return
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


def main():
    PLOTS.mkdir(parents=True, exist_ok=True)
    for stem, title, xlabel, scale in HIST_PROPS:
        plot_histograms(stem, title, xlabel, scale)
    plot_lateral_diffusion()


if __name__ == "__main__":
    main()
