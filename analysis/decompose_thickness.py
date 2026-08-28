#!/usr/bin/env python3
"""Decompose the phosphate--phosphate thickness d_HH into a **tail (core)** term
and a **headgroup-extension** term, **per frame**, to see how much of the
between-system difference in d_HH is core vs headgroup -- and whether the tail
and headgroup extensions trade off (anti-correlate) within a run.

Geometry (all from the SAME ``traj/`` universe, so frame-aligned):

    d_HH(t)  =  d_HC(t)          the apolar core (first-tail-C -> first-tail-C)
              +  H(t)            H(t) = d_HH(t) - d_HC(t), the two headgroup
                                 shells (first-tail-C -> P, x2)

d_HC and d_HH are the lipyphilic ``MembThickness`` of the tail-carbon and
phosphate planes, computed frame-by-frame over the **full trajectory** by
``core_thickness.py`` (keys ``{ff}_{lipid}_lipy_rep{r}`` and ``_dhh_rep{r}``), so
H(t) is a genuine per-frame quantity -- a within-frame tail<->head trade-off is
kept, not averaged away.

Reported:
  * per-system mean d_HH / core d_HC / headgroup H (nm), with per-frame SD;
  * **within-run** per-frame Pearson r(d_HC, H) -- the tail<->head coupling that a
    mean-of-means would hide (computed per replicate on frame anomalies, then
    averaged, so slow between-replicate drift can't inflate it);
  * the between-system variance partition of d_HH into core vs headgroup
    (this uses system means and is, by linearity, identical whether formed per
    frame or from means -- it is the answer to "portion of the *difference*").

    python3 analysis/decompose_thickness.py    # numpy + matplotlib

Reads data/core_thickness.npz.  Writes plots/thickness_decomposition.png.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import figstyle as fs
from palette import FORCEFIELDS
from lipiddefs import LIPIDS, REPS

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
PLOTS = ROOT / "plots"



def per_frame_values():
    """{(ff,lipid): {'hc':[arr_rep,...], 'hh':[...], 'H':[...]}} of frame-aligned
    per-frame arrays (nm), one array per replicate."""
    ct = np.load(DATA / "core_thickness.npz")
    out = {}
    for ff, _, _ in FORCEFIELDS:
        for lipid in LIPIDS:
            hc, hh, H = [], [], []
            for r in REPS:
                kc, kh = f"{ff}_{lipid}_lipy_rep{r}", f"{ff}_{lipid}_dhh_rep{r}"
                if kc not in ct.files or kh not in ct.files:
                    continue
                a, b = np.asarray(ct[kc]), np.asarray(ct[kh])
                n = min(len(a), len(b))          # guard against off-by-one
                a, b = a[:n], b[:n]
                hc.append(a)
                hh.append(b)
                H.append(b - a)
            if hc:
                out[(ff, lipid)] = {"hc": hc, "hh": hh, "H": H}
    return out


def within_run_corr(reps_x, reps_y):
    """Mean over replicates of the per-frame Pearson r, each computed on that
    replicate's frame anomalies (so between-rep drift doesn't inflate it)."""
    rs = []
    for x, y in zip(reps_x, reps_y):
        xa, ya = x - x.mean(), y - y.mean()
        denom = np.sqrt((xa ** 2).sum() * (ya ** 2).sum())
        if denom > 0:
            rs.append(float((xa * ya).sum() / denom))
    return (float(np.mean(rs)), float(np.std(rs))) if rs else (np.nan, np.nan)


def sys_stats(v):
    """mean & per-frame SD (pooled over reps) of hc, hh, H, plus r(hc,H)."""
    def ms(key):
        allf = np.concatenate(v[key])
        # replicate-mean of the per-rep means (project convention for the point)
        return float(np.mean([a.mean() for a in v[key]])), float(np.std(allf))
    (hc_m, hc_s), (hh_m, hh_s), (H_m, H_s) = ms("hc"), ms("hh"), ms("H")
    r_cH = within_run_corr(v["hc"], v["H"])
    r_cHH = within_run_corr(v["hc"], v["hh"])
    return dict(hc=hc_m, hc_sd=hc_s, hh=hh_m, hh_sd=hh_s, H=H_m, H_sd=H_s,
                r_core_head=r_cH, r_core_hh=r_cHH)


def decompose(systems, means):
    """Variance partition of d_HH across systems (uses their means)."""
    hh = np.array([means[s]["hh"] for s in systems])
    hc = np.array([means[s]["hc"] for s in systems])
    H = np.array([means[s]["H"] for s in systems])
    v_hh = np.var(hh)
    if v_hh == 0 or len(systems) < 2:
        return None
    v_hc, v_H = np.var(hc), np.var(H)
    cov = np.mean((hc - hc.mean()) * (H - H.mean()))
    return dict(sd_hh=np.sqrt(v_hh),
                frac_core=(v_hc + cov) / v_hh, frac_head=(v_H + cov) / v_hh,
                sd_core=np.sqrt(v_hc), sd_head=np.sqrt(v_H), cov=cov)


def print_table(stats):
    print(f"\n{'system':16s} {'d_HH':>13} {'core d_HC':>13} {'headgroup H':>13}"
          f" {'core%':>6} {'r(core,H)':>12}")
    for ff, label, _ in FORCEFIELDS:
        for lipid in LIPIDS:
            s = stats.get((ff, lipid))
            if s is None:
                continue
            rc, rcsd = s["r_core_head"]
            print(f"{label+' '+lipid:16s} {s['hh']:6.3f}±{s['hh_sd']:.3f}"
                  f" {s['hc']:6.3f}±{s['hc_sd']:.3f} {s['H']:6.3f}±{s['H_sd']:.3f}"
                  f" {100*s['hc']/s['hh']:5.1f} {rc:+6.2f}±{rcsd:.2f}")
    print("\n  (± on d_HH/d_HC/H is the per-frame SD pooled over reps;"
          " r(core,H) is the within-run per-frame Pearson, mean±SD over reps.)")


def report(name, systems, means):
    d = decompose(systems, means)
    if d is None:
        return
    print(f"\n[{name}]  spread in d_HH: SD = {d['sd_hh']*10:.2f} A "
          f"({d['sd_hh']:.3f} nm)")
    print(f"    tail/core (d_HC)      explains {100*d['frac_core']:5.1f}%   "
          f"(core SD {d['sd_core']:.3f} nm)")
    print(f"    headgroup extension   explains {100*d['frac_head']:5.1f}%   "
          f"(headgroup SD {d['sd_head']:.3f} nm)")


LEGEND_BAND_MM = 5.0     # top strip for the core/headgroup key
FIG_H_MM = 70.0


def plot(stats):
    """Stacked core + headgroup bars, one panel per lipid, at the house
    **180 mm** width (``figstyle``): 9 pt labels, 7 pt ticks, ticks inside.

    Note ``bbox_inches`` is deliberately *not* passed to ``savefig`` -- a tight
    bbox shrink-wraps the canvas and the 180 mm width is the point.
    """
    fs.use()
    fig, axes = fs.figure_grid(1, len(LIPIDS), height_mm=FIG_H_MM,
                              label_band_mm=LEGEND_BAND_MM, sharey=True)
    for ax, lipid in zip(axes, LIPIDS):
        for i, (ff, label, color) in enumerate(FORCEFIELDS):
            s = stats.get((ff, lipid))
            if s is None:
                continue
            hc, H, hh = s["hc"], s["H"], s["hh"]
            ax.bar(i, hc, 0.7, color=color, edgecolor="black", lw=0.5)
            ax.bar(i, H, 0.7, bottom=hc, color=color, edgecolor="black", lw=0.5,
                   hatch="///", alpha=0.45)
            ax.errorbar(i, hh, yerr=s["hh_sd"], color="black", capsize=2, lw=1.0)
            ax.text(i, hc / 2, f"{hc:.2f}", ha="center", va="center",
                    fontsize=fs.TICK_PT, color="white")
            ax.text(i, hc + H / 2, f"{H:.2f}", ha="center", va="center",
                    fontsize=fs.TICK_PT, color="black")
            # within-run tail<->head coupling, inside the foot of the bar
            # (white: grey on the saturated bar colours was unreadable).
            ax.text(i, 0.06, f"r={s['r_core_head'][0]:+.2f}", ha="center",
                    va="bottom", fontsize=fs.TICK_PT - 1, color="white")
        ax.set_title(lipid)
        ax.set_xticks(range(len(FORCEFIELDS)))
        ax.set_xticklabels([l for _, l, _ in FORCEFIELDS])
        fs.categorical_x(ax)          # force fields are categories, not a scale
    axes[0].set_ylabel("thickness (nm)")
    handles = [
        Patch(facecolor="0.55", edgecolor="black", label="core $d_\\mathrm{HC}$ (tails)"),
        Patch(facecolor="0.55", edgecolor="black", hatch="///", alpha=0.45,
              label="headgroup $H=d_\\mathrm{HH}-d_\\mathrm{HC}$"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 1.0), handlelength=1.4, handleheight=0.9)
    PLOTS.mkdir(parents=True, exist_ok=True)
    out = PLOTS / "thickness_decomposition.png"
    fig.savefig(out)
    plt.close(fig)
    w_mm = fig.get_size_inches()[0] * 25.4
    print(f"\nwrote {out}  ({w_mm:.2f} x {fig.get_size_inches()[1] * 25.4:.2f} mm)")


def main():
    vals = per_frame_values()
    stats = {s: sys_stats(v) for s, v in vals.items()}
    means = {s: {"hh": st["hh"], "hc": st["hc"], "H": st["H"]}
             for s, st in stats.items()}
    print_table(stats)

    report("all 9 systems", list(means), means)
    for lipid in LIPIDS:
        systems = [(ff, lipid) for ff, _, _ in FORCEFIELDS if (ff, lipid) in means]
        if len(systems) >= 2:
            report(f"force-field spread within {lipid}", systems, means)

    # Tail-length effect: FF-averaged lipid means.
    lip_means = {}
    for lipid in LIPIDS:
        systems = [(ff, lipid) for ff, _, _ in FORCEFIELDS if (ff, lipid) in means]
        if systems:
            lip_means[("all", lipid)] = {
                k: float(np.mean([means[s][k] for s in systems]))
                for k in ("hh", "hc", "H")}
    report("across lipids (FF-averaged: DOPC-C18 vs DMPx-C14)",
           list(lip_means), lip_means)

    plot(stats)


if __name__ == "__main__":
    main()
