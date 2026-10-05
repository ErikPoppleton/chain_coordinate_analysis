#!/usr/bin/env python3
"""DF tilt stiffness as Fourier modes are added: collective -> single-lipid limit.

For each system the director spectrum <|n_q|^2> (bilayer field
n = (n_up - n_down)/2, as in ``tilt_df.py``) is computed for every mode up to
``--nmax``.  The per-lipid tilt variance is the average of the spectrum over
*all* modes (Parseval), so averaging the spectrum over the modes with
|q| <= q_max and inverting,

    K_eff(q_max) = k_B T / mean_{|q| <= q_max} <|n_q|^2>,

interpolates between the collective (q -> 0) stiffness, when only the longest
wavelengths are included, and the single-lipid value, when every mode is.  With
independent lipids K_eff would be flat at 2 k_B T / (a <n_x^2>); the real-space
methods (ReGiS chi/A, ReSiS kappa) correspond to half of that, so 2x their
values is marked for comparison.

Run (conda env ``lipid64``, from the repository root)::

    PYTHONNOUSERSITE=1 conda run -n lipid64 python analysis/plot_df_mode_convergence.py

Spectra are cached as ``plots/tilt_three_methods/df_modes_{key}_{director}.npz``;
the figure is ``plots/tilt_three_methods/df_mode_convergence.png``.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import tilt_df  # noqa: E402

ROOT = HERE.parent
OUT = ROOT / "plots" / "tilt_three_methods"
KB_KJ = 0.0083144626
SYSTEMS = [("charmm", "DMPC", 1), ("charmm", "DOPC", 1)]


def spectra(ff, lipid, rep, director, nmax, stride):
    key = "%s_%s_%d" % (ff, lipid, rep)
    cache = OUT / ("df_modes_%s_%s.npz" % (key, director))
    if cache.exists():
        d = np.load(cache)
        if int(d["nmax"]) == nmax and int(d["stride"]) == stride:
            return dict(d)
    res = tilt_df.analyse(str(ROOT / "traj" / ("%s_%s.gro" % (ff, lipid))),
                          str(ROOT / "traj" / ("%s_%s%d.xtc" % (ff, lipid, rep))),
                          lipid, director, nmax, 0.0, stride)
    out = dict(q=res["q"], mpar=res["spar"].mean(0), mperp=res["sperp"].mean(0),
               apl=res["apl"].mean(), n_lipids=res["n_lipids"],
               box=res["boxes"].mean(0), nmax=nmax, stride=stride)
    np.savez(cache, **out)
    return out


def cumulative(q, var):
    """K_eff (in k_BT/nm^2) for q_max at each distinct |q| shell."""
    order = np.argsort(q)
    qs, vs = q[order], var[order]
    csum = np.cumsum(vs) / np.arange(1, len(vs) + 1)
    # keep the last entry of each shell
    last = np.r_[np.diff(qs) > 1e-6 * qs[1:], True]
    return qs[last], 1.0 / csum[last], np.arange(1, len(vs) + 1)[last]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--director", default="rsf", choices=["rsf", "levine"])
    ap.add_argument("--nmax", type=int, default=16)
    ap.add_argument("--stride", type=int, default=2)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summ = json.loads((OUT / "summary.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.1), sharey=True)
    report = {}
    for ax, (ff, lip, rep) in zip(axes, SYSTEMS):
        key = "%s_%s_%d" % (ff, lip, rep)
        s = summ[key]
        kT = KB_KJ * s["temperature_K"]
        d = spectra(ff, lip, rep, args.director, args.nmax, args.stride)
        q = d["q"]
        both = 0.5 * (d["mpar"] + d["mperp"])     # per-component, all directions
        qt, Kt, nt = cumulative(q, d["mperp"])
        qb, Kb, nb = cumulative(q, both)
        # per-shell transverse values (no averaging)
        qs, vs = tilt_df.shell_average(q, d["mperp"])
        nleaf = d["n_lipids"] / 2.0
        q_bz = 2 * np.pi / np.sqrt(d["box"].prod()) * np.sqrt(nleaf / np.pi)

        ax.plot(qs, kT / vs, "o", ms=2.5, color="0.65", zorder=1,
                label=r"single shell, transverse")
        ax.plot(qt, kT * Kt, "-", color="#2171b5", lw=1.4,
                label=r"modes $|q|\leq q_{max}$, transverse")
        ax.plot(qb, kT * Kb, "--", color="#2171b5", lw=1.2,
                label=r"modes $|q|\leq q_{max}$, all components")
        r2 = 2 * s["ReSiS"]["kappa_kJmol_nm2"]
        g2 = 2 * s["ReGiS"]["chi_over_A_kJmol_nm2"]
        df0 = s["DF_%s" % args.director]["K_theta_bilayer_kJmol_nm2"]["qcut_2.10"][0]
        ax.axhline(r2, color="#0f7d3b", lw=1.3, label=r"$2\times$ ReSiS $\kappa_\theta$")
        ax.axhline(g2, color="#d4a017", lw=1.1, ls="--", label=r"$2\times$ ReGiS $\chi/A$")
        ax.axhline(df0, color="#2171b5", lw=0.9, ls=":", label=r"DF fit, $q\to0$")
        ax.axvline(q_bz, color="0.4", lw=0.7, ls="-.")
        ax.text(q_bz * 1.03, 6, "one mode\nper lipid", fontsize=6.5, color="0.3", va="bottom", ha="left")
        ax.set_xscale("log")
        ax.set_xlim(0.9, q.max() * 1.05)
        ax.set_ylim(0, None)
        ax.set_xlabel(r"$q_{max}$ (nm$^{-1}$)   [fewer modes $\leftarrow$]")
        ax.set_title("CHARMM %s (rep 1), %s director" % (lip, args.director), fontsize=9)
        top = ax.secondary_xaxis("top")
        tq = [qb[i] for i in np.searchsorted(nb, [4, 16, 64, 256]) if i < len(qb)]
        top.set_ticks(tq)
        top.set_xticklabels([str(nb[np.searchsorted(qb, t)]) for t in tq], fontsize=7)
        top.set_xlabel("modes included", fontsize=8)
        report[key] = {"K_eff_all_modes_transverse": float(kT * Kt[-1]),
                       "K_eff_all_modes_both": float(kT * Kb[-1]),
                       "K_eff_lowest_shell_transverse": float(kT * Kt[0]),
                       "two_x_ReSiS": r2, "two_x_ReGiS": g2, "DF_q0": df0,
                       "q_one_mode_per_lipid": float(q_bz),
                       "n_modes_total": int(nb[-1])}
    axes[0].set_ylabel(r"$K_{eff}$ (kJ mol$^{-1}$ nm$^{-2}$, 310 K)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=7, frameon=False)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.savefig(OUT / "df_mode_convergence.png", dpi=200)
    (OUT / "df_mode_convergence.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
