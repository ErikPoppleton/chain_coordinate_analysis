#!/usr/bin/env python3
"""Diagnostics of the Bayesian finite-size correction (analysis/diffusion_psd.py):
how the accepted draws sit inside the priors.

plots/psd_posterior.png   rows = eta_m, b, R, eta_f, then the *prior-predictive*
                          D_PBC (a fresh prior sample pushed through the PSD
                          model, with D_sim +/- sigma_D shaded); columns =
                          lipids; force fields overlaid.  Filled steps =
                          accepted draws, thin dashed = the prior, both as
                          densities per log10 unit so a prior that is
                          uninformative reads flat-ish.
plots/psd_joint.png       accepted eta_m against R per lipid (the eta_m-R
                          degeneracy), with each force field's prior median
                          of eta_m as a dotted line.

The acceptance rate of the rejection sampler is ~ sqrt(2 pi) sigma_D times
the prior-predictive density of D_PBC at D_sim, so the last row shows
directly why a system accepts few draws: a small sigma_D, a D_sim in the
tail of the prior predictive, or both.

    python3 analysis/plot_psd_posterior.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                              # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import diffusion_psd as psd                                  # noqa: E402
from palette import FORCEFIELDS                              # noqa: E402
from lipiddefs import LIPIDS                                 # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
PLOTS = ROOT / "plots"
N_PRED = 20_000

PARAMS = [  # key, label, scale for display (R, eta_f on linear x)
    ("eta_m", r"$\eta_\mathrm{m}$ ($10^{-11}$ Pa s m)", 1e11),
    ("b", r"$b$ (Pa s m$^{-1}$)", 1.0),
    ("R", r"$R$ (nm)", 1e9),
    ("eta_f", r"$\eta_\mathrm{f}$ (mPa s)", 1e3),
]


def prior_of(key, ff, r):
    """(arithmetic mean, fractional SD) of the prior, SI units."""
    if key == "eta_m":
        return psd.PRIOR_ETA_M
    if key == "b":
        return psd.PRIOR_B
    if key == "eta_f":
        return psd.PRIOR_ETA_F_FF[ff]
    if key == "R":
        return r["R0_nm"] * 1e-9, psd.PRIOR_R_W


def lognormal_pdf_log10(x, mean, w):
    """Density per log10 unit of a log-normal with arithmetic mean/frac SD."""
    s2 = np.log1p(w ** 2)
    mu = np.log(mean) - s2 / 2
    z = (np.log(x) - mu) / np.sqrt(s2)
    return np.exp(-z ** 2 / 2) / np.sqrt(2 * np.pi * s2) * np.log(10)


def prior_predictive(ff, r, seed=1):
    rng = np.random.default_rng(seed)
    p = {"eta_m": psd.PRIOR_ETA_M, "eta_f": psd.PRIOR_ETA_F_FF[ff],
         "b": psd.PRIOR_B}
    em = psd.lognormal(rng, *p["eta_m"], N_PRED)
    ef = psd.lognormal(rng, *p["eta_f"], N_PRED)
    b = psd.lognormal(rng, *p["b"], N_PRED)
    R = psd.lognormal(rng, r["R0_nm"] * 1e-9, psd.PRIOR_R_W, N_PRED)
    L = psd.lognormal(rng, r["L_nm"] * 1e-9, psd.PRIOR_LH_W, N_PRED)
    H = psd.lognormal(rng, r["H_nm"] * 1e-9, psd.PRIOR_LH_W, N_PRED)
    out = np.empty(N_PRED)
    for g in range(0, N_PRED, 500):
        sl = slice(g, g + 500)
        out[sl] = psd.d_pbc(em[sl], ef[sl], R[sl], b[sl], L[sl], H[sl],
                            psd.T_K) * 1e12
    return out


def log_hist(ax, x, edges, colour, filled=True):
    h, _ = np.histogram(np.log10(x), bins=np.log10(edges), density=True)
    ax.stairs(h, edges, color=colour, lw=1.0,
              fill=False, alpha=1.0)
    if filled:
        ax.stairs(h, edges, color=colour, fill=True, alpha=0.15, lw=0)


def main():
    res = json.loads((DATA / "diffusion_psd.json").read_text())
    # the saved npz keeps only 400 thinned draws; rerun the sampler (same
    # seed and priors as diffusion_psd.main, ~20 s) for every accepted draw
    post = {}
    for key, r in res.items():
        if key == "priors":
            continue
        ff = key.split("_")[0]
        p = psd.bayes(r["D_sim"] * 1e-12, r["D_sim_sem"] * 1e-12,
                      r["L_nm"] * 1e-9, r["H_nm"] * 1e-9, r["R0_nm"] * 1e-9,
                      priors={"eta_f": psd.PRIOR_ETA_F_FF[ff]})
        assert len(p["eta_m"]) == r["n_accepted"], key
        post.update({f"{key}_{q}": v for q, v in p.items()})
    nrow = len(PARAMS) + 1
    fig, axes = plt.subplots(nrow, len(LIPIDS), figsize=(11, 2.3 * nrow),
                             squeeze=False)
    # common x range per parameter row, from priors' 0.1-99.9 % and posteriors
    print(f"{'system':16s} {'N_acc':>6s} {'acc %':>6s} {'sigma_D':>7s} "
          f"{'p_pred(D_sim)':>13s} {'pred %ile of D_sim':>18s}")
    for c, lipid in enumerate(LIPIDS):
        for ff, label, colour in FORCEFIELDS:
            key = f"{ff}_{lipid}"
            if key not in res:
                continue
            r = res[key]
            for row, (pk, plabel, sc) in enumerate(PARAMS):
                ax = axes[row, c]
                m, w = prior_of(pk, ff, r)
                s = np.sqrt(np.log1p(w ** 2))
                mu = np.log(m) - s ** 2 / 2
                lo, hi = np.exp(mu - 3.3 * s), np.exp(mu + 3.3 * s)
                edges = np.geomspace(lo, hi, 60)
                log_hist(ax, post[f"{key}_{pk}"] * sc, edges * sc, colour)
                xs = np.geomspace(lo, hi, 300)
                ax.plot(xs * sc, lognormal_pdf_log10(xs, m, w), color=colour,
                        lw=0.8, ls="--")
            # prior predictive of D_PBC
            ax = axes[-1, c]
            dp = prior_predictive(ff, r)
            edges = np.geomspace(0.1, 1000, 70)
            log_hist(ax, dp, edges, colour, filled=False)
            ax.axvspan(r["D_sim"] - r["D_sim_sem"], r["D_sim"] + r["D_sim_sem"],
                       color=colour, alpha=0.25, lw=0)
            # density of D_PBC at D_sim (per um^2/s) and its percentile
            kde_w = r["D_sim_sem"]
            p_at = np.mean(np.exp(-(dp - r["D_sim"]) ** 2 / (2 * kde_w ** 2)))
            pct = 100 * np.mean(dp < r["D_sim"])
            print(f"{key:16s} {r['n_accepted']:6d} "
                  f"{100 * r['n_accepted'] / psd.N_DRAWS:6.2f} "
                  f"{r['D_sim_sem']:7.2f} {p_at:13.3f} {pct:18.1f}")
        for row, (pk, plabel, sc) in enumerate(PARAMS):
            ax = axes[row, c]
            if pk in ("eta_m", "b"):
                ax.set_xscale("log")
            ax.set_xlabel(plabel)
            ax.set_ylabel("density / log$_{10}$")
        ax = axes[-1, c]
        ax.set_xscale("log")
        ax.set_xlabel(r"prior-predictive $D_\mathrm{PBC}$ ($\mu$m$^2$ s$^{-1}$)")
        ax.set_ylabel("density / log$_{10}$")
        axes[0, c].set_title(lipid)
    handles = [plt.Line2D([], [], color=col, lw=1.0, label=lab)
               for _, lab, col in FORCEFIELDS]
    handles += [plt.Line2D([], [], color="0.3", lw=0.8, ls="--", label="prior"),
                plt.Rectangle((0, 0), 1, 1, color="0.3", alpha=0.25,
                              label=r"$D_\mathrm{sim}\pm\sigma_D$")]
    fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = PLOTS / "psd_posterior.png"
    fig.savefig(out, dpi=200)
    print("wrote", out.relative_to(ROOT))

    # ---- joint eta_m vs R
    fig, axes = plt.subplots(1, len(LIPIDS), figsize=(11, 3.6), sharey=True)
    for ax, lipid in zip(axes, LIPIDS):
        for ff, label, colour in FORCEFIELDS:
            key = f"{ff}_{lipid}"
            if key not in res:
                continue
            ax.scatter(post[f"{key}_R"] * 1e9, post[f"{key}_eta_m"] * 1e11,
                       s=1.0, color=colour, alpha=0.08, lw=0, label=label,
                       rasterized=True)
        m, w = psd.PRIOR_ETA_M
        ax.axhline(m * 1e11 / (1 + w ** 2) ** 0.5, color="0.3", ls=":", lw=0.8)
        ax.set_yscale("log")
        ax.set_title(lipid)
        ax.set_xlabel(r"accepted $R$ (nm)")
    axes[0].set_ylabel(r"accepted $\eta_\mathrm{m}$ ($10^{-11}$ Pa s m)")
    leg = axes[0].legend(markerscale=6, frameon=False, loc="lower left")
    for lh in leg.legend_handles:
        lh.set_alpha(1)
    fig.tight_layout()
    out = PLOTS / "psd_joint.png"
    fig.savefig(out, dpi=200)
    print("wrote", out.relative_to(ROOT))


if __name__ == "__main__":
    main()
