#!/usr/bin/env python3
"""Finite-size correction of lipid lateral diffusion: periodic Saffman-Delbrueck
(PSD) model + Venable et al. (2017) Bayesian single-box extrapolation.

Model (Camley et al. 2015; Venable et al. 2017, J. Phys. Chem. B 121:3443),
for a single-leaflet object (a lipid) of radius R in one leaflet of a bilayer
with surface viscosity eta_m (eta_mono = eta_m/2 per leaflet), interleaflet
friction b, in a periodic box of in-plane edge L with water of viscosity
eta_f and a water layer of total thickness 2H between periodic images
(2H = Lz - h, h the P-P thickness d_HH):

    D_PBC = kT/(2 L^2) sum_{k != 0} exp(-k^2 beta^2 R^2/2) A/(A^2 - B^2)
    A = eta_m/2 k^2 + eta_f k coth(2Hk) + b,   B = b + eta_f k csch(2Hk)

and the infinite-system D_inf is the same integrand with H -> inf
(coth -> 1, csch -> 0) integrated over d^2k/(2pi)^2.  beta = 0.828494.

Bayesian step (Venable 2017 appendix): draw theta = (eta_m, eta_f, R, b) from
log-normal priors, accept with probability exp(-(D_PBC(theta)-D_sim)^2 /
2 sigma_D^2), and report the posterior of D_inf (median, 95 % interval) and of
eta_m.  The data pin eta_m given R and eta_f, so a broad eta_m prior is fine;
the D_inf spread then comes mostly from the R and eta_f priors.

Inputs: data/diffusion.json (analysis/diffusion.py: D, L, Lz per system),
lipprops_{area_per_lipid,memb_thickness}.npz (R from APL, h = d_HH).

    python3 analysis/diffusion_psd.py [--check]

Only the bilayer-COM D is corrected: that is the convention the PSD model and
Venable's numbers assume, and per-leaflet COM removal also strips genuine lipid
displacement (Klauda, Brooks & Pastor 2006), so it has no defined relation to
D_PBC or D_inf.
    ->  data/diffusion_psd.json (+ diffusion_psd_posterior.npz, 400 draws/system)
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lipiddefs import FORCEFIELDS, LIPIDS                    # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
KB = 1.380649e-23
BETA = 0.828494
T_K = 310.0

# Priors are log-normal, given as (arithmetic mean, SD as a fraction of the
# mean) -- Venable's convention ("standard deviations of 5 % of the mean"), so a
# 500 % prior has its median at mean/sqrt(26) ~ mean/5.1.
PRIOR_ETA_M = (8.9e-11, 5.0)     # Pa s m; C36 DOPC 298 K (Venable Table 6), 500 %
# Water viscosity, by water model (10 % prior; Venable used 5 %):
#  * CHARMM-modified TIP3P (LJ on H; the CHARMM runs): Venable Table 6, Ewald,
#    0.311 mPa s at 293 K / 0.252 at 323 K -> 0.2776 at 310 K (linear).
#  * original Jorgensen TIP3P (no LJ on H; Lipid21 TP3.itp and SLipids
#    tip3p.itp OWT3/HWT3, checked; SLipids DMPG's TP3 assumed the same):
#    Gonzalez & Abascal 2010 (JCP 132, 096101) 0.321 mPa s at 298 K, scaled to
#    310 K by the CHARMM-TIP3P ratio 0.2776/0.3012 -> 0.296.
ETA_F_CHARMM_TIP3P = 2.8e-4
ETA_F_TIP3P = 0.321e-3 * (0.311 - 0.059 * 17 / 30) / (0.311 - 0.059 * 5 / 30)
PRIOR_ETA_F = (ETA_F_CHARMM_TIP3P, 0.10)          # default (and --check)
PRIOR_ETA_F_FF = {"charmm": (ETA_F_CHARMM_TIP3P, 0.10),
                  "amber": (ETA_F_TIP3P, 0.10),
                  "slipids": (ETA_F_TIP3P, 0.10)}
PRIOR_R_W = 0.25                 # R: mean sqrt(APL/pi) (0.45/0.47 nm in Venable), 25 %
PRIOR_B = (1e8, 5.0)             # Pa s/m (1e7 P/cm, Venable's value), 500 %
PRIOR_LH_W = 0.02                # L and H, 2 % (Venable)
N_DRAWS = 200_000
N_KEEP = 400                     # posterior draws kept for the SI D(L) bands
NMAX = 24                        # lattice half-width; Gaussian cutoff kills k > ~10/nm


def _coth(x):
    return 1.0 / np.tanh(x)


def _csch(x):
    ex = np.exp(-x)
    return 2 * ex / (1 - ex * ex)


def kvectors(L, nmax=None):
    """Distinct nonzero reciprocal-lattice |k| for an L x L box and their
    multiplicities (the sum depends on |k| only).  The Gaussian cutoff sets a
    fixed k_max, so the lattice half-width grows with L (NMAX is sized for
    L ~ 6.5 nm).  ``L`` may be an array (one box per draw): the distinct
    integer |n| do not depend on L, so k = 2 pi |n| / L comes back as
    (n_draws, n_k), the lattice sized for the largest L."""
    L = np.asarray(L, float)
    if nmax is None:
        nmax = max(NMAX, int(np.ceil(NMAX * L.max() / 6.5e-9)))
    n = np.arange(-nmax, nmax + 1)
    n2 = (n[:, None] ** 2 + n[None, :] ** 2).ravel()
    n2, w = np.unique(n2[n2 > 0], return_counts=True)
    if L.ndim == 0:
        return 2 * np.pi / L * np.sqrt(n2), w
    return 2 * np.pi / L[:, None] * np.sqrt(n2)[None, :], w


def _box(L, H):
    """Lattice k as (draws or 1, n_k), plus L and H shaped to broadcast."""
    k, w = kvectors(L)
    k = np.atleast_2d(k)
    L = np.atleast_1d(np.asarray(L, float))
    H = np.atleast_1d(np.asarray(H, float))[:, None]
    return k, w, L, H


def d_pbc(eta_m, eta_f, R, b, L, H, T=T_K):
    """D_PBC (m^2/s); parameters may be arrays (broadcast over draws), L/H in m
    (scalars, or one value per draw)."""
    k, w, L, H = _box(L, H)
    eta_m, eta_f, R, b = (np.atleast_1d(np.asarray(v, float))[:, None]
                          for v in (eta_m, eta_f, R, b))
    A = eta_m / 2 * k ** 2 + eta_f * k * _coth(2 * H * k) + b
    B = b + eta_f * k * _csch(2 * H * k)
    g = np.exp(-k ** 2 * BETA ** 2 * R ** 2 / 2) * A / (A * A - B * B)
    return KB * T / (2 * L ** 2) * (g * w).sum(axis=1)


_KQ = np.geomspace(1e3, 1e11, 3000)            # 1/m, for the D_inf integral


def d_inf(eta_m, eta_f, R, b, T=T_K):
    k = _KQ[None, :]
    eta_m, eta_f, R, b = (np.atleast_1d(np.asarray(v, float))[:, None]
                          for v in (eta_m, eta_f, R, b))
    A = eta_m / 2 * k ** 2 + eta_f * k + b
    f = k * np.exp(-k ** 2 * BETA ** 2 * R ** 2 / 2) * A / (A * A - b * b)
    return KB * T / (4 * np.pi) * np.trapezoid(f, _KQ, axis=1)


def d_pbc_tm(eta_m, eta_f, R, L, H, T=T_K):
    """Transmembrane-object PSD (Venable eq 2a), for the validation cases."""
    k, w, L, H = _box(L, H)
    eta_m, eta_f, R = (np.atleast_1d(np.asarray(v, float))[:, None]
                       for v in (eta_m, eta_f, R))
    g = np.exp(-k ** 2 * BETA ** 2 * R ** 2 / 2) / (
        eta_m * k ** 2 + 2 * eta_f * k * np.tanh(k * H))
    return KB * T / (2 * L ** 2) * (g * w).sum(axis=1)


def d_inf_tm(eta_m, eta_f, R, T=T_K):
    k = _KQ[None, :]
    eta_m, eta_f, R = (np.atleast_1d(np.asarray(v, float))[:, None]
                       for v in (eta_m, eta_f, R))
    f = k * np.exp(-k ** 2 * BETA ** 2 * R ** 2 / 2) / (eta_m * k ** 2 + 2 * eta_f * k)
    return KB * T / (4 * np.pi) * np.trapezoid(f, _KQ, axis=1)


def lognormal(rng, mean, frac_sd, n):
    """Log-normal with arithmetic mean ``mean`` and SD ``frac_sd * mean``."""
    s2 = np.log1p(frac_sd ** 2)
    return np.exp(rng.normal(np.log(mean) - s2 / 2, np.sqrt(s2), n))


def bayes(D_sim, sigma_D, L, H, R0, priors=None, T=T_K, seed=0, n=N_DRAWS,
          chunk=20_000, transmembrane=False):
    """Rejection-sample the posterior (Venable 2017 appendix); SI units.
    ``priors`` overrides the module defaults (keys eta_m, eta_f, b, R_w, LH_w)."""
    p = {"eta_m": PRIOR_ETA_M, "eta_f": PRIOR_ETA_F, "b": PRIOR_B,
         "R_w": PRIOR_R_W, "LH_w": PRIOR_LH_W}
    p.update(priors or {})
    rng = np.random.default_rng(seed)
    acc = {"eta_m": [], "eta_f": [], "R": [], "b": [], "D_pbc": []}
    for s in range(0, n, chunk):
        m = min(chunk, n - s)
        th = {"eta_m": lognormal(rng, *p["eta_m"], m),
              "eta_f": lognormal(rng, *p["eta_f"], m),
              "R": lognormal(rng, R0, p["R_w"], m),
              "b": lognormal(rng, *p["b"], m)}
        Ls = lognormal(rng, L, p["LH_w"], m)
        Hs = lognormal(rng, H, p["LH_w"], m)
        # L and H vary per draw: the lattice depends on L, so evaluate in
        # small groups sharing one L/H sample would bias; loop is cheap enough
        # every draw gets its own L and H, like every other parameter
        # (Venable appendix); evaluated in slices of 500 to bound memory
        dp = np.empty(m)
        for g0 in range(0, m, 500):
            sl = slice(g0, g0 + 500)
            if transmembrane:
                dp[sl] = d_pbc_tm(th["eta_m"][sl], th["eta_f"][sl], th["R"][sl],
                                  Ls[sl], Hs[sl], T)
            else:
                dp[sl] = d_pbc(th["eta_m"][sl], th["eta_f"][sl], th["R"][sl],
                               th["b"][sl], Ls[sl], Hs[sl], T)
        keep = rng.random(m) < np.exp(-(dp - D_sim) ** 2 / (2 * sigma_D ** 2))
        for key in th:
            acc[key].append(th[key][keep])
        acc["D_pbc"].append(dp[keep])
    post = {k: np.concatenate(v) for k, v in acc.items()}
    post["D_inf"] = (d_inf_tm(post["eta_m"], post["eta_f"], post["R"], T)
                     if transmembrane else
                     d_inf(post["eta_m"], post["eta_f"], post["R"], post["b"], T))
    return post


def summary(x):
    lo, med, hi = np.percentile(x, [2.5, 50, 97.5])
    return {"median": float(med), "lo95": float(lo), "hi95": float(hi)}


def check(draws=N_DRAWS):
    """Reproduce Venable 2017's published numbers (units 1e-7 cm^2/s)."""
    c7 = 1e11                       # m^2/s -> 1e-7 cm^2/s
    Pcm, P, Pcm1 = 1e-3, 0.1, 10.0  # P cm -> Pa s m, P -> Pa s, P/cm -> Pa s/m
    nm = 1e-9
    print("deterministic (published in brackets):")
    r = lambda x: float(np.atleast_1d(x)[0]) * c7
    print("  eq 2a, Martini gA-like, L20 H5 R1: D_PBC %.2f [4.45]  D_inf %.2f [7.40]" % (
        r(d_pbc_tm(1.2e-8 * Pcm, 0.007 * P, 1 * nm, 20 * nm, 5 * nm, 323)),
        r(d_inf_tm(1.2e-8 * Pcm, 0.007 * P, 1 * nm, 323))))
    print("  Table 1 WALP23, L15 H2 R0.5:       D_PBC %.3f [0.519] D_inf %.2f [1.25]" % (
        # eta_f from Table 1's L_SD = eta_m/(2 eta_f) = 78 nm -> 0.0096 P
        r(d_pbc_tm(1.5e-7 * Pcm, 0.0096 * P, 0.5 * nm, 15 * nm, 2 * nm, 323)),
        r(d_inf_tm(1.5e-7 * Pcm, 0.0096 * P, 0.5 * nm, 323))))
    print("  Table 7 DPPC exp-matched, 288:     D_PBC %.3f [0.584] D_inf %.2f [1.51]" % (
        r(d_pbc(1.6e-7 * Pcm, 0.00547 * P, 0.45 * nm, 1e7 * Pcm1, 9.6 * nm, 1.46 * nm, 323)),
        r(d_inf(1.6e-7 * Pcm, 0.00547 * P, 0.45 * nm, 1e7 * Pcm1, 323))))
    print("  Table 6 C36 DOPC 288 (eta_m 8.9):  D_PBC %.2f [~0.9]  D_inf %.2f [2.4]" % (
        r(d_pbc(8.9e-8 * Pcm, 0.00311 * P, 0.47 * nm, 1e7 * Pcm1, 9.85 * nm, 1.5 * nm, 298)),
        r(d_inf(8.9e-8 * Pcm, 0.00311 * P, 0.47 * nm, 1e7 * Pcm1, 298))))
    print("Bayesian (median [95%]; published in brackets):")
    cases = [
        ("Case B (gA-like, transmembrane)", True, 4.4e-11, 0.044e-11,
         20 * nm, 5 * nm, 1 * nm, 323,
         {"eta_m": (4e-8 * Pcm, 5.0), "eta_f": (0.01 * P, 0.5), "R_w": 0.25},
         "6.86 (5.59-8.15)"),
        ("C36 DPPC 288", False, 2.3e-11, 0.05e-11, 9.52 * nm, 1.5 * nm,
         0.45 * nm, 323,
         {"eta_m": (3.4e-8 * Pcm, 5.0), "eta_f": (0.00252 * P, 0.05)},
         "5.5 (4.8-6.6)"),
        ("C36 DOPC 288", False, 0.89e-11, 0.05e-11, 9.85 * nm, 1.5 * nm,
         0.47 * nm, 298,
         {"eta_m": (8.9e-8 * Pcm, 5.0), "eta_f": (0.00311 * P, 0.05)},
         "2.2 (2.0-2.7)"),
    ]
    for lab, tm, D, sig, L, H, R, T, pri, pub in cases:
        post = bayes(D, sig, L, H, R, priors=pri, T=T, n=draws, transmembrane=tm)
        sm = summary(post["D_inf"] * c7)
        print("  %-34s %.2f (%.2f-%.2f)  [%s]  n_acc=%d" % (
            lab, sm["median"], sm["lo95"], sm["hi95"], pub, post["D_inf"].size))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="reproduce Venable 2017 Table 6 and exit")
    ap.add_argument("--draws", type=int, default=N_DRAWS)
    args = ap.parse_args()
    if args.check:
        check(args.draws)
        return
    diff = json.loads((DATA / "diffusion.json").read_text())
    apl = np.load(DATA / "lipprops_area_per_lipid.npz")
    thick = np.load(DATA / "lipprops_memb_thickness.npz")
    um = 1e12
    out = {"priors": {"eta_m": PRIOR_ETA_M, "eta_f": PRIOR_ETA_F_FF,
                      "R_width": PRIOR_R_W, "b": PRIOR_B,
                      "width_meaning": "sigma_ln = ln(1 + width)",
                      "T_K": T_K, "draws": args.draws, "com": "bilayer"}}
    samples = {}
    print(f"{'system':14s} {'L':>5s} {'H':>5s} {'R':>5s}  {'D_sim':>11s}  "
          f"{'D_inf [95% CI]':>22s}  {'eta_m [95% CI] (1e-11 Pa s m)':>30s}  acc")
    for ff in FORCEFIELDS:
        for lipid in LIPIDS:
            key = f"{ff}_{lipid}"
            if key not in diff:
                continue
            s = diff[key]
            D = s["D_bilayer"]
            L = s["L_nm"] * 1e-9
            h = float(np.mean(thick[key])) / 10.0                    # A -> nm
            H = (s["Lz_nm"] - h) / 2 * 1e-9
            R0 = np.sqrt(float(np.mean(apl[key])) / 100.0 / np.pi) * 1e-9
            D_sim, sig = D["mean"] / um, D["sem"] / um
            post = bayes(D_sim, sig, L, H, R0, n=args.draws,
                         priors={"eta_f": PRIOR_ETA_F_FF[ff]})
            di, em = summary(post["D_inf"] * um), summary(post["eta_m"] * 1e11)
            pick = np.random.default_rng(1).choice(
                post["D_inf"].size, min(N_KEEP, post["D_inf"].size), replace=False)
            for name in ("eta_m", "eta_f", "R", "b", "D_inf"):
                samples[f"{key}_{name}"] = post[name][pick]
            out[key] = {"L_nm": L * 1e9, "H_nm": H * 1e9, "h_nm": h,
                        "R0_nm": R0 * 1e9, "D_sim": D["mean"], "D_sim_sem": D["sem"],
                        "D_inf": di, "eta_m_1e-11": em,
                        "D_pbc_post": summary(post["D_pbc"] * um),
                        "R_post_nm": summary(post["R"] * 1e9),
                        "n_accepted": int(post["D_inf"].size),
                        "ratio_inf_over_sim": di["median"] / D["mean"]}
            print(f"{key:14s} {L*1e9:5.2f} {H*1e9:5.2f} {R0*1e9:5.3f}  "
                  f"{D['mean']:5.2f}+/-{D['sem']:4.2f}  "
                  f"{di['median']:6.1f} [{di['lo95']:5.1f}, {di['hi95']:6.1f}]  "
                  f"{em['median']:8.2f} [{em['lo95']:5.2f}, {em['hi95']:6.2f}]  "
                  f"{post['D_inf'].size}")
    path = DATA / "diffusion_psd.json"
    path.write_text(json.dumps(out, indent=1))
    np.savez_compressed(DATA / "diffusion_psd_posterior.npz", **samples)
    print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
