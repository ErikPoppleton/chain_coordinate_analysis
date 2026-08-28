#!/usr/bin/env python3
r"""Autocorrelation diagnostics for the **equilibrium** per-frame timeseries.

For each force field / lipid / replicate, and for each per-frame observable, this
computes the **statistical inefficiency** ``g`` and the maximum effective number
of uncorrelated samples ``Neff_max`` via **pymbar** (``pymbar.timeseries``):
``statistical_inefficiency`` for the whole-series ``g_full`` and
``detect_equilibration`` (Chodera's automatic-equilibration estimator) for
``t0`` / ``g_prod`` / ``Neff_max``.

The reported ``g`` column is the **production-region** ``g_prod`` (estimated over
the equilibrated span ``[t0, T]``), per Chodera 2016 -- ``g`` and ``sigma^2`` are
defined on the production portion, and this keeps ``g`` consistent with the
``Neff_max = (T - t0 + 1) / g_prod`` beside it.  ``g_full`` (the whole series,
inflated by any pre-equilibration transient) is retained per-replicate in the
JSON as a diagnostic only -- when ``t0`` is large (a still-drifting run) the two
diverge sharply (e.g. AMBER DMPG APL: g_full ~136 vs g_prod ~18).

Background: MD frames are autocorrelated, so the raw frame count overstates how
much independent information a trajectory carries.  The statistical inefficiency
``g = 1 + 2*tau_int`` (in frames) is the spacing between effectively-independent
samples; ``Neff = N/g``.  ``detect_equilibration`` additionally discards an
optimal equilibration prefix ``t0`` that *maximises* Neff, giving ``Neff_max``.
These are the numbers that tell you, per observable, whether the run-to-run
spread could plausibly be pure within-run sampling noise (``Neff_max`` large) or
must reflect real between-run variance (``Neff_max`` small).

Per-frame observables (the ones with a genuine time axis in ``data/lipprops_*.npz``):
  * ``apl``           -- area per lipid, mean over lipids per frame (A^2)
  * ``thickness``     -- P-P bilayer thickness per frame (A)
  * ``registration``  -- inter-leaflet registration per frame
  * ``scc``           -- chain-averaged S_CC per frame (mean over both tails,
                         all chains, all carbons)
  * ``tilt``          -- per-frame lipid-tilt spread <delta theta^2> (rad^2), the
                         quantity the tilt modulus chi is fitted from (read from
                         the OST pipeline's ``data/tilt_series_{ff}_{lipid}_{rep}.npz``)
``lateral_diffusion`` is excluded: D is one fitted scalar per run and the MSD is
a cumulative (nonstationary) curve, neither of which is a stationary timeseries.

Needs pymbar (installed in the base conda env), so run it there:

    conda run -n base python analysis/stats.py

Prints a per-observable table (mean +/- SD over replicates, plus the worst-case
min Neff_max) and writes the full per-replicate numbers to
``data/equilibrium_timeseries_stats.json``.
"""

import json
from pathlib import Path

import numpy as np
from pymbar import timeseries as pymbar_ts

import autocorr
from lipiddefs import FORCEFIELDS, LIPIDS, REPS

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


# --- statistical inefficiency -------------------------------------------------
# g comes from the shared ``autocorr`` engine (pure numpy) so the SAME estimator
# is used here and in the OST-python-2.7 tilt pipeline; it is bit-identical to
# ``pymbar.timeseries.statistical_inefficiency(fft=True)`` on the project's
# series (validated).  pymbar is retained only for ``detect_equilibration`` (the
# automatic-equilibration t0 scan), which has no pure-numpy equivalent here.

def statistical_inefficiency(A):
    """Statistical inefficiency g of A (shared ``autocorr`` engine)."""
    return float(autocorr.statistical_inefficiency(A))


def detect_equilibration(A, nskip=1):
    """(t0, g_prod, Neff_max) from pymbar's automatic-equilibration detector.

    pymbar picks t0 = argmax Neff(t0); we re-derive g_prod/Neff_max from the
    shared ``autocorr`` engine on the equilibrated span so every reported g is
    from one estimator (the two agree to <1% -- pymbar's own g is kept as the
    t0-selection driver only)."""
    A = np.asarray(A, dtype=float)
    t0, _g_pymbar, _neff_pymbar = pymbar_ts.detect_equilibration(A, nskip=nskip)
    t0 = int(t0)
    prod = A[t0:]
    g_prod = float(autocorr.statistical_inefficiency(prod)) if prod.size > 1 else 1.0
    neff_max = prod.size / g_prod
    return t0, g_prod, float(neff_max)


def analyze_series(A, nskip=None):
    """Full autocorrelation analysis of one per-frame observable series.

    Importable entry point (so table/plot scripts get the equilibrated mean, g
    and its uncertainty in-process, without a separate ``stats.py`` run).
    Returns a dict:
      ``N``          -- number of frames
      ``t0``         -- equilibration cutoff (frames discarded), pymbar eq. 14
      ``g_full``     -- statistical inefficiency of the whole series (diagnostic)
      ``g_prod``     -- statistical inefficiency of the equilibrated span [t0:]
      ``Neff_full`` / ``Neff_max`` -- N/g_full and (N-t0)/g_prod
      ``mean_prod``  -- mean over the equilibrated span (the point estimate)
      ``sd_prod``    -- frame-level SD over [t0:]
      ``sem_within`` -- SD_prod / sqrt(Neff_max), the within-run SEM of the mean
    """
    A = np.asarray(A, dtype=float).ravel()
    N = int(A.size)
    if nskip is None:
        nskip = max(1, N // 250)                    # ~250-point t0 scan
    t0, g_prod, neff_max = detect_equilibration(A, nskip=nskip)
    g_full = statistical_inefficiency(A)
    prod = A[t0:]
    mean_prod = float(prod.mean())
    sd_prod = float(prod.std(ddof=1)) if prod.size > 1 else 0.0
    sem_within = sd_prod / np.sqrt(neff_max) if neff_max > 0 else float("nan")
    return {"N": N, "t0": int(t0),
            "g_full": float(g_full), "Neff_full": float(N / g_full),
            "g_prod": float(g_prod), "Neff_max": float(neff_max),
            "mean_prod": mean_prod, "sd_prod": sd_prod,
            "sem_within": float(sem_within)}


# --- per-frame observable series ---------------------------------------------

def series_for(npzs, ff, lipid, rep):
    """Return {observable: 1-D per-frame series} for one (ff, lipid, rep)."""
    sys = f"{ff}_{lipid}"
    key = f"{sys}_eq{rep}"
    out = {}
    apl = npzs["area_per_lipid"]
    if key in apl.files:
        out["apl"] = apl[key].mean(axis=1)            # (n_frames, n_lipids) -> per frame
    th = npzs["memb_thickness"]
    if key in th.files:
        out["thickness"] = np.asarray(th[key])        # already per frame
    reg = npzs["registration"]
    if key in reg.files:
        out["registration"] = np.asarray(reg[key])
    op = npzs["order_parameter"]
    if key in op.files and "apl" in out:
        F = out["apl"].shape[0]                        # frame count from APL
        arr = op[key]                                  # (2*F*m, n_carbons), [tail, frame, chain]
        nc = arr.shape[1]
        m = arr.shape[0] // (2 * F)
        if 2 * F * m == arr.shape[0]:
            out["scc"] = arr.reshape(2, F, m, nc).mean(axis=(0, 2, 3))
    tilt = DATA / ("tilt_series_%s_%d.npz" % (sys, rep))
    if tilt.exists():                                  # OST per-frame tilt spread
        with np.load(tilt) as tz:
            if "theta_var" in tz.files:
                out["tilt"] = np.asarray(tz["theta_var"], dtype=float)
    ct = npzs.get("core_thickness")                    # optional: d_HC per frame (nm)
    if ct is not None:
        ckey = "%s_lipy_rep%d" % (sys, rep)            # note: _lipy_rep, not _eq
        if ckey in ct.files:
            out["dhc"] = np.asarray(ct[ckey], dtype=float)
    return out


OBSERVABLES = ["apl", "thickness", "dhc", "tilt", "registration", "scc"]
UNITS = {"apl": "A^2", "thickness": "A", "dhc": "nm", "tilt": "rad^2",
         "registration": "", "scc": ""}


def decompose(rows):
    """Within-run vs between-run variance for one system's replicate rows.

    Each run's mean has a within-run sampling error SEM_r = SD_frame / sqrt(Neff_max)
    (variance of the mean of a correlated series).  Across replicates:
      * between_sd  = sample SD (ddof=1) of the per-run means -- the replicate spread,
      * within_sem  = RMS of the per-run SEMs -- the scatter expected from sampling
                      noise alone if the runs were independent well-sampled draws.
    Their ratio decides interpretation: ~1 => the replicate spread IS just sampling
    noise (pooling to ~n_rep*Neff is defensible); >>1 => genuine between-run variance
    (keep n=3).  sigma_between = sqrt(max(0, between_sd^2 - within_sem^2)) is the
    variance-components estimate of the true run-to-run SD."""
    means = np.array([r["mean_prod"] for r in rows])
    sems = np.array([r["sem_within"] for r in rows])
    neff = np.array([r["Neff_max"] for r in rows])
    n = len(rows)
    between_sd = float(means.std(ddof=1)) if n > 1 else 0.0
    within_sem = float(np.sqrt(np.mean(sems ** 2)))
    ratio = between_sd / within_sem if within_sem > 0 else float("inf")
    sigma_between = float(np.sqrt(max(0.0, between_sd ** 2 - within_sem ** 2)))
    if ratio < 1.5:
        verdict = "poolable"
    elif ratio < 3.0:
        verdict = "borderline"
    else:
        verdict = "between-run"
    return {"between_sd": between_sd, "within_sem": within_sem, "ratio": ratio,
            "sigma_between": sigma_between, "verdict": verdict,
            "pooled_n": int(round(n * neff.mean()))}


def main():
    npzs = {}
    for name in ["area_per_lipid", "memb_thickness", "registration", "order_parameter"]:
        p = DATA / f"lipprops_{name}.npz"
        if not p.exists():
            raise SystemExit(f"missing {p}")
        npzs[name] = np.load(p)
    ctp = DATA / "core_thickness.npz"                  # optional: adds the d_HC observable
    if ctp.exists():
        npzs["core_thickness"] = np.load(ctp)
    else:
        print(f"note: {ctp.name} absent -> no d_HC equilibration stats")

    # per-replicate results: results[obs][system] = list of dicts
    results = {obs: {} for obs in OBSERVABLES}
    for ff in FORCEFIELDS:
        for lipid in LIPIDS:
            sys = f"{ff}_{lipid}"
            for rep in REPS:
                series = series_for(npzs, ff, lipid, rep)
                for obs, A in series.items():
                    row = analyze_series(A)
                    row["rep"] = rep
                    results[obs].setdefault(sys, []).append(row)

    # --- report ---
    for obs in OBSERVABLES:
        if not results[obs]:
            continue
        print(f"\n=== {obs}  (g in frames; Neff_max = max effective uncorrelated samples) ===")
        print(f"  {'system':16s} {'N':>5s} {'g_prod (mean±SD)':>18s} "
              f"{'Neff_max (mean±SD)':>20s} {'min Neff_max':>13s}")
        all_g, all_neff = [], []
        for ff in FORCEFIELDS:
            for lipid in LIPIDS:
                sys = f"{ff}_{lipid}"
                rows = results[obs].get(sys)
                if not rows:
                    continue
                # Report the *production-region* g (over [t0, T]), consistent with
                # Neff_max and with Chodera 2016 (g and sigma^2 are defined on the
                # equilibrated portion).  g_full (whole series, transient-inflated)
                # is kept per-replicate in the JSON as a diagnostic only.
                g = np.array([r["g_prod"] for r in rows])
                nf = np.array([r["Neff_max"] for r in rows])
                N = rows[0]["N"]
                all_g += list(g); all_neff += list(nf)
                print(f"  {sys:16s} {N:5d} {g.mean():7.1f} ± {g.std(ddof=0):5.1f} "
                      f"{nf.mean():10.1f} ± {nf.std(ddof=0):6.1f} {nf.min():13.1f}")
        g = np.array(all_g); nf = np.array(all_neff)
        print(f"  {'-- overall --':16s} {'':5s} {g.mean():7.1f} ± {g.std(ddof=0):5.1f} "
              f"{nf.mean():10.1f} ± {nf.std(ddof=0):6.1f} {nf.min():13.1f}")

    # --- variance decomposition: within-run sampling noise vs between-run spread ---
    print("\n\n########  WITHIN-RUN vs BETWEEN-RUN  ########")
    print("ratio = (between-run SD of the 3 replicate means) / (RMS within-run SEM).")
    print("  ~1  -> replicate scatter is just sampling noise; pooling to ~3*Neff is defensible.")
    print("  >>1 -> genuine between-run variance; keep n=3.   (heuristic; n=3, so ratio is noisy.)")
    decomp = {obs: {} for obs in OBSERVABLES}
    for obs in OBSERVABLES:
        if not results[obs]:
            continue
        unit = f" ({UNITS[obs]})" if UNITS[obs] else ""
        print(f"\n=== {obs}{unit} ===")
        print(f"  {'system':16s} {'between-SD':>10s} {'within-SEM':>10s} "
              f"{'ratio':>6s} {'sigma_betw':>10s} {'~pooled n':>9s}  verdict")
        for ff in FORCEFIELDS:
            for lipid in LIPIDS:
                sys = f"{ff}_{lipid}"
                rows = results[obs].get(sys)
                if not rows:
                    continue
                d = decompose(rows)
                decomp[obs][sys] = d
                print(f"  {sys:16s} {d['between_sd']:10.4g} {d['within_sem']:10.4g} "
                      f"{d['ratio']:6.2f} {d['sigma_between']:10.4g} {d['pooled_n']:9d}"
                      f"  {d['verdict']}")

    out = DATA / "equilibrium_timeseries_stats.json"
    out.write_text(json.dumps({"per_replicate": results, "decomposition": decomp},
                              indent=2, sort_keys=True))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
