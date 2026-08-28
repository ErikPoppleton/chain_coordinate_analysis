# -*- coding: utf-8 -*-
"""Dependency-light autocorrelation / block-averaging primitives, shared by the
equilibrium-statistics tooling.

This module is deliberately **pure numpy and Python 2/3 compatible** (no
f-strings, no pymbar): it is imported both by ``stats.py`` (Python 3, base env)
and by ``tilt_modulus.py``, which runs under the OpenStructure **Python 2.7**
interpreter where pymbar is unavailable.  Keeping the core estimator here means
the tilt pipeline can compute a statistical inefficiency (hence a block size for
its within-run chi error) in the same run, without shelling out to ``stats.py``.

``statistical_inefficiency`` reproduces pymbar's definition ``g = 1 + 2*tau_ac``
via the normalised fluctuation autocorrelation summed with the initial-positive
truncation (Chodera 2007; the non-FFT path).  It is validated to agree with
``pymbar.timeseries.statistical_inefficiency(fft=True)`` to within a few percent
on the project's equilibrium series (see ``validate_against_pymbar`` in the
stats-module tests) -- close enough that a block size derived from it matches
what pymbar would choose.
"""

import math

import numpy as np


def statistical_inefficiency(A, mintime=3):
    """Statistical inefficiency ``g = 1 + 2*tau_ac`` (in frames) of a 1-D series.

    ``g`` is the number of correlated samples per effectively-independent one;
    ``Neff = len(A) / g``.  Uses the normalised fluctuation autocorrelation
    ``C(t)`` accumulated over increasing lag, truncated at the first
    non-positive ``C(t)`` past ``mintime`` (the initial-positive-sequence rule
    that keeps the noisy large-lag tail from swamping the sum).
    """
    A = np.asarray(A, dtype=float).ravel()
    N = A.size
    if N < 2:
        return 1.0
    dA = A - A.mean()
    sigma2 = float(np.mean(dA * dA))
    if sigma2 == 0.0:            # a constant series carries no correlation
        return 1.0
    g = 1.0
    t = 1
    while t < N - 1:
        C = float(np.mean(dA[0:N - t] * dA[t:N])) / sigma2
        if C <= 0.0 and t > mintime:
            break
        g += 2.0 * C * (1.0 - float(t) / N)
        t += 1
    return g if g >= 1.0 else 1.0


def neff(A, mintime=3):
    """Effective number of uncorrelated samples ``len(A) / g``."""
    A = np.asarray(A, dtype=float).ravel()
    return A.size / statistical_inefficiency(A, mintime=mintime)


def choose_n_blocks(n_frames, g, min_blocks=5, max_blocks=20):
    """Pick a block count for block-averaging a series of ``n_frames``.

    Targets blocks at least as long as the statistical inefficiency ``g`` (so
    successive blocks are ~uncorrelated), i.e. ``n_blocks ~ n_frames / g``,
    clamped to ``[min_blocks, max_blocks]``.  Returns ``(n_blocks, block_size)``
    with contiguous equal-size blocks (any trailing remainder is dropped by the
    caller via :func:`block_ranges`).  ``block_size < g`` (flagged by the caller)
    means the run is too short/too correlated for the requested block count and
    the resulting within-run error is optimistic.
    """
    g = max(float(g), 1.0)
    nb = int(n_frames // g)
    nb = max(min_blocks, min(max_blocks, nb))
    nb = max(1, min(nb, n_frames))          # never more blocks than frames
    block_size = int(n_frames // nb)
    return nb, block_size


def block_ranges(n_frames, block_size, n_blocks):
    """Contiguous ``[start, stop)`` frame ranges for ``n_blocks`` equal blocks of
    ``block_size`` frames (trailing remainder dropped)."""
    return [(i * block_size, (i + 1) * block_size) for i in range(n_blocks)]


def block_sem(values, ddof=1):
    """Standard error of the mean from block estimates: ``SD(values)/sqrt(n)``.

    ``values`` are one summary statistic per (approximately independent) block
    -- e.g. a chi refit per block.  With correlated blocks this underestimates
    the error, which is why the block size is chosen >= g.
    """
    v = np.asarray(values, dtype=float).ravel()
    n = v.size
    if n < 2:
        return float("nan")
    return float(v.std(ddof=ddof) / np.sqrt(n))


# --- Student-t critical value (two-sided 95%) --------------------------------
# Hardcoded t_{0.975, dof} so no scipy dependency is pulled into the numpy-only
# table / plot scripts.  Interpolated in dof; -> 1.96 asymptotically.
_T95_DOF = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 24, 30, 40, 60,
                     120, 1e9], dtype=float)
_T95_VAL = np.array([12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306,
                     2.262, 2.228, 2.179, 2.131, 2.086, 2.064, 2.042, 2.021,
                     2.000, 1.980, 1.960], dtype=float)


def t95(dof):
    """Two-sided 95% Student-t critical value for ``dof`` degrees of freedom."""
    return float(np.interp(max(1.0, float(dof)), _T95_DOF, _T95_VAL))


def hierarchical_ci_from_estimates(means, sems, neffs=None):
    """Hierarchical 95% CI from **pre-computed** per-replicate point estimates
    and within-run SEMs -- for fitted quantities (the tilt moduli chi / kappa)
    where each replicate already yields a scalar plus a block-averaged within-run
    SEM, not a raw per-frame series.  Same variance model as
    :func:`hierarchical_ci`: ``SE(grand)^2 = max(between_sd^2, within_sem^2)/n``,
    t on ``n-1`` d.o.f. when the replicate spread dominates else on the pooled
    ``neffs`` (per-rep block counts; defaults to a large d.o.f. -> t~1.96).
    Returns the same dict shape, or ``None`` if no usable replicate."""
    pairs = [(float(mu), float(se), (neffs[i] if neffs is not None and i < len(neffs) else None))
             for i, (mu, se) in enumerate(zip(means, sems))
             if mu is not None and np.isfinite(mu)]
    if not pairs:
        return None
    m = np.array([p[0] for p in pairs])
    s = np.array([p[1] for p in pairs])
    ne = [p[2] for p in pairs]
    n = len(m)
    grand = float(m.mean())
    if n == 1:
        within_sem = float(s[0]) if np.isfinite(s[0]) else 0.0
        dof = max(1.0, (ne[0] - 1)) if ne[0] else 1e6
        t = t95(dof)
        return {"mean": grand, "ci_half": t * within_sem, "se": within_sem,
                "t": t, "dof": dof, "between_sd": 0.0, "within_sem": within_sem,
                "sigma_between": 0.0, "regime": "single", "n": 1}
    between_sd = float(m.std(ddof=1))
    with np.errstate(invalid="ignore"):
        within_sem = float(np.sqrt(np.nanmean(s ** 2)))
    if not np.isfinite(within_sem):          # no within-run SEMs -> SD-only CI
        within_sem = 0.0
    se = float(np.sqrt(max(between_sd ** 2, within_sem ** 2) / n))
    if between_sd >= within_sem:
        regime, dof = "between-run", float(n - 1)
    else:
        regime = "pooled"
        dof = (max(1.0, float(np.nansum([x for x in ne if x])) - 1.0)
               if any(ne) else 1e6)
    t = t95(dof)
    sigma_between = float(np.sqrt(max(0.0, between_sd ** 2 - within_sem ** 2)))
    return {"mean": grand, "ci_half": t * se, "se": se, "t": t, "dof": dof,
            "between_sd": between_sd, "within_sem": within_sem,
            "sigma_between": sigma_between, "regime": regime, "n": n}


def hierarchical_ci(series, t0s=None):
    """Point estimate + **hierarchical 95% CI** for a per-frame observable
    measured over several replicate runs.

    ``series`` is a list of 1-D per-frame arrays (one per replicate; ``None`` /
    empty entries are skipped).  ``t0s`` (optional) is a matching list of
    equilibration cutoffs (frames to discard as burn-in, e.g. pymbar's
    ``detect_equilibration`` output from ``stats.py``): each replicate is
    restricted to its production portion ``a[t0:]`` **before** the mean and the
    statistical inefficiency are estimated, per Chodera 2016 (g and sigma^2 are
    defined on the production span, not the full series -- this both removes the
    burn-in bias in the mean and gives the post-t0 g/Neff).  Each replicate is
    reduced to its production mean and an
    autocorrelation-limited within-run SEM ``SD_frame/sqrt(Neff)`` (``Neff =
    N/g``, the statistical inefficiency of *this* run).  Across replicates the
    grand mean's variance is modelled two ways and the larger is taken::

        SE(grand)^2 = max(between_sd^2, within_sem^2) / n

    where ``between_sd`` is the SD of the per-run means (the replicate spread)
    and ``within_sem`` the RMS of the per-run SEMs (the sampling-noise floor).
    This collapses to the pooled pseudoreplication limit when the replicates
    agree (``within`` dominates -> t ~ 1.96 on the pooled effective d.o.f.) and
    to the honest replicate-spread CI when they do not (``between`` dominates
    -> t on ``n-1`` d.o.f.), so between-run systems (e.g. AMBER DMPG) are not
    falsely tightened.  Returns a dict (or ``None`` if no usable replicate):
    ``mean``, ``ci_half`` (95% half-width), ``se``, ``t``, ``dof``,
    ``between_sd``, ``within_sem``, ``sigma_between``, ``regime``, ``n``.
    """
    reps = []
    for i, s in enumerate(series):
        if s is None:
            continue
        a = np.asarray(s, dtype=float).ravel()
        t0 = int(t0s[i]) if t0s is not None and i < len(t0s) and t0s[i] else 0
        a = a[t0:]                               # keep only the production span
        if a.size < 2:
            continue
        g = statistical_inefficiency(a)
        ne = a.size / g
        reps.append({"mean": float(a.mean()), "neff": float(ne), "t0": t0,
                     "sem": float(a.std(ddof=1) / np.sqrt(ne)) if ne > 0 else float("nan")})
    n = len(reps)
    if n == 0:
        return None
    means = np.array([r["mean"] for r in reps])
    sems = np.array([r["sem"] for r in reps])
    grand = float(means.mean())
    if n == 1:                                   # one run: its own within-run SEM
        between_sd, within_sem = 0.0, float(sems[0])
        se, dof, regime = within_sem, max(1.0, reps[0]["neff"] - 1), "single"
    else:
        between_sd = float(means.std(ddof=1))
        within_sem = float(np.sqrt(np.nanmean(sems ** 2)))
        se = float(np.sqrt(max(between_sd ** 2, within_sem ** 2) / n))
        if between_sd >= within_sem:
            regime, dof = "between-run", float(n - 1)
        else:
            regime = "pooled"
            dof = max(1.0, float(np.nansum([r["neff"] for r in reps])) - 1.0)
    t = t95(dof)
    sigma_between = (float(np.sqrt(max(0.0, between_sd ** 2 - within_sem ** 2)))
                     if n > 1 else 0.0)
    return {"mean": grand, "ci_half": t * se, "se": se, "t": t, "dof": dof,
            "between_sd": between_sd, "within_sem": within_sem,
            "sigma_between": sigma_between, "regime": regime, "n": n}


# ---------------------------------------------------------------------------
# Significance testing between two systems, on the same hierarchical variance
# model as the error bars.  A t-test over the raw frames would be
# pseudoreplication (MD frames are correlated and n=3 runs, not n=2500), so the
# comparison uses each system's hierarchical SE and d.o.f. from the dicts
# returned above, combined Welch-style.
# ---------------------------------------------------------------------------

def _betacf(a, b, x, itmax=200, eps=3e-16, fpmin=1e-300):
    """Continued fraction for the incomplete beta function (Lentz's method)."""
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def betai(a, b, x):
    """Regularised incomplete beta function I_x(a, b), pure numpy/stdlib."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
             + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(lbeta) * _betacf(a, b, x) / a
    return 1.0 - math.exp(lbeta) * _betacf(b, a, 1.0 - x) / b


def t_pvalue(t, dof):
    """Two-sided Student-t p-value (no scipy)."""
    dof = max(1.0, float(dof))
    return float(betai(0.5 * dof, 0.5, dof / (dof + float(t) ** 2)))


def compare(ci_a, ci_b):
    """Welch-style two-sided test between two hierarchical-CI results.

    ``ci_a``/``ci_b`` are the dicts from :func:`hierarchical_ci` or
    :func:`hierarchical_ci_from_estimates`; the test uses their hierarchical
    ``se`` (which already carries the max(between, within) variance choice) and
    their ``dof``, so a system whose replicates disagree is *not* credited with
    a significance it has not earned.  Returns ``{delta, se, t, dof, p}``.
    """
    if ci_a is None or ci_b is None:
        return None
    delta = ci_a["mean"] - ci_b["mean"]
    va, vb = ci_a["se"] ** 2, ci_b["se"] ** 2
    se = math.sqrt(va + vb)
    if se == 0.0:
        return {"delta": delta, "se": 0.0, "t": float("inf"), "dof": 1.0,
                "p": 0.0}
    t = delta / se
    dof = (va + vb) ** 2 / (va ** 2 / max(1.0, ci_a["dof"])
                            + vb ** 2 / max(1.0, ci_b["dof"]))
    return {"delta": delta, "se": se, "t": t, "dof": dof,
            "p": t_pvalue(t, dof)}


def stars(p, ns="n.s."):
    """Significance marker: *** p<0.0005, ** p<0.005, * p<0.05, else ``ns``."""
    if p is None:
        return ""
    for thr, mark in ((5e-4, "***"), (5e-3, "**"), (5e-2, "*")):
        if p < thr:
            return mark
    return ns
