#!/usr/bin/env python3
r"""Supplementary (SI) table of the **equilibration analysis** for every
per-frame equilibrium observable: the pymbar-detected equilibration time
``t0``, the production-region **statistical inefficiency** ``g``, the **total
number of samples** (production frames) and the effective number of
uncorrelated samples ``N_eff`` -- one block per (lipid, force field), one row per observable, each
summarised as mean +/- SD across the three replicate simulations -- plus an
**averaging** column saying how the summary table's 95% CI (and so the
significance stars on the figures, which use the same SE and dof) was formed.

**The N_eff column shows the independent-sample count the CI actually used**,
which depends on that regime:

  * ``pooled`` -- the replicates agree within their own sampling noise, so each
    replicate's SEM is SD/sqrt(N_eff) and the dof is sum(N_eff) - 1: the
    column is that **sum** over the three replicates of N_eff = N/g.
  * ``replicate`` -- the replicate spread dominates, so the SE is the SD of the
    three replicate means / sqrt(3) on 2 dof and the within-run N_eff never
    enters it (it only decided the regime): the column is **3**.
  * the <delta theta^2> rows report the tilt modulus kappa_theta fitted on that
    series, whose within-run error is a block average: pooled, the column is
    the total number of blocks over the three replicates (``n_blocks`` in
    lipidator_blocks.json); replicate, 3.  There g sizes the blocks rather than giving N_eff.

The regimes come from ``make_equilibrium_table.regimes()``, i.e. exactly the
computation behind the summary table.

Source is ``data/equilibrium_timeseries_stats.json`` (written by ``stats.py``,
whose ``detect_equilibration`` picks ``t0 = argmax N_eff`` per Chodera 2016).
Frames are converted to ns at **dt = 0.2 ns/frame** (verified from the stored
MSD lag times).  Rows are APL, d_HH, d_HC and the lipid-tilt spread
<delta theta^2> (the per-frame OST quantity the tilt modulus chi is fitted
from); S_CC and registration are omitted -- neither is reported anywhere else.

    python3 analysis/make_equilibration_table.py
    ->  data/equilibration_summary_table.tex   (+ a text preview)

Requires numpy + stdlib.  Run ``stats.py`` first to (re)generate the JSON.
"""

import json
from pathlib import Path

import numpy as np

from lipiddefs import FF_LABELLED as FORCEFIELDS, LIPIDS
import make_equilibrium_table as met

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

NS_PER_FRAME = 0.2                       # verified from lipprops MSD lag times

# (stats.py observable key, display label)
OBS = [
    ("apl",          "APL"),
    ("thickness",    r"$d_{\mathrm{HH}}$"),
    ("dhc",          r"$d_{\mathrm{HC}}$"),
    ("tilt",         r"$\langle\delta\theta^{2}\rangle$"),
]

# summary-table column whose CI regime each row reports
REGIME_COL = {"apl": "apl", "thickness": "thick", "dhc": "dhc", "tilt": "kappa"}
REGIME_LABEL = {"pooled": "pooled", "between-run": "replicate"}

# plain-text row labels for the terminal preview (the LaTeX ones don't strip well)
PLAIN = {"apl": "APL", "thickness": "d_HH", "dhc": "d_HC", "tilt": "<dtheta^2>"}


def load_stats():
    p = DATA / "equilibrium_timeseries_stats.json"
    if not p.exists():
        raise SystemExit("missing %s -- run 'conda run -n base python "
                         "analysis/stats.py' first" % p.name)
    return json.loads(p.read_text())["per_replicate"]


def ms(vals, dec=0):
    """mean +/- SD (sample, ddof=1) of a list -> LaTeX string, or '--'."""
    v = [x for x in vals if x is not None and np.isfinite(x)]
    if not v:
        return "--"
    sd = np.std(v, ddof=1) if len(v) > 1 else 0.0
    return "%.*f $\\pm$ %.*f" % (dec, np.mean(v), dec, sd)


def load_blocks():
    p = DATA / "lipidator_blocks.json"
    return json.loads(p.read_text()) if p.exists() else {}


def used_n(regimes, blocks, ff, lipid, okey, neff):
    """The independent-sample count the summary table's CI used (see module
    docstring): 3 in the replicate regime, else per-replicate N_eff (or, for
    the tilt rows, kappa_theta's blocks per replicate).  Returns a list of
    per-replicate values, or None."""
    regime = regimes.get((ff, lipid, REGIME_COL[okey]), {}).get("regime")
    if regime == "between-run":
        return [3.0]
    if okey == "tilt":
        nb = [blocks.get("%s_%s_%d" % (ff, lipid, n), {}).get("n_blocks")
              for n in (1, 2, 3)]
        nb = [float(x) for x in nb if x is not None]
        return nb or None
    return list(neff)


def n_cell(vals):
    """3 for the replicate regime, else the sum over replicates (pooled)."""
    if not vals:
        return "--"
    return "3" if vals == [3.0] else "%d" % int(round(sum(vals)))


def rows_for(stats, ff, lipid, obs):
    """(t0_ns, g_frames, n_samples, neff) lists over reps, or None if absent."""
    sysn = "%s_%s" % (ff, lipid)
    recs = stats.get(obs, {}).get(sysn)
    if not recs:
        return None
    t0 = np.array([r["t0"] for r in recs], dtype=float)
    N = np.array([r["N"] for r in recs], dtype=float)
    neff = np.array([r["Neff_max"] for r in recs], dtype=float)
    g = np.array([r["g_prod"] for r in recs], dtype=float)
    return (t0 * NS_PER_FRAME, g, (N - t0), neff)


def regime_cell(regimes, ff, lipid, okey):
    r = regimes.get((ff, lipid, REGIME_COL[okey]))
    return REGIME_LABEL.get(r["regime"], "--") if r else "--"


def ratio_cell(regimes, ff, lipid, okey):
    """between_sd / within_sem, the quotient the regime is decided on."""
    r = regimes.get((ff, lipid, REGIME_COL[okey]))
    return "%.2f" % r["ratio"] if r and np.isfinite(r["ratio"]) else "--"


def build_latex(stats, regimes, blocks):
    L = []
    L.append("% Requires \\usepackage{booktabs,multirow,amsmath}")
    L.append("\\begin{table}[t]")
    L.append("\\centering")
    L.append("\\small")
    L.append("\\setlength{\\tabcolsep}{5pt}")
    L.append("\\begin{tabular}{lllcccclc}")
    L.append("\\toprule")
    L.append("Lipid & FF & Observable & $t_0$ & $g$ & $N$ "
             "& $\\sigma_\\mathrm{btw}/\\sigma_\\mathrm{wth}$ & Averaging "
             "& $N_\\mathrm{eff}$ \\\\")
    L.append(" & & & {\\footnotesize (ns)} & {\\footnotesize (frames)} "
             "& {\\footnotesize (frames)} & & & {\\footnotesize (indep.)} \\\\")
    L.append("\\midrule")
    for li, lipid in enumerate(LIPIDS):
        for fi, (ff, fflabel) in enumerate(FORCEFIELDS):
            for oi, (okey, olabel) in enumerate(OBS):
                r = rows_for(stats, ff, lipid, okey)
                if r is None:
                    cells = ["--"] * 3 + [ratio_cell(regimes, ff, lipid, okey),
                                          regime_cell(regimes, ff, lipid, okey),
                                          "--"]
                else:
                    t0, g, nsamp, neff = r
                    cells = [ms(t0), ms(g, 1), "%d" % int(round(np.sum(nsamp))),
                             ratio_cell(regimes, ff, lipid, okey),
                             regime_cell(regimes, ff, lipid, okey),
                             n_cell(used_n(regimes, blocks, ff, lipid, okey, neff))]
                lipcol = ("\\multirow{%d}{*}{%s}" % (len(FORCEFIELDS) * len(OBS), lipid)
                          if fi == 0 and oi == 0 else "")
                ffcol = ("\\multirow{%d}{*}{%s}" % (len(OBS), fflabel)
                         if oi == 0 else "")
                L.append(" & ".join([lipcol, ffcol, olabel] + cells) + " \\\\")
            if fi < len(FORCEFIELDS) - 1:
                L.append("\\cmidrule(l){2-9}")
        if li < len(LIPIDS) - 1:
            L.append("\\midrule")
    L.append("\\bottomrule")
    L.append("\\end{tabular}")
    L.append("\\caption{Equilibration analysis of equilibrium observables.  "
             "$t_0$ is the equilibration time discarded; $g$ is the statistical "
             "inefficiency; $N$ is the total number of production frames over "
             "the three replicates, after discarding those before $t_0$.  "
             "$\\sigma_\\mathrm{btw}/\\sigma_\\mathrm{wth}$ is the standard "
             "deviation of the three replicate means divided by the RMS of their "
             "within-run standard errors (SD$/\\sqrt{(T-t_0)/g}$ per replicate), "
             "which determines whether the runs sample the same distribution "
             "and can be pooled, or whether the difference between replicates "
             "is larger.  Below 1 the replicates are consistent and are "
             "\\emph{pooled}; at or above 1 they are averaged as "
             "\\emph{replicates}.  $N_\\mathrm{eff}$ is the number of "
             "independent samples used to compute the 95\\% confidence "
             "interval.  The $\\langle\\delta\\theta^{2}\\rangle$ rows are the "
             "per-frame spread of lipid tilt angles, whose $t_0$ and $g$ set "
             "the production span and block size used for the tilt modulus "
             "$\\kappa_\\theta$; their $\\sigma_\\mathrm{btw}/\\sigma_\\mathrm{wth}$, "
             "averaging and $N_\\mathrm{eff}$ refer to $\\kappa_\\theta$, whose "
             "within-run error is a block-averaged SEM, so when pooled "
             "$N_\\mathrm{eff}$ is the total number of blocks over the three "
             "replicates.}")
    L.append("\\label{tab:equilibration}")
    L.append("\\end{table}")
    return "\n".join(L) + "\n"


def preview(stats, regimes, blocks):
    hdr = ["lipid", "ff", "obs", "t0(ns)", "g(frames)", "N", "btw/wth",
           "averaging", "Neff"]
    print("  ".join("%-13s" % h for h in hdr))
    for lipid in LIPIDS:
        for ff, fflabel in FORCEFIELDS:
            for okey, olabel in OBS:
                r = rows_for(stats, ff, lipid, okey)
                lab = PLAIN.get(okey, okey)
                if r is None:
                    vals = ["--"] * 3 + [ratio_cell(regimes, ff, lipid, okey),
                                         regime_cell(regimes, ff, lipid, okey),
                                         "--"]
                else:
                    t0, g, nsamp, neff = r
                    vals = ["%.0f±%.0f" % (np.mean(t0), np.std(t0, ddof=1)),
                            "%.1f±%.1f" % (np.mean(g), np.std(g, ddof=1)),
                            "%d" % int(round(np.sum(nsamp))),
                            ratio_cell(regimes, ff, lipid, okey),
                            regime_cell(regimes, ff, lipid, okey),
                            n_cell(used_n(regimes, blocks, ff, lipid, okey, neff))]
                print("  ".join(["%-13s" % lipid, "%-13s" % fflabel, "%-13s" % lab]
                                + ["%-13s" % v for v in vals]))


def main():
    stats = load_stats()
    regimes = met.regimes()
    blocks = load_blocks()
    preview(stats, regimes, blocks)
    out = DATA / "equilibration_summary_table.tex"
    out.write_text(build_latex(stats, regimes, blocks))
    print("\nwrote %s" % out)


if __name__ == "__main__":
    main()
