#!/usr/bin/env python3
r"""Supplementary (SI) table of the **equilibration analysis** for every
per-frame equilibrium observable: the pymbar-detected equilibration time
``t0``, the resulting production **window size**, the **total number of samples**
(production frames) and the effective number of uncorrelated samples
``N_eff`` -- one block per (lipid, force field), one row per observable, each
summarised as mean +/- SD across the three replicate simulations.

Source is ``data/equilibrium_timeseries_stats.json`` (written by ``stats.py``,
whose ``detect_equilibration`` picks ``t0 = argmax N_eff`` per Chodera 2016).
Frames are converted to ns at **dt = 0.2 ns/frame** (verified from the stored
MSD lag times).  Note the total trajectory differs by observable: APL, d_HH,
S_CC and registration are the full **500 ns** equilibrium runs (2501 frames),
while d_HC is computed on the **last 250 ns** (1251 frames, core_thickness).

    python3 analysis/make_equilibration_table.py
    ->  data/equilibration_summary_table.tex   (+ a text preview)

Requires numpy + stdlib.  Run ``stats.py`` first to (re)generate the JSON.
"""

import json
from pathlib import Path

import numpy as np

from lipiddefs import FF_LABELLED as FORCEFIELDS, LIPIDS

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

NS_PER_FRAME = 0.2                       # verified from lipprops MSD lag times

# (stats.py observable key, display label)
OBS = [
    ("apl",          "APL"),
    ("thickness",    r"$d_{\mathrm{HH}}$"),
    ("dhc",          r"$d_{\mathrm{HC}}$"),
    ("scc",          r"$S_{CC}$"),
    ("registration", "Reg."),
]


def total_ns(stats, okey):
    """Full trajectory length (ns) analysed for observable ``okey`` = the max
    frame count over all systems x reps x dt (data-driven, so the table stays
    correct if a source trajectory's length changes -- e.g. d_HC recomputed on
    the full 500 ns)."""
    Ns = [r["N"] for recs in stats.get(okey, {}).values() for r in recs]
    return max(Ns) * NS_PER_FRAME if Ns else None


def load_stats():
    p = DATA / "equilibrium_timeseries_stats.json"
    if not p.exists():
        raise SystemExit("missing %s -- run 'conda run -n base python "
                         "analysis/stats.py' first" % p.name)
    return json.loads(p.read_text())["per_replicate"]


def ms(vals, dec=0):
    """mean +/- SD (population) of a list -> LaTeX string, or '--' if empty."""
    v = [x for x in vals if x is not None and np.isfinite(x)]
    if not v:
        return "--"
    return "%.*f $\\pm$ %.*f" % (dec, np.mean(v), dec, np.std(v))


def rows_for(stats, ff, lipid, obs):
    """(t0_ns, window_ns, n_samples, neff) lists over reps, or None if absent."""
    sysn = "%s_%s" % (ff, lipid)
    recs = stats.get(obs, {}).get(sysn)
    if not recs:
        return None
    t0 = np.array([r["t0"] for r in recs], dtype=float)
    N = np.array([r["N"] for r in recs], dtype=float)
    neff = np.array([r["Neff_max"] for r in recs], dtype=float)
    return (t0 * NS_PER_FRAME, (N - t0) * NS_PER_FRAME, (N - t0), neff)


def traj_len_sentence(stats):
    """Caption sentence stating each observable's analysed trajectory length,
    grouped -- data-driven so it self-corrects if a source length changes."""
    from collections import defaultdict
    groups = defaultdict(list)
    for okey, olabel in OBS:
        T = total_ns(stats, okey)
        if T is not None:
            groups[round(T)].append(olabel)
    if len(groups) == 1:
        return "All observables span the full $%g$\\,ns runs." % list(groups)[0]
    parts = ["%s span $%g$\\,ns" % (", ".join(labs), T)
             for T, labs in sorted(groups.items(), reverse=True)]
    return "; ".join(parts) + "."


def build_latex(stats):
    L = []
    L.append("% Requires \\usepackage{booktabs,multirow,amsmath}")
    L.append("\\begin{table}[t]")
    L.append("\\centering")
    L.append("\\small")
    L.append("\\setlength{\\tabcolsep}{5pt}")
    L.append("\\begin{tabular}{lllcccc}")
    L.append("\\toprule")
    L.append("Lipid & FF & Observable & $t_0$ & Window & $N$ & $N_\\mathrm{eff}$ \\\\")
    L.append(" & & & {\\footnotesize (ns)} & {\\footnotesize (ns)} "
             "& {\\footnotesize (frames)} & {\\footnotesize (uncorr.)} \\\\")
    L.append("\\midrule")
    for li, lipid in enumerate(LIPIDS):
        for fi, (ff, fflabel) in enumerate(FORCEFIELDS):
            for oi, (okey, olabel) in enumerate(OBS):
                r = rows_for(stats, ff, lipid, okey)
                if r is None:
                    cells = ["--"] * 4
                else:
                    t0, win, nsamp, neff = r
                    cells = [ms(t0), ms(win), ms(nsamp), ms(neff)]
                lipcol = ("\\multirow{%d}{*}{%s}" % (len(FORCEFIELDS) * len(OBS), lipid)
                          if fi == 0 and oi == 0 else "")
                ffcol = ("\\multirow{%d}{*}{%s}" % (len(OBS), fflabel)
                         if oi == 0 else "")
                L.append(" & ".join([lipcol, ffcol, olabel] + cells) + " \\\\")
            if fi < len(FORCEFIELDS) - 1:
                L.append("\\cmidrule(l){2-7}")
        if li < len(LIPIDS) - 1:
            L.append("\\midrule")
    L.append("\\bottomrule")
    L.append("\\end{tabular}")
    L.append("\\caption{Equilibration analysis of the per-frame equilibrium "
             "observables (mean $\\pm$ SD over three replicates).  $t_0$ is the "
             "equilibration time discarded (pymbar automatic detection, "
             "$t_0=\\arg\\max N_\\mathrm{eff}$\\cite{chodera2016simple}); Window "
             "$=T-t_0$ is the production span used for the reported means; $N$ is "
             "the number of production frames and $N_\\mathrm{eff}=(T-t_0)/g$ the "
             "effective number of uncorrelated samples (statistical inefficiency "
             "$g$).  Frames $\\to$ ns at $0.2$\\,ns/frame.  " + traj_len_sentence(stats) +
             "  The DMPG bilayers (especially AMBER) show long, replicate-variable "
             "$t_0$ from slow acyl-chain ordering, leaving few uncorrelated samples.}")
    L.append("\\label{tab:equilibration}")
    L.append("\\end{table}")
    return "\n".join(L) + "\n"


def preview(stats):
    hdr = ["lipid", "ff", "obs", "t0(ns)", "window(ns)", "N", "Neff"]
    print("  ".join("%-13s" % h for h in hdr))
    for lipid in LIPIDS:
        for ff, fflabel in FORCEFIELDS:
            for okey, olabel in OBS:
                r = rows_for(stats, ff, lipid, okey)
                lab = olabel.replace("$", "").replace("\\mathrm", "").replace("{", "").replace("}", "").replace("_", "")
                if r is None:
                    vals = ["--"] * 4
                else:
                    t0, win, nsamp, neff = r
                    vals = ["%.0f±%.0f" % (np.mean(t0), np.std(t0)),
                            "%.0f±%.0f" % (np.mean(win), np.std(win)),
                            "%.0f" % np.mean(nsamp),
                            "%.0f±%.0f" % (np.mean(neff), np.std(neff))]
                print("  ".join(["%-13s" % lipid, "%-13s" % fflabel, "%-13s" % lab]
                                + ["%-13s" % v for v in vals]))


def main():
    stats = load_stats()
    preview(stats)
    out = DATA / "equilibration_summary_table.tex"
    out.write_text(build_latex(stats))
    print("\nwrote %s" % out)


if __name__ == "__main__":
    main()
