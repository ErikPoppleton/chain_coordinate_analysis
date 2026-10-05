#!/usr/bin/env python3
r"""Supplementary (SI) table of the **equilibration analysis** for every
per-frame equilibrium observable: the pymbar-detected equilibration time
``t0``, the production-region **statistical inefficiency** ``g``, the **total
number of samples** (production frames) and the effective number of
uncorrelated samples ``N_eff`` -- one block per (lipid, force field), one row per observable, each
summarised as mean +/- SD across the three replicate simulations.

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

# plain-text row labels for the terminal preview (the LaTeX ones don't strip well)
PLAIN = {"apl": "APL", "thickness": "d_HH", "dhc": "d_HC", "tilt": "<dtheta^2>"}


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


QUICK_T0_NS = 40.0      # "equilibrates quickly" threshold for the caption


def burnin_sentence(stats):
    """Caption sentence summarising the burn-ins and the worst-sampled series.

    Computed from the JSON rather than written by hand: a hand-written version
    ("the DMPG bilayers, especially AMBER, show long t0 ... few uncorrelated
    samples") described the old Na+/Mg2+/Ca2+ DMPG runs and silently went stale
    when the potassium rebuild removed that behaviour.
    """
    series = []          # (t0_ns, neff, ff_label, lipid) per replicate x observable
    for lipid in LIPIDS:
        for ff, fflabel in FORCEFIELDS:
            for okey, _ in OBS:
                for r in stats.get(okey, {}).get("%s_%s" % (ff, lipid), []):
                    series.append((r["t0"] * NS_PER_FRAME, r["Neff_max"],
                                   fflabel, lipid))
    if not series:
        return ""
    quick = sum(t0 <= QUICK_T0_NS for t0, _, _, _ in series)
    # The longest burn-in per system, for the two systems with the longest.
    longest = {}
    for t0, _, fflabel, lipid in series:
        key = "%s %s" % (fflabel, lipid)
        longest[key] = max(longest.get(key, 0.0), t0)
    top = sorted(longest.items(), key=lambda kv: -kv[1])[:2]
    min_neff = min(neff for _, neff, _, _ in series)
    return ("  Most series equilibrate quickly ($t_0\\le%g$\\,ns for %d of %d "
            "replicate--observable series); the longest burn-ins are single "
            "replicates (%s), and every series retains "
            "$N_\\mathrm{eff}\\ge%d$ uncorrelated samples."
            % (QUICK_T0_NS, quick, len(series),
               "; ".join("%s, $\\approx%.0f$\\,ns" % (k, round(v, -1))
                         for k, v in top),
               int(np.floor(min_neff))))


def build_latex(stats):
    L = []
    L.append("% Requires \\usepackage{booktabs,multirow,amsmath}")
    L.append("\\begin{table}[t]")
    L.append("\\centering")
    L.append("\\small")
    L.append("\\setlength{\\tabcolsep}{5pt}")
    L.append("\\begin{tabular}{lllcccc}")
    L.append("\\toprule")
    L.append("Lipid & FF & Observable & $t_0$ & $g$ & $N$ & $N_\\mathrm{eff}$ \\\\")
    L.append(" & & & {\\footnotesize (ns)} & {\\footnotesize (frames)} "
             "& {\\footnotesize (frames)} & {\\footnotesize (uncorr.)} \\\\")
    L.append("\\midrule")
    for li, lipid in enumerate(LIPIDS):
        for fi, (ff, fflabel) in enumerate(FORCEFIELDS):
            for oi, (okey, olabel) in enumerate(OBS):
                r = rows_for(stats, ff, lipid, okey)
                if r is None:
                    cells = ["--"] * 4
                else:
                    t0, g, nsamp, neff = r
                    cells = [ms(t0), ms(g, 1), ms(nsamp), ms(neff)]
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
             "$t_0=\\arg\\max N_\\mathrm{eff}$\\cite{chodera2016simple}); $g$ is the "
             "statistical inefficiency of the production span $[t_0,T]$; $N$ is "
             "the number of production frames and $N_\\mathrm{eff}=(T-t_0)/g$ the "
             "effective number of uncorrelated samples.  Frames $\\to$ ns at $0.2$\\,ns/frame.  " + traj_len_sentence(stats) +
             burnin_sentence(stats) + "}")
    L.append("\\label{tab:equilibration}")
    L.append("\\end{table}")
    return "\n".join(L) + "\n"


def preview(stats):
    hdr = ["lipid", "ff", "obs", "t0(ns)", "g(frames)", "N", "Neff"]
    print("  ".join("%-13s" % h for h in hdr))
    for lipid in LIPIDS:
        for ff, fflabel in FORCEFIELDS:
            for okey, olabel in OBS:
                r = rows_for(stats, ff, lipid, okey)
                lab = PLAIN.get(okey, okey)
                if r is None:
                    vals = ["--"] * 4
                else:
                    t0, g, nsamp, neff = r
                    vals = ["%.0f±%.0f" % (np.mean(t0), np.std(t0)),
                            "%.1f±%.1f" % (np.mean(g), np.std(g)),
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
