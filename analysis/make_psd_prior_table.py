#!/usr/bin/env python3
r"""Supplementary (SI) LaTeX table of the **Bayesian finite-size correction's
priors** (analysis/diffusion_psd.py): the shared log-normal priors with their
sources, and below them the per-system values the system-specific priors are
centred on (L, H, R) plus the simulated D and its SEM that enter the
acceptance probability.

Every number is read from ``diffusion_psd`` (the module constants) and
``data/diffusion_psd.json`` (the per-system inputs it actually used), so the
table cannot drift from the fit.  Same style as the two equilibrium tables
(booktabs, \small, footnotesize unit row).

    python3 analysis/make_psd_prior_table.py  ->  data/psd_prior_table.tex
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import diffusion_psd as psd                                  # noqa: E402
from lipiddefs import FORCEFIELDS, FF_LABELS, LIPIDS          # noqa: E402

DATA = Path(__file__).resolve().parent.parent / "data"
OUT = DATA / "psd_prior_table.tex"

CITE_VENABLE = r"\cite{venable2017lipid}"
CITE_GONZALEZ = r"\cite{gonzalez2010shear}"


def pct(w):
    return r"%d\%%" % round(100 * w)


def sci(x):
    """1e8 -> $10^{8}$, 8.9e-11 -> $8.9\times10^{-11}$."""
    m, e = ("%.1e" % x).split("e")
    e = int(e)
    if m == "1.0":
        return r"$10^{%d}$" % e
    return r"$%s\times10^{%d}$" % (m, e)


def shared_rows():
    em, emw = psd.PRIOR_ETA_M
    b, bw = psd.PRIOR_B
    fc, fcw = psd.PRIOR_ETA_F_FF["charmm"]
    ft, ftw = psd.PRIOR_ETA_F_FF["amber"]
    assert psd.PRIOR_ETA_F_FF["slipids"] == psd.PRIOR_ETA_F_FF["amber"]
    lh = pct(psd.PRIOR_LH_W)
    return [
        (r"$\eta_\mathrm{m}$", sci(em),
         r"Pa\,s\,m", pct(emw),
         r"C36 DOPC" + CITE_VENABLE),
        (r"$b$", sci(b), r"Pa\,s\,m$^{-1}$", pct(bw),
         CITE_VENABLE),
        (r"$\eta_\mathrm{f}$ (CHARMM)",
         "$%.2f$" % (fc * 1e3), r"mPa\,s", pct(fcw),
         r"interpolated" + CITE_VENABLE),
        (r"$\eta_\mathrm{f}$ (Lipid21, SLipids)",
         "$%.3f$" % (ft * 1e3), r"mPa\,s", pct(ftw),
         r"scaled" + CITE_GONZALEZ),
        (r"$R$", r"$\sqrt{\mathrm{APL}/\pi}$", "nm",
         pct(psd.PRIOR_R_W), CITE_VENABLE),
        (r"$L$", r"$\langle L_x\rangle$", "nm", lh,
         CITE_VENABLE),
        (r"$H$",
         r"$(\langle L_z\rangle - d_\mathrm{HH})/2$", "nm", lh,
         CITE_VENABLE),
    ]


def build_latex():
    res = json.loads((DATA / "diffusion_psd.json").read_text())
    L = []
    L.append(r"% Requires \usepackage{booktabs,multirow,amsmath}")
    L.append(r"\begin{table}[t]")
    L.append(r"\centering")
    L.append(r"\small")
    L.append(r"\setlength{\tabcolsep}{5pt}")
    # ---- shared priors
    L.append(r"\begin{tabular}{lcccl}")
    L.append(r"\toprule")
    L.append(r"Parameter & Prior mean & Units & Prior SD & Source \\")
    L.append(r" & & & {\footnotesize (\% of mean)} & \\")
    L.append(r"\midrule")
    for row in shared_rows():
        L.append(" & ".join(row) + r" \\")
    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")
    L.append("")
    L.append(r"\vspace{0.8em}")
    L.append("")
    # ---- per-system centres
    L.append(r"\begin{tabular}{llccccc}")
    L.append(r"\toprule")
    L.append(r"Lipid & FF & $L$ & $H$ & $R$ & $D_\mathrm{sim} \pm \sigma_D$"
             r" & $N_\mathrm{acc}$ \\")
    L.append(r" & & {\footnotesize nm} & {\footnotesize nm} & {\footnotesize nm}"
             r" & {\footnotesize µm$^2$\,s$^{-1}$} & {\footnotesize draws} \\")
    for lipid in LIPIDS:
        rows = [(ff, res[f"{ff}_{lipid}"]) for ff in FORCEFIELDS
                if f"{ff}_{lipid}" in res]
        if not rows:
            continue
        L.append(r"\midrule")
        for i, (ff, r) in enumerate(rows):
            first = (r"\multirow{%d}{*}{%s}" % (len(rows), lipid)) if i == 0 else ""
            L.append("%s & %s & $%.2f$ & $%.2f$ & $%.3f$ & $%.1f \\pm %.1f$ & %d \\\\"
                     % (first, FF_LABELS[ff], r["L_nm"], r["H_nm"], r["R0_nm"],
                        r["D_sim"], r["D_sim_sem"], r["n_accepted"]))
    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")
    n = res["priors"]["draws"]
    L.append(
        r"\caption{Priors of the Bayesian single-box finite-size correction of the"
        r" lateral diffusion coefficient\cite{venable2017lipid}.  $D_\infty$ is"
        r" extrapolated with the periodic Saffman--Delbr\"uck model for a"
        r" single-leaflet object\cite{venable2017lipid}, at $T = %d$\,K."
        r"  Every parameter has a log-normal prior with the arithmetic mean and"
        r" standard deviation (as a percentage of the mean) given in the top"
        r" panel; $\eta_\mathrm{m}$ and $b$ are effectively unconstrained"
        r" ($500\%%$), $\eta_\mathrm{f}$ is set by the water model of each force"
        r" field (CHARMM-modified TIP3P with Lennard-Jones sites on the"
        r" hydrogens for CHARMM, the original TIP3P for Lipid21 and SLipids).  $\eta_\mathrm{m}$ is the"
        r" CHARMM36 DOPC value at 298\,K\cite{venable2017lipid}.  The CHARMM"
        r" TIP3P viscosity is interpolated linearly to %d\,K from its values at"
        r" 293 and 323\,K\cite{venable2017lipid}; the TIP3P value"
        r" ($0.321$\,mPa\,s at 298\,K\cite{gonzalez2010shear}) is scaled to"
        r" %d\,K by the same temperature ratio.  $R$, $L$ and $H$ are centred on each system's own values,"
        r" listed in the bottom panel (means over the three replicates; APL and"
        r" $d_\mathrm{HH}$ as in Table~\ref{tab:equilibrium_properties}).  Of"
        r" $%s$ independent draws from the priors, each is accepted with"
        r" probability $\exp[-(D_\mathrm{PBC} - D_\mathrm{sim})^2/2\sigma_D^2]$,"
        r" where $D_\mathrm{PBC}$ is the model prediction for that draw,"
        r" $D_\mathrm{sim}$ the simulated diffusion coefficient (bilayer centre"
        r" of mass removed, fitted over $10\,\mathrm{ns} < t < (T_\mathrm{run} -"
        r" 10\,\mathrm{ns})/2$) and $\sigma_D$ its standard error over the three"
        r" replicates; $N_\mathrm{acc}$ is the number accepted.  $D_\infty$ is"
        r" reported as the median and $95\%%$ credible interval of the accepted"
        r" draws.}"
        % (psd.T_K, psd.T_K, psd.T_K, "{:,}".format(n).replace(",", r"\,")))
    L.append(r"\label{tab:psd_priors}")
    L.append(r"\end{table}")
    return "\n".join(L) + "\n"


def main():
    OUT.write_text(build_latex())
    print(OUT.read_text())
    print("wrote %s" % OUT)


if __name__ == "__main__":
    main()
