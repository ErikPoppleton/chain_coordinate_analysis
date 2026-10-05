#!/usr/bin/env python3
"""Compare the three tilt-modulus methods on the same trajectories.

Reads, from ``plots/tilt_three_methods/``:

* ``df_{ff}_{lipid}_{rep}_{director}.json/.npz``  -- ``tilt_df.py`` output
* ``regis_{ff}_{lipid}_{rep}.json``               -- chi from ``tilt_modulus.py``
* ``logs/resis_{ff}_{lipid}{rep}.log``              -- ReSiS stdout

and writes ``df_spectra.png``, ``tilt_methods_comparison.png`` and
``summary.json`` into the same folder.

Conventions (see analysis/TILT_MODULUS.md, "DF" section):

* DF K_theta is a **bilayer** modulus (Watson 2012 / Levine 2014).
* ReGiS chi / A_lipid and ReSiS kappa are the same per-lipid quantity
  normalised by area.  Their papers call it a bilayer modulus, but for the
  energy (1/2) kappa_theta |t|^2 per monolayer it equals the **monolayer**
  modulus, so it is shown both as reported and doubled.

numpy + matplotlib only.
"""

import json
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "plots" / "tilt_three_methods"
KB_KJ = 0.0083144626
SYSTEMS = [("charmm", "DMPC", 1), ("charmm", "DOPC", 1)]
DIRECTORS = ["levine", "rsf"]
QCUT_TILT = "qcut_2.10"
COL = {"DF": "#2171b5", "ReGiS": "#d4a017", "ReSiS": "#0f7d3b"}


def parse_resis(path):
    txt = path.read_text()
    kap = float(re.search(r"Bilayer Tilt: kappa\s+([\d.eE+-]+)", txt).group(1))
    apl = float(re.search(r"Per Lipid \[bilayer\]:\s+([\d.eE+-]+)", txt).group(1))
    return kap, apl


def load():
    rows = {}
    for ff, lip, rep in SYSTEMS:
        key = "%s_%s_%d" % (ff, lip, rep)
        r = {"df": {}}
        for d in DIRECTORS:
            j = json.loads((OUT / ("df_%s_%s.json" % (key, d))).read_text())
            r["df"][d] = j
        T = r["df"]["levine"]["temperature_K"]
        r["T"] = T
        kap, apl = parse_resis(OUT / "logs" / ("resis_%s_%s%d.log" % (ff, lip, rep)))
        r["resis"] = {"kappa_kT_nm2": kap, "apl_A2": apl}
        rg = OUT / ("regis_%s.json" % key)
        if rg.exists():
            g = json.loads(rg.read_text())
            apl_nm2 = r["df"]["levine"]["apl_nm2"]
            r["regis"] = {"chi_kT": g["chi_kT"], "dchi_kT": g["dchi_kT"],
                          "apl_nm2_used": apl_nm2,
                          "chi_over_A_kT_nm2": g["chi_kT"] / apl_nm2}
        rows[key] = r
    return rows


def plot_spectra(rows):
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.6))
    for i, (key, r) in enumerate(rows.items()):
        lip = key.split("_")[1]
        axt, axb = axes[i]
        for d, mk in zip(DIRECTORS, ["o", "s"]):
            npz = np.load(OUT / ("df_%s_%s.npz" % (key, d)))
            q, mperp, mpar = npz["q"], npz["mperp"], npz["mpar"]
            sp, spar = npz["sem_perp"], npz["sem_par"]
            m = q <= 3.6
            kT = KB_KJ * r["T"]
            yt = kT / mperp[m]
            et = kT * sp[m] / mperp[m] ** 2
            axt.errorbar(q[m] ** 2, yt, yerr=et, fmt=mk, ms=3.5, lw=0.8,
                         mfc="white" if d == "rsf" else COL["DF"],
                         color=COL["DF"], label="director: %s" % d)
            fit = r["df"][d]["tilt"][QCUT_TILT]
            xx = np.linspace(0, 4.5, 50)
            axt.plot(xx, kT * (fit["K_theta_kT_nm2"] + fit["K_tw_kT"] * xx), "-" if d == "levine" else "--",
                     color=COL["DF"], lw=0.9)
            yb = 1.0 / (mpar[m] * q[m] ** 2)
            eb = spar[m] / (mpar[m] ** 2 * q[m] ** 2)
            axb.errorbar(q[m], yb, yerr=eb, fmt=mk, ms=3.5, lw=0.8,
                         mfc="white" if d == "rsf" else "#444444", color="#444444",
                         label="director: %s" % d)
        axt.axvspan(0, 2.1 ** 2, color="0.92", zorder=0)
        axt.set_xlim(0, 13)
        axt.set_xlabel(r"$q^2$ (nm$^{-2}$)")
        axt.set_ylabel(r"$k_BT/\langle|\hat n^\perp_q|^2\rangle$ (kJ mol$^{-1}$ nm$^{-2}$)")
        axt.set_title("%s: transverse (tilt + twist)" % lip, fontsize=9)
        axt.text(0.03, 0.95, "fit region $q\\leq 2.1$ nm$^{-1}$", transform=axt.transAxes,
                 va="top", fontsize=7)
        axb.axvspan(0, 1.5, color="0.92", zorder=0)
        axb.set_xlabel(r"$q$ (nm$^{-1}$)")
        axb.set_ylabel(r"$1/(q^2\langle|\hat n^\parallel_q|^2\rangle)$ ($k_BT$)")
        axb.set_title("%s: longitudinal (bending)" % lip, fontsize=9)
        axb.set_ylim(bottom=0)
        axt.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "df_spectra.png", dpi=200)
    plt.close(fig)


def plot_comparison(rows):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.4), sharey=True)
    for ax, (key, r) in zip(axes, rows.items()):
        lip = key.split("_")[1]
        kT = KB_KJ * r["T"]
        labels, vals, errs, cols, dbl = [], [], [], [], []
        for d in DIRECTORS:
            f = r["df"][d]["tilt"][QCUT_TILT]
            labels.append("DF\n(%s)" % d)
            vals.append(f["K_theta_kT_nm2"] * kT); errs.append(f["K_theta_sem"] * kT)
            cols.append(COL["DF"]); dbl.append(False)
        if "regis" in r:
            labels.append("ReGiS\n$\\chi/A$")
            vals.append(r["regis"]["chi_over_A_kT_nm2"] * kT)
            errs.append(r["regis"]["dchi_kT"] / r["regis"]["apl_nm2_used"] * kT)
            cols.append(COL["ReGiS"]); dbl.append(True)
        labels.append("ReSiS\n$\\kappa_\\theta$")
        vals.append(r["resis"]["kappa_kT_nm2"] * kT); errs.append(0.0)
        cols.append(COL["ReSiS"]); dbl.append(True)
        x = np.arange(len(vals))
        for xi, v, e, c, dd in zip(x, vals, errs, cols, dbl):
            if dd:   # doubled (bilayer-equivalent) shown as hatched extension
                ax.bar(xi, 2 * v, color="white", edgecolor=c, hatch="////", lw=0.8)
            ax.bar(xi, v, color=c, edgecolor=c, lw=0.8)
            if e > 0:
                ax.errorbar(xi, v, yerr=e, color="k", lw=0.8, capsize=2)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_title("CHARMM %s (rep 1)" % lip, fontsize=9)
    axes[0].set_ylabel(r"$K_\theta$ (kJ mol$^{-1}$ nm$^{-2}$, 310 K)")
    fig.text(0.5, 0.005,
             "solid: as reported (DF = bilayer; ReGiS/ReSiS = per-lipid modulus / area, "
             "a monolayer value)   hatched: real-space value x2 (bilayer-equivalent)",
             ha="center", fontsize=6.5)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(OUT / "tilt_methods_comparison.png", dpi=200)
    plt.close(fig)


def main():
    rows = load()
    summary = {}
    for key, r in rows.items():
        T = r["T"]
        kT = KB_KJ * T
        s = {"temperature_K": T, "apl_nm2": r["df"]["levine"]["apl_nm2"],
             "n_lipids": r["df"]["levine"]["n_lipids"],
             "box_nm": r["df"]["levine"]["box_nm"]}
        for d in DIRECTORS:
            f = r["df"][d]["tilt"]
            b = list(r["df"][d]["bend"].values())[0]
            s["DF_%s" % d] = {
                "K_theta_bilayer_kJmol_nm2": {k: [v["K_theta_kT_nm2"] * kT, v["K_theta_sem"] * kT]
                                              for k, v in f.items()},
                "K_theta_bilayer_kT_nm2": {k: [v["K_theta_kT_nm2"], v["K_theta_sem"]]
                                           for k, v in f.items()},
                "K_tw_bilayer_kT": {k: [v["K_tw_kT"], v["K_tw_sem"]] for k, v in f.items()},
                "K_c_bilayer_kT": [b["K_c_kT"], b["K_c_sem"]]}
        if "regis" in r:
            g = r["regis"]
            s["ReGiS"] = {"chi_kT": [g["chi_kT"], g["dchi_kT"]],
                          "chi_over_A_kT_nm2": g["chi_over_A_kT_nm2"],
                          "x2_bilayer_equiv_kT_nm2": 2 * g["chi_over_A_kT_nm2"],
                          "x2_bilayer_equiv_kJmol_nm2": 2 * g["chi_over_A_kT_nm2"] * kT,
                          "chi_over_A_sem_kJmol_nm2": g["dchi_kT"] / g["apl_nm2_used"] * kT,
                          "chi_over_A_kJmol_nm2": g["chi_over_A_kT_nm2"] * kT}
        s["ReSiS"] = {"kappa_kT_nm2": r["resis"]["kappa_kT_nm2"],
                      "x2_bilayer_equiv_kT_nm2": 2 * r["resis"]["kappa_kT_nm2"],
                      "kappa_kJmol_nm2": r["resis"]["kappa_kT_nm2"] * kT,
                      "x2_bilayer_equiv_kJmol_nm2": 2 * r["resis"]["kappa_kT_nm2"] * kT,
                      "apl_A2": r["resis"]["apl_A2"]}
        summary[key] = s
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    plot_spectra(rows)
    plot_comparison(rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
