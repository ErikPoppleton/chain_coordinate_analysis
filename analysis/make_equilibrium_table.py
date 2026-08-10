#!/usr/bin/env python3
r"""Build a LaTeX summary table of the **equilibrium** membrane properties.

Covers every equilibrium-derived observable in ``data/`` -- the lipyphilic
bulk properties, the g_lomepro S_CD order parameter, and the two tilt-modulus
methods (OpenStructure/ost_pymodules and the Lipidator/ReSiS toolkit).  The
pore-formation PMFs, pore cylindrical density maps, and pore-closure
trajectories are deliberately excluded (they are not equilibrium quantities).

Every cell is ``mean +/- SD`` where -- following the project convention -- each
replicate is first reduced to a single scalar (mean over frames / lipids /
carbons / chains) and the mean and **population SD are taken across the three
replicates** (autocorrelated frames would otherwise give a pseudo-replicated,
hairline-thin error bar).  The tilt moduli are read straight from their summary
JSONs, which already apply this same convention.

Inputs (all in ``data/``):
  * ``lipprops_area_per_lipid.npz``   -- APL  (per-rep key ``{ff}_{lipid}_eq{n}``, 2-D frames x lipids)
  * ``lipprops_memb_thickness.npz``   -- bilayer thickness (per-rep, 1-D per frame, Angstrom)
  * ``lipprops_registration.npz``     -- inter-leaflet registration (per-rep, 1-D per frame)
  * ``lipprops_lateral_diffusion.npz``-- fitted D, key ``{ff}_{lipid}_D`` (3 reps, cm^2/s)
  * ``lipprops_order_parameter.npz``  -- S_CC samples, key ``{ff}_{lipid}_eq{n}`` (chain-frames x carbons)
  * ``{ff}_{lipid}_{rep}_scd_sn{1,2}.dat`` -- g_lomepro -S_CD (col2 = mean, col3 = SD-over-lipids)
  * ``tilt_moduli_summary.json``      -- ost_pymodules chi (kJ/mol/rad^2)
  * ``lipidator_moduli_summary.json`` -- Lipidator kappa_theta (kJ/mol/nm^2)

Writes ``data/equilibrium_summary_table.tex`` (drop-in; needs
``\usepackage{booktabs,multirow,amsmath}``) and prints a plain-text preview
plus a report of any missing systems.

    python3 analysis/make_equilibrium_table.py
"""

import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

FORCEFIELDS = [("charmm", "CHARMM"), ("amber", "AMBER"), ("slipids", "SLipids")]
LIPIDS = ["DOPC", "DMPC", "DMPG"]
REPS = [1, 2, 3]


def summarise_reps(per_rep):
    """[scalar per rep] -> (mean, population SD) or None if empty."""
    vals = [v for v in per_rep if v is not None and np.isfinite(v)]
    if not vals:
        return None
    return float(np.mean(vals)), float(np.std(vals, ddof=0))


def rep_scalar_from_npz(npz, ff, lipid, reduce):
    """Reduce each per-rep array ``{ff}_{lipid}_eq{n}`` to one scalar."""
    out = []
    for n in REPS:
        key = "%s_%s_eq%d" % (ff, lipid, n)
        out.append(reduce(npz[key]) if key in npz.files else None)
    return out


def scd_rep_scalars(ff, lipid):
    """Chain-averaged -S_CD per rep, pooling both tails (col 2 of each .dat)."""
    out = []
    for n in REPS:
        vals = []
        for sn in (1, 2):
            f = DATA / ("%s_%s_%d_scd_sn%d.dat" % (ff, lipid, n, sn))
            if f.exists():
                arr = np.loadtxt(f)
                if arr.ndim == 1:
                    arr = arr[None, :]
                vals.append(arr[:, 1])            # col 2 = mean -S_CD per carbon
        out.append(float(np.concatenate(vals).mean()) if vals else None)
    return out


def load():
    need = ["lipprops_area_per_lipid.npz", "lipprops_memb_thickness.npz",
            "lipprops_registration.npz", "lipprops_lateral_diffusion.npz",
            "lipprops_order_parameter.npz", "tilt_moduli_summary.json",
            "lipidator_moduli_summary.json"]
    missing_files = [f for f in need if not (DATA / f).exists()]
    if missing_files:
        raise SystemExit("missing input files: %s" % ", ".join(missing_files))
    npz = {k: np.load(DATA / ("lipprops_%s.npz" % k))
           for k in ["area_per_lipid", "memb_thickness", "registration",
                     "lateral_diffusion", "order_parameter"]}
    tilt = json.load(open(DATA / "tilt_moduli_summary.json"))
    lipd = json.load(open(DATA / "lipidator_moduli_summary.json"))
    return npz, tilt, lipd


def collect():
    npz, tilt, lipd = load()
    rows = {}          # (ff, lipid) -> {param: (mean, sd) or None}
    missing = []       # (ff, lipid, param) with no data
    for ff, _ in FORCEFIELDS:
        for lipid in LIPIDS:
            cell = {}
            # --- lipyphilic ---
            cell["apl"] = summarise_reps(rep_scalar_from_npz(
                npz["area_per_lipid"], ff, lipid, lambda a: float(a.mean())))
            cell["thick"] = summarise_reps(rep_scalar_from_npz(
                npz["memb_thickness"], ff, lipid, lambda a: float(a.mean()) / 10.0))  # A -> nm
            cell["reg"] = summarise_reps(rep_scalar_from_npz(
                npz["registration"], ff, lipid, lambda a: float(a.mean())))
            dkey = "%s_%s_D" % (ff, lipid)
            cell["diff"] = (summarise_reps([v * 1e7 for v in npz["lateral_diffusion"][dkey]])
                            if dkey in npz["lateral_diffusion"].files else None)  # cm^2/s -> 1e-7
            cell["scc"] = summarise_reps(rep_scalar_from_npz(
                npz["order_parameter"], ff, lipid, lambda a: float(a.mean())))
            # --- g_lomepro ---
            cell["scd"] = summarise_reps(scd_rep_scalars(ff, lipid))
            # --- tilt moduli (already mean/SD over reps in the JSONs) ---
            k = "%s_%s" % (ff, lipid)
            cell["chi"] = ((tilt[k]["chi_kJ_per_mol"], tilt[k]["sd_kJ_per_mol"])
                           if k in tilt else None)
            cell["kappa"] = ((lipd[k]["tilt_kJ_per_mol_nm2"], lipd[k]["sd_kJ_per_mol_nm2"])
                             if k in lipd else None)
            rows[(ff, lipid)] = cell
            for p, v in cell.items():
                if v is None:
                    missing.append((ff, lipid, p))
    return rows, missing


# (param key, header, unit, decimals)
COLS = [
    ("apl",   "APL",             r"\AA$^2$",                          1),
    ("thick", r"$d_{\mathrm{HH}}$", "nm",                            2),
    ("reg",   "Reg.",            "--",                                3),
    ("diff",  r"$D$",            r"$10^{-7}$\,cm$^2$\,s$^{-1}$",      2),
    ("scc",   r"$\langle S_{CC}\rangle$", "--",                      3),
    ("scd",   r"$\langle -S_{CD}\rangle$", "--",                     3),
    ("chi",   r"$\chi$",         r"kJ\,mol$^{-1}$\,rad$^{-2}$",       1),
    ("kappa", r"$\kappa_\theta$", r"kJ\,mol$^{-1}$\,nm$^{-2}$",       1),
]
GROUPS = [("lipyphilic", 5), ("g\\_lomepro", 1), ("tilt modulus", 2)]


def fmt(cell, dec):
    if cell is None:
        return "--"
    m, s = cell
    return "$%.*f \\pm %.*f$" % (dec, m, dec, s)


def build_latex(rows):
    ncol = 2 + len(COLS)
    lines = []
    lines.append("% Requires \\usepackage{booktabs,multirow,amsmath} (and \\AA via textcomp/mathcomp or siunitx).")
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\setlength{\\tabcolsep}{4pt}")
    lines.append("\\begin{tabular}{ll" + "c" * len(COLS) + "}")
    lines.append("\\toprule")
    # group header row
    grp = ["\\multicolumn{2}{c}{}"]
    start = 2
    cmids = []
    for name, span in GROUPS:
        grp.append("\\multicolumn{%d}{c}{%s}" % (span, name))
        cmids.append("\\cmidrule(lr){%d-%d}" % (start + 1, start + span))
        start += span
    lines.append(" & ".join(grp) + " \\\\")
    lines.append(" ".join(cmids))
    # column header + unit rows
    hdr = ["Lipid", "FF"] + [h for _, h, _, _ in COLS]
    lines.append(" & ".join(hdr) + " \\\\")
    units = ["", ""] + ["{\\footnotesize %s}" % u for _, _, u, _ in COLS]
    lines.append(" & ".join(units) + " \\\\")
    lines.append("\\midrule")
    for li, lipid in enumerate(LIPIDS):
        for fi, (ff, fflabel) in enumerate(FORCEFIELDS):
            cell = rows[(ff, lipid)]
            first = ("\\multirow{%d}{*}{%s}" % (len(FORCEFIELDS), lipid)
                     if fi == 0 else "")
            vals = [fmt(cell[k], dec) for k, _, _, dec in COLS]
            lines.append(" & ".join([first, fflabel] + vals) + " \\\\")
        if li != len(LIPIDS) - 1:
            lines.append("\\midrule")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\caption{Equilibrium membrane properties by lipid and force field "
                 "(mean $\\pm$ SD over three replicate simulations). "
                 "$d_{\\mathrm{HH}}$: phosphate--phosphate bilayer thickness; "
                 "Reg.: inter-leaflet registration; $D$: lateral diffusion coefficient; "
                 "$\\langle S_{CC}\\rangle$ / $\\langle -S_{CD}\\rangle$: chain-averaged "
                 "order parameters (lipyphilic / g\\_lomepro); "
                 "$\\chi$: per-lipid tilt modulus (OpenStructure/ost\\_pymodules); "
                 "$\\kappa_\\theta$: area-normalised tilt modulus (Lipidator/ReSiS). "
                 "$T=310$\\,K.}")
    lines.append("\\label{tab:equilibrium_properties}")
    lines.append("\\end{table}")
    return "\n".join(lines) + "\n"


def preview(rows):
    hdr = ["lipid", "ff"] + [k for k, _, _, _ in COLS]
    print("  ".join("%-14s" % h for h in hdr))
    for lipid in LIPIDS:
        for ff, fflabel in FORCEFIELDS:
            cell = rows[(ff, lipid)]
            out = ["%-14s" % lipid, "%-14s" % fflabel]
            for k, _, _, dec in COLS:
                c = cell[k]
                out.append("%-14s" % ("--" if c is None
                                      else "%.*f+-%.*f" % (dec, c[0], dec, c[1])))
            print("  ".join(out))


def main():
    rows, missing = collect()
    preview(rows)
    tex = build_latex(rows)
    out = DATA / "equilibrium_summary_table.tex"
    out.write_text(tex)
    print("\nwrote %s" % out)
    if missing:
        print("\nMISSING (no data found -- likely not pulled from the cluster):")
        for ff, lipid, p in missing:
            print("  %s_%s : %s" % (ff, lipid, p))
    else:
        print("\nAll parameters present for all 9 systems x 3 replicates.")


if __name__ == "__main__":
    main()
