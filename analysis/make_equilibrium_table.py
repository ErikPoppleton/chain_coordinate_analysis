#!/usr/bin/env python3
r"""Build a LaTeX summary table of the **equilibrium** membrane properties.

Covers the equilibrium-derived observables in ``data/`` -- the lipyphilic
bulk properties and the Lipidator/ReSiS tilt modulus -- plus one pore column:
the **pore-formation barrier Delta G** from the chain-coordinate PMF, exactly the
value Figure 2 plots (``draw_pore_energy.barrier()``, basin to pore-side peak).
The other pore branches (cylindrical density maps, closure trajectories) are
excluded.  ``collect()`` still returns the g_lomepro ``<-S_CD>`` and the
ost_pymodules ``chi`` (the vs-barrier figures plot them), but neither is
tabulated any more.

Error scheme (per column, by what data is available):
  * The **per-frame lipyphilic observables** (APL, d_HH, d_HC) carry a
    **hierarchical 95% confidence interval** (``autocorr.hierarchical_ci``):
    each replicate contributes an autocorrelation-limited within-run SEM
    (SD/sqrt(Neff), Neff = N/g via the statistical inefficiency; Chodera 2016),
    and the grand-mean SE takes ``max(between_sd^2, within_sem^2)/n`` -- the
    pooled pseudoreplication limit where the replicates agree, the honest
    replicate-spread CI where they do not (so AMBER/SLipids DMPG are not falsely
    tightened).  The point estimate is unchanged (grand mean over frames & reps).
  * The **tilt moduli** (chi, kappa) also get a hierarchical 95% CI, but from
    pre-computed per-replicate estimates: each run yields a scalar plus a
    block-averaged within-run SEM (``chi_block_sem`` in ``tilt_moduli.json``,
    ``kappa_block_sem`` in ``lipidator_blocks.json``), combined with the
    replicate spread the same ``max(between^2, within^2)`` way.
  * **D and <-S_CD>** keep ``mean +/- SD`` across the three replicate scalars --
    a single fitted scalar / profile per run, with no within-run error series.
  * **Delta G** is the full 100 ns profile's barrier +/- the SD of the same
    barrier over the five consecutive 20 ns partial profiles -- a convergence
    spread (one PMF per system, no replicates).

Inputs (all in ``data/``):
  * ``lipprops_area_per_lipid.npz``   -- APL  (per-rep key ``{ff}_{lipid}_eq{n}``, 2-D frames x lipids)
  * ``lipprops_memb_thickness.npz``   -- P-P thickness d_HH (per-rep, 1-D per frame, Angstrom)
  * ``core_thickness.npz``            -- d_HC, lipyphilic key ``{ff}_{lipid}_lipy_rep{n}`` (nm; optional)
  * ``lipprops_registration.npz``     -- inter-leaflet registration (per-rep, 1-D per frame)
  * ``lipprops_lateral_diffusion.npz``-- fitted D, key ``{ff}_{lipid}_D`` (3 reps, cm^2/s)
  * ``{ff}_{lipid}_{rep}_scd_sn{1,2}.dat`` -- g_lomepro -S_CD (col2 = mean, col3 = SD-over-lipids)
  * ``tilt_moduli_summary.json``      -- ost_pymodules chi (kJ/mol/rad^2)
  * ``lipidator_moduli_summary.json`` -- Lipidator kappa_theta (kJ/mol/nm^2)
  * ``{ff}_{lipid}_profile.xvg`` + ``_part{1..5}_profile.xvg`` -- gmx wham PMFs (Delta G)

Writes ``data/equilibrium_summary_table.tex`` (drop-in; needs
``\usepackage{booktabs,multirow,amsmath}``) and prints a plain-text preview
plus a report of any missing systems.

    python3 analysis/make_equilibrium_table.py
"""

import json
import os
from pathlib import Path

import numpy as np

import autocorr
from lipiddefs import FF_LABELLED as FORCEFIELDS, LIPIDS, REPS

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def series_reps(npz, tmpl, ff, lipid, transform):
    """[per-frame series per rep] from ``npz`` under keys ``tmpl % (ff,lipid,n)``,
    each mapped through ``transform`` (e.g. mean-over-lipids, Angstrom->nm)."""
    out = []
    for n in REPS:
        k = tmpl % (ff, lipid, n)
        out.append(transform(np.asarray(npz[k])) if k in npz.files else None)
    return out


def ci_cell(series_list, t0s=None):
    """(mean, 95% CI half-width) from the hierarchical (autocorrelation-aware)
    estimator, or None -- for the per-frame observables (APL/d_HH/d_HC).
    ``t0s`` are the per-replicate pymbar equilibration cutoffs (burn-in dropped
    before the mean and g are estimated, per Chodera 2016)."""
    r = autocorr.hierarchical_ci(series_list, t0s=t0s)
    return (r["mean"], r["ci_half"]) if r else None


# stats.py observable name for each per-frame table column.
STATS_OBS = {"apl": "apl", "thick": "thickness", "dhc": "dhc"}


def load_t0map():
    """{(stats_obs, system, rep): t0} from stats.py's equilibrium-timeseries JSON
    (pymbar ``detect_equilibration``).  Returns None (no burn-in trimming) if the
    JSON is absent -- so run ``stats.py`` first for the de-biased means/CIs."""
    p = DATA / "equilibrium_timeseries_stats.json"
    if not p.exists():
        print("note: %s absent -> equilibrium CIs computed WITHOUT burn-in "
              "trimming; run 'conda run -n base python analysis/stats.py' first "
              "for pymbar equilibration detection." % p.name)
        return None
    d = json.loads(p.read_text())["per_replicate"]
    m = {}
    for obs, systems in d.items():
        for sysn, recs in systems.items():
            for r in recs:
                m[(obs, sysn, int(r["rep"]))] = int(r["t0"])
    return m


def t0_reps(t0map, col, ff, lipid):
    """Per-replicate t0 list for table column ``col``, or None if unavailable."""
    if t0map is None or col not in STATS_OBS:
        return None
    sysn = "%s_%s" % (ff, lipid)
    return [t0map.get((STATS_OBS[col], sysn, n), 0) for n in REPS]


def summarise_reps(per_rep):
    """[scalar per rep] -> (mean, population SD) or None if empty.

    Retained for the *fitted* quantities (D, <-S_CD>, chi, kappa) that have no
    per-frame series for the autocorrelation-aware CI -- their error stays the
    SD across replicates (chi/kappa additionally have block SEMs available in
    their per-rep JSONs once the tilt/lipidator block runs complete)."""
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


RT_310 = 2.577          # kJ/mol at 310 K (chi is per-lipid k_BT/rad^2 -> kJ/mol/rad^2)


def moduli_ci(perrep, ff, lipid, val_key, sem_key):
    """(mean, hierarchical-95%-CI half-width) in kJ/mol units for a tilt modulus,
    from its per-replicate JSON (value + block-averaged within-run SEM, both in
    k_BT units -> * RT).  Combines the per-rep block SEMs (within-run) with the
    replicate spread (between-run) via ``autocorr.hierarchical_ci_from_estimates``
    -- the same estimator the per-frame observables use.  Returns None if absent."""
    if perrep is None:
        return None
    means, sems, neffs = [], [], []
    for n in REPS:
        rec = perrep.get("%s_%s_%d" % (ff, lipid, n))
        if rec is None or rec.get(val_key) is None:
            means.append(None); sems.append(float("nan")); neffs.append(None)
            continue
        means.append(rec[val_key] * RT_310)
        s = rec.get(sem_key)
        sems.append(s * RT_310 if s is not None else float("nan"))
        neffs.append(rec.get("n_blocks"))
    r = autocorr.hierarchical_ci_from_estimates(means, sems, neffs)
    return (r["mean"], r["ci_half"]) if r else None


def pore_barrier(ff, lipid):
    """(Delta G, SD over the five 20 ns partials) in kJ/mol, or None.

    Imported lazily: ``draw_pore_energy`` itself imports this module (for
    ``collect()``), so a top-level import would be circular."""
    import draw_pore_energy as dpe
    full = dpe.barrier(ff, lipid)
    if full is None:
        return None
    parts = dpe.partial_barriers(ff, lipid)
    return (full, float(np.std(parts, ddof=1)) if len(parts) > 1 else float("nan"))


def load():
    need = ["lipprops_area_per_lipid.npz", "lipprops_memb_thickness.npz",
            "lipprops_lateral_diffusion.npz", "tilt_moduli.json"]
    missing_files = [f for f in need if not (DATA / f).exists()]
    if missing_files:
        raise SystemExit("missing input files: %s" % ", ".join(missing_files))
    npz = {k: np.load(DATA / ("lipprops_%s.npz" % k))
           for k in ["area_per_lipid", "memb_thickness", "lateral_diffusion"]}
    # per-replicate moduli (chi + block SEM; kappa_full + block SEM) for the
    # hierarchical CI -- not the mean/SD summary JSONs.
    tiltrep = json.load(open(DATA / "tilt_moduli.json"))
    lipblk = (json.load(open(DATA / "lipidator_blocks.json"))
              if (DATA / "lipidator_blocks.json").exists() else None)
    # d_HC (hydrophobic core, lipyphilic first-tail-carbon metric); optional.
    ct = (np.load(DATA / "core_thickness.npz")
          if (DATA / "core_thickness.npz").exists() else None)
    return npz, tiltrep, lipblk, ct


def collect():
    npz, tiltrep, lipblk, ct = load()
    t0map = load_t0map()          # pymbar equilibration cutoffs (per obs/system/rep)
    rows = {}          # (ff, lipid) -> {param: (mean, sd) or None}
    missing = []       # (ff, lipid, param) with no data
    for ff, _ in FORCEFIELDS:
        for lipid in LIPIDS:
            cell = {}
            # --- lipyphilic (per-frame observables -> hierarchical 95% CI) ---
            # APL per frame = mean over lipids; d_HH per frame (A->nm); d_HC per
            # frame (already nm).  The point estimate is unchanged (grand mean
            # over frames & reps); only the error becomes the autocorrelation-
            # aware CI in place of the SD across replicate means.
            cell["apl"] = ci_cell(series_reps(
                npz["area_per_lipid"], "%s_%s_eq%d", ff, lipid, lambda a: a.mean(axis=1)),
                t0_reps(t0map, "apl", ff, lipid))
            cell["thick"] = ci_cell(series_reps(
                npz["memb_thickness"], "%s_%s_eq%d", ff, lipid, lambda a: a.astype(float) / 10.0),
                t0_reps(t0map, "thick", ff, lipid))
            cell["dhc"] = (ci_cell(series_reps(
                ct, "%s_%s_lipy_rep%d", ff, lipid, lambda a: a.astype(float)),
                t0_reps(t0map, "dhc", ff, lipid))
                if ct is not None else None)
            dkey = "%s_%s_D" % (ff, lipid)
            cell["diff"] = (summarise_reps([v * 1e8 for v in npz["lateral_diffusion"][dkey]])
                            if dkey in npz["lateral_diffusion"].files else None)  # cm^2/s -> um^2/s
            # --- g_lomepro ---
            cell["scd"] = summarise_reps(scd_rep_scalars(ff, lipid))
            # --- tilt moduli: hierarchical CI from per-rep value + block SEM ---
            cell["chi"] = moduli_ci(tiltrep, ff, lipid, "chi_kT", "chi_block_sem")
            cell["kappa"] = moduli_ci(lipblk, ff, lipid, "kappa_full", "kappa_block_sem")
            # --- pore-formation barrier (Figure 2's value) +/- partial spread ---
            cell["dg"] = pore_barrier(ff, lipid)
            rows[(ff, lipid)] = cell
            for p, v in cell.items():
                if v is None:
                    missing.append((ff, lipid, p))
    return rows, missing


# (param key, header, unit, decimals)
COLS = [
    ("apl",   "APL",             r"\AA$^2$",                          1),
    ("thick", r"$d_{\mathrm{HH}}$", "nm",                            2),
    ("dhc",   r"$d_{\mathrm{HC}}$", "nm",                            2),
    ("diff",  r"$D$",            r"µm$^2$\,s$^{-1}$",                 1),
    # The g_lomepro <-S_CD> is still computed (collect() returns it, and
    # plot_property_vs_barrier.py plots it) but is no longer tabulated.
    # The ost_pymodules per-lipid chi is still computed (collect() returns it,
    # and the vs-barrier figures plot it) but is no longer tabulated: the two
    # tilt moduli measure the same thing, so the table carries only the
    # area-normalised Lipidator/ReSiS kappa_theta.
    ("kappa", r"$\kappa_\theta$", r"kJ\,mol$^{-1}$\,nm$^{-2}$",       1),
    ("dg",    r"$\Delta G_{\mathrm{pore}}$", r"kJ\,mol$^{-1}$",           1),
]
GROUPS = [
    (r"lipyphilic\cite{smith2021lipyphilic}", 4),
    (r"Lipidator\cite{allolio2018local}", 1),
    (r"gmx wham\cite{hub2010gwham}", 1),
]

# Nagle 2017 (Table 1, X-ray diffuse scattering, T = 30 C) reports K_theta in
# mN/m; the table's kappa_theta is in kJ/mol/nm^2.  1 mN/m = 1e-21 J/nm^2, so
# x N_A -> 0.6022 kJ/mol/nm^2.  No monolayer/bilayer factor is applied: Nagle
# compares a Khelashvili-Harries real-space simulation value (the method family
# Lipidator implements) directly against the X-ray one.
MN_PER_M_TO_KJ_MOL_NM2 = 0.60221


def _nagle_kappa(k_mn, err_mn):
    return r"$%.1f \pm %.1f$\cite{nagle2017experimentally}" % (
        k_mn * MN_PER_M_TO_KJ_MOL_NM2, err_mn * MN_PER_M_TO_KJ_MOL_NM2)

# Experimental reference values + citations, hardcoded from the literature (one
# extra "Exp" row per lipid).  Keyed by COLS column key; a missing key -> blank
# cell.  Except where a temperature is noted inline, values are for T = 30 C.
EXPERIMENTAL = {
    "DOPC": {
        "apl":   r"$72.5$\cite{nagle2000structure}",
        "thick": r"$3.69$\cite{nagle2000structure}",
        "dhc":   r"$2.71$\cite{nagle2000structure}",
        "diff":  r"$\approx12$ ($308$\,K)\cite{filippov2003effect}",
        "kappa": _nagle_kappa(89, 4),          # 89 +/- 4 mN/m
    },
    "DMPC": {
        "apl":   r"$59.6$\cite{nagle2000structure}",
        "thick": r"$3.60$\cite{nagle2000structure,schuy2006thermal}",
        "dhc":   r"$2.62$\cite{nagle2000structure}",
        "diff":  r"$\approx11$ ($308$\,K)\cite{filippov2003effect}",
        "kappa": _nagle_kappa(44, 2),          # 44 +/- 2 mN/m
    },
    "DMPG": {
        "apl":   r"$65.1$\cite{pan2012molecular}",
        "thick": r"$3.25$\cite{pan2012molecular}",
        "dhc":   r"$2.36$\cite{pan2012molecular}",
        "diff":  r"$11.8$ ($309$ K)\cite{pali1989restricted}",
    },
}


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
    grp = ["\\multicolumn{2}{c}{Software}"]
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
    for lipid in LIPIDS:
        for fi, (ff, fflabel) in enumerate(FORCEFIELDS):
            cell = rows[(ff, lipid)]
            first = ("\\multirow{%d}{*}{%s}" % (len(FORCEFIELDS), lipid)
                     if fi == 0 else "")
            vals = [fmt(cell[k], dec) for k, _, _, dec in COLS]
            lines.append(" & ".join([first, fflabel] + vals) + " \\\\")
        # experimental reference row for this lipid
        exp = EXPERIMENTAL.get(lipid, {})
        exp_vals = [exp.get(k, "") for k, _, _, _ in COLS]
        lines.append(" & ".join(["", "Exp"] + exp_vals) + " \\\\")
        lines.append("\\midrule")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\caption{Equilibrium membrane properties by lipid and force field "
                 "(mean $\\pm$ uncertainty over three replicate simulations). "
                 "The uncertainty is a hierarchical $95\\%$ confidence interval "
                 "combining each replicate's within-run error with the between-replicate "
                 "spread: for the per-frame observables (APL, $d_{\\mathrm{HH}}$, "
                 "$d_{\\mathrm{HC}}$) the within-run error is the autocorrelation-limited "
                 "standard error (statistical inefficiency $g$\\cite{chodera2016simple}), "
                 "and for the tilt modulus $\\kappa_\\theta$ it is a "
                 "block-averaged SEM; for $D$ (a single fitted scalar per run) it is "
                 "the SD across the three replicates. "
                 "$d_{\\mathrm{HH}}$: phosphate--phosphate bilayer thickness; "
                 "$d_{\\mathrm{HC}}$: hydrophobic-core thickness (first-tail-carbon "
                 "MembThickness; agrees with the tail mass-density width at "
                 "$500$\\,kg\\,m$^{-3}$ to $\\sim0.02$\\,nm); "
                 "$D$: lateral diffusion coefficient; "
                 "$\\kappa_\\theta$: area-normalised lipid tilt modulus "
                 "(Lipidator/ReSiS; experimental values converted from mN\\,m$^{-1}$); "
                 "$\\Delta G_{\\mathrm{pore}}$: pore-formation barrier, from the "
                 "chain-coordinate PMF basin to the highest point on its pore side "
                 "(one $100$\\,ns umbrella-sampling PMF per system; $\\pm$ is the SD of the "
                 "same barrier over five consecutive $20$\\,ns sub-windows). "
                 "Simulations: $T=310$\\,K. Except where indicated, experimental values "
                 "are for $T= 30^{\\circ}$\\,C ($303.15$\\,K)}")
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
