#!/usr/bin/env python3
"""Autocorrelation diagnostics for the HMMM equilibrium timeseries.

The HMMM counterpart of ``analysis/stats.py``: per force field / ensemble /
replicate it computes the equilibration cutoff ``t0`` (pymbar
``detect_equilibration``), the statistical inefficiency ``g`` and the maximum
number of effective uncorrelated samples ``Neff_max`` for each per-frame
observable, and writes them to ``HMMM/data/equilibrium_timeseries_stats.json``.
That file is the ``t0`` source of truth the HMMM plots read to drop the burn-in
and to build their hierarchical CIs.

pymbar lives in the **base** conda env, so::

    PYTHONNOUSERSITE=1 conda run -n base python HMMM/analysis/stats.py

Observables: ``apl`` (per-frame mean area per lipid over the lipids) and
``thickness`` (d_HH).  Lateral diffusion is excluded for the same reason as in
the parent project -- D is one fitted scalar per run and the MSD is
nonstationary.

The per-series maths (``analyze_series``) and the within-/between-run variance
decomposition (``decompose``) are imported from the parent project's
``analysis/stats.py`` rather than re-implemented, so the two branches cannot
drift apart.  They are loaded by explicit path because both modules are called
``stats``.
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                      # HMMM/
PARENT_ANALYSIS = ROOT.parent / "analysis"

sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PARENT_ANALYSIS))    # autocorr, lipiddefs, palette, figstyle

from hmmm_defs import FORCEFIELDS, ENSEMBLES, REPS, DT_NS, system as system_key
import stability as stab


def _parent_stats():
    """Import the parent project's ``analysis/stats.py`` under a distinct name.

    A plain ``import stats`` would find this file, so the module is loaded from
    its path."""
    spec = importlib.util.spec_from_file_location(
        "parent_stats", PARENT_ANALYSIS / "stats.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


OBSERVABLES = ["apl", "thickness"]
UNITS = {"apl": "A^2", "thickness": "A"}


def series_for(npzs, ff, ens, rep, records=None):
    """``{observable: 1-D per-frame series}`` for one run, cut at its stability time.

    The truncation has to happen **before** the equilibration analysis, not
    after: on a run that broke up, pymbar's ``detect_equilibration`` latches
    onto the breakup as the "equilibration" and returns a ``t0`` *later* than
    the moment the membrane stopped being a membrane (552 ns against a 401 ns
    stability time for charmm_npt1), which leaves no valid window at all.  Run
    on the stable window it returns a burn-in inside that window, as intended.
    """
    key = "%s_rep%d" % (system_key(ff, ens), rep)
    out = {}
    apl = npzs["area_per_lipid"]
    if key in apl.files:
        out["apl"] = np.asarray(apl[key], dtype=float).mean(axis=1)
    thick = npzs["memb_thickness"]
    if key in thick.files:
        out["thickness"] = np.asarray(thick[key], dtype=float)
    if records:
        for obs, A in list(out.items()):
            n = stab.stable_frames(records, ff, ens, rep, n_available=len(A))
            out[obs] = A[:n]
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--no-truncate", dest="truncate", action="store_false",
                    help="analyse the whole trajectory instead of each run's "
                         "stable window")
    args = ap.parse_args()

    ps = _parent_stats()
    analyze_series, decompose = ps.analyze_series, ps.decompose
    records = stab.load() if args.truncate else {}

    npzs = {}
    for name in ("area_per_lipid", "memb_thickness"):
        p = args.data_dir / ("lipprops_%s.npz" % name)
        if not p.exists():
            raise SystemExit("missing %s -- run lipid_properties.py first" % p)
        npzs[name] = np.load(p)

    results = {obs: {} for obs in OBSERVABLES}
    for ff in FORCEFIELDS:
        for ens in ENSEMBLES:
            sysname = system_key(ff, ens)
            for rep in REPS:
                for obs, A in series_for(npzs, ff, ens, rep, records).items():
                    row = analyze_series(A)
                    row["rep"] = rep
                    results[obs].setdefault(sysname, []).append(row)

    for obs in OBSERVABLES:
        if not results[obs]:
            continue
        print("\n=== %s  (t0 and g in frames; %.3g ns per frame) ===" % (obs, DT_NS))
        print("  %-16s %5s %8s %18s %20s" % ("system", "N", "t0 (ns)",
                                             "g_prod (mean+-SD)",
                                             "Neff_max (mean+-SD)"))
        for ff in FORCEFIELDS:
            for ens in ENSEMBLES:
                rows = results[obs].get(system_key(ff, ens))
                if not rows:
                    continue
                t0 = np.array([r["t0"] for r in rows]) * DT_NS
                g = np.array([r["g_prod"] for r in rows])
                nf = np.array([r["Neff_max"] for r in rows])
                print("  %-16s %5d %8.0f %9.1f +- %5.1f %11.1f +- %6.1f"
                      % (system_key(ff, ens), rows[0]["N"], t0.mean(),
                         g.mean(), g.std(ddof=0), nf.mean(), nf.std(ddof=0)))

    print("\n########  WITHIN-RUN vs BETWEEN-RUN  ########")
    decomp = {obs: {} for obs in OBSERVABLES}
    for obs in OBSERVABLES:
        if not results[obs]:
            continue
        print("\n=== %s (%s) ===" % (obs, UNITS[obs]))
        print("  %-16s %10s %10s %6s %10s  verdict"
              % ("system", "between-SD", "within-SEM", "ratio", "sigma_betw"))
        for ff in FORCEFIELDS:
            for ens in ENSEMBLES:
                sysname = system_key(ff, ens)
                rows = results[obs].get(sysname)
                if not rows:
                    continue
                d = decompose(rows)
                decomp[obs][sysname] = d
                print("  %-16s %10.4g %10.4g %6.2f %10.4g  %s"
                      % (sysname, d["between_sd"], d["within_sem"], d["ratio"],
                         d["sigma_between"], d["verdict"]))

    out = args.out or (args.data_dir / "equilibrium_timeseries_stats.json")
    out.write_text(json.dumps({"per_replicate": results,
                               "decomposition": decomp},
                              indent=2, sort_keys=True))
    print("\nwrote %s" % out)


if __name__ == "__main__":
    main()
