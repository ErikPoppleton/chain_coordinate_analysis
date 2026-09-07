#!/usr/bin/env python3
"""Run the Lipidator ReSiS tool over all 9 systems x 3 replicates and collect
the bilayer tilt modulus.  Reads the GROMACS xtc natively (no conversion), using
the per-force-field director libraries in ``~/software/lipidator-toolkit/
directors/{ff}.ldx`` and the index files ``data/lipidator_{ff}_{lipid}.ndx``
(from ``make_lipidator_index.py``).

Writes:
  * ``data/lipidator_moduli.json``          -- per-replicate tilt kappa + APL
  * ``data/lipidator_moduli_summary.json``  -- mean +/- SD across replicates

The tilt modulus is reported by ReSiS **area-normalised in k_BT/nm^2** (contrast
the per-lipid k_BT/rad^2 from the ost_pymodules pipeline; the two relate by
kappa[k_BT/nm^2] = chi[k_BT/rad^2]/A_lipid).  Requires only the stdlib.

    python3 analysis/run_lipidator.py
"""

import argparse
import json
import re
import subprocess
from collections import defaultdict
from pathlib import Path

from lipiddefs import FORCEFIELDS, LIPIDS, REPS

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
TRAJ = ROOT / "traj"
LIPKIT = Path.home() / "software/lipidator-toolkit"
RESIS = LIPKIT / "build/resis"
DIRECTORS = LIPKIT / "directors"

TILT_RE = re.compile(r"Bilayer Tilt:\s*kappa\s+([-\d.eE]+)\s+kt/nm")
AREA_RE = re.compile(r"Per Lipid \[bilayer\]:\s+([-\d.eE]+)")


def run_one(ff, lipid, rep):
    xtc = TRAJ / ("%s_%s%d.xtc" % (ff, lipid, rep))
    ldx = DIRECTORS / ("%s.ldx" % ff)
    ndx = DATA / ("lipidator_%s_%s.ndx" % (ff, lipid))
    for f in (xtc, ldx, ndx):
        if not f.exists():
            print("  skip %s_%s_%d: missing %s" % (ff, lipid, rep, f.name))
            return None
    proc = subprocess.run([str(RESIS), str(xtc), str(ldx), str(ndx)],
                          capture_output=True, text=True)
    m, a = TILT_RE.search(proc.stdout), AREA_RE.search(proc.stdout)
    if proc.returncode != 0 or not m:
        print("  FAIL %s_%s_%d (exit %d)%s" % (ff, lipid, rep, proc.returncode,
              " -- no tilt line" if proc.returncode == 0 else ""))
        return None
    tilt, area = float(m.group(1)), (float(a.group(1)) if a else None)
    print("  %s_%s_%d: tilt = %.3f kt/nm^2  (APL %.1f A^2)" %
          (ff, lipid, rep, tilt, area or 0))
    return {"ff": ff, "lipid": lipid, "rep": rep,
            "tilt_kt_per_nm2": tilt, "apl_A2": area}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--forcefields", nargs="+", default=FORCEFIELDS)
    ap.add_argument("--lipids", nargs="+", default=LIPIDS)
    ap.add_argument("--reps", nargs="+", type=int, default=REPS)
    args = ap.parse_args()

    if not RESIS.exists():
        raise SystemExit("resis binary not found at %s" % RESIS)
    DATA.mkdir(parents=True, exist_ok=True)
    # Merge into any existing file: a subset run (e.g. one lipid re-run after a
    # rebuild) must not drop the systems it skipped.  The summary below is then
    # recomputed from the merged set, so it always covers all 9 systems.
    moduli = DATA / "lipidator_moduli.json"
    results = json.loads(moduli.read_text()) if moduli.exists() else {}
    for ff in args.forcefields:
        for lipid in args.lipids:
            for rep in args.reps:
                res = run_one(ff, lipid, rep)
                if res:
                    results["%s_%s_%d" % (ff, lipid, rep)] = res
    with open(moduli, "w") as fh:
        json.dump(results, fh, indent=2, sort_keys=True)

    # aggregate mean +/- SD across replicates (project convention)
    import statistics as st
    by = defaultdict(list)
    for r in results.values():
        by[(r["ff"], r["lipid"])].append(r["tilt_kt_per_nm2"])
    summary = {}
    for (ff, lipid), vals in by.items():
        summary["%s_%s" % (ff, lipid)] = {
            "tilt_kt_per_nm2": st.fmean(vals),
            "sd_kt_per_nm2": st.pstdev(vals) if len(vals) > 1 else 0.0,
            "n_rep": len(vals)}
    with open(DATA / "lipidator_moduli_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)
    print("\nwrote data/lipidator_moduli.json (%d reps) + _summary.json"
          % len(results))


if __name__ == "__main__":
    main()
