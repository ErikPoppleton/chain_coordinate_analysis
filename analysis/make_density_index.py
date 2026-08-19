#!/usr/bin/env python3
"""Build per-force-field/-lipid GROMACS index files for ``gmx density`` from the
full-system (water + ion) ``traj/{ff}_{lipid}.gro`` topologies (one replicate's
downloaded ``md.gro`` per system), one ``data/density_{ff}_{lipid}.ndx`` each.
Groups written:

    water          all water atoms          (TIP3 / TP3 / SOL)
    sodium         Na+ ions                 (SOD / Na+ / NA)
    magnesium      Mg2+ ions                (MG)
    calcium        Ca2+ ions                (CAL / CA)
    phosphorus     lipid phosphate P        (name P / P31)
    glycerol_C2    glycerol sn-2 carbon     (name C2)
    midpoint_C     mid-chain carbon         both chains: DOPC -> upper C=C carbon
                                            (C9); DMPC/DMPG -> C7
    terminal_C     last acyl carbon         both chains: C18 (DOPC) / C14 (DMPx)

The last two use constant group names (varying only the underlying carbon) so a
`gmx density` loop needs no per-lipid conditionals.

Atom naming (verified against the gro files):
  * CHARMM & SLipids -- one lipid residue; acyl carbons ``C2{n}`` (sn-1) /
    ``C3{n}`` (sn-2), glycerol ``C2``, phosphate ``P``.
  * AMBER (Lipid17) -- head residue (``PC``/``PGR``) carries glycerol ``C2`` and
    phosphate ``P31``; each of the two tail residues (``OL``/``MY``) carries the
    acyl carbons ``C1{n}`` (so a chain-position group matches 2x the lipid count).

Force field / lipid / water / ion names are detected from the residue names, so
no filename parsing is needed.  Uses ``MDAnalysis.selections.gromacs.Selection-
Writer`` to emit the ndx; groups are built from resname/name masks (not selection
strings) so the AMBER ``Na+``/``Cl-`` names don't trip the selection parser.

Run in the lipyphilic/MDAnalysis env, e.g.::

    conda run -n lipid64 python analysis/make_density_index.py
    python analysis/make_density_index.py traj/charmm_DOPC.gro   # a subset

Empty groups (an ion species absent from a system) are skipped with a warning,
since ``gmx density`` errors on an empty group.
"""

import sys
from pathlib import Path

import numpy as np
import MDAnalysis as mda
from MDAnalysis.selections.gromacs import SelectionWriter

# Force-field vocabulary lives in lipiddefs; re-exported here because the other
# index builders import it from this module.
from lipiddefs import (SCHEME, WATER, SODIUM, MAGNESIUM, CALCIUM,
                       PHOSPHATE_NAMES as PHOSPHATE,
                       TAIL_NC, AMBER_HEAD, AMBER_TAIL,
                       chain_atom_names, lipid_resnames, midpoint_carbon)

ROOT = Path(__file__).resolve().parent.parent
TRAJ = ROOT / "traj"          # canonical full-system gros: traj/{ff}_{lipid}.gro
DATA = ROOT / "data"


def detect(resnames):
    """Infer (ff, scheme, lipid) from the residue names present.  Water resnames
    distinguish CHARMM (TIP3) from SLipids (SOL); AMBER shows OL/MY/TP3."""
    resn = set(resnames)
    if resn & {"OL", "MY"} or "TP3" in resn:
        ff = "amber"
        lipid = "DOPC" if "OL" in resn else ("DMPG" if "PGR" in resn else "DMPC")
    elif "TIP3" in resn:
        ff = "charmm"
    elif "SOL" in resn:
        ff = "slipids"
    else:
        sys.exit(f"cannot detect force field from resnames {sorted(resn)}")
    scheme = SCHEME[ff]
    if scheme == "single":
        present = [l for l in TAIL_NC if l in resn]
        if len(present) != 1:
            sys.exit(f"cannot detect a single lipid resname (found {present})")
        lipid = present[0]
    return ff, scheme, lipid


def build_groups(u, scheme, lipid):
    """Return an ordered list of (name, AtomGroup) for the requested groups."""
    resn = u.atoms.resnames
    name = u.atoms.names
    lipid_resn = lipid_resnames(scheme, lipid)
    in_lipid = np.isin(resn, list(lipid_resn))

    def by_resname(names):
        return u.atoms[np.isin(resn, list(names))]

    def by_atomname(names):
        return u.atoms[np.isin(name, list(names)) & in_lipid]

    groups = [
        ("water", by_resname(WATER)),
        ("sodium", by_resname(SODIUM)),
        ("magnesium", by_resname(MAGNESIUM)),
        ("calcium", by_resname(CALCIUM)),
        ("phosphorus", by_atomname(PHOSPHATE)),
        ("glycerol_C2", by_atomname({"C2"})),
    ]
    # Constant group names (so a density loop needs no per-lipid conditionals);
    # only the underlying carbon position varies: DOPC -> upper C=C carbon (C9),
    # DMPC/DMPG -> mid-chain C7; terminal -> last acyl carbon (C18 / C14).
    mid = midpoint_carbon(lipid)
    groups.append(("midpoint_C", by_atomname(chain_atom_names(scheme, mid))))
    groups.append(("terminal_C", by_atomname(chain_atom_names(scheme, TAIL_NC[lipid]))))
    return groups


def process(gro, out_dir):
    u = mda.Universe(str(gro))
    ff, scheme, lipid = detect(u.atoms.resnames)
    groups = build_groups(u, scheme, lipid)
    out = out_dir / f"density_{ff}_{lipid}.ndx"
    written = []
    with SelectionWriter(str(out), mode="w") as ndx:
        for gname, ag in groups:
            if ag.n_atoms == 0:
                print(f"  WARNING: group '{gname}' is empty; skipping")
                continue
            ndx.write(ag, name=gname)
            written.append((gname, ag.n_atoms))
    print(f"wrote {out}  ({ff} {lipid})")
    for gname, n in written:
        print(f"    {gname:16s} {n:7d} atoms")
    return out


def main():
    args = sys.argv[1:]
    files = [Path(a) for a in args] if args else sorted(TRAJ.glob("*.gro"))
    if not files:
        sys.exit(f"no .gro files (looked in {TRAJ})")
    DATA.mkdir(parents=True, exist_ok=True)
    for gro in files:
        if not gro.exists():
            print(f"skip {gro}: not found")
            continue
        process(gro, DATA)


if __name__ == "__main__":
    main()
