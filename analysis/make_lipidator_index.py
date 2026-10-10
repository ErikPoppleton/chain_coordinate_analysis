#!/usr/bin/env python3
"""Build GROMACS index files for the Lipidator toolkit: one group per lipid
type (named to match the director library) listing the **1-based global index
of the first atom of each lipid**.  The toolkit adds the director's within-lipid
offsets to these, so the reference must be each lipid's first atom in trajectory
order.

  * CHARMM/SLipids: one residue per lipid -> the residue's first atom.
  * AMBER (Lipid21): three residues per lipid (tail-head-tail); we group the
    lipid residues into consecutive triples in index order and take the first
    atom of each triple.  (Residue info is only used here to *build* the index;
    the toolkit itself reads only the xtc.)

Writes ``data/lipidator_{ff}_{lipid}.ndx`` from ``traj/{ff}_{lipid}.gro``.

    conda run -n lipid64 python analysis/make_lipidator_index.py            # all 9
    conda run -n lipid64 python analysis/make_lipidator_index.py traj/amber_DMPC.gro
"""

import sys
from pathlib import Path

import numpy as np
import MDAnalysis as mda
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_density_index as mdi
from lipiddefs import FORCEFIELDS, LIPIDS, base_lipid

ROOT = Path(__file__).resolve().parent.parent
TRAJ = ROOT / "traj"
DATA = ROOT / "data"


def first_atoms(gro, label=None):
    """Return (ff, lipid, list of 1-based global first-atom indices per lipid).

    ``label`` overrides the detected lipid name for the *output filename* only
    (e.g. ``DMPG_JC`` for the Na+/JC ion condition, whose residues are still
    plain ``DMPG``); the index group inside keeps the base lipid name, because
    ReSiS matches that group against a director block in the shipped
    ``{ff}.ldx`` library.
    """
    u = mda.Universe(str(gro))
    ff, scheme, lipid = mdi.detect(u.atoms.resnames)
    if scheme == "amber":
        resn = [mdi.AMBER_HEAD[lipid], mdi.AMBER_TAIL[lipid]]
        res = u.select_atoms("resname %s" % " ".join(resn)).residues
        res = sorted(res, key=lambda r: r.atoms.indices[0])   # index order
        if len(res) % 3 != 0:
            sys.exit("AMBER %s: %d lipid residues, not a multiple of 3"
                     % (lipid, len(res)))
        firsts = [res[3 * j].atoms.indices[0] + 1 for j in range(len(res) // 3)]
    else:
        res = u.select_atoms("resname %s" % lipid).residues
        firsts = [r.atoms.indices[0] + 1 for r in res]
    return ff, (label or lipid), firsts


def write_ndx(ff, lipid, firsts):
    out = DATA / ("lipidator_%s_%s.ndx" % (ff, lipid))
    with open(out, "w") as f:
        # Group name = base lipid: ReSiS looks it up in the director library.
        f.write("[ %s ]\n" % base_lipid(lipid))
        for i in range(0, len(firsts), 15):
            f.write(" ".join(str(x) for x in firsts[i:i + 15]) + "\n")
    print("wrote %s  ([ %s ], %d lipids, first index %d)"
          % (out, lipid, len(firsts), firsts[0]))


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    args = sys.argv[1:]
    gros = [Path(a) for a in args] if args else [
        TRAJ / ("%s_%s.gro" % (ff, lip)) for ff in FORCEFIELDS for lip in LIPIDS]
    for gro in gros:
        if not gro.exists():
            print("skip %s: not found" % gro)
            continue
        # The filename already carries the label ({ff}_{label}.gro), which is
        # what distinguishes an ion condition (DMPG_JC) from the base system.
        stem = gro.stem
        label = (stem.split("_", 1)[1]
                 if stem.split("_", 1)[0] in FORCEFIELDS and "_" in stem
                 else None)
        ff, lipid, firsts = first_atoms(gro, label)
        write_ndx(ff, lipid, firsts)


if __name__ == "__main__":
    main()
