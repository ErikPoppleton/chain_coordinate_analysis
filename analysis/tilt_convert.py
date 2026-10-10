#!/usr/bin/env python3
"""Stage 1 of the tilt-modulus pipeline: convert a water-containing GROMACS
trajectory into the topology-PDB + DCD pair that OpenStructure reads.

OpenStructure only reads PDB/CRD structures and CHARMM **DCD** trajectories, runs
under Python 2.7 (no MDAnalysis), and loads a multi-model PDB only by holding
every frame's entity in memory.  So we read the ``gro``+``xtc`` here (MDAnalysis,
fast binary xtc) and emit a compact ``data/tilt_{ff}_{lipid}.pdb`` (frame-0
topology) + ``.dcd`` (all frames) that OST streams frame-by-frame.

Force-field-specific handling (so ``ost_pymodules``' ``AnalyzeLipidTiltAndSplay``
sees each lipid as ONE residue with a head->tail vector inside it):

  * **Heavy atoms only.**  Hydrogens are dropped from lipids and water.  The
    water/lipid interface density (which defines the membrane normal) and the
    tilt director (phosphate/glycerol-C2 -> terminal carbons) are all heavy-atom
    quantities, and dropping H is what lets the AMBER merge below produce unique
    atom names.  Applied to all three force fields so the density is built
    identically for a fair comparison.

  * **AMBER residue merge.**  Lipid21 splits each lipid into three residues in
    tail-head-tail order (``MY``/``OL``, ``PC``/``PGR``, ``MY``/``OL``).  We give
    all three one resname (``LIP``) and a shared resid so OST reads them as one
    residue.  Both tails otherwise carry identical carbon names (``C1{n}``),
    which OST would merge -- so the **second** tail's carbons are renamed
    ``C1{n}`` -> ``CB{n}`` (kept starting with C so the element still reads as
    carbon).  The tilt selections then read the last-3 carbons of each chain as
    ``C1{12,13,14}`` (tail A) + ``CB{12,13,14}`` (tail B).  CHARMM/SLipids are
    already one residue per lipid with distinct sn-1 (``C2x``) / sn-2 (``C3x``)
    names and need no renaming.

  * **Water** is reduced to its oxygen and rebuilt as one residue per molecule
    with resids above the lipids.  This is required because the AMBER gro numbers
    *every* water with the same resid (a Lipid21/GROMACS quirk) -- MDAnalysis
    then sees one giant water residue and OST collapses all the identically named
    oxygens into a single atom.  Rebuilding one-residue-per-water fixes it and is
    harmless for CHARMM/SLipids (already per-molecule).  Ions are dropped.

The bilayer is assumed already whole and centred in z (cluster-side trjconv
output); no re-centring is done.

Run in the MDAnalysis env::

    conda run -n lipid64 python analysis/tilt_convert.py tmp/charmm_DMPC_md.gro traj/charmm_DMPC.xtc
    conda run -n lipid64 python analysis/tilt_convert.py TOP TRAJ -o data/tilt_charmm_DMPC_1
"""

import argparse
from pathlib import Path

import numpy as np
import MDAnalysis as mda

import make_density_index as mdi  # detect(ff/scheme/lipid) + resname tables

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

LIP_RESNAME = "LIP"   # uniform AMBER lipid resname on the OST side


def prepare(top, traj):
    """Return (merged_universe, u, lipids, waters, ff, lipid, n_lipids).

    ``merged`` holds the heavy lipid atoms (one residue per lipid, AMBER merged)
    + one-residue-per-molecule water oxygens, with names/resids/resnames fixed up
    for OST.  ``lipids``/``waters`` are the source views (into ``u``) whose
    positions are copied into ``merged`` frame-by-frame when writing the DCD."""
    u = mda.Universe(top, traj)
    ff, scheme, lipid = mdi.detect(u.atoms.resnames)

    lip_resn = ([mdi.AMBER_HEAD[lipid], mdi.AMBER_TAIL[lipid]]
                if scheme == "amber" else [lipid])
    heavy = "and not (name H*)"
    lipids = u.select_atoms("resname %s %s" % (" ".join(lip_resn), heavy))
    waters = u.select_atoms("resname %s %s" % (" ".join(sorted(mdi.WATER)), heavy))
    if lipids.n_atoms == 0 or waters.n_atoms == 0:
        raise SystemExit("no lipids (%d) or waters (%d) -- need a water-"
                         "containing topology" % (lipids.n_atoms, waters.n_atoms))

    lip_res = lipids.residues
    if scheme == "amber":
        n_res = lip_res.n_residues
        if n_res % 3 != 0:
            raise SystemExit("AMBER lipid residues (%d) not a multiple of 3"
                             % n_res)
        n_lipids = n_res // 3
        # residues run tail-head-tail per lipid; rename the 2nd tail's carbons
        # C1{n} -> CB{n} so both tails are distinct in the merged residue.
        loc = np.unique(lipids.resindices, return_inverse=True)[1]  # 0..n_res-1
        names = np.array(lipids.names, dtype=object)
        heads = {mdi.AMBER_HEAD[lipid]}
        for j in range(n_lipids):
            if lip_res[3 * j + 1].resname not in heads:
                raise SystemExit("unexpected AMBER residue order at lipid %d" % j)
        for i in np.where(loc % 3 == 2)[0]:
            if names[i].startswith("C1"):
                names[i] = "CB" + names[i][2:]
        lipids.names = list(names)
        lip_res.resids = (np.arange(n_res) // 3 + 1).tolist()
        lip_res.resnames = [LIP_RESNAME] * n_res
    else:
        n_lipids = lip_res.n_residues

    # Rebuild water as one residue per oxygen (see module docstring: the AMBER
    # gro lumps all waters into one residue).  resids stacked above the lipids.
    wn = waters.n_atoms
    wu = mda.Universe.empty(wn, n_residues=wn, atom_resindex=np.arange(wn),
                            trajectory=True)
    wu.add_TopologyAttr("names", list(waters.names))
    wu.add_TopologyAttr("resnames", list(waters.resnames))
    wu.add_TopologyAttr("resids", (np.arange(wn) + n_lipids + 1).tolist())
    wu.atoms.positions = waters.positions

    merged = mda.Merge(lipids, wu.atoms)
    return merged, u, lipids, waters, ff, lipid, n_lipids


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("top")
    ap.add_argument("traj")
    ap.add_argument("-o", "--out", default=None,
                    help="output basename (default data/tilt_{ff}_{lipid})")
    args = ap.parse_args()

    merged, u, lipids, waters, ff, lipid, n_lipids = prepare(args.top, args.traj)
    print("detected %s %s: %d lipids, %d atoms kept (heavy lipid + water O)"
          % (ff, lipid, n_lipids, merged.atoms.n_atoms))

    DATA.mkdir(parents=True, exist_ok=True)
    base = Path(args.out) if args.out else DATA / ("tilt_%s_%s" % (ff, lipid))
    # Carry the box into the frames so the DCD records it -- the no-tile
    # analysis runs PBC=False and ignores it, but the PBC-tiling path
    # (ExtendTrajectoryToNeighboringUnitCells) needs the per-frame cell vectors.
    merged.dimensions = u.dimensions
    merged.atoms.write(str(base) + ".pdb")         # frame-0 topology (fixed up)
    nframes = 0
    with mda.Writer(str(base) + ".dcd", merged.atoms.n_atoms) as w:
        for _ in u.trajectory:
            merged.dimensions = u.dimensions
            merged.atoms.positions = np.concatenate(
                [lipids.positions, waters.positions])
            w.write(merged.atoms)
            nframes += 1
    print("wrote %s.pdb + %s.dcd  (%d frames)" % (base, base, nframes))


if __name__ == "__main__":
    main()
