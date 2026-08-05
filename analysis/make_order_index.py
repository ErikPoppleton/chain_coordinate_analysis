#!/usr/bin/env python3
"""Generate the GROMACS index (.ndx) file that g_lomepro's ``-order`` (S_CD /
S_CD deuterium order parameter) analysis needs, for the lipids/force fields in
this study.

Why this exists
---------------
The equilibrium branch (``lipid_properties.py``) computes the **S_CC** order
parameter (orientation of each C-C bond to the bilayer normal).  g_lomepro
instead computes **S_CD** (the H-C-H / deuterium order parameter), which is what
NMR experiments report, using the molecular-frame construction of Gapsys et al.
(``reference/gapsys2013computational.pdf``).  g_lomepro reads its acyl-chain
atoms from a GROMACS index file; this script writes that file.

What g_lomepro's ``-order`` expects (verified against the g_lomepro source,
``order.c`` / ``g_lomepro.c``)
------------------------------------------------------------------------------
* Two groups, **sn-1 chain** then **sn-2 chain**, each listing that chain's
  carbons *per lipid, in chain order*, concatenated lipid after lipid
  (``idorder1[i + j*order_atom_num1]`` -> block ``j`` is lipid ``j``'s chain).
* For carbon at chain position ``i`` it builds the frame from neighbours
  ``i-1`` and ``i+1`` (loop ``for(i=1; i<N-1; i++)``), so the **first and last
  atom of each chain get no S_CD** -- they serve only as neighbours.  To obtain
  an S_CD for the first aliphatic carbon (C2), the **carbonyl carbon is included
  as the first atom** of each chain.  A 14-carbon chain (C1 carbonyl .. C14)
  therefore yields S_CD for C2..C13 -- matching the paper's C2/C7/C13 labels.
* ``-unsat`` (0/1/2) = number of unsaturated chains.  Each double bond needs a
  separate group of **exactly two atoms per lipid**: the two ``sp2`` carbons
  ``Ci`` then ``Ci+1`` (``idunsat[found_first_unsat + j*2]``).  For ``-unsat 1``
  the unsaturated chain must be sn-1; for ``-unsat 2`` g_lomepro reads the sn-1
  double-bond group first, then the sn-2 one.

The three lipids here are symmetric (di-oleoyl DOPC, di-myristoyl DMPC/DMPG), so
which chain is called sn-1 vs sn-2 is immaterial -- both carry the same fatty
acid.  DOPC is oleoyl (18:1 cis-Delta9) -> ``-unsat 2``; DMPC/DMPG are myristoyl
(14:0) -> ``-unsat 0``.

Atom naming (verified geometrically against ``snapshots/*.gro``)
---------------------------------------------------------------
* CHARMM & SLipids: one residue per lipid.  sn-1 = ``C21``(carbonyl)..``C2N``,
  sn-2 = ``C31``..``C3N`` (N=18 DOPC, 14 DMPC/DMPG).  DOPC double bond
  ``C29=C210`` (sn-1), ``C39=C310`` (sn-2).
* AMBER (Lipid17): each lipid is three residues -- tail, head, tail.  The two
  carbonyls live in the head (``PC``/``PGR``): ``C11`` for the chain *before*
  the head, ``C21`` for the chain *after*.  Tail residues (``OL`` oleoyl / ``MY``
  myristoyl) name their carbons ``C1x`` = fatty-acid carbon ``x`` (C12..C1N).
  So sn-1 = head ``C11`` + tail ``C12..C1N``; sn-2 = head ``C21`` + tail
  ``C12..C1N``.  DOPC double bond ``C19=C110`` in each tail.  In the raw
  ``md.gro`` every residue is numbered ``1`` and adjacent tails merge, so
  residues are split on atom-name restart, not on resid.

Usage
-----
    python3 analysis/make_order_index.py md.gro                 # auto-detect
    python3 analysis/make_order_index.py md.gro -o order.ndx
    python3 analysis/make_order_index.py md.gro --ff amber --lipid DOPC

Index serials are 1-based positions in the file, i.e. GROMACS atom serials, so
generate the ``.ndx`` from the *same* structure whose atom order matches the
trajectory you pass to g_lomepro (``md.gro`` matches ``inbox.xtc``).

Requires only numpy + the standard library.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# Aliphatic carbons per acyl chain, counting the carbonyl as C1 (so a 14:0 tail
# is C1..C14).  DOPC is oleoyl (18:1), DMPC/DMPG are myristoyl (14:0).
TAIL_NC = {"DOPC": 18, "DMPC": 14, "DMPG": 14}
# cis double-bond position: carbons (d, d+1) are the two sp2 carbons.  Oleoyl is
# Delta9 -> C9=C10.  Saturated lipids are absent from this dict.
DOUBLE_BOND = {"DOPC": 9}

# Force-field residue vocabulary.
AMBER_HEAD = {"DOPC": "PC", "DMPC": "PC", "DMPG": "PGR"}
AMBER_TAIL = {"DOPC": "OL", "DMPC": "MY", "DMPG": "MY"}
PHOSPHATE_NAMES = {"P", "P31"}  # CHARMM/SLipids "P", AMBER "P31"


def read_gro(path):
    """Return a list of atoms ``(serial, resname, name, xyz)`` with 1-based
    serials = file order (= GROMACS atom serials, what an .ndx refers to)."""
    lines = Path(path).read_text().splitlines()
    if len(lines) < 3:
        sys.exit(f"{path}: not a valid .gro file")
    natoms = int(lines[1])
    atoms = []
    for i, ln in enumerate(lines[2:2 + natoms]):
        resname = ln[5:10].strip()
        name = ln[10:15].strip()
        xyz = np.array([float(ln[20:28]), float(ln[28:36]), float(ln[36:44])])
        atoms.append((i + 1, resname, name, xyz))
    if len(atoms) != natoms:
        sys.exit(f"{path}: header says {natoms} atoms but found {len(atoms)}")
    return atoms


def split_residues(atoms):
    """Split the atom stream into residues *without trusting resid*, which is
    unreliable in raw Lipid17 output (every residue numbered 1, adjacent tails
    merging).  A new residue starts whenever the residue name changes or an atom
    name repeats (each residue carries every atom name at most once).

    Returns a list of ``(resname, {name: (serial, xyz)})`` in file order."""
    residues = []
    cur = None
    cur_resn = None
    for serial, resname, name, xyz in atoms:
        if resname != cur_resn or name in cur:
            cur = {}
            residues.append((resname, cur))
            cur_resn = resname
        cur[name] = (serial, xyz)
    return residues


def detect(atoms):
    """Infer (scheme, lipid) from the residue names present.  CHARMM and SLipids
    share the single-residue naming scheme, so only 'amber' vs 'single' matters
    for building the index; the lipid identity sets tail length / unsaturation."""
    resn = {a[1] for a in atoms}
    if resn & {"OL", "MY"}:
        scheme = "amber"
        lipid = "DOPC" if "OL" in resn else ("DMPG" if "PGR" in resn else "DMPC")
    else:
        scheme = "single"
        present = [l for l in TAIL_NC if l in resn]
        if len(present) != 1:
            sys.exit("could not detect a single lipid resname (found "
                     f"{present or 'none'}); pass --ff/--lipid explicitly")
        lipid = present[0]
    return scheme, lipid


def _pick(res_map, names, ctx):
    out = []
    for n in names:
        if n not in res_map:
            sys.exit(f"{ctx}: atom '{n}' missing from residue "
                     f"(has {sorted(res_map)})")
        out.append(res_map[n][0])
    return out


def build_single(residues, lipid):
    """CHARMM/SLipids: sn-1 = C21..C2N, sn-2 = C31..C3N within one residue."""
    n = TAIL_NC[lipid]
    sn1_names = [f"C2{k}" for k in range(1, n + 1)]  # C21 (carbonyl) first
    sn2_names = [f"C3{k}" for k in range(1, n + 1)]
    lip_res = [rm for rn, rm in residues if rn == lipid]
    sn1, sn2, db1, db2 = [], [], [], []
    for rm in lip_res:
        sn1 += _pick(rm, sn1_names, f"{lipid} sn-1")
        sn2 += _pick(rm, sn2_names, f"{lipid} sn-2")
        if lipid in DOUBLE_BOND:
            d = DOUBLE_BOND[lipid]
            db1 += _pick(rm, [f"C2{d}", f"C2{d + 1}"], f"{lipid} sn-1 =")
            db2 += _pick(rm, [f"C3{d}", f"C3{d + 1}"], f"{lipid} sn-2 =")
    membrane = sorted(s for rm in lip_res for (s, _) in rm.values())
    return len(lip_res), sn1, sn2, db1, db2, membrane


def build_amber(residues, lipid):
    """AMBER Lipid17: pair each head (PC/PGR) with its two flanking tail residues
    and split them into the C11-chain (sn-1) and C21-chain (sn-2) by which
    carbonyl each tail's first carbon (C12) sits next to."""
    head_rn, tail_rn = AMBER_HEAD[lipid], AMBER_TAIL[lipid]
    n = TAIL_NC[lipid]
    tail_names = [f"C1{k}" for k in range(2, n + 1)]  # C12..C1N (no carbonyl)

    def carbonyl(head_map, tail_map, ctx):
        """Return which head carbonyl (C11 or C21) this tail belongs to."""
        c12 = tail_map.get("C12")
        if c12 is None:
            sys.exit(f"{ctx}: tail residue lacks C12")
        cands = [(c, head_map[c][1]) for c in ("C11", "C21") if c in head_map]
        if len(cands) != 2:
            sys.exit(f"{ctx}: head lacks C11/C21 (has {sorted(head_map)})")
        return min(cands, key=lambda kv: np.linalg.norm(c12[1] - kv[1]))[0]

    sn1, sn2, db1, db2, membrane = [], [], [], [], []
    nlip = 0
    for p, (rn, rm) in enumerate(residues):
        if rn != head_rn:
            continue
        before = next((residues[q] for q in range(p - 1, -1, -1)
                       if residues[q][0] == tail_rn), None)
        after = next((residues[q] for q in range(p + 1, len(residues))
                      if residues[q][0] == tail_rn), None)
        if before is None or after is None:
            sys.exit(f"AMBER {lipid}: head #{nlip + 1} is not flanked by two "
                     f"{tail_rn} tails")
        # Assign each flanking tail to its carbonyl; expect one C11 and one C21.
        pair = {}
        for tail in (before[1], after[1]):
            pair[carbonyl(rm, tail, f"AMBER {lipid}")] = tail
        if set(pair) != {"C11", "C21"}:
            sys.exit(f"AMBER {lipid}: head #{nlip + 1} tails both map to the "
                     "same carbonyl")
        t1, t2 = pair["C11"], pair["C21"]
        sn1 += [rm["C11"][0]] + _pick(t1, tail_names, f"{lipid} sn-1 tail")
        sn2 += [rm["C21"][0]] + _pick(t2, tail_names, f"{lipid} sn-2 tail")
        if lipid in DOUBLE_BOND:
            d = DOUBLE_BOND[lipid]
            db1 += _pick(t1, [f"C1{d}", f"C1{d + 1}"], f"{lipid} sn-1 =")
            db2 += _pick(t2, [f"C1{d}", f"C1{d + 1}"], f"{lipid} sn-2 =")
        membrane += ([s for (s, _) in rm.values()]
                     + [s for (s, _) in t1.values()]
                     + [s for (s, _) in t2.values()])
        nlip += 1
    return nlip, sn1, sn2, db1, db2, sorted(membrane)


def phosphates(atoms):
    return [s for (s, rn, nm, _) in atoms if nm in PHOSPHATE_NAMES]


def write_group(fh, name, serials):
    fh.write(f"[ {name} ]\n")
    for i in range(0, len(serials), 15):
        fh.write(" ".join(str(s) for s in serials[i:i + 15]) + "\n")
    fh.write("\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("structure", help="GROMACS .gro whose atom order matches the "
                                       "trajectory (e.g. md.gro)")
    ap.add_argument("-o", "--out", help="output .ndx (default: order_{ff}_{lipid}.ndx)")
    ap.add_argument("--ff", choices=["charmm", "amber", "slipids"],
                    help="override force-field detection")
    ap.add_argument("--lipid", choices=list(TAIL_NC),
                    help="override lipid detection")
    args = ap.parse_args()

    atoms = read_gro(args.structure)
    scheme, lipid = detect(atoms)
    if args.lipid:
        lipid = args.lipid
    if args.ff:
        scheme = "amber" if args.ff == "amber" else "single"
    ff = args.ff or ("amber" if scheme == "amber" else "charmm/slipids")

    residues = split_residues(atoms)
    builder = build_amber if scheme == "amber" else build_single
    nlip, sn1, sn2, db1, db2, membrane = builder(residues, lipid)
    if nlip == 0:
        sys.exit(f"no {lipid} lipids found in {args.structure}")

    # Every lipid must contribute the same chain length (g_lomepro divides
    # norder/lipid_num and assumes uniform blocks).
    per1, per2 = len(sn1) // nlip, len(sn2) // nlip
    if per1 * nlip != len(sn1) or per2 * nlip != len(sn2):
        sys.exit("chains are not uniform across lipids -- broken/incomplete "
                 "residues in the structure?")

    out = Path(args.out) if args.out else Path(f"order_{ff.replace('/', '_')}_{lipid}.ndx")
    unsat = 2 if lipid in DOUBLE_BOND else 0
    with open(out, "w") as fh:
        write_group(fh, "Membrane", membrane)
        write_group(fh, "Phosphates", phosphates(atoms))
        write_group(fh, "sn1", sn1)
        write_group(fh, "sn2", sn2)
        if unsat:
            write_group(fh, "sn1_double_bond", db1)
            write_group(fh, "sn2_double_bond", db2)

    print(f"wrote {out}")
    print(f"  force field : {ff}")
    print(f"  lipid       : {lipid}  ({nlip} lipids)")
    print(f"  chain atoms : sn1={per1}, sn2={per2} per lipid "
          f"(carbonyl + C2..C{TAIL_NC[lipid]}) -> S_CD for C2..C{TAIL_NC[lipid] - 1}")
    print(f"  g_lomepro   : run with  -unsat {unsat}")
    order = ["Membrane (lipid group)", "sn1", "sn2"]
    if unsat:
        order += ["sn1_double_bond", "sn2_double_bond"]
    print("  group order g_lomepro will prompt for (order-parameter part):")
    print("    " + " -> ".join(order))


if __name__ == "__main__":
    main()
