#!/usr/bin/env python3
"""Renumber the lipid residues of an AMBER (Lipid17) GROMACS ``.gro``.

GROMACS writes every Lipid17 residue with resid ``1``, and a lipid's three
residues (tail ``OL``/``MY`` -- head ``PC``/``PGR`` -- tail) sit contiguously,
so the sn-2 tail of one lipid and the sn-1 tail of the next merge into a single
80-atom residue as far as MDAnalysis is concerned.  That breaks every tool that
reasons per residue: `make_lipidator_index` and `tilt_convert` both refuse
(``lipid residues not a multiple of 3``), and `core_thickness` would key its
leaflet assignment off the wrong atoms.

This restores the numbering the working copies already carry: **resid = the
1-based lipid number**, shared by that lipid's three residues (so a residue
boundary is a resname change *within* a lipid and a resid change *between*
lipids -- exactly what MDAnalysis splits on).  Residue boundaries are found the
way `make_order_index` finds them, ignoring the resid entirely: a new residue
starts whenever the resname changes **or an atom name repeats**.

Only the leading contiguous lipid block is touched -- solvent keeps whatever
numbering it has (`tilt_convert` rebuilds water per molecule itself).  The file
is rewritten in place, and only columns 1-5 of the lipid lines change.

Usage:
    python3 analysis/renumber_gro.py traj/amber_DMPG.gro
    python3 analysis/renumber_gro.py traj/*.gro     # non-AMBER files are skipped
    python3 analysis/renumber_gro.py --check traj/amber_DMPG.gro   # report only
"""

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lipiddefs import AMBER_HEAD, AMBER_TAIL

HEADS = set(AMBER_HEAD.values())
TAILS = set(AMBER_TAIL.values())
LIPID_RESN = HEADS | TAILS


def split_lipid_residues(lines):
    """[(resname, [line indices])] for the leading contiguous lipid block."""
    residues = []
    seen = set()
    cur_resn = None
    for i, line in enumerate(lines):
        resn = line[5:10].strip()
        name = line[10:15].strip()
        if resn not in LIPID_RESN:
            break                      # end of the leading lipid block
        if resn != cur_resn or name in seen:
            residues.append((resn, []))
            seen = set()
            cur_resn = resn
        residues[-1][1].append(i)
        seen.add(name)
    return residues


def renumber(path, check=False):
    text = path.read_text().splitlines(True)
    if len(text) < 3:
        sys.exit(f"{path}: not a .gro file")
    head, atoms, tail = text[:2], text[2:-1], text[-1:]

    residues = split_lipid_residues(atoms)
    if not residues:
        print(f"  {path.name}: no AMBER lipid residues -- skipped")
        return False

    resnames = [r for r, _ in residues]
    if len(residues) % 3:
        sys.exit(f"{path}: {len(residues)} lipid residues, not a multiple of 3")
    for j in range(0, len(residues), 3):
        if not (resnames[j] in TAILS and resnames[j + 1] in HEADS
                and resnames[j + 2] in TAILS):
            sys.exit(f"{path}: lipid {j // 3 + 1} is not tail-head-tail "
                     f"({resnames[j:j + 3]})")
    n_lipids = len(residues) // 3

    changed = 0
    for j, (_, idx) in enumerate(residues):
        resid = "%5d" % (j // 3 + 1)
        for i in idx:
            if atoms[i][:5] != resid:
                atoms[i] = resid + atoms[i][5:]
                changed += 1

    print(f"  {path.name}: {n_lipids} lipids / {len(residues)} residues, "
          f"{changed} atom lines renumbered")
    if changed and not check:
        path.write_text("".join(head + atoms + tail))
    return bool(changed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gro", nargs="+", type=Path)
    ap.add_argument("--check", action="store_true",
                    help="report what would change without rewriting")
    ap.add_argument("--backup", type=Path, default=None,
                    help="copy each original here before rewriting")
    args = ap.parse_args()

    for gro in args.gro:
        if args.backup and not args.check:
            args.backup.mkdir(parents=True, exist_ok=True)
            shutil.copy2(gro, args.backup / gro.name)
        renumber(gro, check=args.check)


if __name__ == "__main__":
    main()
