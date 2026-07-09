#!/usr/bin/env python3
"""
Reorder the atoms of every DOPC molecule in a GROMACS .gro file so that the
per-molecule atom order matches the order defined in a .itp topology file,
then renumber the (global) atom-index column sequentially.

Only DOPC residues are touched; ions, water, and everything else are left
exactly where they are. Reordering is done by atom name, which is unique
within a DOPC molecule, so the mapping is unambiguous.

GRO fixed-column format (per line):
    cols  0:5   residue number
    cols  5:10  residue name
    cols 10:15  atom name
    cols 15:20  atom number  (renumbered here)
    cols 20:    coordinates (+ optional velocities) -- preserved verbatim

Usage:
    python3 reorder_dopc.py [input.gro] [topology.itp] [output.gro]
Defaults: step5_input.gro  DOPC.itp  step5_input_reordered.gro
"""
import sys

RESNAME = "DOPC"


def parse_itp_order(itp_path):
    """Return the list of DOPC atom names in the order given by the [ atoms ]
    section of the .itp file."""
    names = []
    in_atoms = False
    with open(itp_path) as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith("["):
                # a new directive block begins
                in_atoms = stripped.strip("[] \t").startswith("atoms")
                continue
            if not in_atoms:
                continue
            if not stripped or stripped.startswith(";"):
                continue
            # columns: nr type resnr residue atom cgnr charge mass
            names.append(stripped.split()[4])
    return names


def main():
    in_gro = sys.argv[1] if len(sys.argv) > 1 else "step5_input.gro"
    itp = sys.argv[2] if len(sys.argv) > 2 else "DOPC.itp"
    out_gro = sys.argv[3] if len(sys.argv) > 3 else "step5_input_reordered.gro"

    itp_order = parse_itp_order(itp)
    itp_index = {name: i for i, name in enumerate(itp_order)}
    n_per_mol = len(itp_order)
    print(f"Loaded {n_per_mol} atom names from {itp}")

    with open(in_gro) as f:
        lines = f.readlines()

    title = lines[0]
    natoms = int(lines[1])
    body = lines[2:2 + natoms]
    box = lines[2 + natoms:]

    # Reorder DOPC residues in place. Because the file lists all atoms of a
    # residue contiguously, we process one residue-block at a time.
    out_body = []
    i = 0
    n_reordered = 0
    while i < len(body):
        resname = body[i][5:10].strip()
        if resname != RESNAME:
            out_body.append(body[i])
            i += 1
            continue

        block = body[i:i + n_per_mol]

        # sanity checks
        resnum = block[0][0:5]
        if any(ln[5:10].strip() != RESNAME or ln[0:5] != resnum for ln in block):
            raise ValueError(
                f"DOPC residue starting at body line {i} is not {n_per_mol} "
                f"contiguous atoms of a single residue")
        block_names = [ln[10:15].strip() for ln in block]
        if set(block_names) != set(itp_order):
            raise ValueError(
                f"Atom-name set of DOPC residue at body line {i} does not "
                f"match the topology")

        block_sorted = sorted(block, key=lambda ln: itp_index[ln[10:15].strip()])
        out_body.extend(block_sorted)
        n_reordered += 1
        i += n_per_mol

    print(f"Reordered {n_reordered} {RESNAME} molecules")

    # Renumber the global atom-index column (cols 15:20), wrapping at 100000
    # as GROMACS does.
    for idx in range(len(out_body)):
        ln = out_body[idx]
        atomnum = (idx + 1) % 100000
        out_body[idx] = ln[:15] + f"{atomnum:>5d}" + ln[20:]

    with open(out_gro, "w") as f:
        f.write(title)
        f.write(f"{natoms}\n")
        f.writelines(out_body)
        f.writelines(box)

    print(f"Wrote {out_gro} ({natoms} atoms)")


if __name__ == "__main__":
    main()
