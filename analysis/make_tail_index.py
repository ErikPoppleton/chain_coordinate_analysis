#!/usr/bin/env python3
"""Build a GROMACS ``.ndx`` of the **hydrophobic acyl-tail atoms** for each force
field / lipid, for a ``gmx density`` mass-density profile of the membrane core.

This feeds the *second* d_HC (hydrophobic-core thickness) metric: the tail mass
density profile is a slab centred on the bilayer mid-plane, and d_HC is taken as
the **width of that profile at a density threshold of 500 kg/m^3** (last 50 ns of
the equilibration, per the reference method).  ``gmx density -dens mass`` reads
the masses from the ``.tpr`` and the atom list from this ``.ndx``.

Writes ``data/tail_{ff}_{lipid}.ndx`` with two groups:
  * ``tails``    -- every aliphatic acyl-chain atom of both chains **plus their
                    hydrogens** (all-atom, so the kg/m^3 scale matches an all-atom
                    tail density and the 500 kg/m^3 threshold is meaningful).  The
                    ester **carbonyl and the glycerol/head are excluded** -- those
                    are the polar interface, not the hydrophobic core.
  * ``Membrane`` -- all lipid atoms (a convenient centering group for
                    ``gmx density -center``).

Tail atom set per force field (verified against the ``traj/*.gro`` naming):
  * CHARMM / SLipids -- one lipid residue; aliphatic carbons ``C2{2..N}`` (sn-1)
    and ``C3{2..N}`` (sn-2) -- i.e. everything past the carbonyl ``C2 1``/``C3 1``
    -- plus their bonded hydrogens (found by geometry, so the differing CHARMM vs
    SLipids H-naming does not matter).
  * AMBER (Lipid21) -- each tail is its own residue ``OL`` (DOPC) / ``MY`` (DMPx)
    holding exactly the aliphatic carbons ``C1{2..N}`` + hydrogens (the carbonyl
    ``C1 1``/``C2 1`` lives in the head residue), so the whole tail residue *is*
    the hydrophobic tail -- no name masking or bond guessing needed.

Serials are 1-based file positions == GROMACS serials, so generate the ``.ndx``
from the same structure whose atom order matches the trajectory -- here the
water-containing ``traj/{ff}_{lipid}.gro`` (matches ``traj/{ff}_{lipid}{rep}.xtc``
and ``traj/{ff}_{lipid}.tpr``).  Output lands in the gitignored ``data/``.

    python3 analysis/make_tail_index.py                    # all traj/*.gro
    python3 analysis/make_tail_index.py traj/charmm_DMPC.gro   # a subset

Requires numpy + MDAnalysis.  ``gmx density`` itself is run separately (the tpr
here is a chain-coordinate-fork build that stock gmx on this box can't read, so
the density step is done cluster-side -- the printed command shows how).
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import MDAnalysis as mda
from MDAnalysis.selections.gromacs import SelectionWriter

from make_density_index import detect
from lipiddefs import TAIL_NC, AMBER_TAIL, lipid_resnames

ROOT = Path(__file__).resolve().parent.parent
TRAJ = ROOT / "traj"
DATA = ROOT / "data"


def tail_atoms(u, scheme, lipid):
    """AtomGroup of the hydrophobic acyl-tail atoms (aliphatic C + their H)."""
    resn = u.atoms.resnames
    if scheme == "amber":
        # The OL/MY tail residue is exactly the aliphatic chain (carbonyl is in
        # the head), so the whole residue -- carbons and hydrogens -- is the tail.
        return u.atoms[np.isin(resn, [AMBER_TAIL[lipid]])]

    # CHARMM / SLipids: one residue per lipid.  Take the aliphatic carbons
    # (C2{2..N}, C3{2..N} -- past the carbonyl C2 1 / C3 1) and add their bonded
    # hydrogens by geometry, so we don't depend on the H-naming convention.
    n = TAIL_NC[lipid]
    carbon_names = ({f"C2{i}" for i in range(2, n + 1)}
                    | {f"C3{i}" for i in range(2, n + 1)})
    lipid_atoms = u.atoms[np.isin(resn, [lipid])]
    carbons = lipid_atoms[np.isin(lipid_atoms.names, list(carbon_names))]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")     # vdw-radius guess chatter
        lipid_atoms.guess_bonds()           # populates u.bonds among lipid atoms
    hydrogens = u.select_atoms("name H* and bonded group carb", carb=carbons)
    return carbons + hydrogens


def membrane_atoms(u, scheme, lipid):
    """AtomGroup of all lipid atoms (centering group for ``gmx density``)."""
    resn = u.atoms.resnames
    return u.atoms[np.isin(resn, list(lipid_resnames(scheme, lipid)))]


def process(gro):
    u = mda.Universe(str(gro))
    ff, scheme, lipid = detect(u.atoms.resnames)
    tails = tail_atoms(u, scheme, lipid)
    memb = membrane_atoms(u, scheme, lipid)
    if tails.n_atoms == 0:
        sys.exit(f"{gro.name}: no tail atoms selected (check naming)")

    DATA.mkdir(parents=True, exist_ok=True)
    out = DATA / f"tail_{ff}_{lipid}.ndx"
    with SelectionWriter(str(out), mode="w") as ndx:
        ndx.write(tails, name="tails")
        ndx.write(memb, name="Membrane")

    n_lip = memb.n_residues if scheme == "single" else memb.n_residues // 3
    per_lip = tails.n_atoms / n_lip if n_lip else float("nan")
    print(f"{ff}_{lipid}: {tails.n_atoms} tail atoms "
          f"({per_lip:.0f}/lipid over ~{n_lip} lipids), "
          f"{memb.n_atoms} membrane atoms -> {out.name}")
    return ff, lipid


def print_gmx_hint(systems):
    print("\nRun gmx density cluster-side (last 50 ns, mass density, kg/m^3),")
    print("centering on the membrane and outputting the tail slab; d_HC is then")
    print("the width of the profile at 500 kg/m^3:\n")
    print("  # per system {ff}_{lipid}, per replicate rep in 1 2 3:")
    print("  #   t0 = (trajectory end in ps) - 50000")
    print('  echo "Membrane tails" | gmx density \\')
    print("      -f traj/{ff}_{lipid}${rep}.xtc -s traj/{ff}_{lipid}.tpr \\")
    print("      -n data/tail_{ff}_{lipid}.ndx -dens mass -center -symm \\")
    print("      -sl 200 -b ${t0} -o data/{ff}_{lipid}_${rep}_taildens.xvg")
    print("\nThen `python3 analysis/plot_core_thickness.py` parses the .xvg "
          "(width @ 500 kg/m^3) and compares it to the lipyphilic metric.")


def main():
    args = sys.argv[1:]
    gros = [Path(a) for a in args] if args else sorted(
        p for p in TRAJ.glob("*.gro") if "inbox" not in p.name)
    if not gros:
        sys.exit(f"no .gro files found (looked in {TRAJ})")
    done = [process(g) for g in gros]
    print_gmx_hint(done)


if __name__ == "__main__":
    main()
