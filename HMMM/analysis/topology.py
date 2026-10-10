"""Build MDAnalysis universes for the HMMM runs, repairing the AMBER residue
numbering on the way.

MDAnalysis can't read the GROMACS 2026 ``.tpr``, so (as in the parent project)
the ``.gro`` is the topology.  The AMBER (Lipid21) ``.gro`` numbers **every**
lipid residue ``1`` and lays a lipid out tail-head-tail (``VL``-``PC``-``VL``),
so MDAnalysis merges the whole 17920-atom lipid block into a couple of giant
residues.  That is fatal here rather than merely untidy: every lipyphilic
analysis sizes its output by ``membrane.n_residues`` and indexes leaflets with
``membrane.residues[...]``, so a 256-phosphate selection spanning one residue
yields one "lipid".

Rather than rewrite the downloaded trajectories' topology in place, this module
writes a **renumbered copy** into ``HMMM/data/topol/{stem}.gro`` the first time
it is needed and reuses it afterwards.  Molecule boundaries are found the way
``make_order_index``/``renumber_gro`` find them -- ignoring the resid entirely,
a new molecule starts at a resname change **or a repeated atom name** -- and
each molecule gets its own resid.  That is enough for the phosphate-keyed
analyses (one phosphate per lipid, in AMBER's ``PC`` head residue) and needs no
knowledge of how many residues a lipid has.

Files that already resolve to one residue per lipid (CHARMM, and presumably
SLipids) are used unchanged.
"""

import sys
from pathlib import Path

import MDAnalysis as mda

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hmmm_defs import PHOSPHATE_SEL

ROOT = Path(__file__).resolve().parent.parent      # HMMM/
TRAJ = ROOT / "traj"
TOPOL_CACHE = ROOT / "data" / "topol"


def split_molecules(atom_lines):
    """``[[line index, ...], ...]``, one list per molecule.

    A new molecule starts whenever the residue name changes or an atom name
    repeats within the current one -- the resid column is ignored, which is the
    whole point (AMBER's is uniformly ``1``).
    """
    molecules = []
    seen = set()
    cur_resn = None
    for i, line in enumerate(atom_lines):
        resn = line[5:10].strip()
        name = line[10:15].strip()
        if resn != cur_resn or name in seen:
            molecules.append([])
            seen = set()
            cur_resn = resn
        molecules[-1].append(i)
        seen.add(name)
    return molecules


def renumbered_gro(gro, out):
    """Write ``gro`` to ``out`` with one unique resid per molecule.

    Only columns 1-5 (the resid field) of the atom lines change; the title,
    count, coordinates and box pass through verbatim.  Resids wrap modulo
    100000, the width of the ``.gro`` field.
    """
    text = gro.read_text().splitlines(True)
    head, atoms, tail = text[:2], text[2:-1], text[-1:]
    for n, idx in enumerate(split_molecules(atoms)):
        resid = "%5d" % (n % 100000)
        for i in idx:
            atoms[i] = resid + atoms[i][5:]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(head + atoms + tail))
    return out


def topology_for(gro, quiet=False):
    """Path to a ``.gro`` whose residues are one per molecule.

    Returns ``gro`` itself when its phosphates already land in as many residues
    as there are phosphates, otherwise the (cached) renumbered copy.
    """
    u = mda.Universe(str(gro))
    phos = u.select_atoms(PHOSPHATE_SEL)
    if phos.n_atoms and phos.n_atoms == phos.n_residues:
        return gro
    cached = TOPOL_CACHE / gro.name
    if not cached.exists() or cached.stat().st_mtime < gro.stat().st_mtime:
        renumbered_gro(gro, cached)
        if not quiet:
            print("    renumbered topology -> %s"
                  % cached.relative_to(ROOT.parent))
    check = mda.Universe(str(cached)).select_atoms(PHOSPHATE_SEL)
    if check.n_atoms != check.n_residues:
        raise RuntimeError("%s: %d phosphates still span %d residues after "
                           "renumbering" % (gro.name, check.n_atoms,
                                            check.n_residues))
    return cached


def build_universe(stem, traj_dir=TRAJ, quiet=False):
    """Universe for one run stem (``charmm_npat2``) from its ``.gro`` + ``.xtc``."""
    traj_dir = Path(traj_dir)
    gro, xtc = traj_dir / (stem + ".gro"), traj_dir / (stem + ".xtc")
    if not (gro.exists() and xtc.exists()):
        raise FileNotFoundError("missing %s.gro or %s.xtc in %s"
                                % (stem, stem, traj_dir))
    return mda.Universe(str(topology_for(gro, quiet=quiet)), str(xtc))
