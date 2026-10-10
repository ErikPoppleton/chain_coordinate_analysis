#!/usr/bin/env python3
"""Generate lipid-director libraries (`.ldx`) for the Lipidator toolkit for any
of our force fields, from the atom NAMES of a recipe (so it is not tied to one
force field's atom numbering).

Recipe (matched to the shipped CHARMM/SLipids library so all force fields are
comparable):
  * head [P][1]     : phosphate + glycerol C2 + both ester carbonyls
                      + the first two acyl carbons of each chain
  * tail [CHAIN1][2]: the last three acyl carbons of each chain
  * surface [C2][3] : head atoms (element mass) + tail atoms weighted 5.0
Director = normalized COM(head) -> COM(tail).

Atom-name schemes:
  * CHARMM / SLipids  (one residue/lipid): phosphate `P`, glycerol `C2`,
    carbonyls `C21`/`C31`, sn-1 acyl `C2{k}`, sn-2 acyl `C3{k}`.
  * AMBER (Lipid21, three residues/lipid, tail-head-tail): phosphate `P31`,
    glycerol `C2`, carbonyls `C11`/`C21` (head residue), acyl `C1{k}` in each
    tail residue.  Positions are counted within the concatenated lipid.

The atom numbers written are 1-based positions within a lipid, counted from its
first atom (what the toolkit adds the index-file offset to).

    conda run -n lipid64 python analysis/make_director.py            # amber + slipids
    conda run -n lipid64 python analysis/make_director.py --check-charmm
"""

import sys
from pathlib import Path

import MDAnalysis as mda
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_density_index as mdi
from lipiddefs import TERM

ROOT = Path(__file__).resolve().parent.parent
TRAJ = ROOT / "traj"
DIRECTORS = Path.home() / "software/lipidator-toolkit/directors"


def recipe_positions(ff, lipid):
    """Return (head, tail) lists of (position, mass) for the director, with
    positions 1-based within the lipid in trajectory atom order."""
    n = TERM[lipid]
    u = mda.Universe(str(TRAJ / ("%s_%s.gro" % (ff, lipid))))
    _, scheme, _ = mdi.detect(u.atoms.resnames)

    if scheme == "amber":
        resn = [mdi.AMBER_HEAD[lipid], mdi.AMBER_TAIL[lipid]]
        tri = sorted(u.select_atoms("resname %s" % " ".join(resn)).residues[:3],
                     key=lambda r: r.atoms.indices[0])
        base = tri[0].atoms.indices[0]
        tailA = {a.name: a.index - base + 1 for a in tri[0].atoms}
        head_r = {a.name: a.index - base + 1 for a in tri[1].atoms}
        tailB = {a.name: a.index - base + 1 for a in tri[2].atoms}
        head = [(head_r["P31"], "P"), (head_r["C2"], "C"),
                (head_r["C11"], "C"), (head_r["C21"], "C"),
                (tailA["C12"], "C"), (tailA["C13"], "C"),
                (tailB["C12"], "C"), (tailB["C13"], "C")]
        last3 = ["C1%d" % c for c in (n - 2, n - 1, n)]
        tail = [(tailA[c], "C") for c in last3] + [(tailB[c], "C") for c in last3]
    else:  # charmm / slipids: single residue, CHARMM-style names
        res = u.select_atoms("resname %s" % lipid).residues[0]
        pos = {a.name: a.index - res.atoms.indices[0] + 1 for a in res.atoms}
        head = [(pos["P"], "P"), (pos["C2"], "C"),
                (pos["C31"], "C"), (pos["C21"], "C"),
                (pos["C22"], "C"), (pos["C23"], "C"),
                (pos["C32"], "C"), (pos["C33"], "C")]
        tail = ([(pos["C2%d" % c], "C") for c in (n - 2, n - 1, n)] +
                [(pos["C3%d" % c], "C") for c in (n - 2, n - 1, n)])
    return head, tail


def block(ff, lipid):
    head, tail = recipe_positions(ff, lipid)
    surf = head + [(p, "5.0") for p, _ in tail]
    fmt = lambda pairs: " ".join("%d %s" % (p, m) for p, m in pairs)
    return ("#%s\n[P][1]\n%s\n[CHAIN1][2]\n%s\n[C2][3]\n%s\n\n\n"
            % (lipid, fmt(head), fmt(tail), fmt(surf)))


def has_lipid(text, lipid):
    return ("#%s\n" % lipid) in text


def check_charmm():
    """Confirm the recipe reproduces the shipped charmm.ldx position sets for
    whichever of our lipids the shipped library actually contains."""
    import re
    shipped = (DIRECTORS / "charmm.ldx").read_text()
    ok = True
    for lipid in ("DMPC", "DOPC", "DMPG"):
        m = re.search(r"#%s\n(.*?)\n\n" % lipid, shipped, re.S)
        if not m:
            print("charmm %s: not in shipped charmm.ldx (will be generated)" % lipid)
            continue
        want = set(int(t) for t in re.findall(r"(?:^|\s)(\d+)\s+[A-Za-z0-9.]",
                                              m.group(1)))
        head, tail = recipe_positions("charmm", lipid)
        got = set(p for p, _ in head) | set(p for p, _ in tail)
        if not got <= want:
            ok = False
        print("charmm %s: recipe reproduces shipped director = %s"
              % (lipid, got <= want))
    return ok


def append_missing(ff, libname, lipids):
    """Append any of `lipids` missing from directors/<libname> (created if
    absent), using the validated recipe.  Idempotent."""
    path = DIRECTORS / libname
    text = path.read_text() if path.exists() else ""
    missing = [l for l in lipids if not has_lipid(text, l)]
    if not missing:
        print("%s already complete for %s" % (libname, ",".join(lipids)))
        return
    if text and not text.endswith("\n\n"):
        text += "\n\n"
    text += "".join(block(ff, l) for l in missing)
    path.write_text(text)
    print("wrote %s: added %s" % (path, ", ".join(missing)))


def main():
    if "--check-charmm" in sys.argv:
        check_charmm()
        return
    if not check_charmm():
        sys.exit("recipe does not reproduce shipped charmm.ldx -- aborting")
    LIPS = ("DMPC", "DOPC", "DMPG")
    # Fill every gap with the validated recipe, without disturbing existing
    # entries: CHARMM lacks DMPG; SLipids lacks all three; AMBER has no library.
    append_missing("charmm", "charmm.ldx", LIPS)
    append_missing("slipids", "slipids.ldx", LIPS)
    append_missing("amber", "amber.ldx", LIPS)


if __name__ == "__main__":
    main()
