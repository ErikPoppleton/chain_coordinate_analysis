"""Single source of truth for the HMMM systems and their force-field vocabulary.

The HMMM branch is the same three force fields (CHARMM / AMBER / SLipids)
applied to a **highly mobile membrane mimetic** bilayer: the DOPC tails are
truncated to short (6:0) acyl chains and the hydrophobic core is filled with
the organic solvent **1,1-dichloroethane (DCLE)**, so the lipids diffuse orders
of magnitude faster than in a full bilayer while the headgroup region stays
realistic.

What differs from the parent (full-bilayer) project, and therefore why this is
a separate module rather than an import of ``analysis/lipiddefs.py``:

* the second axis is the **ensemble**, not the lipid -- every system is the same
  HMMM DOPC-mimetic, run as ``npt`` / ``npat`` / ``posre`` (3 replicates each);
* the acyl chains are 6 carbons, not 14/18, and CHARMM terminates them with
  ``C2T``/``C3T`` rather than a numbered carbon (SLipids keeps ``C26``/``C36``);
* AMBER's tail residue is ``VL`` (valeroyl), not ``OL``/``MY``;
* there is **no water and no ions** in these trajectories -- the solvent is
  DCLE, named ``DCLE`` by CHARMM and ``303`` by AMBER/SLipids.  Every index-group
  builder ported from the parent project has to learn that residue.

Kept pure stdlib (no numpy/MDAnalysis) so an index builder or a py2-only tool
can import it.
"""

# --- the systems -----------------------------------------------------------

FORCEFIELDS = ["charmm", "amber", "slipids"]
FF_LABELS = {"charmm": "CHARMM", "amber": "AMBER", "slipids": "SLipids"}
FF_LABELLED = [(ff, FF_LABELS[ff]) for ff in FORCEFIELDS]

# Pressure-coupling / restraint ensembles -- the HMMM analogue of the parent
# project's lipid axis.  ``posre`` restrains the lipid headgroups, ``npat``
# fixes the lateral area, ``npt`` lets the box relax in all directions.
ENSEMBLES = ["npt", "npat", "posre"]
ENS_LABELS = {"npt": "NPT", "npat": "NPAT", "posre": "posres"}

REPS = [1, 2, 3]

# Trajectories are ``traj/{ff}_{ens}{rep}.{gro,xtc}``.
def stem(ff, ens, rep):
    """Filename stem of one run: ``charmm_npat2``."""
    return "%s_%s%d" % (ff, ens, rep)


def system(ff, ens):
    """npz/json key for one (force field, ensemble) system: ``charmm_npat``."""
    return "%s_%s" % (ff, ens)


DT_NS = 1.0          # ns per frame (1 ns sampling, 1 us runs -> 1001 frames)

# --- residue / atom vocabulary --------------------------------------------

# Residue layout, as in the parent project: CHARMM/SLipids keep a whole lipid
# in one residue; AMBER (Lipid17) splits it tail-head-tail.
SCHEME = {"charmm": "single", "amber": "amber", "slipids": "single"}

# CHARMM/SLipids name the single lipid residue after the parent lipid even
# though the tails are truncated.  A build made with an explicitly short-tailed
# topology may instead call it DVPC (di-valeroyl), so both are accepted
# wherever a residue name is *matched*; SINGLE_RESN is what we write.
SINGLE_RESN = "DOPC"
SINGLE_RESN_ALIASES = set(["DOPC", "DVPC"])
AMBER_HEAD = "PC"
AMBER_TAIL = "VL"            # valeroyl -- the short HMMM tail residue

LIPID_RESNAMES = SINGLE_RESN_ALIASES | set([AMBER_HEAD, AMBER_TAIL])

# One phosphorus per lipid: CHARMM/SLipids "P", AMBER (Lipid17) "P31".  Same
# selection as the parent project, so lipyphilic code ports unchanged.
PHOSPHATE_NAMES = set(["P", "P31"])
PHOSPHATE_SEL = "name P P31"

# --- the organic solvent ---------------------------------------------------

# 1,1-dichloroethane.  CHARMM calls the residue DCLE; the AMBER/SLipids
# topologies call it 303 (the residue name in DCLE.itp is the numeral).  Both
# name the two carbons C1/C2; the chlorines and hydrogens differ.
DCLE = set(["DCLE", "303"])
DCLE_SEL = "resname DCLE 303"
DCLE_CARBONS = set(["C1", "C2"])
DCLE_CHLORINE = set(["CL11", "CL12",      # CHARMM
                     "CL3", "CL4"])       # AMBER / SLipids
DCLE_HYDROGEN = set(["H13", "H21", "H22", "H23",    # CHARMM
                     "H5", "H6", "H7", "H8"])       # AMBER / SLipids
# The CHCl2 carbon -- the natural single-site centre of a DCLE molecule, and
# what the radial distribution functions are keyed on.  **The two topologies
# swap C1 and C2**: CHARMM's C1 carries the chlorines (C1, CL11, CL12, H13, C2,
# H21-H23) while the AMBER/SLipids DCLE.itp calls that carbon C2 (C2, H8, CL3,
# CL4, C1, H5-H7) -- the same swap `setup_scripts/reorder_gro.py` documents.
# This table is the expectation; `rdf.py` verifies it geometrically per file,
# because getting it wrong silently shifts every peak by a C-C bond length.
DCLE_CENTRE = {"charmm": "C1", "amber": "C2", "slipids": "C2"}
# The solvent is the only non-lipid species: these trajectories carry no water
# and no ions.
SOLVENT_RESNAMES = set(DCLE)

# --- acyl chains -----------------------------------------------------------

# Carbons per acyl chain counting the carbonyl as C1, i.e. a 6:0 chain is
# C1..C6.  Saturated, so there is no double bond.
TAIL_NC = 6

# CHARMM/SLipids: sn-1 C21(carbonyl) C22..C25 + terminal, sn-2 C31..C35 +
# terminal.  **The terminal methyl is spelled differently by the two**: CHARMM
# ``C2T``/``C3T``, SLipids the numbered ``C26``/``C36``.  Both spellings are
# matched (each force field only uses one), so a selection keyed on the old
# CHARMM-only names doesn't silently come up empty for SLipids.
# AMBER: carbonyls C11/C21 live in the PC head; each VL tail residue runs
# C12..C16.


def chain_atom_names(scheme, position):
    """Acyl-carbon atom name(s) at a chain position (1 = carbonyl), covering
    both chains -- the HMMM analogue of ``lipiddefs.chain_atom_names``.

    CHARMM/SLipids separate the chains by name (sn-1 ``C2x``, sn-2 ``C3x``) and
    spell the terminal carbon ``C2T``/``C3T`` (CHARMM) or ``C26``/``C36``
    (SLipids) -- both are returned; AMBER's two tail residues share
    ``C1x``, so one name matches both, and its carbonyl (position 1) is instead
    the head-residue pair ``C11``/``C21``.
    """
    if scheme == "amber":
        if position == 1:
            return set(["C11", "C21"])           # carbonyls, in the PC head
        return set(["C1%d" % position])
    if position == TAIL_NC:
        # terminal methyls: CHARMM C2T/C3T, SLipids C26/C36
        return set(["C2T", "C3T", "C2%d" % position, "C3%d" % position])
    return set(["C2%d" % position, "C3%d" % position])


def lipid_resnames(scheme, aliases=True):
    """Set of residue names making up one lipid under ``scheme``.

    ``aliases`` includes the tolerated single-residue spellings (DOPC/DVPC),
    which is what a *selection* wants; pass False for the canonical one."""
    if scheme == "amber":
        return set([AMBER_HEAD, AMBER_TAIL])
    return set(SINGLE_RESN_ALIASES) if aliases else set([SINGLE_RESN])


def tail_atom_names(scheme):
    """Every aliphatic (post-carbonyl) acyl-carbon name of both chains."""
    names = set()
    for pos in range(2, TAIL_NC + 1):
        names |= chain_atom_names(scheme, pos)
    return names


def head_carbon_names(scheme):
    """Lipid carbon names that are *not* aliphatic tail carbons.

    AMBER keeps them in its own residue (``PC``), so this is only needed for the
    single-residue force fields, where the glycerol/choline/carbonyl carbons and
    the acyl chains share one residue and have to be told apart by name.
    Returns ``None`` for AMBER, where ``resn PC`` already does the job.
    """
    if scheme == "amber":
        return None
    return set(["C1", "C2", "C3",                       # glycerol
                "C11", "C12", "C13", "C14", "C15",      # choline
                "C21", "C31"])                          # ester carbonyls
