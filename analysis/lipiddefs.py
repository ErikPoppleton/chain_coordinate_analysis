"""Single source of truth for the systems studied and their force-field
vocabulary.

Every analysis script needs some subset of the same facts: which force fields /
lipids / replicates exist, how each force field names its water, ions,
phosphate and acyl carbons, how many carbons a tail has, and where the double
bond sits.  They used to be re-declared in a dozen files, so this module holds
them once and everything else imports from here.

Kept **pure stdlib and Python 2/3 compatible** (like ``autocorr.py``) because
``tilt_modulus.py`` runs under OpenStructure's Python 2.7.

Force-field "schemes"
---------------------
CHARMM and SLipids put a whole lipid in **one residue** (scheme ``"single"``),
naming the sn-1 chain ``C2x`` and the sn-2 chain ``C3x``.  AMBER (Lipid17)
splits each lipid into **tail-head-tail residues** (scheme ``"amber"``), so both
chains are separate residues that share the carbon naming ``C1x``.  Almost every
force-field conditional in the repo is really a test on this scheme.
"""

# --- the systems -----------------------------------------------------------

# Force-field keys, in the canonical order used by every plot/table.  The keys
# are also the filename prefixes (``{ff}_{lipid}...``).
FORCEFIELDS = ["charmm", "amber", "slipids"]
FF_LABELS = {"charmm": "CHARMM", "amber": "AMBER", "slipids": "SLipids"}
# [(key, display label), ...] in canonical order -- the table scripts' shape.
FF_LABELLED = [(ff, FF_LABELS[ff]) for ff in FORCEFIELDS]

LIPIDS = ["DOPC", "DMPC", "DMPG"]
REPS = [1, 2, 3]

# Residue layout per force field (see the module docstring).
SCHEME = {"charmm": "single", "amber": "amber", "slipids": "single"}

# --- acyl chains -----------------------------------------------------------

# Carbons per acyl chain, counting the carbonyl as C1 (so a 14:0 tail is
# C1..C14).  DOPC is di-oleoyl (18:1), DMPC/DMPG are di-myristoyl (14:0).
TAIL_NC = {"DOPC": 18, "DMPC": 14, "DMPG": 14}
TERM = TAIL_NC          # alias: "last acyl carbon" reads better in tilt code

# cis double-bond position: carbons (d, d+1) are the two sp2 carbons.  Oleoyl is
# Delta9 -> C9=C10.  Saturated lipids are absent from this dict.
DOUBLE_BOND = {"DOPC": 9}
# Density-profile mid-chain carbon for the saturated lipids (the unsaturated
# ones use their upper C=C carbon, i.e. DOUBLE_BOND[lipid], instead).
CHAIN_MID = 7

# --- residue / atom vocabulary --------------------------------------------

# AMBER (Lipid17) residue names: one head + two identical tail residues.
AMBER_HEAD = {"DOPC": "PC", "DMPC": "PC", "DMPG": "PGR"}
AMBER_TAIL = {"DOPC": "OL", "DMPC": "MY", "DMPG": "MY"}

# One phosphorus per lipid: CHARMM/SLipids "P", AMBER (Lipid17) "P31".
PHOSPHATE_NAMES = set(["P", "P31"])
PHOSPHATE_SEL = "name P P31"        # the MDAnalysis/lipyphilic form

# Solvent and ions.  The water resname also pins the force field (see
# ``make_density_index.detect``).
WATER = set(["TIP3", "TP3", "SOL"])
WATER_NAME = {"charmm": "TIP3", "amber": "TP3", "slipids": "SOL"}
WATER_OXYGEN = set(["O", "OW", "OH2", "OW1"])   # water O across TP3/TIP3/SOL
SODIUM = set(["SOD", "Na+", "NA"])
MAGNESIUM = set(["MG"])
CALCIUM = set(["CAL", "CA"])


# --- small shared helpers --------------------------------------------------

def lipid_resnames(scheme, lipid):
    """Set of residue names making up one lipid under ``scheme``."""
    if scheme == "amber":
        return set([AMBER_HEAD[lipid], AMBER_TAIL[lipid]])
    return set([lipid])


def chain_atom_names(scheme, position):
    """Acyl-carbon atom name(s) at a chain position, covering both chains.

    Single-residue force fields separate the chains by name (sn-1 ``C2x``,
    sn-2 ``C3x``); AMBER's two tail residues share ``C1x``, so one name already
    matches both."""
    if scheme == "amber":
        return set(["C1%d" % position])
    return set(["C2%d" % position, "C3%d" % position])


def midpoint_carbon(lipid):
    """Chain position of the density-profile ``midpoint_C`` group: the upper
    (head-side) C=C carbon for unsaturated lipids, else the mid-chain carbon."""
    return DOUBLE_BOND.get(lipid, CHAIN_MID)


def n_unsat(lipid):
    """g_lomepro's ``-unsat``: 2 double-bond groups (sn-1 + sn-2) or 0."""
    return 2 if lipid in DOUBLE_BOND else 0
