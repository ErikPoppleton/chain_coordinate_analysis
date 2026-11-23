#!/bin/bash

# Generate a Pymol command file to display the last cylinder for the current directory.
#
# Force field and lipid come from the "ff_lipid" directory (e.g. amber_DOPC) --
# the same convention as setup_scripts/setup_pull.sh.  On the cluster the run
# directory sits several levels below it (umbrella windows, close replicates,
# ...), so the path is walked upwards to find it rather than just reading the
# parent directory name.  They can also be passed explicitly:
#
#     cylinder_pml.sh [ff] [lipid]
#
# The three force fields name things differently:
#
#   * lipid tails -- AMBER splits each lipid into a head and a tail residue, so
#     hiding the tails is just "hide everything, resn <tail>".  CHARMM and
#     SLipids put the whole lipid in one residue, so the tails have to be picked
#     out by atom name instead: the two acyl chains are the C2* / C3* carbons,
#     minus the glycerol backbone (C2, C3) and the ester carbonyls (C21, C31),
#     which live in AMBER's *head* residue and so must stay visible here too.
#     The hydrogens are then picked up by bond, via "neighbor".
#
#   * water and ions -- different residue names in each force field; the
#     selections below simply list all of them.

set -e

OUT="cyl_style.pml"

# Top of the simulation tree; the "ff_lipid" directory always lives below it.
SIM_ROOT="Simulations"

is_ff() {
    case "$1" in amber|charmm|slipids) return 0 ;; *) return 1 ;; esac
}

is_lipid() {
    case "$1" in DOPC|DMPC|DMPG) return 0 ;; *) return 1 ;; esac
}

# Walk up from the working directory until a path component parses as
# "ff_lipid", stopping at $SIM_ROOT.  Both halves have to be recognised, so a
# directory that merely contains an underscore is not mistaken for the run name.
find_ff_lipid_dir() {
    local d="$PWD" base
    while [ -n "$d" ] && [ "$d" != "/" ]; do
        base=${d##*/}
        [ "$base" = "$SIM_ROOT" ] && break   # walked past it without a match
        if is_ff "$(echo "$base" | cut -d '_' -f1 | tr '[:upper:]' '[:lower:]')" &&
           is_lipid "$(echo "$base" | cut -d '_' -f2 | tr '[:lower:]' '[:upper:]')"; then
            echo "$base"
            return 0
        fi
        d=${d%/*}
    done
    return 1
}

if [ -n "$1" ] && [ -n "$2" ]; then
    FF=$1
    LIPID=$2
else
    FF_LIPID_DIR=$(find_ff_lipid_dir) || {
        echo "Could not find an 'ff_lipid' directory (e.g. amber_DOPC) in the path:"
        echo "  $PWD"
        echo "Expected one below a '${SIM_ROOT}' directory, or pass them explicitly:"
        echo "  $(basename $0) <ff> <lipid>"
        exit 1
    }
    FF=$(echo "$FF_LIPID_DIR" | cut -d '_' -f1)
    LIPID=$(echo "$FF_LIPID_DIR" | cut -d '_' -f2)
fi

LIPID=$(echo $LIPID | tr '[:lower:]' '[:upper:]')
FF=$(echo $FF | tr '[:upper:]' '[:lower:]')

if !([ $FF = amber ] || [ $FF = charmm ] || [ $FF = slipids ]); then
    echo "Unknown forcefield: $FF"
    echo "Forcefield must be one of: 'amber', 'charmm', 'slipids'"
    echo "Forcefield is obtained from directory name which must be 'ff_lipid' e.g. amber_DOPC"
    exit 1
fi

if !([ $LIPID = DOPC ] || [ $LIPID = DMPC ] || [ $LIPID = DMPG ]); then
    echo "Unknown lipid: $LIPID"
    echo "Lipid must be one of: 'DOPC', 'DMPC', 'DMPG'"
    echo "Lipid is obtained from directory name which must be 'ff_lipid' e.g. amber_DOPC"
    exit 1
fi

# Water and ion residue names: every name any of the force fields uses, rather
# than a per-force-field list, because the rebuilt DMPG systems broke the
# per-force-field pattern (K+ counterions everywhere, and SLipids DMPG came back
# with AMBER-style TP3 water and K+/Cl- ions).  The +/- in the AMBER ion names
# are selection-list characters to Pymol and have to be backslash-escaped
# (quoting them does not work).
WATER='resn TIP3+TP3+SOL'
IONS='resn SOD or resn Na\+ or resn NA or resn POT or resn K\+ or resn CLA or resn Cl\- or resn CL or resn MG or resn CAL or resn CA'
ION_COLORS='color sodium, resn SOD or resn Na\+ or resn NA
color potassium, resn POT or resn K\+
color chlorine, resn CLA or resn Cl\- or resn CL
color magnesium, resn MG
color calcium, resn CAL or resn CA'

# How to select the lipid tails.
if [ $FF = amber ]; then
    # AMBER keeps the acyl chains in their own residue: DOPC -> oleoyl (OL),
    # DMPC/DMPG -> myristoyl (MY).  (A case, not a bash-4 associative array, so
    # this still runs under the bash 3.2 that ships with macOS.)
    case $LIPID in
        DOPC) AMBER_TAIL="OL" ;;
        DMPC|DMPG) AMBER_TAIL="MY" ;;
    esac
    TAIL_SELECT="select tail_atoms, resn ${AMBER_TAIL}"
else
    # CHARMM/SLipids: one residue per lipid, so pick the chains out by atom name
    # and grow the selection over bonds to catch their hydrogens.
    TAIL_SELECT="select tail_carbons, resn ${LIPID} and name C2*+C3* and not name C2+C3+C21+C31
select tail_atoms, tail_carbons or (hydro and neighbor tail_carbons)
delete tail_carbons"
fi

# The newest log = the numerically highest job id (a plain sort would rank
# slurm.999.err above slurm.1000.err).
ERR=$(ls | grep -E '^slurm\.[0-9]+\.err$' | sort -t. -k2,2n | tail -n 1 || true)
if [ -z "$ERR" ]; then
    echo "No slurm.*.err file found in $(pwd) -- cannot read the cylinder geometry."
    exit 1
fi

CYL_ARGS=$(tail -n 30 ${ERR} | grep -e 'Cylinder center' -e 'Cylinder spanning' | awk '{print $5, $6}' | tr -s '\n' ' ' | tr -s ' ' ', ')
CYL_ARGS="${CYL_ARGS}u=nm"

cat > ${OUT} <<EOF
cyl ${CYL_ARGS}
turn x, 90
set sphere_scale, 0.5
set transparency_mode, 3

# Hide the lipid tails so the pore is visible through the membrane.
${TAIL_SELECT}
hide everything, tail_atoms
delete tail_atoms

# Water and ions (every force field's names).
# "hide everything" first because Pymol's automatic styling on load is not the
# same for every water name (TP3 comes up as spheres, TIP3/SOL as lines), and
# nb_spheres is no good here: it only draws atoms with no bonds, and Pymol
# infers the O-H bonds when it reads the .gro.
hide everything, ${WATER}
show spheres, ${WATER}
show spheres, ${IONS}

# Colour the ions explicitly.  Pymol infers the element from the atom name (a
# .gro carries no element column) and gets several of these wrong, differently
# per force field: CHARMM SOD comes out sulfur-yellow and CAL green, AMBER Na+
# comes out white.  Without this the same ion changes colour between figures.
${ION_COLORS}
EOF

echo "wrote ${OUT} (${FF}, ${LIPID})"
