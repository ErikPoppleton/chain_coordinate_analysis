#! /bin/bash
set -e

RUNDIR=$(pwd)
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
LIPID=$(echo ${PWD##*/} | cut -d '_' -f2)
FF=$(echo ${PWD##*/} | cut -d '_' -f1)
LIPID=$(echo $LIPID | tr '[:lower:]' '[:upper:]')
FF=$(echo $FF | tr '[:upper:]' '[:lower:]')

if !([ $FF = amber ] || [ $FF = charmm ] || [ $FF = slipids ]); then
    echo "Unknown forcefield: $FF"
    echo "Forcefield must be one of: 'amber', 'charmm', 'slipids'"
    echo "Forcefield is obtained from directory name which must be 'ff_lipid' e.g. amber_DOPC"
    exit 1
fi

if ! declare -A assoc 2>/dev/null; then
    echo "Associative arrays not supported! This script requires at least Bash 4."
    exit 1
fi

if [ $FF = amber ]; then
   declare -A amber_tails
   declare -A amber_heads 
   amber_tails["DOPC"]="OL"
   amber_tails["DMPC"]="MY"
   amber_tails["DMPG"]="MY"
   amber_heads["DOPC"]="PC"
   amber_heads["DMPC"]="PC"
   amber_heads["DMPG"]="PGR"

   SEL=${amber_tails[$LIPID]}


else
    SEL=$LIPID
fi

# Gromacs had a bug with default groups in make_ndx which was fixed in 2022, so we need a newer version to make the initial file
module purge
module load gromacs/2025.1 && wait $!
echo "$SEL System" | gmx trjconv -f step6.6_equilibration.gro -s step6.6_equilibration.tpr -pbc mol -center -o npt.gro
echo -e "q" | gmx make_ndx -f npt.gro -o pull.ndx

# For everything else, we can use the chain coordinate fork.
module purge
module load gcc/8 cuda/11.6 && wait $!
export GMX=/u/epoppleton/software/gromacs-chain-coordinate/build/bin/gmx_mpi
source ~/software/gromacs-chain-coordinate/build/scripts/GMXRC.bash
if [ $FF = 'charmm' ] || [ $FF = 'slipids' ]; then
    $GMX select -f npt.gro -s npt.gro -on memb.ndx -select "\"MEMB\" resname $LIPID"
    $GMX select -f npt.gro -s npt.gro -on solv.ndx -select "\"SOLV\" !resname $LIPID"
    $GMX select -f npt.gro -s npt.gro -on tail.ndx -select '"tail" name "C2*" or name "C3*"'
else
    $GMX select -f npt.gro -s npt.gro -on memb.ndx -select "\"MEMB\" resname ${amber_heads[$LIPID]} or resname $SEL"
    $GMX select -f npt.gro -s npt.gro -on solv.ndx -select "\"SOLV\" !(resname ${amber_heads[$LIPID]} or resname $SEL)"
    $GMX select -f npt.gro -s npt.gro -on tail.ndx -select "\"tail\" resname $SEL"
fi
cat memb.ndx >> pull.ndx
cat solv.ndx >> pull.ndx
cat tail.ndx >> pull.ndx
rm memb.ndx solv.ndx tail.ndx
{ echo -e "a O* | a N* | a P*"; echo -e "q"; } | $GMX make_ndx -f npt.gro -n pull.ndx -o pull.ndx
sed -E -i 's/\[ .*O\*.*\]/[ ONP ]/' pull.ndx #If N or P not present, they won't be in the name. There are always waters with O.
cp ${SCRIPT_DIR}/../setup_files/pull_${FF}.mdp ./pull.mdp
cp ${SCRIPT_DIR}/../setup_files/submit_pull.sh .
sed -i "s/LIPID/$LIPID/" submit_pull.sh

# Change the fixed pore center to (3, 3)
eval "$(cat submit_pull.sh | grep 'export SLICE')"
$GMX grompp -f pull.mdp -c npt.gro -p topol.top -n pull.ndx -o pull.tpr
