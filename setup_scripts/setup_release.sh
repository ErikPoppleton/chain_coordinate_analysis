#!/bin/bash

set -e

module purge
module load gcc/8 cuda/11.6 && wait $!
export GMX=/u/epoppleton/software/gromacs-chain-coordinate/build/bin/gmx_mpi
source ~/software/gromacs-chain-coordinate/build/scripts/GMXRC.bash

read X < <(cat ../submit_pull.sh | grep "FREEZE_CYLINDER_X=" | cut -d "=" -f 2);
read Y < <(cat ../submit_pull.sh | grep "FREEZE_CYLINDER_Y=" | cut -d "=" -f 2);
eval "$(cat submit_umbrella.sh | grep 'export SLICE')"
for i in {1..10}; do
	mkdir close${i} && cd close${i}
	cp ../umbrella.mdp ./close.mdp
	cp ../submit_umbrella.sh ./submit_close.sh
	sed -i -e 's/\(^nsteps *= \)[0-9]*\(.*$\)/\12500000 \2/' -e 's/15000/0/' close.mdp
	sed -i -e '/^\$/d' -e 's/md.tpr/close.tpr/' -e 's/deffnm md/deffnm close/' -e "s/umb23/close$i/" submit_close.sh
	$GMX grompp -f close.mdp -t ../md.cpt -c ../md.gro -n ../../pull.ndx -p ../../topol.top -o close.tpr
	sbatch submit_close.sh
	cd ..
done
