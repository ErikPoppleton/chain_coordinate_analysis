#!/bin/bash

set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

module purge
module load gromacs/2026.3
for ff in amber charmm slipids; do 
	for l in DMPC DOPC DMPG; do
		cd ${ff}_${l}
		for r in 1 2 3; do
		    rm -r eq${r}
		    mkdir eq${r}
	       	    cd eq${r}
		    cp ../step6.6_equilibration.cpt .
		    cp ../npt.gro .
		    cp ${SCRIPT_DIR}/../setup_files/eq_${ff}.mdp ./md.mdp
		    cp ${SCRIPT_DIR}/../setup_files/submit_eq.sh .
		    sed -i "s/JOBNAME/${ff:0:1}${l}/" submit_eq.sh
		    gmx grompp -f md.mdp -t step6.6_equilibration.cpt -c npt.gro -n ../pull.ndx -p ../topol.top -o md.tpr
		    JOBID=$(sbatch submit_eq.sh  2>&1 | awk '{print $(NF)}')
		    echo $JOBID
		    for x in {1..2}; do
			JOBID=$(sbatch --dependency=afterany:${JOBID} submit_eq.sh 2>&1 | awk '{print $(NF)}')
			echo $JOBID
		    done
		    cd ..

	        done
		cd ..
	done
done
