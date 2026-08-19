import numpy as np
import matplotlib.pyplot as plt
import subprocess
import os

def read_xvg(xvg_file):
    metadata = {}
    data = []
    with open(xvg_file, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            if line.startswith('@'):
                line = line.strip('@ ').split()
                metadata[line[0]] = ' '.join(line[1:])
            else: 
                line = line.split()
                if data == []:
                    data = [[] for _ in line]
                for i, d in enumerate(line):
                    data[i].append(float(d))

    data = np.array(data)
    return metadata, data

# Check that files exist before starting
rep = os.getcwd().split('/')[-1][-1]
toploc = 'topol.top'

# The SLURM submit templates come from this repository's setup_files/, located
# relative to this script rather than an absolute path, so a clone anywhere works.
SETUP_FILES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           'setup_files')
SUBMIT_NPT = os.path.join(SETUP_FILES, 'submit_npt.sh')
SUBMIT_UMBRELLA = os.path.join(SETUP_FILES, 'submit_umbrella.sh')

importantfiles = ['umbrella_npt.mdp', 'umbrella.mdp', 'md.xtc', 'pull.tpr', 'pull.ndx', toploc]
missing  = [f for f in importantfiles if not os.path.isfile(f)]
missing += [f for f in (SUBMIT_NPT, SUBMIT_UMBRELLA) if not os.path.isfile(f)]
assert not missing, f"Files not found: {', '.join(missing)}"

# Generate the pullx file based on the production number of slices (27 instead of 35)
print("Starting rerun...")
result = subprocess.run(f'''
                module load gcc/8 cuda/11.6;
                export GMX=/u/epoppleton/software/gromacs-chain-coordinate/build/bin/gmx_mpi;
                source ~/software/gromacs-chain-coordinate/build/scripts/GMXRC.bash;
                read X < <(cat submit_pull.sh | grep "FREEZE_CYLINDER_X=" | cut -d "=" -f 2);
                read Y < <(cat submit_pull.sh | grep "FREEZE_CYLINDER_Y=" | cut -d "=" -f 2);
                export SLICE_COORD_N=27;
                export SLICE_COORD_D=0.1;
                export SLICE_COORD_CYL_R=0.9;
                export SLICE_COORD_FREEZE_CYLINDER_XIREF=0.8;
                export SLICE_COORD_FREEZE_CYLINDER_X=$X;
                export SLICE_COORD_FREEZE_CYLINDER_Y=$Y;
                export OMP_NUM_THREADS=1;
                $GMX mdrun -v -rerun md.xtc -s pull.tpr -deffnm rerun -ntomp 1;
                ''',
               shell=True, executable='/bin/bash')

# Read the pullx file into Python
metadata, pull_data = read_xvg(f'rerun_pullx.xvg')

# Using the target xi values from the 2017 paper
target_cv = np.round(np.concatenate([np.linspace(0, 0.65, 11), np.linspace(0.7, 1.0, 13)]), 2)
target_id = np.array([np.argmin(np.abs(pull_data[1] - xi)) for xi in target_cv])

# Get the frames corresponding to the xi values
print("Converting trajectories...")
result = subprocess.run(f'''
                module load gcc/8 cuda/11.6;
                export GMX=/u/epoppleton/software/gromacs-chain-coordinate/build/bin/gmx_mpi;
                source ~/software/gromacs-chain-coordinate/build/scripts/GMXRC.bash;
                TS=$($GMX check -e rerun.edr 2>&1 | grep "frames with a timestep" | cut -d ' ' -f 8);
                i=0;
                for id in {' '.join(target_id.astype(str))}; do echo "0" | $GMX trjconv -f md.xtc -s pull.tpr -dump $((id*TS)) -o conf${{i}}.gro; echo "Wrote file conf${{i}}.gro"; ((i++)); done;
               ''', shell=True, executable='/bin/bash')

# Make a nice plot showing the chosen frames
fig, ax = plt.subplots()
ax.hlines(target_cv, xmin=np.zeros_like(target_cv), xmax=np.full(len(target_cv), 60000), colors=['r' for _ in target_cv], zorder=0)
ax.plot(pull_data[0], pull_data[1], c='k', linewidth=0.5)
ax.scatter(pull_data[0][target_id], pull_data[1][target_id], s=3)
ax.set_ylabel(r'$\xi$')
ax.set_xlabel('Frames')
fig.savefig('umbrella_frames.png', dpi=300)

# Create a run directory for each umbrella
subprocess.run(f'''
               i=0;
               module load gcc/8 cuda/11.6;
               export GMX=/u/epoppleton/software/gromacs-chain-coordinate/build/bin/gmx_mpi;
               source ~/software/gromacs-chain-coordinate/build/scripts/GMXRC.bash;
               read X < <(cat submit_pull.sh | grep "FREEZE_CYLINDER_X=" | cut -d "=" -f 2);
               read Y < <(cat submit_pull.sh | grep "FREEZE_CYLINDER_Y=" | cut -d "=" -f 2);
               export SLICE_COORD_N=27;
               export SLICE_COORD_D=0.1;
               export SLICE_COORD_CYL_R=0.9;
               export SLICE_COORD_FREEZE_CYLINDER_XIREF=0.8;
               export SLICE_COORD_FREEZE_CYLINDER_X=$X;
               export SLICE_COORD_FREEZE_CYLINDER_Y=$Y;
               for xi in {' '.join(target_cv.astype(str))}; do \
                if (( $(echo "$xi >= 0.7" | bc -l) )); then \
                    k=15000;
                else  
                    k=5000; 
                fi;
                mkdir umbrella${{i}};
                mv conf${{i}}.gro umbrella${{i}}/;
                cp umbrella_npt.mdp umbrella${{i}}/;
                cp umbrella.mdp umbrella${{i}}/;
                sed -i "s/SETME/${{xi}}/g" umbrella${{i}}/umbrella_npt.mdp;
                sed -i "s/SETME/${{xi}}/g" umbrella${{i}}/umbrella.mdp;
                sed -i "s/SETKTOO/1000/g" umbrella${{i}}/umbrella_npt.mdp;
                sed -i "s/SETKTOO/${{k}}/g" umbrella${{i}}/umbrella.mdp;
                cd umbrella${{i}};
                $GMX grompp -f umbrella_npt.mdp -c conf${{i}}.gro -n ../pull.ndx -p ../{toploc} -o npt.tpr; 
                cp "{SUBMIT_NPT}" .;
                cp "{SUBMIT_UMBRELLA}" .;
                sed -i "s/JOBNAME/umb${{i}}/" submit_npt.sh;
                sed -i "s/JOBNAME/umb${{i}}/" submit_umbrella.sh;
                cd ..;
                ((i++));
               done;
               ''', shell=True, executable='/bin/bash')
