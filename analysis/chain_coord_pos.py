import numpy as np
import matplotlib.pyplot as plt
import re
import os

traj = []
flag = False
# The newest SLURM log, i.e. the highest job id among slurm.<jobid>.err (by
# number, so it stays right if job ids gain a digit).
errfiles = [f for f in os.listdir('.') if re.fullmatch(r'slurm\.\d+\.err', f)]
if not errfiles:
    raise SystemExit("No slurm.*.err file found in %s" % os.getcwd())
errfile = max(errfiles, key=lambda f: int(f.split('.')[1]))
with open(errfile, 'r') as f:
    for l in f:
        if re.search("Slice +N_p", l):
            flag = True
            conf = []
            continue
        if not flag:
            continue
        else:
            if l.startswith('-'):
                flag = False
                traj.append(conf)
            else:
                l = l.strip().split()
                conf.append(float(l[1]))

traj = np.array(traj)
                
slices = np.arange(27)
means = [np.divide(np.sum(conf*slices), np.sum(conf)) for conf in traj]

fig, ax = plt.subplots()
ax.plot(np.arange(len(means)), means, linewidth=0.5)
ax.set_ylim(-1, len(slices))   # the full slice range, 0..26
fig.savefig('chain_pos.png', dpi=200)
