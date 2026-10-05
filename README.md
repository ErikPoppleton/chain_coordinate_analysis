# Lipid-membrane pore formation along a chain coordinate

Molecular-dynamics study of pore formation and pore stability in single-component
lipid bilayers, comparing three force fields — **CHARMM**, **AMBER Lipid17**,
and **SLipids** — across three lipids — **DOPC**, **DMPC**, and **DMPG**.
Pores are opened along the *chain coordinate* ξ, a collective variable that
counts the connected chain of polar atoms spanning the membrane, and the free
energy of pore nucleation is obtained by umbrella sampling along it.

The project has two stages, and this repository holds both:

| Stage | Where it runs | Documented in |
|---|---|---|
| **Simulation setup** — build the systems, equilibrate, steer a pore open, run the umbrella windows, WHAM | an HPC cluster with SLURM + the GROMACS *chain-coordinate* fork | **this README** |
| **Analysis** — membrane properties, tilt moduli, PMFs, pore density/occupancy, tables and figures | mixed local/cluster | **[`analysis/README.md`](analysis/README.md)** |

If you already have the trajectories and just want the numbers and figures, go
straight to the [analysis README](analysis/README.md).

> **What this is.** These are the scripts as they were actually run, on one
> specific cluster. Paths like `/u/epoppleton/software/gromacs-chain-coordinate`
> and `module load gcc/8 cuda/11.6` are hard-coded throughout, and several steps
> were run interactively rather than scripted. Treat the contents of
> `setup_scripts/` and `setup_files/` as a documented record plus a starting
> template, not as a turnkey pipeline. Adapting them is mostly a matter of
> fixing the module loads, the `$GMX` path, and the SLURM headers.

---

## 1. Requirements

| Requirement | Why |
|---|---|
| [**`gromacs-chain-coordinate`**](https://gitlab.com/cbjh/gromacs-chain-coordinate) — a fork of **GROMACS 2021** | provides `pull-coord1-geometry = pore-slices`, the chain-coordinate CV. **Stock GROMACS cannot run, continue, or even `-rerun` these systems.** Everything that touches the CV must use the fork's `gmx_mpi` |
| A recent **stock GROMACS** (2022+; 2025.1/2026.3 were used here) | `make_ndx` default groups (fixed in 2022), the plain equilibrium runs, and trajectory wrangling |
| **SLURM** | every production run is submitted through `sbatch` |
| **Bash ≥ 4** | `setup_pull.sh` uses associative arrays |
| Python 3 + numpy/matplotlib | `setup_scripts/select_umbrella_frames.py` |

Build GROMACS_chain_coordinate per its installation instructions. Once compiled, these two bash commands must be run before any command which uses the chain coordinate:

```bash
export GMX=/path/to/gromacs-chain-coordinate/build/bin/gmx_mpi
source /path/to/gromacs-chain-coordinate/build/scripts/GMXRC.bash
```

## 2. The chain coordinate ξ

The CV is configured through **`SLICE_COORD_*` environment variables read at run time** — *not* through the `.mdp` (the `.mdp` only says
`pull-coord1-geometry = pore-slices` and sets `pull-coord1-init`/`-k`). They are
exported inside each submit script, so a rerun or a manual `mdrun` must export
the same values or it will measure a different coordinate:

| Variable | Steered pull | Umbrella / production | Meaning |
|---|---|---|---|
| `SLICE_COORD_N` | 35 | **27** | number of slices spanning the membrane |
| `SLICE_COORD_D` | 0.1 | 0.1 | slice thickness (nm) |
| `SLICE_COORD_CYL_R` | 0.9 | 0.9 | radius of the CV cylinder (nm) |
| `SLICE_COORD_FREEZE_CYLINDER_XIREF` | 0.8 | 0.8 | ξ above which the cylinder centre is frozen |
| `SLICE_COORD_FREEZE_CYLINDER_X/Y` | 3.0 / 3.0 | inherited from `submit_pull.sh` | frozen pore centre in the membrane plane (nm) |
| `SLICE_COORD_NST_OUTPUT` | 50000 | 50000 | how often the cylinder geometry is written to the log |

The slice count deliberately **differs between the steered pull (35) and
production (27)** because 27 slices (which gives the target flat membrane ξ=0.2) during pulling would sometimes result in bent membranes or hanging droplets rather than complete pores. Umbrella starting frames are picked from a
*rerun* of the pull trajectory at the production setting rather than from the
pull's own `pullx` (step 4 below). The pulled group is `ONP` — all O, N and P
atoms — against the lipid-`tail` reference group.

The two groups, plus `MEMB`/`SOLV`, live in the per-system **`pull.ndx`**, which
every later step (including the analysis-side `trjconv` pass) selects from.

## 3. Directory convention

Everything keys off a per-system directory named **`{ff}_{lipid}`** —
`{ff}` ∈ `charmm, amber, slipids`, `{lipid}` ∈ `DOPC, DMPC, DMPG`. `setup_pull.sh`
parses the force field and lipid **out of this directory name**, and so do the
analysis-side scripts. A finished system looks like:

```
{ff}_{lipid}/
├── topol.top, npt.gro, pull.ndx          # system + index groups (MEMB/SOLV/tail/ONP)
├── pull.mdp, umbrella.mdp, umbrella_npt.mdp
├── submit_pull.sh, submit_npt.sh, submit_umbrella.sh, submit_wham.sh
├── eq{1,2,3}/                            # unbiased equilibrium replicates
│   └── md.tpr, md.xtc, md.gro, md.cpt
├── md.xtc, pull.tpr, md_pullx.xvg        # the steered pulling run
├── umbrella{0..23}/                      # umbrella windows along ξ
│   └── npt.{tpr,gro,cpt}, md.tpr, md.xtc, md_pullx.xvg
│   └── umbrella23/close{1..10}/          # metastability replicates (ξ=1 window only)
└── profile.xvg, hist.xvg, bs.xvg         # gmx wham output
```

## 4. The pipeline

### Step 0 — build the systems  *(not scripted here)*

Bilayers of 128 lipids + water + Na counterions + 150 mM NaCl + 5 mM MgCl + 2 mM CaCl were
built per force field using the [CHARMM-GUI](https://www.charmm-gui.org/) Membrane Builder which generates the 7-step minimization and equilibration pipeline ending with the `step6.6_equilibration.{gro,cpt,tpr}` files the
scripts consume. 

CHARMM-GUI only has parameters for the CHARMM36 and Amber Lipid21 forcefields. The Slipids topology files for each lipid were built by hand using the forcefield files from the [Slipids2020 repository](https://bitbucket.org/alexander_lyubartsev/slipids-2020/src/master/Slipids_2020.ff/). This required re-ordering the DOPC `.gro` file generated by CHARMM-GUI with `setup_scripts/reorder_gro.py`.

Reference configurations for every force field / lipid pair are in
`snapshots/*_example.gro` — useful for checking residue and atom names.

### Step 1 — index groups, MDPs, and the pull `.tpr`

Run **from inside** each `{ff}_{lipid}/` directory:

```bash
REPO=/path/to/chain_coordinate_analysis     # wherever you cloned this
$REPO/setup_scripts/setup_pull.sh
```

It:

1. parses `FF`/`LIPID` from the directory name and refuses unknown values;
2. centres the membrane (`trjconv -pbc mol -center`) → `npt.gro`, using **stock
   GROMACS ≥ 2022** because older `make_ndx` had a default-group bug;
3. builds `pull.ndx` with the four groups the CV and all later analysis need —
   `MEMB` (lipids), `SOLV` (everything else), `tail` (the pull reference: acyl
   carbons `C2*`/`C3*` for CHARMM/SLipids, the whole `OL`/`MY` tail residue for
   AMBER), and `ONP` (all O + N + P atoms, the pulled polar group);
4. copies in the force-field-specific `pull_{ff}.mdp`, `umbrella_{ff}.mdp`,
   `umbrella_npt_{ff}.mdp` and the submit scripts;
5. `grompp`s `pull.tpr` with the fork.

The AMBER branch is the only genuinely force-field-dependent part: Lipid17
splits a lipid into head (`PC`/`PGR`) + two tail (`OL`/`MY`) residues, so the
membrane and tail selections are built from bash associative arrays instead of a
single lipid resname.

### Step 2 — unbiased equilibrium replicates

```bash
$REPO/setup_scripts/setup_equilibrium.sh   # run from the directory holding all nine {ff}_{lipid}/
```

Creates `eq{1,2,3}/` per system, each `grompp`ed from the same equilibration
checkpoint, and submits **three chained jobs** (`sbatch --dependency=afterany`)
per replicate so a run survives the 24 h wall clock. `eq_{ff}.mdp` writes a
compressed frame every 100 000 steps = **200 ps**, which is the 0.2 ns/frame the
analysis assumes.

These runs are pure equilibrium MD — no pulling — and they are what the
[equilibrium half of the analysis](analysis/README.md) consumes.

### Step 3 — steer a pore open

```bash
sbatch submit_pull.sh                   # from {ff}_{lipid}/
```

`pull_{ff}.mdp` pulls at `pull-coord1-rate = 1.6e-5` (ξ per ps) with
`pull-coord1-k = 5000` from `pull-coord1-init = 0.20` over 60 ns, writing
`md.xtc` + `md_pullx.xvg`. The commented second `mdrun` line in the submit script
is the continuation form (`-cpi md.cpt`), used if the job hits the wall clock (which it likely will because GROMACS 2021 is significantly slower than GROMACS 2026).

### Step 4 — pick the umbrella windows

```bash
python $REPO/setup_scripts/select_umbrella_frames.py   # from each {ff}_{lipid}/ directory
```

1. **Reruns** the pull trajectory with the *production* CV settings
   (`SLICE_COORD_N=27`) → `rerun_pullx.xvg`, so the ξ values match what the
   windows will actually restrain.
2. Picks **24 target ξ values** — `linspace(0, 0.65, 11)` then
   `linspace(0.7, 1.0, 13)` (Based on [Hub & Awasthi 2017](https://pubs.acs.org/jctcce/article/13/5/2352/752738/Probing-a-Continuous-Polar-Defect-A-Reaction)), deliberately denser above ξ = 0.7 around the expected energy barrier, and captures the frame closest to each target ξ value via `trjconv -dump`.
3. Writes `umbrella_frames.png`, a sanity plot of ξ(t) with the chosen frames
   marked.
4. Creates `umbrella{0..23}/`, filling the `SETME` (ξ reference) and `SETKTOO`
   (force constant) placeholders in the MDPs: **k = 5000** below ξ = 0.7 and
   **k = 15000** at/above it, matching the denser spacing. Then `grompp`s each
   window's NPT `.tpr`.

Then, per window, a short restrained NPT followed by production:

```bash
for i in {0..23}; do (cd umbrella${i} && sbatch submit_npt.sh); done
# when the NPT jobs finish (submit_umbrella.sh grompps md.tpr from npt.cpt/npt.gro):
for i in {0..23}; do (cd umbrella${i} && sbatch submit_umbrella.sh); done
```

`umbrella_npt_{ff}.mdp` runs 10 ns at k = 1000 to relax the window;
`umbrella_{ff}.mdp` runs 50 ns per submission. Both take the frozen pore centre X/Y and `SLICE_COORD_FREEZE_CYLINDER_XIREF`
from `submit_pull.sh`, so the whole system keeps one common pore axis.

### Step 5 — the ξ = 1 window and the metastability replicates

`umbrella23` (ξ = 1, the fully open pore) was extended to **200 ns** rather than
50 for the pore geometry/occupancy analysis.

From inside `umbrella23/`:

```bash
$REPO/setup_scripts/setup_release.sh
```

This spawns `close{1..10}/`: ten 5 ns replicates restarted from the ξ = 1
configuration with **the restraint switched off**. The pore is then free to close or persist,
and the resulting `close_pullx.xvg` traces analyzed for metastability.

### Step 6 — WHAM

From each `{ff}_{lipid}/`, list the windows and submit:

```bash
rm -f pullx_files.dat tpr_files.dat
touch pullx_files.dat tpr_files.dat
for x in {0..23}; do
    echo "umbrella${x}/md_pullx.xvg" >> pullx_files.dat
    echo "umbrella${x}/md.tpr"       >> tpr_files.dat
done
sbatch submit_wham.sh
```

`submit_wham.sh` runs the fork's `gmx wham` with bootstrap error estimation
(`-bs-method traj`) and integrated autocorrelation times (`-ac`), producing
`profile.xvg`, `hist.xvg`, `iac.xvg`, `bs.xvg`. The consecutive 10 ns
sub-profiles used for the convergence figures come from the same file lists with
`-b`/`-e` — see §4a of the [analysis README](analysis/README.md).

## 5. MDP reference

| File | Stage | Key settings |
|---|---|---|
| `eq_{ff}.mdp` | unbiased equilibrium | 500 ns, 2 fs, no pulling, frame every 200 ps |
| `pull_{ff}.mdp` | steered pulling | 60 ns, `pull-coord1-rate = 1.6e-5`, `k = 5000`, `init = 0.20` |
| `umbrella_npt_{ff}.mdp` | per-window relaxation | 10 ns, `rate = 0`, `k = 1000` |
| `umbrella_{ff}.mdp` | window production | 50 ns/job, `rate = 0`, `k = {5000,15000} (ξ<0.7,ξ≥0.7)` |

All runs: leap-frog, 2 fs, h-bond constraints, v-rescale thermostat at
**310 K**, c-rescale semiisotropic barostat at 1 bar.

The three force fields differ only in their non-bonded treatment, as each
force field prescribes:

| | `rcoulomb` / `rvdw` | `vdw-modifier` | `DispCorr` | `rlist` |
|---|---|---|---|---|
| CHARMM | 1.2 | force-switch (from 1.0) | no | 1.4 |
| AMBER | 1.0 | potential-shift | EnerPres | 1.3 |
| SLipids | 1.5 | force-switch (from 1.4) | EnerPres | 1.7 |

## 6. Visualising a run

`analysis/cylinder_pml.sh`, run from a run directory, reads the CV cylinder
geometry out of the newest `slurm.*.err` and writes a `cyl_style.pml` that draws
it in PyMOL, with the lipid tails hidden so the pore is visible. It detects the
force field and lipid by walking **up** the path to the `{ff}_{lipid}` directory
(stopping at a directory named `Simulations`), since run directories are nested
several levels below it; `cylinder_pml.sh <ff> <lipid>` overrides detection.

## 7. Adapting to other force fields / lipids

On the setup side, three places encode the force field:

1. **`setup_pull.sh`** — the `amber_heads`/`amber_tails` associative arrays and
   the `MEMB`/`SOLV`/`tail` selections. A new force field that keeps one residue
   per lipid follows the CHARMM/SLipids branch; one that splits head and tail
   residues follows the AMBER branch.
2. **`setup_files/*_{ff}.mdp`** — copy an existing set and change the non-bonded
   block to whatever the force field prescribes.
3. **Directory names** — `{ff}_{lipid}` is load-bearing at every later step.

The analysis side has its own single source of truth for the same facts
(`analysis/lipiddefs.py`) — see
§6 of the [analysis README](analysis/README.md).

## 8. Repository layout

| Path | Contents |
|---|---|
| `setup_scripts/` | the setup drivers described above |
| `setup_files/` | MDP templates and SLURM submit scripts, per force field |
| `analysis/` | all analysis and plotting code — see [`analysis/README.md`](analysis/README.md) |
| `snapshots/` | one example configuration per force field / lipid, for checking names |
| `traj/`, `data/`, `plots/` | analysis inputs/outputs (gitignored) |

---

## Next: the analysis

Everything downstream of the trajectories — equilibrium membrane properties,
order parameters, two independent tilt moduli, the PMFs along ξ, pore geometry
and occupancy, pore metastability, and the manuscript tables and figures — is
documented in **[`analysis/README.md`](analysis/README.md)**.
