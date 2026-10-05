# Lipid-membrane equilibrium and pore-formation analysis

Reproducible analysis of membrane properties for three force fields
(**CHARMM**, **AMBER**, **SLipids**) across three lipids (**DOPC**, **DMPC**,
**DMPG**). Two halves: the **equilibrium** properties (§3) — area per lipid,
bilayer thickness, hydrophobic-core thickness, order parameters, lateral
diffusion, and two independent tilt moduli, each with autocorrelation-aware
error bars, plus a manuscript summary table and an SI equilibration table — and
the **pore-formation** simulations (§4) — PMFs along the chain coordinate ξ,
pore geometry and occupancy, and pore metastability.

This README documents what actually happened, there are ocassionally conventions/decisions that look insane in hindsight (duplicated trajectory files, thickness being computed multiple times, the local/cluster file paths, some things in scripts while others were run interactivley). If there's some portion of this you would actually use, open a GitHub issue and I'll try to clean it up for you.

> **Scope.** This README covers the **analysis** stage: you have the
> trajectories and want to reproduce the tables and figures. Producing them
> (system building, steered pulling, umbrella sampling on the custom GROMACS
> *chain-coordinate* fork) is a separate **simulation-setup** stage, documented
> in the [repository-root README](../README.md).
>
> **All commands below are written to be run from the repository root**
> (`python3 analysis/plot_pmf.py`, `data/…`, `traj/…`), even though this file
> lives in `analysis/`. The scripts themselves derive their paths from their own
> location, so they work from any working directory.

---

## 1. The two-copy trajectory layout (read this first)

The analysis described here was split across two machines because of difficulties getting certain programs installed on both machines. Throughout the README, they are labeled as *local*, which in the original paper was an arm64 device running MacOS and *cluster* which was an x86 device running Red Hat Enterprise Linux.

| Requirement | Only available | Tools that need it |
|---|---|---|
| GROMACS 2026.3  (reads the `tpx v138` `.tpr`) | **cluster** | Trajectory wrangling, `gmx density` |
| `g_lomepro` | **cluster** | `lipid_properties.py` |
| OpenStructure + OST_pymodules (Python 2.7)| **local** | `tilt_modulus.py` |
| Lipidator `resis` binary | **local** | `tilt_modulus.py` |
| MDAnalysis + lipyphilic (Python 3) | both | most scripts |

Because this repository reflects what was actually run rather than adding a complex path resolver after-the-fact, the scripts in this repo require **two
copies of the same trajectories** depending on where the original analysis was run, each with its own naming convention:

| Copy | Path convention | Contents | Consumed by |
|---|---|---|---|
| **A — cluster, lipid-only** | `~/Simulations/{ff}_{lipid}/eq{i}/` | `memb.gro` + `inbox.xtc` (MEMB group only, waters removed) | `lipid_properties.py`, `g_lomepro`, `gmx density` |
| **B — local, full system** | `traj/{ff}_{lipid}{i}.xtc` (+ shared `traj/{ff}_{lipid}.gro`) | full system incl. water/ions | `tilt_convert.py`, `core_thickness.py`, `run_lipidator*.py` |

`{ff}` ∈ `{charmm, amber, slipids}`, `{lipid}` ∈ `{DOPC, DMPC, DMPG}`, `{i}` ∈
`{1, 2, 3}` (replicates). **The filename convention is load-bearing** — the
scripts parse `ff`/`lipid`/`rep` out of the paths; there is no manifest.

Both copies are produced **on the cluster** from the same raw `md.xtc` /
`md.tpr` by one `trjconv` preprocessing pass (both `-pbc mol -center`ed on the
membrane, using the `MEMB` group from `../pull.ndx`); they differ only in the
output selection — the lipid-only `inbox.xtc` (copy A) or the full-system
`full.xtc` (copy B). The exact `trjconv` loop is in **§3, Step 0a**.

`inbox.xtc` (+ a matching lipid-only `memb.gro`) stays cluster-side

`full.xtc` needs to be downloaded to `traj/` as `traj/{ff}_{lipid}{i}.xtc`. Additionally, **one** replicate's
full-system `md.gro` per system (→ `traj/{ff}_{lipid}.gro`) as the shared
topology (the atom order — lipid, ions, water — is identical across replicates). Keeping water
in copy B is what lets OpenStructure build the water/lipid density interface for the tilt normal, which the lipid-only copy A cannot provide. The
full-system `.gro` files drive the `gmx density` index builder.

All computed data lands in **`data/`** (gitignored; large `.xvg`/`.npz` live
here), all figures in **`plots/`**.

---

## 2. Dependencies

Almost all of the analysis is Python 3. The versions this pipeline was last run
with:

| Package | Version | Needed by |
|---|---|---|
| Python | 3.13 | everything (except the tilt-χ compute) |
| numpy | 2.5 | everything |
| scipy | 1.18 | (MDAnalysis / pymbar dependency) |
| MDAnalysis | 2.10 | index building, lipid properties, core thickness, tilt conversion, lipidator blocks |
| lipyphilic | 0.12.1 | lipid properties, core thickness |
| matplotlib | 3.11 | all plotting |
| pymbar | 4.2 | `stats.py` (equilibration detection) |
| statsmodels | 0.14 | `stats.py` (pymbar's FFT autocorrelation path) |

External binaries: **GROMACS 2026.3** and **g_lomepro**
on the cluster; the **Lipidator `resis`** binary locally.

**How you install the Python packages is up to you** — a dedicated project
environment, or all of it in one existing environment; conda/mamba/pip/uv,
whatever you use. The pure
plotting scripts need only numpy + matplotlib, and every script derives its paths
from its own location (`Path(__file__)`), so they run from anywhere. *(Aside: on
Apple silicon this pipeline required a dedicated x86 environment because one
of the lipyphilic-stack packages — `lipyphilic` and its dependency `freud` — shipped
only an x86 build.)*

**The exception — OpenStructure + OST_pymodules (Python 2.7).**
[OST_pymodules](https://github.com/njohner/ost_pymodules) (used to compute tilt modulus) was last updated in 2018 and is written in Python 2. Therefore, we also had to compile a Python 2.7-compatible version of [OpenStructure](https://openstructure.org) ([v1.7.0](https://git.scicore.unibas.ch/schwede/openstructure/-/tags)), so the tilt-modulus
*compute* step (`tilt_modulus.py`, driven by `run_tilt.sh`) must run in its own
**Python 2.7 + OpenStructure** environment, built **with** the IPLT image
libraries (needed for the water/lipid density interface). This was terrible to get compiled and the reason for the duplicated trajectory paths. If you decide to follow me in this folly, good luck.

---

## 3. Run order

The analysis pipeline has a series of steps which must be run in order to complete successfully. Within each step, there are branches which are independent and can be run in any order/in parallel.
The sequence is:  
**Step 0 preprocessing → all raw-data branches → `stats.py` → the two production-span
tilt steps (`refit_tilt_t0.py`, `run_lipidator_blocks.py`) → tables/plotting.**
The pore-formation half of the project is a separate pipeline — see §4.

### Step 0 — preprocessing: trajectories, downloads, index files

Nothing else runs until this is done. It has three parts, in order:
**(0a)** make the two trajectory copies on the cluster, **(0b)** download the
full-system copy plus the topology `.gro` files, **(0c)** build the GROMACS
`.ndx` index files the cluster-side analyses need.

#### 0a — `trjconv` the two copies  *(cluster, GROMACS 2026)*

From the `Simulations/` root, turn each raw `md.xtc`/`md.tpr` into `inbox.xtc`
(copy A, lipid-only) and `full.xtc` (copy B, full system). Both are `-pbc mol
-center`ed on the `MEMB` group of `../pull.ndx`; only the output selection
differs:

```bash
for ff in charmm amber slipids; do
    for l in DMPC DOPC DMPG; do
        for i in 1 2 3; do
            cd ${ff}_${l}/eq${i};
            echo "MEMB MEMB" | gmx trjconv -f md.xtc -s md.tpr -n ../pull.ndx -pbc mol -center -o inbox.xtc;
            echo "MEMB System" | gmx trjconv -f md.xtc -s md.tpr -n ../pull.ndx -pbc mol -center -o full.xtc;
            echo "MEMB MEMB" | gmx trjconv -f md.gro -s md.tpr -n ../pull.ndx -pbc mol -center -o memb.gro;
            cd ../../;
        done;
    done;
done
```

`inbox.xtc` (copy A) stays on the cluster next to `memb.gro` while `full.xtc` (copy B) will be downloaded.

#### 0b — download copy B and the topologies  *(cluster → local)*

Pull down, for the local branches (B, C, D and the index builders below):

- every replicate's `full.xtc` → `traj/{ff}_{lipid}{i}.xtc`;
- **one** replicate's full-system `md.gro` per system → `traj/{ff}_{lipid}.gro`
  (atom order is identical across replicates).

The `.gro` files are not optional: MDAnalysis 2.10 cannot read the GROMACS 2026 `.tpr` format.

**AMBER `.gro` files need their lipid residues renumbered.** GROMACS numbers
every Lipid17 residue `1`, so MDAnalysis merges each lipid's sn-2 tail with the
next lipid's sn-1 (257 residues instead of 384) and `make_lipidator_index` /
`tilt_convert` refuse with "lipid residues not a multiple of 3". Run this on
every freshly downloaded AMBER topology — it is idempotent, and a no-op on a
file that is already numbered:

```bash
python analysis/renumber_gro.py --check traj/amber_*.gro   # report first
python analysis/renumber_gro.py traj/amber_*.gro           # rewrite in place
```

#### 0c — build the index files  *(local; the `.ndx` are used cluster-side)*

Three `.ndx` builders turn the downloaded `.gro` files into the atom selections
the cluster-side analyses need. They must run **before** Branches A and B:

```bash
python analysis/make_density_index.py    # traj/{ff}_{lipid}.gro -> data/density_{ff}_{lipid}.ndx (9 groups incl. potassium)
python analysis/make_tail_index.py       # traj/{ff}_{lipid}.gro -> data/tail_{ff}_{lipid}.ndx
python analysis/make_order_index.py      # traj/{ff}_{lipid}.gro -> data/order_{ff}_{lipid}.ndx
```

Rebuilding a system with a different ion set or solvent naming is handled
automatically — the builders detect the force field from the **lipid residue
layout**, not the water name — but re-run them whenever a `.gro` changes, since
the ion groups are read from it (the DMPG systems now carry potassium: CHARMM
`POT`, AMBER/SLipids `K+`, with the `sodium` group empty and skipped).

Because the tools that consume them run cluster-side, copy the finished
`.ndx` back, one per system, into `{ff}_{lipid}/` — i.e. one level above the
`eq{i}` run directories, so a run directory reaches them as `../density.ndx`:

```bash
CLUSTER=user@cluster.example.org      # your ssh target
REMOTE=~/Simulations                  # the Simulations/ root on the cluster
for ff in charmm amber slipids; do
    for l in DMPC DOPC DMPG; do
        scp data/density_${ff}_${l}.ndx ${CLUSTER}:${REMOTE}/${ff}_${l}/density.ndx;
        scp data/tail_${ff}_${l}.ndx    ${CLUSTER}:${REMOTE}/${ff}_${l}/tail.ndx;
        scp data/order_${ff}_${l}.ndx   ${CLUSTER}:${REMOTE}/${ff}_${l}/lomepro.ndx;
    done;
done
```

### Branch A — lipyphilic properties + S_CD  *(cluster: MDAnalysis/lipyphilic + g_lomepro)*

```bash
# reads copy A (~/Simulations/{ff}_{lipid}/eq{i}/{memb.gro,inbox.xtc})
python analysis/lipid_properties.py --nprocs 9
```

Computes APL, thickness, registration, S_CC, lateral diffusion via lipyphilic →
`data/lipprops_*.npz`; Computes S_CD via `g_lomepro` →
`data/{ff}_{lipid}_{rep}_scd_sn{1,2}.dat`, reading the `{ff}_{lipid}/lomepro.ndx`
uploaded in Step 0c (a system without one is skipped).

### Branch B — hydrophobic-core thickness d_HC + equilibrium density profiles  *(cluster gmx density; local MDAnalysis/lipyphilic)*

`d_HC` is computed by two independent metrics — a tail mass-density width from
`gmx density` (cluster) and a lipyphilic first-tail-carbon `MembThickness`
(local) — so run the cluster half first and have its profiles in `data/` before
the local half. The same `gmx density` pass also produces the **equilibrium
density profiles** (water/ions/lipid moieties along z), so both live here.

**1. Cluster `gmx density`** against the `tail.ndx` and `density.ndx` copied
back in Step 0c — one pass, two outputs, from the `Simulations/` root:

```bash
for ff in charmm amber slipids; do
    for l in DMPC DOPC DMPG; do
        for i in 1 2 3; do
            cd ${ff}_${l}/eq${i};
            echo "Membrane tails" | gmx density -f full.xtc -s md.tpr -n ../tail.ndx -center -o taildens.xvg;
            if [[ $l == "DMPG" ]]; then
                echo "terminal_C water potassium phosphorus glycerol_C2 midpoint_C terminal_C" | gmx density -f full.xtc -s md.tpr -n ../density.ndx -center -ng 6 -o density.xvg;
            else
                echo "terminal_C water sodium magnesium calcium phosphorus glycerol_C2 midpoint_C terminal_C" | gmx density -f full.xtc -s md.tpr -n ../density.ndx -center -ng 8 -o density.xvg;
            fi
            cd ../../;
        done;
    done;
done
```

The first group selected is the centering group (`Membrane`); the rest are the
output groups (`-ng 8` for the eight density groups). `make_density_index.py`
skips an ion group that is absent from a system (`gmx density` errors on an
empty group), so if it warned about one, drop that name from the selection and
lower `-ng` for that system.

**2. Download the profiles** into `data/` under the names the plotting scripts
parse: `taildens.xvg` → `data/{ff}_{lipid}_{rep}_taildens.xvg` (kept per
replicate) and `density.xvg` → `data/{ff}_{lipid}_density.xvg` (one per system;
`plot_density.py` maps the columns to group names via the `@ sN legend`
header). The density profiles need nothing further — `plot_density.py` in the
figures step turns them into `plots/density_profiles.png`.

**3. Local lipyphilic metric** — reads copy B (`traj/`) and, on top of `d_HC`,
stores the frame-aligned `d_HH` the thickness decomposition needs:

```bash
python analysis/core_thickness.py            # -> data/core_thickness.npz
```

### Branch C — tilt modulus, OpenStructure  *(local; stage 2 needs the Python 2.7 + OpenStructure env)* — see §2

```bash
# reads copy B; stage 1 converts (Python 3 / MDAnalysis), stage 2 computes chi (Python 2.7 / OpenStructure); resumable
./analysis/run_tilt.sh                        # -> data/tilt_moduli.json + data/tilt_series_*.npz
```

`run_tilt.sh` loops ff×lipid×rep: `tilt_convert.py` (gro+xtc → pdb+dcd) →
`tilt_modulus.py` (χ + block-averaged within-run SEM) → deletes the bulky dcd.
Set the path to your OpenStructure interpreter at the top of `run_tilt.sh`.

χ is then **refit on the production span** once `stats.py` (Step E) has supplied
`t₀` — cheap, since the raw per-frame tilts are already in `tilt_series_*.npz`:

```bash
# AFTER Step E; rewrites data/tilt_moduli.json in place (idempotent)
~/miniconda3/envs/openstructure/bin/python analysis/refit_tilt_t0.py
```

### Branch D — tilt modulus, Lipidator/ReSiS  *(local; needs the `resis` binary)* — **needs Branch C + Step E**

```bash
python analysis/make_lipidator_index.py   # -> data/lipidator_{ff}_{lipid}.ndx
python analysis/run_lipidator.py          # -> data/lipidator_moduli.json  (kappa per rep, whole run)
# production-span kappa + within-run error: runs resis on trajectory slices [t0:];
# reuses t0 from equilibrium_timeseries_stats.json and g_var from tilt_moduli.json
python analysis/run_lipidator_blocks.py   # -> data/lipidator_blocks.json
```
Note that Lipidator requires 'directors', custom formatted atom selections for determining lipid orientation. The directors used by `run_lipidator.py` were built with `make_director.py`:
```shell
python analysis/make_director.py          # -> <lipidator-toolkit>/directors/*.ldx
```
The new directors generated here have been [pull requested](https://github.com/allolio/lipidator-toolkit/pull/2) into the Lipidator repo, but at the time of writing, it has not been accepted.

### Step E — equilibration statistics  *(local; needs pymbar + statsmodels)* — **needs A + B**

```bash
python analysis/stats.py                   # -> data/equilibrium_timeseries_stats.json
```

Per (ff, lipid, rep, observable): pymbar `detect_equilibration` → t₀, and the
statistical inefficiency `g` / `Neff` on the production span `[t₀, T]`.
Observables are APL, d_HH, d_HC, the per-frame tilt spread ⟨δθ²⟩ (from Branch
C's `tilt_series_*.npz`), registration and S_CC — so run it **after** Branches
A/B/C. **Run this before any table/plot below**, and before `refit_tilt_t0.py`
and `run_lipidator_blocks.py`, all of which read t₀ from its JSON to drop the
burn-in (critical for the slow-equilibrating AMBER/SLipids DMPG systems).

### Step F — tables & figures  *(local; needs numpy + matplotlib)*
Requires all previous steps to have completed successfully and the `.npz`/`.dat` files to have been downloaded from the cluster-side `data/` directory.

```bash
python analysis/plot_tilt_modulus.py         # -> tilt summary + plots/tilt_modulus.png
python analysis/plot_lipidator.py            # -> lipidator summary + plots/lipidator_tilt_modulus.png
python analysis/make_equilibrium_table.py    # -> data/equilibrium_summary_table.tex
python analysis/make_equilibration_table.py  # -> data/equilibration_summary_table.tex (SI)
python analysis/plot_lipid_properties.py     # -> plots/lipprop_*.png (traj plots mark t0)
python analysis/plot_core_thickness.py       # -> plots/core_thickness*.png
python analysis/decompose_thickness.py       # -> plots/thickness_decomposition.png
python analysis/plot_density.py              # -> plots/density_profiles.png
```

**Dependency summary:** `A,B,C → E` (E's ⟨δθ²⟩ observable needs C's tilt series);
`E → refit_tilt_t0, D` (both slice to t₀; D also reuses tilt `g_var`);
`E → make_equilibrium_table, make_equilibration_table, plot_lipid_properties` (t₀);
`D → make_equilibrium_table, plot_lipidator` (κ CI);
`C + refit_tilt_t0 → make_equilibrium_table, plot_tilt_modulus` (χ CI).

---

## 4. Pore-formation analysis

Everything above is equilibrium bilayer analysis. This section covers the second
half of the project: the **pore-formation** simulations, in which a pore is
forced open along the chain coordinate, ξ, and umbrella-sampled along the opening pathway.

Those simulations need [**gromacs-chain-coordinate**](https://gitlab.com/cbjh/gromacs-chain-coordinate),
a fork of **GROMACS 2021** that adds the `pull-coord1-geometry = pore-slices`
collective variable; stock GROMACS cannot run (or re-run) them. As above, this
README assumes the simulations **have already been run** — it documents the
analysis only.

**Assumed cluster layout**, per system `{ff}_{lipid}/`:

| Path | What it is |
|---|---|
| `umbrella{i}/` | the umbrella windows along ξ, 50 ns each (`md.tpr`, `md.xtc`, `md_pullx.xvg`) |
| `umbrella23/` | the **final window, ξ = 1** (the open pore) — run **200 ns** instead of 50, for pore geometry |
| `umbrella23/close{1..10}/` | 10 × 5 ns metastability replicates, restarted from the final ξ = 1 configuration (`setup_scripts/setup_release.sh`) |

### 4a — PMFs along ξ  *(cluster, `gmx wham` from the fork)*

The full-profile `gmx wham` submit script (`submit_wham.sh`) is placed in the simulation directory
during setup. 

The lists of pullx/tpr files required by `gmx wham` can be generated with:
```shell
rm pullx_files.dat tpr_files.dat
touch pullx_files.dat tpr_files.dat
for x in {0..23}; do 
    echo "umbrella${x}/md_pullx.xvg" >> pullx_files.dat; 
    echo "umbrella${x}/md.tpr" >> tpr_files.dat; 
done
```

The five 10 ns sub-profiles (used for the convergence figures) are
computed by splitting the same window list into consecutive 10 ns slices. Run from each `{ff}_{lipid}/` directory:

```bash
export GMX=/path/to/gromacs-chain-coordinate/build/bin/gmx_mpi
source /path/to/gromacs-chain-coordinate/build/scripts/GMXRC.bash
for i in {1..5}; do
START=$(((i-1)));
END=$((i));
srun -n 1 -c 10 -t 01:00:00 --mem 16000 -p interactive --export=ALL $GMX wham -it tpr_files.dat -ix pullx_files.dat -o part${i}_profile.xvg -hist part${i}_hist.xvg -unit kJ -ac -oiact part${i}_iac.xvg -bs-method traj -bsres part${i}_bs.xvg -b ${START}0000 -e ${END}0000 &
done
```

Download the profiles into `data/` labeled by forcefield (this naming convention is load-bearing):

| Downloaded as | From |
|---|---|
| `data/{ff}_{lipid}_profile.xvg` | the full-length `gmx wham` profile |
| `data/{ff}_{lipid}_part{1..5}_profile.xvg` | `part${i}_profile.xvg` above (consecutive 20 ns windows) |
| `data/{ff}_{lipid}_20-100_profile.xvg` | optional: a single `-b 20000` run (80 ns window test for DMPC) |

```bash
python analysis/plot_pmf.py             # -> plots/pmf_profiles.png (1x3, partials drawn faintly behind)
python analysis/plot_range_vs_time.py   # -> plots/range_vs_{time,starttime}_{lipid}.png (PMF spread vs sampling)
```

### 4b — Pore geometry and occupancy  *(cluster compute, local plots)*

`pore_density.py` builds cylindrically-averaged **(radius, z)** density maps
around the pore centre from the 200 ns ξ = 1 window, plus the per-frame count of
atoms inside the CV cylinder (r = 0.9 nm, h = 2.7 nm — set at `grompp` time by the `SLICE_COORD_CYL_R`, `SLICE_COORD_N` and `SLICE_COORD_D` environment variables).

It needs three files per run directory: `md.xtc` + `md.gro` (the **full**
water+ion system) and `rerunerr.log`, the chain-coordinate rerun log, which
prints `Cylinder center at (X/Y/Z): cx cy cz` for every frame. Generate `rerunerr.log with:

```bash
# in {ff}_{lipid}/umbrella23/, with $GMX = the gromacs-chain-coordinate gmx_mpi
eval "$(cat submit_umbrella.sh | grep 'export SLICE')"
export SLICE_COORD_NST_OUTPUT=1
$GMX mdrun -v -rerun md.xtc -s md.tpr -deffnm rerun -ntomp 1 &> rerunerr.log
```


With the logs in place, compute the maps:

```bash
# cluster, MDAnalysis env; discovers ~/Simulations/{ff}_{lipid}/umbrella23/ for all nine systems
python analysis/pore_density.py
```
See `pore_density.py -h` for options on running on specific lipids or other simulation paths.

This will write `data/poredens_{ff}_{lipid}.npz` (density maps) and
`data/poredens_counts_{ff}_{lipid}.npz` (per-frame occupancy). Download both,
then plot locally:

```bash
python analysis/plot_pore_density.py    # -> plots/poredens_{water,phosphorus,glycerol_c2,midpoint_c}.png
                                        #    + plots/poredens_zoom_{water,phosphorus}.png
python analysis/plot_pore_counts.py     # -> plots/poredens_counts.png, poredens_counts_middle.png
```

Both read only the `.npz` files (numpy + matplotlib); panels with no data are
left blank, so a partial download still plots.

### 4c — Pore metastability  *(the `close` replicates)*

`setup_release.sh`, run from `umbrella23/`, spawns `close{1..10}/` — 10 × 5 ns
runs restarted from the final ξ = 1 configuration (`md.cpt`/`md.gro`) with the
umbrella force constant zeroed, so the pore is free to close while ξ keeps being
recorded. Download the ξ
traces as `data/{ff}_{lipid}_close{1..10}_pullx.xvg`:

```bash
python analysis/plot_close_pullx.py     # -> plots/close_pullx_{lipid}.png (one panel per ff, 10 replicates)
```

### 4c′ — Snapshot renders  *(local; needs PyMOL)*

The rendered pore images used by `make_figures.py`, Figure 2d and Figure 4g–i
are produced from the configurations in `snapshots/`:

```bash
# if a pore sits within ~1 nm of a box face, recentre it across PBC first
conda run -n lipid64 python analysis/center_pore_gro.py     # -> *_centred.gro + *_cyl_centred.pml
python analysis/render_snapshots.py     # -> snapshots/{ff}_{lipid}_{last,close}.png (18 panels, ~16 s)
python analysis/render_snapshots.py --ff charmm --lipid DMPC --set last   # a subset
python analysis/render_snapshots.py --dry-run                            # write the .pml only
```

Inputs are `snapshots/{ff}_{lipid}_last_conf.gro` (end-of-umbrella pore) and
`{ff}_{lipid}_close.gro` (closed pore); a `*_renumbered.gro` is preferred where
one exists. The chain-coordinate cylinder drawn in the `last` panels cannot be
recomputed locally — it comes from the run's `slurm.*.err` via
`cylinder_pml.sh` — so its geometry is read back out of the committed
`snapshots/{ff}_{lipid}_cyl_style.pml`, which stays the source of truth for it.
The `cyl` command itself is the `~/.pymol/draw_cylinder.py` plugin loaded from
`.pymolrc`, so do **not** run PyMOL with `-k`.

The clipping planes are widened past the box after framing, since both PyMOL's
auto-zoom (onto the small cylinder) and `zoom` otherwise leave a slab shallower than
the system and slice the foreground spheres open.

Output is 1358 × 1358 px on a transparent ground — 57.5 mm at 600 dpi, the
panel size `make_figures.py` places at native scale. Re-run
`analysis/make_figures.py`, `figures/fig02_pore_energy.py` and
`figures/fig04_pore_occupancy.py` afterwards to pick the new renders up.

### Branch E — DMPG ion-condition comparison  *(local; the SI figure)*

The DMPG systems exist in two ion conditions (K⁺ Åqvist, and the original
Na⁺ Joung–Cheetham runs analysed under the label `DMPG_JC`). The local half:

```bash
python analysis/renumber_gro.py traj/amber_DMPG_JC.gro          # AMBER only, once
conda run -n lipid64 python analysis/make_lipidator_index.py traj/*_DMPG_JC.gro
conda run -n lipid64 python analysis/core_thickness.py --lipids DMPG_JC
conda run -n lipid64 python analysis/ion_binding.py --lipids DMPG DMPG_JC   # -> data/ion_binding.npz (g(r) + N(r))
python analysis/run_lipidator.py --lipids DMPG_JC
LIPIDS=DMPG_JC ./analysis/run_tilt.sh
conda run -n base python analysis/stats.py --lipids DOPC DMPC DMPG DMPG_JC
conda run -n base python analysis/stats.py --data-dir data/data_Na --lipids DMPG
~/miniconda3/envs/openstructure/bin/python analysis/refit_tilt_t0.py
conda run -n lipid64 python analysis/run_lipidator_blocks.py --lipids DMPG_JC
python figures/figSI_dmpg_ions.py                               # -> plots/final/SI_dmpg_ions.*
python analysis/render_snapshots.py --ff amber --lipid DMPG_Na --set last   # the Na+ pore snapshot
python figures/figSI_dmpg_pmf_ions.py                           # -> plots/final/SI_dmpg_pmf_ions.*
```

The Na⁺ cluster-side properties live in `data/data_Na/` (lipprops + S_CD, keyed
by the plain `DMPG` name — the directory is the condition marker); everything
computed locally is keyed by the `DMPG_JC` label in the shared `data/`. Order
matters as usual: `stats.py` before `refit_tilt_t0.py` before the κ blocks.

### 4d — Equilibrium properties vs the pore barrier

Each system's **pore-nucleation
barrier** (the min→max range of its full PMF) scattered against the equilibrium
properties from §3.

```bash
python analysis/plot_property_vs_barrier.py       # -> plots/property_vs_barrier.png (6 properties)
python analysis/plot_thick_kappa_vs_barrier.py    # -> plots/thick_kappa_vs_barrier.png (d_HC + kappa only)
```

Both reuse `make_equilibrium_table.collect()` for the property means and error
bars, so they need everything §3 Step F needs **plus** all nine
`data/{ff}_{lipid}_profile.xvg` from 4a.

---

## 5. Error-bar model (what the bars mean)

Point-estimate figures/tables use a **hierarchical 95% confidence interval**
(`analysis/autocorr.py`, `hierarchical_ci*`): each replicate contributes a
within-run error, combined with the between-replicate spread as
`SE = sqrt(max(between_sd², within_sem²) / n)` — tight/pooled where replicates
agree, honestly wide where they don't (e.g. AMBER DMPG). The within-run error is
the autocorrelation-limited SEM (`SD/√Neff`, Chodera 2016) for per-frame
observables (APL, d_HH, d_HC), and a block-averaged SEM for the tilt moduli
(χ, κ). `D` and `⟨−S_CD⟩` — a single fitted scalar/profile per run — keep the SD
across replicates. The property-distribution **histograms** deliberately show the
full pooled distribution, not an error bar.

---

## 6. Adapting to different force fields / lipids

Almost everything you need to change lives in **`analysis/lipiddefs.py`**, the
single source of truth for which systems exist and how each force field names
things. It is pure stdlib and Python 2/3 compatible (the OpenStructure stage
runs under Python 2.7), and every other script imports from it:

| In `lipiddefs.py` | What it defines |
|---|---|
| `FORCEFIELDS`, `FF_LABELS`, `LIPIDS`, `REPS` | the systems — keys (= filename prefixes), display labels, replicate numbers |
| `SCHEME` | residue layout per force field: `"single"` (CHARMM/SLipids, one residue per lipid) vs `"amber"` (Lipid17 tail–head–tail) |
| `TAIL_NC` (alias `TERM`) | carbons per acyl chain, carbonyl counted as C1 |
| `DOUBLE_BOND`, `CHAIN_MID` | cis double-bond position per lipid (absent ⇒ saturated) and the mid-chain carbon used for the saturated density group |
| `AMBER_HEAD`, `AMBER_TAIL` | AMBER head / tail residue names per lipid |
| `PHOSPHATE_NAMES`, `PHOSPHATE_SEL` | the one-phosphorus-per-lipid selection (`P` / `P31`) |
| `WATER`, `WATER_NAME`, `WATER_OXYGEN`, `SODIUM`, `MAGNESIUM`, `CALCIUM` | solvent/ion resnames (the water resname is also what pins the force field in `detect()`) |
| `lipid_resnames`, `chain_atom_names`, `midpoint_carbon`, `n_unsat` | the small helpers that turn the above into selections |

So:

1. **A new lipid** — add it to `LIPIDS`, give it a `TAIL_NC` entry, an
   `AMBER_HEAD`/`AMBER_TAIL` entry, and (if unsaturated) a `DOUBLE_BOND` entry.
   Optionally add an `EXPERIMENTAL` row in `make_equilibrium_table.py` for the
   table's `Exp` line; leave it out and the row stays blank.
2. **A new force field** — add its key to `FORCEFIELDS` + `FF_LABELS`, its
   residue layout to `SCHEME`, its water/ion resnames to `WATER`/`WATER_NAME`
   (+ `SODIUM`/…), and a colour in `palette.COLORS` (`palette.FORCEFIELDS` is
   built from `lipiddefs` + those colours, so the plots pick it up
   automatically). If it names acyl carbons unlike either existing scheme,
   extend `chain_atom_names`/`lipid_resnames` and the `detect()` functions in
   `make_density_index.py` / `make_order_index.py`.
3. **Filenames** — name the trajectories exactly `{ff}_{lipid}{i}` (copy B) and
   `{ff}_{lipid}/eq{i}` (copy A); the load-bearing convention is how every script
   finds and labels the data.

Anything genuinely toolkit-specific stays with its tool: the OST director
selections in `tilt_modulus.selections()`, the Lipidator director recipe in
`make_director.py`, and the g_lomepro chain construction in
`make_order_index.py` — all of which read their chain lengths and residue names
from `lipiddefs`.

---

## 7. Output artifacts

**Tables** (`data/`, drop-in LaTeX; need `booktabs,multirow,amsmath` + a
bibliography with the cited toolkit/`chodera2016simple` keys):

| File | Contents |
|---|---|
| `equilibrium_summary_table.tex` | main manuscript table: APL, d_HH, d_HC, D, ⟨−S_CD⟩, χ, κ — mean ± hierarchical 95% CI |
| `equilibration_summary_table.tex` | SI: per-observable t₀ / production window / N / N_eff (mean ± SD over reps) |

**Key figures** (`plots/`):

| File | Contents |
|---|---|
| `lipprop_{area_per_lipid,memb_thickness,registration}.png` | overlaid property histograms |
| `lipprop_traj_*.png` | per-frame trajectories (rep 1) with the pymbar t₀ marked |
| `lipprop_core_thickness.png`, `core_thickness*.png` | d_HC histogram, bar chart, density profiles |
| `lipprop_scd.png` | S_CD per-carbon order-parameter profile |
| `tilt_modulus.png`, `lipidator_tilt_modulus.png` | the two tilt moduli (χ, κ) with hierarchical-CI bars |
| `density_profiles.png` | z-density profiles of water/ions/lipid markers |

Pore-formation figures (§4):

| File | Contents |
|---|---|
| `pmf_profiles.png` | PMFs along ξ, 1×3 lipid panels, 20 ns partials behind |
| `range_vs_time_{lipid}.png`, `range_vs_starttime_{lipid}.png` | PMF energy range vs sampling length / window start |
| `poredens_{water,phosphorus,glycerol_c2,midpoint_c}.png` | cylindrically-averaged (r, z) density maps around the pore |
| `poredens_zoom_{water,phosphorus}.png` | the same, restricted to the CV cylinder |
| `poredens_counts{,_middle}.png` | per-frame occupancy of the CV cylinder (and its middle third) |
| `close_pullx_{lipid}.png` | ξ(t) of the 10 pore-metastability replicates |
| `property_vs_barrier.png`, `thick_kappa_vs_barrier.png` | equilibrium properties vs pore-nucleation barrier (§4d — bridges §3 and §4) |

**Intermediate data** (`data/`, reusable so the slow steps run once):
`lipprops_*.npz`, `core_thickness.npz`, `tilt_moduli.json` + `tilt_series_*.npz`,
`lipidator_moduli.json` + `lipidator_blocks.json`, the `*_summary.json` files,
and `equilibrium_timeseries_stats.json` (the t₀/g/Neff source of truth); from
§4, `poredens_{ff}_{lipid}.npz` + `poredens_counts_{ff}_{lipid}.npz` and the
`gmx wham` `*_profile.xvg` files.

---

Simulation setup — how these trajectories were produced — is documented in
the [repository-root README](../README.md).
