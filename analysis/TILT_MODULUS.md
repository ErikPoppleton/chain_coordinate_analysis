# Running the tilt-modulus calculations (OpenStructure, ReSiS and DF)

The commands that produce the **raw** lipid tilt modulus with the two
independent real-space toolkits used in this project, plus a Fourier-space
director-fluctuation (DF) analysis (section 3):

| | OpenStructure (`ost_pymodules`) | Lipidator (ReSiS, `resis`) |
|---|---|---|
| Quantity | χ, per lipid | κ_θ, area-normalised |
| Native unit | k_BT/rad² | k_BT/nm² |
| Reads | PDB + CHARMM DCD (converted from gro+xtc) | GROMACS xtc directly |
| Python | 2.7 (OpenStructure) for the analysis, 3 (MDAnalysis) for conversion | none; a C++ binary plus Python 3 helpers |
| Output | `data/tilt_moduli.json` | `data/lipidator_moduli.json` |

The two measure the same modulus and are related by κ_θ = χ / A_lipid. They
agree to about 10 %, which serves as a cross-check.

The DF analysis (`analysis/tilt_df.py`, section 3) measures a *different*
quantity: a bilayer modulus read from the long-wavelength limit of the
collective director spectrum. It does not reproduce the real-space numbers, and
it isn't meant to (section 4).

This covers only the raw analysis. The equilibration cut (`refit_tilt_t0.py`),
the block-averaged errors (`run_lipidator_blocks.py`) and the hierarchical
confidence intervals used for the paper are a separate layer on top. See
`analysis/README.md` and `CLAUDE.md` for those.

Run every command **from the repository root**.

---

## 0. Inputs and environments

### Trajectories

Each system needs the **full-system** (water + ions) trajectories:

```
traj/{ff}_{lipid}.gro          shared topology (one replicate's md.gro)
traj/{ff}_{lipid}{rep}.xtc     one per replicate, rep = 1, 2, 3
```

with `ff` ∈ {`charmm`, `amber`, `slipids`} and `lipid` ∈ {`DOPC`, `DMPC`,
`DMPG`}.

- **Water is required.** OpenStructure takes the membrane normal from the
  water/lipid density interface.
- **The bilayer must be whole and centred** before you start, as written by
  the cluster-side `gmx trjconv -pbc mol -center` (centred on `MEMB`). Neither
  tool re-centres.
- The lipid-only `memb.gro`/`inbox.xtc` copies used by lipyphilic do **not**
  work here.

### Environments and tools

| what | where on this machine |
|---|---|
| MDAnalysis env | conda env `lipid64` (always with `PYTHONNOUSERSITE=1`: a stray `~/.local` scipy otherwise breaks the MDAnalysis import) |
| OpenStructure Python 2.7, **built with the image (IPLT) libraries** | `~/miniconda3/envs/openstructure/bin/python` |
| `ost_pymodules` | `~/software/ost_pymodules` (put on `sys.path` by `tilt_modulus.py`) |
| ReSiS binary | `~/software/lipidator-toolkit/build/resis` |
| director libraries | `~/software/lipidator-toolkit/directors/{charmm,amber,slipids}.ldx` |

OpenStructure has to be built with the image libraries: the interface density
is built with `img.*`. `tilt_modulus.py` does the two pieces of setup that the
interactive `ost` launcher would normally do: it imports `ost.img`, and it calls
`SetPrefixPath` (from `$DNG_ROOT`/`$OST_ROOT`, or by walking up from
`ost.__file__`).

### AMBER only: renumber the gro first

Lipid17 numbers every residue `1`. MDAnalysis then merges each lipid's sn-2
tail with the next lipid's sn-1, and both `tilt_convert.py` and
`make_lipidator_index.py` refuse to run ("lipid residues not a multiple of 3").
Fix the gro in place first. Non-AMBER files are skipped:

```bash
python3 analysis/renumber_gro.py --check traj/amber_*.gro   # report only
python3 analysis/renumber_gro.py traj/amber_*.gro           # rewrite in place
```

---

## 1. OpenStructure / `ost_pymodules` → χ (k_BT/rad²)

The method follows Khelashvili et al. 2013, JCTC 9:3866.

- **Director:** from the P + glycerol-C2 (mass-weighted) centre to the centre
  of mass of the last three carbons of each chain.
- **Normal:** the local water/lipid interface normal at each lipid.
- **Tilt θ:** the angle between the director and the normal.
- **Fit:** χ comes from the fit `−ln[P(θ)/sin θ] = (χ/2) θ²`.

Two stages are needed because OpenStructure cannot read xtc and has no
MDAnalysis.

### Stage 1: convert gro + xtc to PDB + DCD (`lipid64`)

```bash
PYTHONNOUSERSITE=1 conda run -n lipid64 python analysis/tilt_convert.py \
    traj/charmm_DMPC.gro traj/charmm_DMPC1.xtc -o data/tilt_charmm_DMPC_1
# -> data/tilt_charmm_DMPC_1.pdb  (frame-0 topology)
#    data/tilt_charmm_DMPC_1.dcd  (all frames, with box)
```

The output basename **must** be `data/tilt_{ff}_{lipid}_{rep}`. Stage 2 parses
the force field, lipid and replicate back out of it.

What the conversion does, so the library sees each lipid as one residue:

- keeps **heavy atoms only** (lipids and water);
- reduces water to its oxygens and rebuilds it as one residue per molecule
  (AMBER gives all waters one resid otherwise);
- drops ions;
- for AMBER, merges each lipid's three residues into one `LIP` residue and
  renames the second tail's carbons from `C1{n}` to `CB{n}` (OpenStructure
  would otherwise merge the two identically named tails).

### Stage 2: compute χ (OpenStructure Python 2.7)

```bash
~/miniconda3/envs/openstructure/bin/python analysis/tilt_modulus.py \
    data/tilt_charmm_DMPC_1 --stride 1 --density-stride 10
```

| option | default | meaning |
|---|---|---|
| `--stride` | 1 | analyse every *N*-th frame. **Keep it at 1.** A coarse stride undersamples the interface density, which gives noisy normals, inflated tilts and a **suppressed** χ (stride 20 gave ≈ 11 against ≈ 14 at stride 1 for CHARMM DMPC). |
| `--density-stride` | 10 | frame stride for building the *average* interface density; 10 is enough |
| `--within-normals` | 10.0 | patch radius (Å) on the smoothed interface used for each normal |
| `--density-cutoff` | 0.3 | density level that defines the water/lipid boundary |
| `--nbins` | 90 | histogram bins for the P(θ) fit |

Several basenames may be passed in one call.

#### Output

- **`data/tilt_moduli.json`**, keyed `{ff}_{lipid}_{rep}` and merged into the
  existing file. The raw result is **`chi_kT`** (pooled fit over all frames ×
  lipids, k_BT/rad²) with its fit error **`dchi_kT`**. Multiply by RT
  (2.577 kJ/mol at 310 K) for kJ mol⁻¹ rad⁻². The block-SEM keys alongside it
  belong to the statistics layer.
- **`data/tilt_series_{ff}_{lipid}_{rep}.npz`**: the raw per-frame, per-lipid
  tilt angles (radians).
- **`plots/tilt/{ff}_{lipid}/`**: the library's tilt-distribution and PMF-fit
  plots.

`refit_tilt_t0.py` (statistics layer) later **rewrites** `chi_kT` in place
with the value refitted on the equilibrated span, and keeps the whole-run value
as `chi_kT_full`. A `chi_kT` read straight after `tilt_modulus.py` is the
whole-run fit.

The library function `AnalyzeLipidTiltAndSplay` is **not** used: its splay step
has a bug (an unguarded `GetBoolProp('')`). `tilt_modulus.py` calls the
library's tilt pieces directly instead (`GetBoundaryBetweenViews` →
`AssignNormalsToLipids` → `AnalyzeLipidTilts` → `FitTiltDistribution`). Note
that `tilt_modulus.py`'s own docstring still describes the old route and a
phosphate-only head; the code is as described here.

### Batch driver: every system × replicate

```bash
./analysis/run_tilt.sh                                  # all 9 systems x 3 reps
FFS="amber" LIPIDS="DMPG" REPS="1 2" ./analysis/run_tilt.sh   # a subset
FRESH=1 ./analysis/run_tilt.sh                          # start tilt_moduli.json clean
```

For each `{ff}_{lipid}_{rep}`, the driver:

1. runs stage 1;
2. runs stage 2;
3. deletes the bulky `.dcd` (the small `.pdb` is kept).

It skips systems whose `traj/` files are missing and can be resumed. `STRIDE`
and `DENS_STRIDE` pass through to stage 2. The OpenStructure interpreter path
is set at the top of the script (`OST_PY=`).

**PBC tiling is not needed.** The library's example replicates the unit cell,
but that is for a curved hexagonal phase. For these flat bilayers, tiling
changed CHARMM DMPC's χ by 0.5 % (14.03 → 14.10), so the pipeline runs without
it. The box is still written into the DCD, so tiling could be switched on.

---

## 2. Lipidator / ReSiS → κ_θ (k_BT/nm²)

ReSiS reads **only the xtc**: coordinates and atom count, no names or residues.
The lipids are defined entirely by two files:

- an **index** listing each lipid's first atom;
- a **director library** (`.ldx`) giving, within one lipid, which atoms make up
  the head, tail and surface groups.

Because of this, AMBER's three-residues-per-lipid layout doesn't matter to it.

### Step 1: director libraries (`lipid64`, one-off)

```bash
PYTHONNOUSERSITE=1 conda run -n lipid64 python analysis/make_director.py --check-charmm
PYTHONNOUSERSITE=1 conda run -n lipid64 python analysis/make_director.py
```

- The first command checks that the recipe reproduces the shipped CHARMM
  library's DMPC/DOPC offsets. It aborts if not.
- The second appends the entries the shipped libraries lack (CHARMM DMPG;
  SLipids DOPC/DMPC/DMPG) and writes `amber.ldx` for all three lipids, into
  `~/software/lipidator-toolkit/directors/`. It is idempotent.

The recipe (positions are 1-based within the lipid, counted from its first
atom):

| group | atoms |
|---|---|
| head `[P][1]` | phosphate, glycerol C2, both ester carbonyls, the first two acyl carbons of each chain |
| tail `[CHAIN1][2]` | the last three acyl carbons of each chain |
| surface `[C2][3]` | the head atoms (element mass) plus the tail atoms (weight 5.0) |

The director is COM(head) → COM(tail).

Two ways a library breaks **silently**, so check by hand after any edit:

- **A duplicated `#LIPID` block double-counts that lipid.** The APL halves and
  κ doubles.
- **A group header with leading whitespace** (`" [C2][3]"`) fails to parse.
  Every lipid block after it then never loads, and `resis` segfaults on the
  empty selection.

Keep exactly one block per lipid, headers flush-left, and two blank lines after
each block.

### Step 2: index files (`lipid64`)

```bash
PYTHONNOUSERSITE=1 conda run -n lipid64 python analysis/make_lipidator_index.py      # all 9
PYTHONNOUSERSITE=1 conda run -n lipid64 python analysis/make_lipidator_index.py traj/amber_DMPC.gro
# -> data/lipidator_{ff}_{lipid}.ndx
```

- Each file has one group, `[ {LIPID} ]`, whose name must match the director
  block.
- It lists the 1-based global index of each lipid's first atom: the residue's
  first atom for CHARMM/SLipids, the first atom of each tail–head–tail triple
  for AMBER.
- The serials come from `traj/{ff}_{lipid}.gro`, which matches the xtc atom
  order.

### Step 3: run ReSiS

One replicate by hand:

```bash
~/software/lipidator-toolkit/build/resis \
    traj/charmm_DMPC1.xtc \
    ~/software/lipidator-toolkit/directors/charmm.ldx \
    data/lipidator_charmm_DMPC.ndx
```

The results are printed to stdout. The two lines that matter:

```
Bilayer Tilt: kappa <value> kt/nm^2     <- the tilt modulus, k_BT/nm^2
Per Lipid [bilayer]: <value>            <- area per lipid, Å^2
```

All systems and replicates (stdlib only):

```bash
python3 analysis/run_lipidator.py                                  # all 9 x 3
python3 analysis/run_lipidator.py --forcefields amber --lipids DMPG --reps 1 2 3
```

This runs `resis` on each `traj/{ff}_{lipid}{rep}.xtc` with
`directors/{ff}.ldx` and `data/lipidator_{ff}_{lipid}.ndx`, and parses the two
lines above. It writes:

- **`data/lipidator_moduli.json`**: per replicate, `tilt_kt_per_nm2` and
  `apl_A2`, merged into the existing file, so a subset run keeps the other
  systems.
- **`data/lipidator_moduli_summary.json`**: mean ± SD across replicates.

`resis` has no frame-range option, so it always analyses the **whole** xtc you
give it. Multiply by RT (2.577 kJ/mol at 310 K) for kJ mol⁻¹ nm⁻².

**Building `resis`** is only needed if recompiling. The shipped toolkit no
longer compiles on current Apple clang (`minimath.hpp` redefines
`std::tuple`). The `2112d9e` shims commit on the toolkit's master fixes this:

```bash
cd ~/software/lipidator-toolkit && cmake -S . -B build && make -C build
```

---

## 3. Director fluctuations (DF) → K_θ (k_BT/nm², bilayer)

Fourier-space method of Watson et al. 2012 (PRL 109:028102), with the theory of
Watson et al. 2011 (JCP 135:244701), as applied to CHARMM36 by Levine et al.
2014 (JACS 136:13582) and Venable et al. 2015. Script: `analysis/tilt_df.py`
(conda env `lipid64`, MDAnalysis + numpy). It reads gro + xtc directly. No
water, conversion or index file is needed. Bilayers must be whole, as for the
other two tools.

What it does:

- **Director per lipid, head → tail.**
  - `--director levine` (default, as in Levine 2014): midpoint of P and C2 → midpoint of the two terminal methyl carbons.
  - `--director rsf`: this project's ReGiS definition (P + C2 → COM of the last three carbons of each chain).
  - K_θ is director-sensitive at about the 10 % level, so state which one was used.
- **Leaflet** from the sign of the director z component.
- **Per-leaflet Fourier transform, summed directly over lipid positions.** There is no grid, so no empty-cell choices:
  `n_q^a = (√A / N_a) Σ_i n_i,xy e^{-i q·r_i}`, then bilayer `n̂_q = ½(n_q^up − n_q^down)`.
- **Longitudinal / transverse split** relative to q, for every mode (n_x, n_y) up to `--nmax` (default 6).
- **Fits.**
  - `1/⟨|n̂⊥|²⟩ = K_θ + K_tw q²` for q ≤ `--qcut-tilt`. Default cutoffs are 1.5, 2.1 and 2.5 nm⁻¹; the headline value is **2.1 nm⁻¹**, which keeps the three lowest q-shells for both the 6.2 nm (DMPC) and 6.6 nm (DOPC) boxes.
  - `1/⟨|n̂∥|²⟩ = K_c q²` for q ≤ `--qcut-bend` (default 1.5).
  - Fits are weighted by block errors. Errors are SEM over `--nblocks` (default 10) time blocks.
- **Units.** Report K_θ in **kJ mol⁻¹ nm⁻²** (the json key `K_theta_kJmol_nm2`, at `--temperature`, default 310 K). The fits are done internally in k_BT/nm² (K_θ) and k_BT (K_tw, K_c); the json also gives K_tw and K_c in kJ/mol.
  **K_θ is a bilayer modulus** (= 2 × the monolayer coefficient of ½κ_θ|m|²).

```bash
PYTHONNOUSERSITE=1 conda run -n lipid64 python analysis/tilt_df.py --selftest
# selftest: expected K_theta(bilayer) = 20.00 kT/nm^2, recovered 20.03

PYTHONNOUSERSITE=1 conda run -n lipid64 python analysis/tilt_df.py \
    traj/charmm_DMPC.gro traj/charmm_DMPC1.xtc --lipid DMPC --director levine \
    -o plots/tilt_three_methods/df_charmm_DMPC_1_levine
# -> .json (fits; K_theta_kJmol_nm2 is the value to report) and .npz (spectra, block SEMs); about 1 min per trajectory
```

The self-test feeds independent Gaussian per-lipid tilts of known monolayer
stiffness through the same transform and fit. Recovering 2× that stiffness
confirms the normalisation and the bilayer convention.

**Limits of these systems for DF.**

- With 64 lipids per leaflet the boxes are about 6.2–6.6 nm, so the smallest q is about 1 nm⁻¹. K_θ is an extrapolation to q = 0 from roughly 1–2 nm⁻¹.
- The answer moves with the q cutoff by about 10–15 % (DOPC 35.9 → 43.2 kJ mol⁻¹ nm⁻² for 1.5 → 2.5 nm⁻¹). Treat that as a systematic error on top of the block SEM.
- Venable 2015 report DF converged at 648 lipids (288 "on trust"). The 128-lipid systems here are below that.
- K_c from the longitudinal spectrum has **not** reached its small-q plateau in these boxes, so the K_c values are indicative only.

Comparison of all three methods on the same trajectories:
`python3 analysis/plot_tilt_three_methods.py` → `plots/tilt_three_methods/`
(see the README there).

---

## 4. Cross-check

The two results should satisfy κ_θ ≈ χ / A_lipid, with A_lipid in nm² from the
same run (`apl_A2 / 100`). For example, CHARMM DMPC rep 1 gives χ = 13.1 k_BT/rad²
at A = 0.604 nm², so χ/A = 21.7 k_BT/nm², against ReSiS κ_θ = 20.5 k_BT/nm².
Agreement to about 10 % is the expected level. A κ_θ near **2×** χ/A almost always means a duplicated director block
(section 2, step 1).

**Do not expect DF to agree with χ/A or κ_θ.**

- **Conventions differ.** The real-space number χ/A (≈ κ_θ) is the per-lipid modulus divided by the area. For the energy ½κ_θ|t|² per monolayer that is the **monolayer** modulus, although the ReGiS/ReSiS papers label it "bilayer". DF K_θ is a **bilayer** modulus. Double the real-space value before comparing.
- **They measure different things.** DF reads the q → 0 collective tilt; real-space methods read the per-lipid (all-wavelength) tilt variance.
- **On these trajectories** (rep 1, 310 K):

| | DF K_θ (levine director, bilayer) | ReGiS χ/A | ReSiS κ_θ |
|---|---|---|---|
| CHARMM DMPC | 29.8 ± 1.2 | 55.9 | 53.9 |
| CHARMM DOPC | 41.4 ± 2.1 | 45.3 | 37.8 |

  All values in kJ mol⁻¹ nm⁻² at 310 K. The real-space values are as reported, i.e. monolayer by the convention above.
- **The two families rank the lipids oppositely.** DF puts DOPC above DMPC, as Venable 2015 DF and X-ray experiment do. Both real-space tools put DMPC above DOPC.

