# HMMM analysis

Force-field comparison (CHARMM / AMBER / SLipids) of a **highly mobile membrane
mimetic** (HMMM) bilayer: DOPC with its acyl chains truncated to 6:0 and the
hydrophobic core filled with **1,1-dichloroethane (DCLE)**.  Each force field is
run in three ensembles — `npt`, `npat`, `posre` — with three 1 µs replicates
each, sampled at 1 ns/frame.

Everything runs **locally**; unlike the parent project there is no cluster step
and no g_lomepro.  All three force fields are in `traj/` (SLipids landed
2026-09-24).

AMBER NPAT was re-run with the correct box (9.91 nm laterally, matching the
CHARMM/SLipids NPAT boxes; the first runs used 10.70 nm) and re-analysed on
2026-09-28.  All three reps stay a bilayer for the full microsecond.

## Layout

```
HMMM/traj/{ff}_{ens}{rep}.{gro,xtc}     full system (no water, no ions — DCLE is the solvent)
HMMM/data/                              computed .npz / .json  (gitignored, as in the parent project)
HMMM/data/topol/                        cached, residue-renumbered AMBER .gro topologies
HMMM/plots/final/                       production figures
```

## Run order

```bash
# 1. compute (lipyphilic + MDAnalysis) -> HMMM/data/lipprops_*.npz, box.npz
PYTHONNOUSERSITE=1 conda run -n lipid64 python HMMM/analysis/lipid_properties.py --nprocs 6

# 2. equilibration / autocorrelation (needs pymbar, which lives in base)
PYTHONNOUSERSITE=1 conda run -n base python HMMM/analysis/stats.py

# 3. stability: per-frame lateral SD of d_HH, then the stability time
PYTHONNOUSERSITE=1 conda run -n lipid64 python HMMM/analysis/thickness_variation.py --nprocs 6
PYTHONNOUSERSITE=1 conda run -n base    python HMMM/analysis/stability.py

# 4. equilibration stats -- AFTER stability.py, which it truncates on
PYTHONNOUSERSITE=1 conda run -n base python HMMM/analysis/stats.py

# 5. diffusion (lipid + DCLE) and solvent RDFs, both over the stable window
PYTHONNOUSERSITE=1 conda run -n lipid64 python HMMM/analysis/diffusion.py --nprocs 6
PYTHONNOUSERSITE=1 conda run -n lipid64 python HMMM/analysis/rdf.py --nprocs 6

# 6. snapshots -> HMMM/plots/snapshots/{ff}_{ens}{rep}.png
PYTHONNOUSERSITE=1 conda run -n lipid64 python HMMM/analysis/render_snapshots.py
#    + the two large AMBER NPAT runs -> big_hmmm_{stable,broken}.png
python3 HMMM/analysis/render_big_hmmm.py

# 7. figures -> HMMM/plots/final/
PYTHONNOUSERSITE=1 conda run -n base python HMMM/figures/fig01_hmmm_stability.py
PYTHONNOUSERSITE=1 conda run -n base python HMMM/figures/fig02_hmmm_properties.py
PYTHONNOUSERSITE=1 conda run -n base python HMMM/figures/figSI_snapshots.py
python3 HMMM/figures/figSI_big_hmmm.py
```

**The order matters.** `stability.py` needs `thickness_variation.py`; `stats.py`,
`diffusion.py` and `rdf.py` all read `stability.json` and truncate on it; both
figures read all of the above. Re-running `thickness_variation.py` invalidates
everything downstream.

`PYTHONNOUSERSITE=1` is not optional on this machine — a stray `~/.local`
scipy shadows the env's and breaks the MDAnalysis import.

Each step takes a `--forcefields` / `--ensembles` / `--reps` subset; step 1
**updates** the `.npz` files in place rather than rewriting them, so running it
for SLipids alone later will not clobber the CHARMM/AMBER results.

## Modules

| file | what it is |
|---|---|
| `hmmm_defs.py` | single source of truth for the systems and the force-field vocabulary — force fields, ensembles, replicates, residue/atom names, the DCLE solvent, the 6-carbon chains.  Pure stdlib; **index-group builders ported from the parent project get their DCLE names from here.** |
| `topology.py` | universe construction, incl. repairing AMBER's uniformly-`1` residue numbering (cached into `data/topol/`) |
| `lipid_properties.py` | the lipyphilic compute step: area per lipid, `d_HH`, MSD/`D`, plus the per-frame box vectors |
| `stats.py` | pymbar `t0`, statistical inefficiency `g`, `Neff`, and the within-/between-run variance decomposition |
| `thickness_variation.py` | per-frame **lateral** SD of d_HH — the stability signal |
| `stability.py` | the stability time per run, and which systems get hatched |
| `diffusion.py` | lateral D for the lipid headgroups **and** the DCLE solvent, over the stable window |
| `rdf.py` | DCLE–DCLE and C6–DCLE radial distribution functions + running coordination numbers |
| `plot_rdf.py` | styles them into `plots/rdf.png` and `plots/rdf_cn.png` |
| `rdf_time.py` / `plot_rdf_time.py` | diagnostic: RDF first-peak position in 25 ns windows over the **whole** run (rep 1, no stability cut) → `plots/rdf_peak_time.png`; the peak does not move at breakup |
| `render_snapshots.py` | headless PyMOL renders of every run's final configuration |
| `render_big_hmmm.py` | oblique (45°, edge-on) renders of the two large AMBER NPAT final frames, `traj/big_hmmm_{stable,broken}.gro` |
| `draw_stability.py` / `draw_properties.py` | `draw_*(ax)` painters for the two main-text figures |
| `../figures/fig01_hmmm_stability.py` | figure 1 — stability |
| `../figures/fig02_hmmm_properties.py` | figure 2 — properties |
| `../figures/figSI_snapshots.py` | SI galleries — every run's final configuration, one figure per force field |
| `../figures/figSI_big_hmmm.py` | SI figure — the two large AMBER NPAT runs, **a** stable / **b** broken |

Shared infrastructure (`palette`, `figstyle`, `autocorr`, and the parent
`stats.analyze_series`/`decompose`) is imported from `../../analysis`, so the
two branches use one colour scheme, one house figure style and one error model.

## Things that bit, and how they are handled

**AMBER numbers every lipid residue `1`.**  Lipid21 lays a lipid out
tail–head–tail (`VL`–`PC`–`VL`) with a shared resid, so MDAnalysis merges the
entire 17 920-atom lipid block into a couple of giant residues.  That is fatal
rather than untidy: every lipyphilic analysis sizes its output by
`membrane.n_residues` and indexes leaflets with `membrane.residues[...]`, so a
256-phosphate selection would describe *one* lipid.  `topology.py` writes a
renumbered copy (one resid per molecule, boundaries found by resname change *or
repeated atom name*) into `data/topol/` and uses that as the topology; the
downloaded `traj/` files are never modified.

**Two CHARMM NPT runs fail.**  `charmm_npt1` (790 frames) and `charmm_npt3`
(891 frames) hit a pressure-coupling failure: the box contracts laterally and
stretches along z from ~400 ns, then runs away to 2.3 × 2.3 × 270 nm over the
last ~100 frames.  **Every available frame is analysed as-is** — the blow-up is
real output, not something the pipeline hides — so those runs carry enormous
tails (`d_HH` to ~39 nm, APL down to ~6 Å²).  The figure sets its y limits from
the well-behaved groups and marks an off-panel whisker with a caret, rather
than letting one broken run set the scale for the other eight systems.  The
full per-frame box trace of every run is kept in `data/box.npz`.

**NPAT fixes the lateral area by construction.**  The per-frame *mean* APL of an
NPAT run is therefore a constant, and the three replicates agree to ~1e-14 Å².
The box degenerates to a line and a significance test would return p ≈ 0 for a
difference that was imposed, not measured — so a group with a numerically zero
hierarchical standard error gets no bracket, and says so in the printed table.

**The reference lipid is arguably the wrong one.**  These HMMMs were built to
reproduce **DOPC's thickness**, which `d_HH` (panel b) duly checks.  APL and `D`
are *not* expected to match DOPC and would be better judged against a **DSPC**
bilayer, whose headgroup packing the HMMM construction preserves.  No DSPC data
exists in this repository, so the figure carries DOPC and the caption has to
carry the caveat.

## Vocabulary notes (for the index-group builders still to be ported)

* Solvent is **DCLE**, `resname DCLE` (CHARMM) / `resname 303` (AMBER, SLipids) —
  8 atoms, carbons `C1`/`C2`, chlorines `CL11`,`CL12` (CHARMM) or `CL3`,`CL4`.
  There is **no water and no ions** in these systems.
* Lipid residues: CHARMM one residue still named `DOPC` (SLipids `DVPC`); AMBER head `PC`
  + two `VL` (valeroyl) tails.
* Chains are 6 carbons counting the carbonyl.  **CHARMM terminates them
  `C2T`/`C3T`, SLipids `C26`/`C36`** — a selection keyed on either spelling alone
  silently selects nothing for the other force field.  `chain_atom_names`
  returns both (each force field uses only one), which gives the same 512
  terminal carbons in all three.
* SLipids names the single lipid residue `DVPC`, not `DOPC`; it is in
  `SINGLE_RESN_ALIASES`, so selections match it.
* Phosphate is `name P P31`, as in the parent project.

## Solvent RDFs (`rdf.py` + `plot_rdf.py`)

Two pairs, both keyed on DCLE's **CHCl2 carbon** as the molecule's single site:
`dcle_dcle` (solvent packing, self-pair excluded) and `c6_dcle` (solvent around
the acyl-tail terminus — chain position 6, `C2T`/`C3T` for CHARMM, `C26`/`C36`
for SLipids and `C16` for AMBER's `VL` residues). Range 0–2 nm in 0.1 Å bins; `--stride` (5 by
default) subsamples frames, which a smooth ensemble average tolerates easily.

- **CHARMM and AMBER swap C1 and C2.** CHARMM's `C1` carries the two chlorines;
  the AMBER/SLipids `DCLE.itp` calls that carbon `C2` — the same swap
  `setup_scripts/reorder_gro.py` documents. Hard-coding the wrong one shifts
  every peak by a C–C bond length and would never raise, so `rdf.py` finds the
  centre **geometrically** per file (the candidate carbon closest to that
  molecule's chlorines) and only *reports* a disagreement with the expectation
  in `hmmm_defs.DCLE_CENTRE`. A `name C*` glob does **not** work for this: it
  matches `CL11`/`CL3`, and a chlorine is trivially the atom nearest the
  chlorines. The candidates are named explicitly.
- **g(r) does not level off at 1.** MDAnalysis normalises against the
  box-average density and DCLE is confined to the bilayer core, so its local
  density is ~2–3× the box average. Peak positions are unaffected, peak heights
  are scaled; the plateau is stored per curve and drawn as a dotted line, and
  `plot_rdf.py --scale-plateau` divides it out. The **coordination number**
  (`rdf_cn.png`, the integral of the same histogram) sidesteps the issue
  entirely and is the number to quote.
- **Collapsed frames are dropped, and only those.** Minimum image needs
  `r_max ≤ L/2`; the two failed CHARMM NPT runs contract to a 2.3 nm box where a
  2 nm cutoff is not computable. Those frames are skipped and counted in the
  output — a correctness bound, not a judgement about the data.

## Snapshots (`render_snapshots.py`)

Headless PyMOL, whole `.pml` generated from Python, one panel per run:
lipids as spheres with **tail carbons `gray30`, headgroup carbons `gray80`,
phosphorus `orange`**, and DCLE as **pale-green sticks**. Not AMBER-only — the
selections come from `hmmm_defs`, so the single-residue force fields (chains
separated by atom name, terminal methyls `C2T`/`C3T` or `C26`/`C36`) work too.

- **Always the run's final configuration**, read from `traj/{ff}_{ens}{rep}.gro`
  with no trajectory attached and no way to ask for another frame. Some of these
  runs end with the membrane broken up and that is the point of the panel;
  hunting for a frame that looks better would be choosing the picture rather
  than reporting the run. (Attaching the `.xtc` and forgetting to seek silently
  renders frame 0 — the *starting* structure — which is exactly what happened
  once here.)
- **Wrapped, cut where it is emptiest.** `-pbc mol` keeps molecules whole but
  lets them sit outside the box, which renders as lipid patches floating clear
  of the membrane. The system is translated so the **emptiest slice of the
  lipid z-histogram** lands on the box edge and then wrapped by residue. For an
  intact bilayer that is the gap between periodic images, so the membrane comes
  out centred — the same answer as centring on the mean phosphate z, but it also
  survives a membrane that has broken up or reoriented.
- **Framing** is a square window the width of the in-plane extent, centred on the
  median lipid z, rather than `zoom all`: 8 Å of buffer (`zoom` fits atom
  *centres*, so a smaller buffer clips the outermost spheres).
- **The two collapsed runs get their own camera.** `charmm_npt1`/`npt3` end as
  ~4 × 4 × 270 nm filaments; fitting all 270 nm gives a hairline however it is
  turned. They are detected from the `.gro` box line (`box_z > 3 ×` in-plane)
  and shown turned **60°** instead of 90°, under a wide-angle (50°) perspective
  camera and pulled back to ~6 column widths, so the column runs into the screen
  and fades out with the rest of the box outside the frame. `--collapsed-turn`
  and `--collapsed-window` tune it; a *smaller* angle points the long box more
  directly away from the camera.
- **A translucent DCLE surface was tried and rejected** (`show surface` on an
  extracted object, `solvent_radius 3.2`, `vdw 2`, 50 % transparent). It is
  prettier for an intact bilayer, but it degenerates into disjoint shards for
  exactly the broken-up runs the panels exist to show.
- PyMOL lives in the conda **base** env while the structure preparation needs
  MDAnalysis from `lipid64`, so `find_pymol()` falls back to the conda root's
  `bin` when `pymol` is not on `PATH`.

## Stability (`thickness_variation.py` + `stability.py`)

`lipid_properties.py` stores one `d_HH` per frame — the mean over the membrane
plane. `thickness_variation.py` keeps the other half of that frame: the
**standard deviation of d_HH across the plane** (each leaflet's phosphates
binned onto a 6×6 grid, per-cell leaflet separation, SD over cells). A flat
bilayer is uniform; buckling, opening or breaking up makes the leaflet
separation depend strongly on *where* it is measured, and the spatial SD climbs
an order of magnitude long before the *mean* thickness looks wrong.

**Stability time** = the first frame at which the 25 ns rolling mean of that SD
exceeds **1.0 nm**; a run that never crosses is stable for its whole
trajectory. The threshold sits in the gap between the runs that hold together
(plateaux at 0.17 / 0.35 / 0.55–0.75 nm, no step) and the two that do not
(clean steps out of their plateau to 1.0–1.6 and to 20 nm), and it reproduces
the by-eye classification of every run. `stability.py` also prints the
*relative* rule (2× each run's own first-100 ns plateau) as a cross-check; for
CHARMM and AMBER the two agree to 15–40 ns. **It is useless for SLipids
NPT/NPAT**, which break up *inside* the 100 ns baseline window: the baseline is
itself already >1 nm, so the relative rule reports 111–1001 ns for runs whose
final frames are plainly not bilayers. Trust the absolute rule there.

Results: **charmm_npt** breaks at 401 / — / 461 ns (rep 2 survives),
**charmm_npat** at 500 ns (reps 2–3 survive), **slipids_npt** at 30 / 41 / 12 ns
and **slipids_npat** at 13 / 15 / 88 ns; everything else (all of AMBER, and
posres in every force field) runs the full microsecond. So `charmm_npt`,
`charmm_npat`, `slipids_npt` and `slipids_npat` are the four hatched systems.

The SLipids breakup is fast and total: the lateral SD leaves its 0.19 nm start
within ~10–20 ns and sits at 1.1–2 nm for the rest of the run, with no CHARMM-
style step, while the NPAT runs keep a normal mean `d_HH` (~3.7 nm) — the
final frames show lipids reorganised around DCLE channels rather than a
bilayer, and SLipids NPT stretches the box to 12–28 nm along z. Their stable
windows are therefore only 12–88 frames, so every SLipids NPT/NPAT property is
computed from ~3–18 RDF frames and ~8–37 effective samples — noisy, and hatched
for that reason.

**Truncation has to come before the equilibration analysis, not after.** On a
run that broke up, pymbar's `detect_equilibration` latches onto the breakup and
returns a `t0` *later* than the stability time (552 ns against 401 ns for
`charmm_npt1`), leaving no valid window at all. `stats.py` therefore slices to
the stable window first and finds `t0` inside it.

## Main-text figures

**Figure 1 — stability** (`fig01_hmmm_stability.py`): 2×3, 180 × 118 mm.
**a** a run that survived (AMBER posres), **b** the lateral-SD traces c is read
from, **c** how long each system stayed a bilayer (bars = mean over reps, dots =
individual runs, dashed line = the full µs), **d** destabilised but never
broken (AMBER NPT), **e** broken (CHARMM NPAT), **f** crashed (CHARMM NPT).
(b and c were swapped on 2026-10-10.) Panel b is on a **log** y axis — the runs span 0.14 nm (a restrained flat
bilayer) to 30 nm (a filament), and a linear axis would put every stable run on
the baseline.

**Figure 2 — properties** (`fig02_hmmm_properties.py`): 2×3, 180 × 112 mm.
**a** lipid headgroup diffusion, **b** d_HH, **c** APL (all three against a full
DOPC bilayer past a dashed rule), **d** DCLE diffusion (HMMM only), **e**
DCLE–DCLE and **f** C6–DCLE g(r). Both diffusion panels are on a log axis: the
HMMM lipids are 10–30× a DOPC bilayer and DCLE another ~20× faster again.

**Colour carries two axes at once** in both figures: the force field picks the
light→dark ramp (`palette.CMAPS`, the pore project's closing-trajectory ramps)
and the ensemble picks the shade — every third colour of that ten-step ramp, so
NPT is darkest and posres lightest. The DOPC reference group has no ensemble, so
it takes the force field's solid categorical colour (`palette.COLORS`).

**The red `//\\` hatch** marks a system with any replicate short of the full
microsecond. Truncation keeps those numbers honest but cannot make them
equivalent — they come from less trajectory, and from a system on its way to
falling apart. It is drawn as a second, face-less patch over the artist, because
matplotlib ties hatch colour to edge colour and the bars keep a black outline;
on the RDF panels the SD *band* is hatched rather than the line restyled, so the
curves stay comparable. The hatch is a **properties-figure** convention only: Figure 1's stability-time
bars (panel c) are not hatched (removed 2026-10-10), since the bar heights
themselves show which systems fell short.

**NPAT fixes the lateral area by construction**, so its per-frame mean APL is a
constant and its box in panel c degenerates to a line. That is not a plotting
bug.

**SI galleries** (`figSI_snapshots.py`): one figure per force field, 3×3 at
180 × 172 mm, **replicates across the columns and pressure-coupling condition
down the rows**. Figure 1 shows four runs chosen to illustrate the four
outcomes; this is the rest of the evidence, so a reader can check that an
outcome belongs to the condition rather than to the run that was picked. Each
panel carries its run's **stability time**, in red where it fell short of the
microsecond — the CHARMM NPT row in particular is only interpretable with the
numbers on it (runs 1 and 3 collapse at 401 and 461 ns, run 2 survives). No
panel letters: a gallery is addressed by its row and column labels, and nine
more letters would only compete with them. A force field with no renders yet is
skipped rather than producing a grid of "missing" boxes.

**Large-system SI figure** (`figSI_big_hmmm.py` → `SI_big_hmmm.*`, 180 × 86 mm, no titles,
1×2): the final configurations of two large HMMM runs (904 lipids, AMBER, NPAT,
~19.6 × 17.0 × 8.1 nm hexagonal box), one that stayed a bilayer and one that
broke up, rendered by `render_big_hmmm.py` in the usual HMMM style. The camera
is **45° above the membrane plane, facing an edge of the box**: only `turn x,
-45` from the top-down view, no spin about z. It is orthoscopic and **framed on
the eight corners of the rectangular brick** (+15 Å), rather than on the atoms,
so both panels share one scale and the broken run cannot change the framing.
The brick, not the hexagonal cell, because `trjconv -pbc mol` (default `-ur
rect`) puts every molecule's centre in `[0, a_x) × [0, b_y) × [0, c_z)`, and the
cell's slanted `b` vector would add ~10 nm of empty width. The camera is placed
by view-matrix arithmetic rather than `zoom`, which letterboxes in headless
mode. PyMOL's `get_view()[:9]` is the model→camera rotation stored
**column-major**. The gros are rendered as downloaded (already `-pbc mol` on the
cluster), with no re-wrapping.

An earlier exploratory figure (`fig01_hmmm_vs_dopc.py` →
`HMMM_equilibrium.*`, with its painter `draw_hmmm_equilibrium.py`) was
**retired and deleted** on 2026-09-24. It drew figure 2's a–c panels but cut
only the pymbar burn-in, not the stability time, so for the four unstable
systems it reported post-breakup geometry (e.g. SLipids NPT `d_HH` 6.10 nm and
APL 44.7 Å² against 4.11 nm / 69.5 Å² on the stable window). `fig02_hmmm_properties.py`
is the only source of those numbers now.
