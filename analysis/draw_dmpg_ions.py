#!/usr/bin/env python3
"""Painters for the DMPG ion-condition SI figure (K+ Aqvist vs Na+ Joung-Cheetham).

Background: the AMBER Lipid21 paper reports that anionic lipids over-order and
thicken with Joung-Cheetham ions -- which is what the original DMPG runs used
(Na+ JC for AMBER and SLipids) and what produced the thickening this project
first measured.  Lipid21 suggests only potassium with the older Aqvist
parameters behaves acceptably for anionic lipids, so the DMPG systems were
re-run with K+ (Aqvist parameters for AMBER and SLipids).  The two conditions
differ substantially -- including for CHARMM, whose parameter set was the same
in both -- so this figure puts them side by side.

The panel painters mirror :mod:`draw_equilibrium` (Figure 1), with two changes:

* the categorical x axis is the **ion condition** (two columns) rather than the
  lipid (three); every panel is DMPG only, force fields as the coloured series;
* the per-carbon order-parameter panel overlays **all six** curves (3 force
  fields x 2 conditions), the JC condition in a lighter tint of the same
  force-field colour (position cannot encode the condition there).

Where the two conditions' inputs live
-------------------------------------
The K+ condition is the project's current data: ``data/lipprops_*.npz``,
``data/core_thickness.npz`` (keys ``{ff}_DMPG_*``), ``data/lipidator_blocks.json``
(keys ``{ff}_DMPG_{rep}``) and ``data/*_DMPG_{rep}_scd_sn{1,2}.dat``.

The Na+/JC condition is split by *how* it was produced, which is why it is not
all in one place:

* the cluster-side lipyphilic properties (APL, d_HH, D) were re-run into a
  sibling directory, ``data/data_Na/lipprops_*.npz``, whose keys are the plain
  ``{ff}_DMPG_eq{rep}`` -- the *directory* is the condition marker;
* the local analyses run off ``traj/{ff}_DMPG_JC{rep}.xtc``, so their outputs
  land in the shared ``data/`` under the **label** ``DMPG_JC``
  (``core_thickness.npz`` key ``{ff}_DMPG_JC_lipy_rep{r}``,
  ``lipidator_blocks.json`` key ``{ff}_DMPG_JC_{rep}``).

Each condition also needs its own pymbar ``t0`` (the production span the boxes
and CIs use): ``data/equilibrium_timeseries_stats.json`` for K+, and
``data/data_Na/equilibrium_timeseries_stats.json`` for JC (produce it with
``stats.py --data-dir data/data_Na --lipids DMPG``).

Any panel whose inputs are absent is left empty and reported, so the figure can
be built while parts of a condition are still being computed.
"""

import json
from pathlib import Path

import numpy as np
import matplotlib.ticker as mticker

import autocorr
import figstyle
import draw_equilibrium as de
from lipiddefs import REPS
from palette import FORCEFIELDS, CMAPS

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
NA_DATA = DATA / "data_Na"

# (key, x-axis label, lipprops directory, local-output label)
CONDITIONS = [
    # "\AA" is a LaTeX macro mathtext does not implement (it renders as the
    # literal text), so the Angstrom is the Unicode character, set outside math.
    ("K",  "K$^+$ \u00c5q", DATA,    "DMPG"),
    ("JC", "Na$^+$ JC",      NA_DATA, "DMPG_JC"),
]
COND_KEYS = [c[0] for c in CONDITIONS]

XTICK_PAD = 8.0         # pt -- baseline-anchored x tick labels (see _categorical)
RVIEW = 6.0             # A -- x limit of the ion-coordination panel
FIRST_SHELL = 4.5       # A -- first-shell boundary quoted in the run log
JC_TINT = 0.42          # CMAPS position for the lighter (JC) shade
# The lighter JC tint is *required* in the -S_CD panel, where all six curves
# share one axis and position cannot encode the condition.  On the categorical
# panels (a-d) the condition is already the x group, so the tint is optional
# there -- on by default because it makes the two columns read as one system in
# two states; set False for flat force-field colours in those panels.
TINT_CATEGORICAL = True

BOX_W = de.BOX_W
OFFSETS = de.OFFSETS
WHIS = de.WHIS
D_SCALE = de.D_SCALE
DT_NS = de.DT_NS
RT_310 = de.RT_310
SMOOTH_NS = de.SMOOTH_NS


def shade(ff, cond, colour, categorical=False):
    """Force-field colour for one condition: flat for K+, a lighter tint for JC.

    ``categorical`` marks the bar/box panels, where the tint is cosmetic and
    can be switched off with :data:`TINT_CATEGORICAL`.
    """
    if cond == "K" or (categorical and not TINT_CATEGORICAL):
        return colour
    return CMAPS[ff](JC_TINT)


_DUP_WARNED = set()


def duplicate_of_k(stem, data_dir):
    """True if ``data_dir``'s copy of ``lipprops_{stem}.npz`` is the K+ one.

    The Na+ lipprops are a *separate* set of simulations, so byte-identical
    arrays mean the wrong files were copied in -- which would draw the same
    data in both columns and read as "the ion condition changes nothing".
    This has already happened twice in this project's re-run cycles (a stale
    g_lomepro copy, a stale tail-density copy), so it is checked rather than
    trusted, and the offending condition is dropped from the panel.
    """
    if data_dir == DATA:
        return False
    a, b = DATA / f"lipprops_{stem}.npz", data_dir / f"lipprops_{stem}.npz"
    if not (a.exists() and b.exists()):
        return False
    ka, kb = np.load(a), np.load(b)
    shared = [k for k in kb.files if k in ka.files and "DMPG" in k]
    if not shared:
        return False
    same = all(np.array_equal(np.asarray(ka[k]), np.asarray(kb[k]))
               for k in shared)
    if same and stem not in _DUP_WARNED:
        _DUP_WARNED.add(stem)
        print(f"  !! {b} is byte-identical to the K+ {a.name} for every DMPG "
              f"key -- these are the K+ data, not Na+/JC; dropping that "
              f"condition from this panel")
    return same


def _stats(data_dir):
    path = data_dir / "equilibrium_timeseries_stats.json"
    if not path.exists():
        return None
    with open(path) as fh:
        return json.load(fh)


def load(prop):
    """``{(ff, cond): {"series": [...], "ci": ...}}`` for one per-frame property.

    Same treatment as :func:`draw_equilibrium.load` -- each replicate sliced to
    its own pymbar production span, the CI hierarchical over those spans -- but
    the two conditions come from different directories, each with its own
    ``t0``.  A condition with no stats JSON is skipped rather than silently
    analysed from frame 0, which would mix an equilibration transient into one
    column only.
    """
    stem, obs, _, scale = de.PROPS[prop]
    out = {}
    for cond, _, data_dir, _ in CONDITIONS:
        npz_path = data_dir / f"{stem}.npz"
        stats = _stats(data_dir)
        if not npz_path.exists() or stats is None:
            continue
        if duplicate_of_k(stem.replace("lipprops_", ""), data_dir):
            continue
        npz = np.load(npz_path)
        per_rep = stats["per_replicate"].get(obs, {})
        for ff, _, _ in FORCEFIELDS:
            key = f"{ff}_DMPG"
            recs = {r["rep"]: r for r in per_rep.get(key, [])}
            series, t0s = [], []
            for rep in REPS:
                name = f"{key}_eq{rep}"
                if name not in npz:
                    continue
                a = np.asarray(npz[name], dtype=float)
                if a.ndim == 2:                 # APL is (n_frames, n_lipids)
                    a = a.mean(axis=1)
                series.append(a * scale)
                t0s.append(int(recs.get(rep, {}).get("t0", 0)))
            if not series:
                continue
            out[(ff, cond)] = {
                "series": [s[t0:] for s, t0 in zip(series, t0s)],
                "ci": autocorr.hierarchical_ci(series, t0s),
            }
    return out


def _categorical(ax):
    ax.set_xticks(range(len(CONDITIONS)), [c[1] for c in CONDITIONS])
    ax.set_xlim(-0.5, len(CONDITIONS) - 0.5)
    figstyle.categorical_x(ax)
    # Matplotlib top-aligns x tick labels, so a taller glyph pushes its whole
    # label down: the ring on the Angstrom in "K+ Aq" dropped it below
    # "Na+ JC".  Baseline alignment lines the two up on their baselines
    # regardless of what sits above the cap height.
    for label in ax.get_xticklabels():
        label.set_verticalalignment("baseline")
    # With baseline anchoring the text no longer hangs from its top edge, so
    # the tick pad has to cover the font's descent (the "q" of Aq) itself.
    ax.tick_params(axis="x", which="major", pad=XTICK_PAD)


def _across_conditions(prop, cis, verbose):
    """Print the K+ vs Na+/JC test per force field.

    The drawn brackets compare force fields *within* a condition (the Figure 1
    convention this figure follows).  The comparison this figure is actually
    about runs the other way -- the same force field under the two ion
    conditions -- so it is reported in the run log rather than crowding the
    panel with brackets that span the two columns.
    """
    if not verbose or len(cis) < 2:
        return
    a, b = cis.get(COND_KEYS[0]), cis.get(COND_KEYS[1])
    if not a or not b:
        return
    rows = []
    for fi, (ff, _, _) in enumerate(FORCEFIELDS):
        if fi >= len(a) or fi >= len(b) or a[fi] is None or b[fi] is None:
            continue
        rows.append((ff, autocorr.compare(a[fi], b[fi])))
    # Holm across the three force fields' K+ vs Na+/JC tests for this property.
    adj = autocorr.holm([cmp["p"] for _, cmp in rows])
    for (ff, cmp), pa in zip(rows, adj):
        print(f"  {prop:10s} {ff:8s} K+ vs Na+/JC  "
              f"delta={cmp['delta']:+.3f}  dof={cmp['dof']:.1f}  "
              f"p={cmp['p']:.2e}  p_holm={pa:.2e}  "
              f"{autocorr.stars(pa, ns='n.s.')}")


def draw_box(ax, prop, show_ns=True, star_pt=6.0, verbose=False):
    """Box-and-whisker of a per-frame property, two ion-condition columns.

    Box = the per-frame distribution over the production span pooled across
    replicates; brackets = the hierarchical Welch test between force fields
    *within* a condition (the Figure 1 convention).
    """
    data = load(prop)
    cis, tops = {}, {}
    for ci_, (cond, _, _, _) in enumerate(CONDITIONS):
        cis[cond], tops[cond] = [], -np.inf
        for fi, (ff, _, colour) in enumerate(FORCEFIELDS):
            entry = data.get((ff, cond))
            cis[cond].append(None if entry is None else entry["ci"])
            if entry is None:
                continue
            pooled = np.concatenate(entry["series"])
            bp = ax.boxplot([pooled], positions=[ci_ + OFFSETS[fi]],
                            widths=BOX_W, whis=WHIS, showfliers=False,
                            patch_artist=True, manage_ticks=False)
            for patch in bp["boxes"]:
                patch.set(facecolor=shade(ff, cond, colour, True),
                          edgecolor="black", linewidth=0.7)
            for part in ("whiskers", "caps"):
                for line in bp[part]:
                    line.set(color="black", linewidth=0.7)
            for line in bp["medians"]:
                line.set(color="black", linewidth=0.9)
            tops[cond] = max(tops[cond], np.percentile(pooled, WHIS[1]))
            if verbose:
                print(f"  {prop:10s} {cond:3s} {ff:8s} "
                      f"mean {entry['ci']['mean']:.3f} "
                      f"+- {entry['ci']['ci_half']:.3f} ({entry['ci']['regime']})")
    de.sig_brackets(ax, cis, tops, show_ns=show_ns, star_pt=star_pt,
                    verbose=verbose, groups=COND_KEYS)
    _across_conditions(prop, cis, verbose)
    ax.set_ylabel(de.PROPS[prop][2])
    _categorical(ax)
    return data


def draw_diffusion(ax, verbose=False):
    """Grouped bars of D, two ion-condition columns (+- SD across replicates)."""
    for ci_, (cond, _, data_dir, _) in enumerate(CONDITIONS):
        npz_path = data_dir / "lipprops_lateral_diffusion.npz"
        if not npz_path.exists():
            print(f"  no lateral diffusion for {cond} ({npz_path})")
            continue
        if duplicate_of_k("lateral_diffusion", data_dir):
            continue
        npz = np.load(npz_path)
        for fi, (ff, ff_label, colour) in enumerate(FORCEFIELDS):
            key = f"{ff}_DMPG_D"
            if key not in npz:
                continue
            d = np.asarray(npz[key], dtype=float) * D_SCALE
            x = ci_ + OFFSETS[fi]
            e = d.std(ddof=1) if d.size > 1 else 0.0
            ax.bar([x], [d.mean()], BOX_W, color=shade(ff, cond, colour, True),
                   edgecolor="black", linewidth=0.7)
            ax.errorbar([x], [d.mean()], yerr=[e], fmt="none", ecolor="black",
                        elinewidth=0.7, capsize=1.5)
            if verbose:
                print(f"  D {cond:3s} {ff:8s} {d.mean():.2f} +- {e:.2f} um^2/s")
    ax.set_ylabel(r"$D$ ($\mathrm{\mu m^2\,s^{-1}}$)")
    ax.set_ylim(bottom=0)
    _categorical(ax)


def draw_kappa(ax, show_ns=True, star_pt=6.0, verbose=False):
    """Grouped bars of the Lipidator kappa_theta, two ion-condition columns.

    Both conditions read the same ``lipidator_blocks.json``; they are told apart
    by the label in the key (``{ff}_DMPG_{rep}`` vs ``{ff}_DMPG_JC_{rep}``).
    """
    path = DATA / "lipidator_blocks.json"
    if not path.exists():
        print(f"  no {path.name}; kappa panel empty")
        _categorical(ax)
        return
    with open(path) as fh:
        blocks = json.load(fh)
    cis, tops = {}, {}
    for ci_, (cond, _, _, label) in enumerate(CONDITIONS):
        cis[cond], tops[cond] = [], -np.inf
        for fi, (ff, ff_label, colour) in enumerate(FORCEFIELDS):
            recs = [blocks.get(f"{ff}_{label}_{r}") for r in REPS]
            recs = [r for r in recs if r]
            if not recs:
                cis[cond].append(None)
                continue
            means = [r["kappa_full"] * RT_310 for r in recs]
            sems = [r.get("kappa_block_sem", np.nan) * RT_310 for r in recs]
            neffs = [r.get("n_blocks", 1) for r in recs]
            ci = autocorr.hierarchical_ci_from_estimates(means, sems, neffs)
            cis[cond].append(ci)
            x = ci_ + OFFSETS[fi]
            ax.bar([x], [ci["mean"]], BOX_W, color=shade(ff, cond, colour, True),
                   edgecolor="black", linewidth=0.7)
            ax.errorbar([x], [ci["mean"]], yerr=[ci["ci_half"]], fmt="none",
                        ecolor="black", elinewidth=0.7, capsize=1.5)
            tops[cond] = max(tops[cond], ci["mean"] + ci["ci_half"])
            if verbose:
                print(f"  kappa {cond:3s} {ff:8s} {ci['mean']:.1f} "
                      f"+- {ci['ci_half']:.1f} kJ/mol/nm2 ({ci['regime']})")
    de.sig_brackets(ax, cis, tops, show_ns=show_ns, star_pt=star_pt,
                    verbose=verbose, groups=COND_KEYS)
    _across_conditions("kappa", cis, verbose)
    ax.set_ylabel(r"$\kappa_\theta$ (kJ\,mol$^{-1}$\,nm$^{-2}$)"
                  .replace("\\,", " "))
    ax.set_ylim(bottom=0)
    _categorical(ax)


def scd_profiles(cond_label, data_dir):
    """``{ff: (carbons, mean, SD)}`` of -S_CD for one condition.

    g_lomepro output is per replicate per tail; the two tails are averaged
    within a replicate (they differ by <0.01 for these symmetric lipids), then
    the line is the mean over replicates and the band their SD.  The JC files
    are looked for both as ``data_Na/{ff}_DMPG_{rep}_...`` (re-run alongside the
    other cluster-side Na+ analyses) and as ``data/{ff}_DMPG_JC_{rep}_...``
    (re-run under the local label), so either convention works.
    """
    out = {}
    for ff, _, _ in FORCEFIELDS:
        per_rep, carbons = [], None
        for rep in REPS:
            tails = []
            for sn in (1, 2):
                for path in (data_dir / f"{ff}_DMPG_{rep}_scd_sn{sn}.dat",
                             DATA / f"{ff}_{cond_label}_{rep}_scd_sn{sn}.dat"):
                    if path.exists():
                        arr = np.loadtxt(path)
                        carbons = arr[:, 0]
                        tails.append(arr[:, 1])
                        break
            if tails:
                per_rep.append(np.mean(tails, axis=0))
        if per_rep:
            per_rep = np.array(per_rep)
            out[ff] = (carbons, per_rep.mean(axis=0),
                       per_rep.std(axis=0, ddof=1) if len(per_rep) > 1
                       else np.zeros(per_rep.shape[1]))
    return out


def draw_scd(ax, verbose=False):
    """All six -S_CD profiles overlaid: 3 force fields x 2 ion conditions.

    Condition is encoded by *tint* (JC lighter) rather than position, so the
    six curves can share one axis; the in-panel key shows the two tints in one
    force field's hue so the convention reads without six legend entries.
    """
    from matplotlib.lines import Line2D
    carbons_seen = None
    drawn = []
    for cond, cond_label, data_dir, label in CONDITIONS:
        profiles = scd_profiles(label, data_dir)
        if not profiles:
            print(f"  no S_CD files for the {cond} condition")
            continue
        for ff, ff_label, colour in FORCEFIELDS:
            if ff not in profiles:
                continue
            carbons, mean, sd = profiles[ff]
            carbons_seen = carbons
            col = shade(ff, cond, colour)
            ax.plot(carbons, mean, color=col, marker="o", markersize=1.6,
                    markeredgewidth=0, lw=1.0 if cond == "K" else 0.9,
                    zorder=3 if cond == "K" else 2)
            ax.fill_between(carbons, mean - sd, mean + sd, color=col,
                            alpha=0.18, lw=0)
            drawn.append(cond)
            if verbose:
                print(f"  S_CD {cond:3s} {ff:8s} <-S_CD> = {mean.mean():.4f}")
    ax.set_xlabel("acyl carbon")
    ax.set_ylabel(r"$-S_\mathrm{CD}$")
    if carbons_seen is not None:
        ax.set_xticks(np.arange(int(carbons_seen[0]),
                                int(carbons_seen[-1]) + 1, 2, dtype=int))
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    # Tint key, in CHARMM's hue: same colour at two lightnesses = the condition.
    ff0, _, colour0 = FORCEFIELDS[0]
    keys = [(c, l) for c, l, _, _ in CONDITIONS if c in drawn]
    if len(keys) > 1:
        ax.legend([Line2D([], [], color=shade(ff0, c, colour0), lw=1.2)
                   for c, _ in keys], [l for _, l in keys],
                  loc="lower left", frameon=False, handlelength=1.4,
                  borderpad=0.1, labelspacing=0.25, borderaxespad=0.2)


def draw_dhc_traj(ax, cond, verbose=False):
    """Per-frame d_HC(t) for one ion condition, all replicates of all three ff.

    Replicate shades follow Figure 1f (the ``palette.CMAPS`` ramps), so the
    within-condition run-to-run spread reads the same way in both panels; the
    two panels share a y range so the conditions can be compared by eye.
    """
    label = dict((c[0], c[3]) for c in CONDITIONS)[cond]
    title = dict((c[0], c[1]) for c in CONDITIONS)[cond]
    path = DATA / "core_thickness.npz"
    if not path.exists():
        print(f"  no {path.name}; d_HC panel ({cond}) empty")
        ax.set_title(title)
        return False
    npz = np.load(path)
    window = int(round(SMOOTH_NS / DT_NS))
    shades = np.linspace(0.9, 0.3, 10)[::4]
    found = False
    for ff, ff_label, colour in FORCEFIELDS:
        cmap = CMAPS[ff]
        for ri, rep in enumerate(REPS):
            key = f"{ff}_{label}_lipy_rep{rep}"
            if key not in npz:
                continue
            found = True
            y = np.asarray(npz[key], dtype=float)
            t = np.arange(len(y)) * DT_NS
            col = cmap(shades[ri])
            ax.plot(t, y, color=col, lw=0.4, alpha=0.20)
            ax.plot(t, de._smooth(y, window), color=col, lw=0.9)
            if verbose:
                print(f"  d_HC {cond:3s} {ff:8s} rep{rep}  {y[0]:.2f} -> "
                      f"{y[-100:].mean():.2f} nm")
    if not found:
        print(f"  no d_HC series for the {cond} condition "
              f"(core_thickness.npz has no {label} keys)")
    ax.set_xlabel("time (ns)")
    ax.set_ylabel(r"$d_\mathrm{HC}$ (nm)")
    ax.set_xlim(0, None)
    ax.set_title(title)
    return found


def draw_ion_binding(ax, kind="gr", ref="box", verbose=False):
    """Counterion pair distribution g(r) around the lipid phosphate.

    This is the *mechanism* panel: Lipid21 attributes the over-ordering of
    anionic lipids to Joung-Cheetham cations binding the headgroup too
    strongly, and the P...cation PDF shows that directly -- the first-shell
    peak is an order of magnitude taller for the JC builds.  ``kind="nr"``
    switches to the cumulative coordination number instead (both are in the
    npz); g(r) is the default because the contrast is far starker in the
    peak than in the integral.

    Line is the mean over replicates, band their SD, JC in the lighter tint (as
    in the S_CD panel -- both conditions share one axis here).
    """
    path = DATA / "ion_binding.npz"
    if not path.exists():
        print(f"  no {path.name}; ion-binding panel empty "
              f"(run ion_binding.py --lipids DMPG DMPG_JC)")
        return
    npz = np.load(path)
    r = np.asarray(npz["r_A"], dtype=float)
    for cond, cond_label, _, label in CONDITIONS:
        for ff, ff_label, colour in FORCEFIELDS:
            keys = [f"{ff}_{label}_{kind}_rep{rep}" for rep in REPS]
            reps = [npz[k] for k in keys if k in npz]
            if not reps:
                continue
            a = np.array(reps, dtype=float)
            if kind == "gr" and ref == "farfield":
                far = a[:, r >= 8.0].mean(axis=1, keepdims=True)
                a = a / np.where(far > 0, far, 1.0)
            mean = a.mean(axis=0)
            sd = a.std(axis=0, ddof=1) if len(a) > 1 else np.zeros_like(mean)
            col = shade(ff, cond, colour)
            ax.plot(r, mean, color=col, lw=1.0 if cond == "K" else 0.9,
                    zorder=3 if cond == "K" else 2)
            ax.fill_between(r, mean - sd, mean + sd, color=col, alpha=0.18,
                            lw=0)
            if verbose:
                ion = str(npz.get(f"{ff}_{label}_ion", "?"))
                nk = f"{ff}_{label}_nr_rep{REPS[0]}"
                extra = (f", N({FIRST_SHELL:.1f} A) = "
                         f"{npz[nk][int(np.searchsorted(r, FIRST_SHELL))]:.2f}"
                         if nk in npz else "")
                print(f"  g(r) {cond:3s} {ff:8s} {ion:8s} "
                      f"peak = {mean.max():.1f} at r = "
                      f"{r[int(np.argmax(mean))]:.1f} A, "
                      f"far-field g = {mean[r >= 8.0].mean():.2f}{extra}")
    ax.set_xlabel(r"$r$ from phosphate ($\mathrm{\AA}$)")
    ax.set_ylabel(r"counterion $g(r)$" if kind == "gr"
                  else r"counterions per lipid, $N(r)$")
    # Stop at RVIEW: the first shell (~3.7 A) and its decay are the content,
    # and the npz keeps the full 10 A for anyone who wants the diffuse layer.
    ax.set_xlim(0, RVIEW)
    top = max([l.get_ydata()[r <= RVIEW].max() for l in ax.get_lines()
               if len(l.get_ydata()) == len(r)] or [1.0])
    ax.set_ylim(0, top * 1.10)


def legend_handles():
    """Force-field key (colours shown at full strength, i.e. the K+ tint)."""
    return de.legend_handles()
