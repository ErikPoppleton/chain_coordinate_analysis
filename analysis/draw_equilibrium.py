#!/usr/bin/env python3
"""Panel painters for the main-text **equilibrium force-field comparison**.

Pure ``draw_*(ax)`` functions: they create no figure and save nothing, so the
same panel can go into an exploratory one-off or into a
``figures/figNN_*.py`` production figure (see :mod:`figstyle`).

Panels
------
``draw_box(ax, prop)``     box-and-whisker of a per-frame distribution -- area
                           per lipid (A^2) or head-head thickness d_HH (nm) --
                           lipids along x, a group of three force fields each,
                           with pairwise significance brackets.
``draw_diffusion(ax)``     grouped bars of the lateral diffusion coefficient D
                           (um^2/s), mean +- SD across the three replicates.

What the boxes and the stars mean are **two different things**, deliberately:

* the **box** is the distribution of the per-frame observable over the
  production span, pooled across replicates (box = quartiles, whiskers = the
  5th/95th percentiles, no fliers) -- i.e. how much the membrane fluctuates;
* the **stars** come from :func:`autocorr.compare`, a Welch-style test on the
  *hierarchical* standard errors, which combine each run's
  autocorrelation-limited SEM with the between-replicate spread.  A t-test over
  the raw frames would be pseudoreplication -- 2501 correlated frames are not
  2501 samples, and the real n is 3 runs -- and would call almost every pair
  significant.

Both the production span (pymbar ``t0``) and the per-replicate within-run SEMs
are read from ``data/equilibrium_timeseries_stats.json``, so **run ``stats.py``
first**.  Requires numpy + matplotlib only.
"""

import json
from pathlib import Path

import numpy as np
import matplotlib.ticker as mticker

import autocorr
import figstyle
from lipiddefs import LIPIDS, REPS
from palette import FORCEFIELDS, CMAPS

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

# prop key -> (npz stem, stats.py observable name, axis label, scale)
PROPS = {
    "apl": ("lipprops_area_per_lipid", "apl",
            r"APL ($\mathrm{\AA}^2$)", 1.0),
    "thickness": ("lipprops_memb_thickness", "thickness",
                  r"$d_\mathrm{HH}$ (nm)", 0.1),      # npz is in Angstrom
}

D_SCALE = 1e8                    # cm^2/s -> um^2/s
WHIS = (5, 95)                   # whisker percentiles
BOX_W = 0.22
OFFSETS = np.array([-0.26, 0.0, 0.26])


def _stats_json():
    path = DATA / "equilibrium_timeseries_stats.json"
    if not path.exists():
        raise SystemExit(f"{path} missing -- run `conda run -n base python "
                         f"analysis/stats.py` first")
    with open(path) as fh:
        return json.load(fh)


def load(prop):
    """``{(ff, lipid): {"series": [per-rep arrays], "ci": hierarchical CI}}``.

    Each replicate series is sliced to its pymbar production span ``[t0:]`` and
    scaled to the plotted unit; the CI comes from :func:`autocorr.hierarchical_ci`
    on those same spans.
    """
    stem, obs, _, scale = PROPS[prop]
    npz = np.load(DATA / f"{stem}.npz")
    per_rep = _stats_json()["per_replicate"].get(obs, {})
    out = {}
    for ff, _, _ in FORCEFIELDS:
        for lipid in LIPIDS:
            key = f"{ff}_{lipid}"
            recs = {r["rep"]: r for r in per_rep.get(key, [])}
            series, t0s = [], []
            for rep in REPS:
                name = f"{key}_eq{rep}"
                if name not in npz:
                    continue
                a = np.asarray(npz[name], dtype=float)
                if a.ndim == 2:              # APL is (n_frames, n_lipids)
                    a = a.mean(axis=1)
                series.append(a * scale)
                t0s.append(int(recs.get(rep, {}).get("t0", 0)))
            if not series:
                continue
            out[(ff, lipid)] = {
                "series": [s[t0:] for s, t0 in zip(series, t0s)],
                "ci": autocorr.hierarchical_ci(series, t0s),
            }
    return out


def pairwise(cis):
    """``[((i, j), compare-dict), ...]`` for one group of force fields.

    ``cis`` is the per-force-field list of hierarchical-CI dicts (None where a
    system is missing).  Adjacent pairs come first so their brackets stack
    innermost-first.
    """
    out = []
    for i, j in ((0, 1), (1, 2), (0, 2)):
        if i >= len(cis) or j >= len(cis) or cis[i] is None or cis[j] is None:
            continue
        out.append(((i, j), autocorr.compare(cis[i], cis[j])))
    return out


def _bracket(ax, x1, x2, y, height, label, fontsize):
    """A significance bracket spanning ``x1..x2`` with ``label`` above it."""
    ax.plot([x1, x1, x2, x2], [y, y + height, y + height, y],
            lw=0.7, color="black", clip_on=False, solid_joinstyle="miter")
    ax.text(0.5 * (x1 + x2), y + height, label, ha="center", va="bottom",
            fontsize=fontsize, clip_on=False)


def sig_brackets(ax, cis, tops, show_ns=True, star_pt=6.0, verbose=False):
    """Stack pairwise significance brackets over each group of a grouped plot.

    ``cis`` is ``{lipid: [ci per force field]}`` and ``tops`` ``{lipid: y of the
    tallest artist in that group}``.  Every group's ladder starts at the same
    height (the tallest artist in the panel), so the brackets read as one row.
    """
    lo, hi = ax.get_ylim()
    span = hi - lo
    step = 0.075 * span
    base = max(tops.values()) + 0.06 * span
    highest = base
    for lipid in LIPIDS:
        if lipid not in cis:
            continue
        li = LIPIDS.index(lipid)
        level = 0
        for (i, j), cmp in pairwise(cis[lipid]):
            if cmp is None:
                continue
            mark = autocorr.stars(cmp["p"], ns="n.s.")
            if verbose:
                ffs = [f for f, _, _ in FORCEFIELDS]
                print(f"  {lipid:5s} {ffs[i]:8s} vs {ffs[j]:8s}  "
                      f"delta={cmp['delta']:+.3f}  t={cmp['t']:+.2f}  "
                      f"dof={cmp['dof']:.1f}  p={cmp['p']:.2e}  {mark}")
            if mark == "n.s." and not show_ns:
                level += 1
                continue
            y = base + level * step
            _bracket(ax, li + OFFSETS[i], li + OFFSETS[j], y, 0.018 * span,
                     mark, star_pt)
            highest = max(highest, y + step)
            level += 1
    ax.set_ylim(lo, max(hi, highest + 0.03 * span))


def draw_box(ax, prop, show_ns=True, star_pt=6.0, verbose=False):
    """Grouped box-and-whisker of ``prop`` with pairwise significance brackets.

    ``show_ns`` also brackets the non-significant pairs as ``n.s.`` (there are
    only a couple, so labelling them is clearer than a gap in the ladder)."""
    data = load(prop)
    label = PROPS[prop][2]
    cis, tops = {}, {}
    for li, lipid in enumerate(LIPIDS):
        cis[lipid] = []
        tops[lipid] = -np.inf
        for fi, (ff, ff_label, colour) in enumerate(FORCEFIELDS):
            entry = data.get((ff, lipid))
            cis[lipid].append(None if entry is None else entry["ci"])
            if entry is None:
                continue
            pooled = np.concatenate(entry["series"])
            bp = ax.boxplot([pooled], positions=[li + OFFSETS[fi]],
                            widths=BOX_W, whis=WHIS, showfliers=False,
                            patch_artist=True, manage_ticks=False)
            for patch in bp["boxes"]:
                patch.set(facecolor=colour, edgecolor="black", linewidth=0.7)
            for part in ("whiskers", "caps"):
                for line in bp[part]:
                    line.set(color="black", linewidth=0.7)
            for line in bp["medians"]:
                line.set(color="black", linewidth=0.9)
            tops[lipid] = max(tops[lipid], np.percentile(pooled, WHIS[1]))

    sig_brackets(ax, cis, tops, show_ns=show_ns, star_pt=star_pt,
                 verbose=verbose)
    ax.set_xticks(range(len(LIPIDS)), LIPIDS)
    ax.set_xlim(-0.5, len(LIPIDS) - 0.5)
    ax.set_ylabel(label)
    figstyle.categorical_x(ax)
    return data


def draw_diffusion(ax, verbose=False):
    """Grouped bars of the lateral diffusion coefficient (um^2/s).

    Error bars are the **SD across the three replicates**: D is one fitted
    scalar per run, so there is no within-run series to autocorrelation-correct
    (the same convention the summary table uses).
    """
    npz_path = DATA / "lipprops_lateral_diffusion.npz"
    if not npz_path.exists():
        raise SystemExit(f"{npz_path} missing")
    npz = np.load(npz_path)
    for fi, (ff, ff_label, colour) in enumerate(FORCEFIELDS):
        xs, ys, es = [], [], []
        for li, lipid in enumerate(LIPIDS):
            key = f"{ff}_{lipid}_D"
            if key not in npz:
                continue
            d = np.asarray(npz[key], dtype=float) * D_SCALE
            xs.append(li + OFFSETS[fi])
            ys.append(d.mean())
            es.append(d.std(ddof=1) if d.size > 1 else 0.0)
            if verbose:
                print(f"  {lipid:5s} {ff:8s} D = {d.mean():.3f} "
                      f"+- {es[-1]:.3f} um^2/s  (n={d.size})")
        if not xs:
            continue
        ax.bar(xs, ys, BOX_W, color=colour, edgecolor="black", linewidth=0.7,
               label=ff_label)
        ax.errorbar(xs, ys, yerr=es, fmt="none", ecolor="black",
                    elinewidth=0.7, capsize=1.5)
    ax.set_xticks(range(len(LIPIDS)), LIPIDS)
    ax.set_xlim(-0.5, len(LIPIDS) - 0.5)
    ax.set_ylabel(r"$D$ ($\mathrm{\mu m^2\,s^{-1}}$)")
    ax.set_ylim(bottom=0)
    figstyle.categorical_x(ax)


def legend_handles():
    """Shared force-field key, for a figure-level legend."""
    from matplotlib.patches import Patch
    return ([Patch(facecolor=c, edgecolor="black", linewidth=0.7)
             for _, _, c in FORCEFIELDS],
            [label for _, label, _ in FORCEFIELDS])


# --------------------------------------------------------------- second row
DT_NS = 0.2                      # 200 ps per frame
RT_310 = 2.577                   # kJ/mol at 310 K, for kT -> kJ/mol
SMOOTH_NS = 5.0                  # rolling mean applied to the d_HC trajectory


def draw_kappa(ax, show_ns=True, star_pt=6.0, verbose=False):
    """Grouped bars of the Lipidator/ReSiS tilt modulus kappa_theta.

    Per-replicate kappa + its block-averaged within-run SEM come from
    ``lipidator_blocks.json``; the error bar is the **hierarchical 95% CI**
    (the repo-wide convention for chi/kappa), and the unit is kJ/mol/nm^2
    (kappa[kT/nm^2] x RT at 310 K).
    """
    path = DATA / "lipidator_blocks.json"
    if not path.exists():
        raise SystemExit(f"{path} missing -- run run_lipidator_blocks.py first")
    with open(path) as fh:
        blocks = json.load(fh)
    cis = {lipid: [] for lipid in LIPIDS}
    tops = {lipid: -np.inf for lipid in LIPIDS}
    for fi, (ff, ff_label, colour) in enumerate(FORCEFIELDS):
        xs, ys, es = [], [], []
        for li, lipid in enumerate(LIPIDS):
            recs = [blocks.get(f"{ff}_{lipid}_{r}") for r in REPS]
            recs = [r for r in recs if r]
            if not recs:
                cis[lipid].append(None)
                continue
            ci = autocorr.hierarchical_ci_from_estimates(
                [r["kappa_full"] for r in recs],
                [r.get("kappa_block_sem", np.nan) for r in recs],
                [r.get("n_blocks") for r in recs])
            # The CI is tested in kT/nm^2 -- the unit only rescales both sides.
            cis[lipid].append(ci)
            xs.append(li + OFFSETS[fi])
            ys.append(ci["mean"] * RT_310)
            es.append(ci["ci_half"] * RT_310)
            tops[lipid] = max(tops[lipid], ys[-1] + es[-1])
            if verbose:
                print(f"  {lipid:5s} {ff:8s} kappa = {ys[-1]:.1f} +- {es[-1]:.1f} "
                      f"kJ/mol/nm^2  ({ci['regime']}, n={ci['n']})")
        if not xs:
            continue
        ax.bar(xs, ys, BOX_W, color=colour, edgecolor="black", linewidth=0.7,
               label=ff_label)
        ax.errorbar(xs, ys, yerr=es, fmt="none", ecolor="black",
                    elinewidth=0.7, capsize=1.5)
    ax.set_xticks(range(len(LIPIDS)), LIPIDS)
    ax.set_xlim(-0.5, len(LIPIDS) - 0.5)
    ax.set_ylabel(r"$\kappa_\theta$ (kJ mol$^{-1}$ nm$^{-2}$)")
    ax.set_ylim(bottom=0)
    figstyle.categorical_x(ax)
    sig_brackets(ax, cis, tops, show_ns=show_ns, star_pt=star_pt,
                 verbose=verbose)


def scd_profiles(lipid):
    """``{ff: (carbons, mean over reps, SD over reps)}`` of -S_CD.

    g_lomepro writes one file per replicate per tail (columns: carbon number,
    mean -S_CD, SD over lipids x frames).  The two tails are averaged *within*
    a replicate -- for these symmetric di-oleoyl / di-myristoyl lipids the
    sn-1/sn-2 profiles differ by <0.01 -- then the line is the mean over
    replicates and the band their SD.
    """
    out = {}
    for ff, _, _ in FORCEFIELDS:
        per_rep = []
        carbons = None
        for rep in REPS:
            tails = []
            for sn in (1, 2):
                path = DATA / f"{ff}_{lipid}_{rep}_scd_sn{sn}.dat"
                if not path.exists():
                    continue
                arr = np.loadtxt(path)
                carbons = arr[:, 0]
                tails.append(arr[:, 1])
            if tails:
                per_rep.append(np.mean(tails, axis=0))
        if per_rep:
            per_rep = np.array(per_rep)
            out[ff] = (carbons, per_rep.mean(axis=0),
                       per_rep.std(axis=0, ddof=1) if len(per_rep) > 1
                       else np.zeros(per_rep.shape[1]))
    return out


def draw_scd(ax, lipid="DMPG", verbose=False):
    """Per-carbon -S_CD profile (g_lomepro) for one lipid, force fields overlaid.

    Carbon number runs along **x** here (rather than down an inverted y as in
    the standalone ``plots/lipprop_scd.png``) so the panel reads like its row
    neighbours; the band is the SD across the three replicates.
    """
    profiles = scd_profiles(lipid)
    for ff, ff_label, colour in FORCEFIELDS:
        if ff not in profiles:
            continue
        carbons, mean, sd = profiles[ff]
        ax.plot(carbons, mean, color=colour, label=ff_label, marker="o",
                markersize=2.0, markeredgewidth=0)
        ax.fill_between(carbons, mean - sd, mean + sd, color=colour,
                        alpha=0.20, lw=0)
        if verbose:
            print(f"  {lipid} {ff:8s} <-S_CD> = {mean.mean():.4f}")
    ax.set_xlabel("acyl carbon")
    ax.set_ylabel(r"$-S_\mathrm{CD}$")
    # Carbon number is an integer index, so ticks sit on the integers only --
    # no 2.5, no minor ticks between.  The mirrored top ticks stay (unlike the
    # categorical lipid axes): the carbons are ordered, evenly spaced positions
    # along the chain, so a top scale still means something.
    if profiles:
        first = next(iter(profiles.values()))[0]
        ax.set_xticks(np.arange(int(first[0]), int(first[-1]) + 1, dtype=int))
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.set_title(lipid)
    return profiles


def _smooth(y, window):
    """Centred rolling mean, edges shortened rather than padded."""
    if window < 2:
        return y
    kernel = np.ones(window) / window
    out = np.convolve(y, kernel, mode="same")
    half = window // 2
    for i in list(range(half)) + list(range(len(y) - half, len(y))):
        lo, hi = max(0, i - half), min(len(y), i + half + 1)
        out[i] = y[lo:hi].mean()
    return out


def draw_dhc_traj(ax, lipid="DMPG", verbose=False):
    """Per-frame hydrophobic-core thickness d_HC(t) for one lipid.

    All three replicates of each force field are drawn (raw trace faint, a
    5 ns rolling mean on top), each a different shade of its force field's
    colour -- the same light->dark ramps ``plot_close_pullx.py`` uses for its
    10 closing replicates (``palette.CMAPS``), sampled ``[::4]`` out of those
    ten shades so three well-separated ones come out.  This is the point of the
    panel: it shows the
    slow acyl-chain ordering that pymbar discards as burn-in for the DMPGs, and
    the run-to-run spread behind their between-run error bars.  d_HC stands in
    for the chain order parameter here -- the two are strongly correlated, and
    d_HC is the one available per frame.
    """
    path = DATA / "core_thickness.npz"
    if not path.exists():
        raise SystemExit(f"{path} missing -- run core_thickness.py first")
    npz = np.load(path)
    window = int(round(SMOOTH_NS / DT_NS))
    shades = np.linspace(0.9, 0.3, 10)[::4]      # 3 of the 10 close-plot shades
    for ff, ff_label, colour in FORCEFIELDS:
        cmap = CMAPS[ff]
        for ri, rep in enumerate(REPS):
            key = f"{ff}_{lipid}_lipy_rep{rep}"
            if key not in npz:
                continue
            y = np.asarray(npz[key], dtype=float)
            t = np.arange(len(y)) * DT_NS
            shade = cmap(shades[ri])
            ax.plot(t, y, color=shade, lw=0.4, alpha=0.20)
            ax.plot(t, _smooth(y, window), color=shade, lw=0.9,
                    label=ff_label if rep == REPS[0] else None)
            if verbose:
                print(f"  {lipid} {ff:8s} rep{rep}  d_HC {y[0]:.2f} -> "
                      f"{y[-100:].mean():.2f} nm")
    ax.set_xlabel("time (ns)")
    ax.set_ylabel(r"$d_\mathrm{HC}$ (nm)")
    ax.set_xlim(0, None)
    ax.set_title(lipid)
