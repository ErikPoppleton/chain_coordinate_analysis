#!/usr/bin/env python3
"""Panel painters for the HMMM **properties** figure.

Pure ``draw_*(ax)`` functions -- they create no figure and save nothing.

``draw_diffusion(ax, species)``  grouped bars of the lateral diffusion
                                 coefficient, for the lipid headgroups (with the
                                 full DOPC bilayer as a reference group) or for
                                 the DCLE solvent (HMMM only -- a full bilayer
                                 has no organic solvent to compare against).
``draw_box(ax, prop)``           box-and-whisker of d_HH or the area per lipid,
                                 again against DOPC.
``draw_rdf(ax, pair)``           the DCLE-DCLE and C6-DCLE radial distribution
                                 functions, one curve per (force field,
                                 ensemble).

Two conventions that run through the whole figure
-------------------------------------------------
**Everything is cut at its run's stability time.**  A property measured after
the membrane stopped being a membrane is not a membrane property, so every
per-frame series is sliced to ``stability.py``'s window before it is pooled, the
diffusion coefficients are refitted on that window by ``diffusion.py``, and the
RDFs are binned over it by ``rdf.py``.  The burn-in cut (pymbar ``t0``) is
applied inside that window -- ``HMMM/analysis/stats.py`` is run on the truncated
series for exactly this reason.

**A system with any replicate short of the full microsecond is hatched**
``//\\\\`` **in red.**  Truncation keeps those numbers honest but cannot make
them equivalent: they come from less trajectory, and from a system that was on
its way to falling apart.  The hatch says so on the face of the panel.  It is
drawn as a second, face-less patch over the artist, because matplotlib ties
hatch colour to edge colour and the bars keep a black outline.

Colour is the same scheme as the stability figure: the **force field** picks the
light-to-dark ramp (``palette.CMAPS``) and the **ensemble** the shade within it.
The DOPC reference group has no ensemble, so it takes the force field's solid
categorical colour (``palette.COLORS``).

Reads ``data/{lipprops_*,diffusion,rdf}.npz``, ``data/stability.json`` and
``data/equilibrium_timeseries_stats.json`` from both this project and the parent
one.  numpy + matplotlib only.
"""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PARENT = ROOT.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PARENT / "analysis"))

import autocorr
import figstyle
from palette import COLORS
import stability as stab
from draw_stability import colour, HATCH, HATCH_COLOUR
from hmmm_defs import (FORCEFIELDS, FF_LABELS, ENSEMBLES, ENS_LABELS, REPS,
                       system as system_key)

DATA = ROOT / "data"
PARENT_DATA = PARENT / "data"

REFERENCE_LIPID = "DOPC"
GROUPS = ENSEMBLES + [REFERENCE_LIPID]
GROUP_LABELS = [ENS_LABELS[e] for e in ENSEMBLES] + [REFERENCE_LIPID]
SEPARATOR_X = len(ENSEMBLES) - 0.5

PROPS = {
    "thickness": ("lipprops_memb_thickness", "thickness",
                  r"$d_\mathrm{HH}$ (nm)", 0.1),      # npz is in Angstrom
    "apl": ("lipprops_area_per_lipid", "apl", r"APL ($\mathrm{\AA}^2$)", 1.0),
}
SPECIES_LABEL = {"lipid": r"$D_\mathrm{lipid}$ ($\mathrm{\mu m^2\,s^{-1}}$)",
                 "dcle": r"$D_\mathrm{DCLE}$ ($\mathrm{\mu m^2\,s^{-1}}$)"}
RDF_LABEL = {"dcle_dcle": ("DCLE–DCLE", r"$g(r)$, DCLE–DCLE"),
             "c6_dcle": ("C6–DCLE", r"$g(r)$, C6–DCLE")}

D_SCALE = 1e8                    # cm^2/s -> um^2/s
WHIS = (5, 95)
BOX_W = 0.22
OFFSETS = np.array([-0.26, 0.0, 0.26])


def _stats(path, obs):
    if not path.exists():
        raise SystemExit("%s missing -- run the matching stats.py first" % path)
    with open(path) as fh:
        return json.load(fh)["per_replicate"].get(obs, {})


def group_colour(ff, group):
    """Ramp-by-force-field, shade-by-ensemble; the reference lipid has no shade."""
    return COLORS[ff] if group == REFERENCE_LIPID else colour(ff, group)


def flagged(records, group):
    """``{ff: True}`` for the systems whose hatch should be drawn."""
    if group == REFERENCE_LIPID:
        return {ff: False for ff in FORCEFIELDS}
    return {ff: stab.any_unstable(records, ff, group) for ff in FORCEFIELDS}


def _hatch_bar(ax, x, height, width, bottom=0.0):
    ax.bar([x], [height], width, bottom=bottom, facecolor="none",
           edgecolor=HATCH_COLOUR, linewidth=0.0, hatch=HATCH, zorder=3)


def _hatch_box(ax, x, lo, hi, width):
    from matplotlib.patches import Rectangle
    ax.add_patch(Rectangle((x - 0.5 * width, lo), width, hi - lo,
                           facecolor="none", edgecolor=HATCH_COLOUR,
                           linewidth=0.0, hatch=HATCH, zorder=3))


def load(prop):
    """``{(ff, group): {"series": [...], "ci": ...}}``, stability- and t0-cut."""
    stem, obs, _, scale = PROPS[prop]
    records = stab.load()
    hm_npz = np.load(DATA / ("%s.npz" % stem))
    hm_stats = _stats(DATA / "equilibrium_timeseries_stats.json", obs)
    ref_npz = np.load(PARENT_DATA / ("%s.npz" % stem))
    ref_stats = _stats(PARENT_DATA / "equilibrium_timeseries_stats.json", obs)

    out = {}
    for ff in FORCEFIELDS:
        for group in GROUPS:
            hmmm = group != REFERENCE_LIPID
            npz = hm_npz if hmmm else ref_npz
            per_obs = hm_stats if hmmm else ref_stats
            key = system_key(ff, group) if hmmm else "%s_%s" % (ff, group)
            fmt = key + ("_rep%d" if hmmm else "_eq%d")
            recs = {r["rep"]: r for r in per_obs.get(key, [])}
            series, t0s = [], []
            for rep in REPS:
                name = fmt % rep
                if name not in npz:
                    continue
                a = np.asarray(npz[name], dtype=float)
                if a.ndim == 2:                  # APL is (n_frames, n_lipids)
                    a = a.mean(axis=1)
                if hmmm:
                    a = a[:stab.stable_frames(records, ff, group, rep,
                                              n_available=len(a))]
                series.append(a * scale)
                t0s.append(int(recs.get(rep, {}).get("t0", 0)))
            if not series:
                continue
            out[(ff, group)] = {
                "series": [s[t0:] for s, t0 in zip(series, t0s)],
                "ci": autocorr.hierarchical_ci(series, t0s),
            }
    return out


def _categorical_axis(ax, groups, labels, separator=None):
    ax.set_xticks(range(len(groups)), labels)
    ax.set_xlim(-0.5, len(groups) - 0.5)
    if separator is not None:
        ax.axvline(separator, color="0.55", lw=0.7, ls=(0, (3, 2)), zorder=0)
    figstyle.categorical_x(ax)


def draw_box(ax, prop, verbose=False):
    """Grouped box-and-whisker of ``prop`` over the ensembles plus DOPC.

    The box is the pooled distribution of the per-frame observable over each
    replicate's production window (quartile box, 5th/95th-percentile whiskers,
    no fliers).  y limits come from the well-behaved groups so one run that
    blew up cannot flatten the rest, and a caret marks any whisker that leaves
    the panel.
    """
    data = load(prop)
    records = stab.load()
    pooled, boxes = {}, {}
    for gi, group in enumerate(GROUPS):
        flags = flagged(records, group)
        for fi, ff in enumerate(FORCEFIELDS):
            entry = data.get((ff, group))
            if entry is None:
                continue
            vals = np.concatenate(entry["series"])
            pooled[(ff, group)] = vals
            x = gi + OFFSETS[fi]
            bp = ax.boxplot([vals], positions=[x], widths=BOX_W, whis=WHIS,
                            showfliers=False, patch_artist=True,
                            manage_ticks=False)
            for patch in bp["boxes"]:
                patch.set(facecolor=group_colour(ff, group), edgecolor="black",
                          linewidth=0.7)
            for part in ("whiskers", "caps"):
                for line in bp[part]:
                    line.set(color="black", linewidth=0.7)
            for line in bp["medians"]:
                line.set(color="black", linewidth=0.9)
            boxes[(ff, group)] = (x, np.percentile(vals, 25),
                                  np.percentile(vals, 75), flags.get(ff, False))
            if verbose:
                print("  %-6s %-8s median %.3f  [%.3f, %.3f]%s"
                      % (group, ff, np.median(vals),
                         np.percentile(vals, WHIS[0]),
                         np.percentile(vals, WHIS[1]),
                         "  UNSTABLE" if flags.get(ff) else ""))

    # y limits from the groups whose whisker span is not an order of magnitude
    # above the typical one -- the IQR would not catch a run that is fine for
    # most of its frames and then blows up in the tail.
    spans = {k: float(np.diff(np.percentile(v, WHIS))[0])
             for k, v in pooled.items()}
    if spans:
        typical = np.median(list(spans.values()))
        stable = [k for k, s in spans.items() if s <= 10 * max(typical, 1e-12)]
        if stable:
            lo = min(np.percentile(pooled[k], WHIS[0]) for k in stable)
            hi = max(np.percentile(pooled[k], WHIS[1]) for k in stable)
            pad = 0.12 * (hi - lo) if hi > lo else 0.1 * max(abs(hi), 1.0)
            ax.set_ylim(lo - pad, hi + pad)
            for k, vals in pooled.items():
                x = boxes[k][0]
                if np.percentile(vals, WHIS[1]) > hi + pad:
                    ax.plot([x], [hi + pad], marker="^", color="black",
                            markersize=2.5, markeredgewidth=0, clip_on=False)
                if np.percentile(vals, WHIS[0]) < lo - pad:
                    ax.plot([x], [lo - pad], marker="v", color="black",
                            markersize=2.5, markeredgewidth=0, clip_on=False)
    for x, q1, q3, flag in boxes.values():
        if flag:
            _hatch_box(ax, x, q1, q3, BOX_W)

    ax.set_ylabel(PROPS[prop][2])
    _categorical_axis(ax, GROUPS, GROUP_LABELS, SEPARATOR_X)
    return data


def draw_diffusion(ax, species="lipid", reference=True, verbose=False):
    """Grouped bars of the lateral diffusion coefficient, on a log axis.

    Error bars are the SD across the three replicates -- D is one fitted scalar
    per run, so there is no within-run series to autocorrelation-correct (the
    parent project's convention).  The log axis is not decoration: the HMMM
    lipids diffuse 10-30x faster than a full DOPC bilayer and DCLE another
    10-20x faster again, which a linear axis would render as one tall bar and
    a row of stubs.
    """
    records = stab.load()
    hm = np.load(DATA / "diffusion.npz")
    ref = np.load(PARENT_DATA / "lipprops_lateral_diffusion.npz") \
        if reference else None
    groups = GROUPS if reference else list(ENSEMBLES)
    labels = GROUP_LABELS if reference else [ENS_LABELS[e] for e in ENSEMBLES]

    values = {}
    for ff in FORCEFIELDS:
        for group in groups:
            if group == REFERENCE_LIPID:
                key = "%s_%s_D" % (ff, group)
                if ref is not None and key in ref:
                    values[(ff, group)] = np.asarray(ref[key], float) * D_SCALE
            else:
                key = "%s_%s_D" % (system_key(ff, group), species)
                if key in hm:
                    values[(ff, group)] = np.asarray(hm[key], float) * D_SCALE
    if not values:
        return

    floor = 10 ** np.floor(np.log10(min(v.mean() for v in values.values())) - 0.3)
    ax.set_yscale("log")
    for gi, group in enumerate(groups):
        flags = flagged(records, group)
        for fi, ff in enumerate(FORCEFIELDS):
            d = values.get((ff, group))
            if d is None:
                continue
            mean = float(d.mean())
            sd = float(d.std(ddof=1)) if d.size > 1 else 0.0
            x = gi + OFFSETS[fi]
            ax.bar([x], [mean - floor], BOX_W, bottom=floor,
                   color=group_colour(ff, group), edgecolor="black",
                   linewidth=0.7)
            # A bar on a log axis starts at the floor, and an SD wider than the
            # mean would otherwise ask for a negative lower cap.
            ax.errorbar([x], [mean], yerr=[[min(sd, mean - floor * 1.001)], [sd]],
                        fmt="none", ecolor="black", elinewidth=0.7, capsize=1.5)
            if flags.get(ff):
                _hatch_bar(ax, x, mean - floor, BOX_W, bottom=floor)
            if verbose:
                print("  %-6s %-8s D = %8.2f +- %6.2f um^2/s%s"
                      % (group, ff, mean, sd,
                         "  UNSTABLE" if flags.get(ff) else ""))
    ax.set_ylim(floor, None)
    ax.set_ylabel(SPECIES_LABEL[species])
    _categorical_axis(ax, groups, labels,
                      SEPARATOR_X if reference else None)


def draw_rdf(ax, pair, verbose=False):
    """One RDF panel: every (force field, ensemble) overlaid.

    Line is the mean over replicates, band their SD.  A system with an unstable
    replicate gets its band hatched rather than its line restyled, so the curve
    itself stays comparable with the others.

    The plateau is not 1: MDAnalysis normalises g(r) against the box-average
    density and DCLE is confined to the bilayer core, so the bulk level sits
    near 1.7-2.0.  Peak positions are unaffected; heights are scaled by that
    factor.
    """
    records = stab.load()
    npz = np.load(DATA / "rdf.npz")
    for ff in FORCEFIELDS:
        for ens in ENSEMBLES:
            sysname = system_key(ff, ens)
            if "%s_bins" % sysname not in npz:
                continue
            keys = ["%s_%s_rep%d" % (sysname, pair, r) for r in REPS]
            ys = [npz[k] for k in keys if k in npz]
            if not ys:
                continue
            r = np.asarray(npz["%s_bins" % sysname]) / 10.0       # A -> nm
            ys = np.vstack(ys)
            mean = ys.mean(axis=0)
            sd = ys.std(axis=0, ddof=1) if len(ys) > 1 else np.zeros_like(mean)
            col = colour(ff, ens)
            ax.plot(r, mean, color=col, lw=1.0)
            unstable = stab.any_unstable(records, ff, ens)
            ax.fill_between(r, mean - sd, mean + sd, color=col, alpha=0.20,
                            lw=0.0,
                            hatch=HATCH if unstable else None,
                            edgecolor=HATCH_COLOUR if unstable else None)
            if verbose:
                lo = r > 0.2
                i = int(np.argmax(mean * lo))
                print("  %-6s %-8s first peak %.3f nm, g = %.2f%s"
                      % (ens, ff, r[i], mean[i],
                         "  UNSTABLE" if unstable else ""))
    ax.set_xlabel("$r$ (nm)")
    ax.set_ylabel(RDF_LABEL[pair][1])
    ax.set_xlim(0, 2.0)
    ax.set_ylim(bottom=0)


def legend_handles():
    """The shared (force field, ensemble) key, plus the unstable marker."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles, labels = [], []
    for ff in FORCEFIELDS:
        for ens in ENSEMBLES:
            handles.append(Line2D([0], [0], color=colour(ff, ens), lw=1.4))
            labels.append("%s %s" % (FF_LABELS[ff], ENS_LABELS[ens]))
    handles.append(Patch(facecolor="white", edgecolor=HATCH_COLOUR,
                         linewidth=0.0, hatch=HATCH))
    labels.append(r"unstable (< 1 $\mathrm{\mu}$s)")
    return handles, labels
