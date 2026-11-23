#!/usr/bin/env python3
"""Panel painters for the PC lipids with vs without ions.

Pure ``draw_*(ax)`` functions -- they create no figure and save nothing --
for ``figures/figSI_noions.py``.  The zwitterionic lipids (DOPC, DMPC) were
also run with **no ions at all** (label ``{lipid}_noions``); this compares
them with the ordinary runs (Na+/Cl- salt, plus the small Mg2+/Ca2+ count).

``draw_diffusion(ax)``     lateral diffusion D, bars +- SD across replicates
``draw_box(ax, prop)``     box-and-whisker of d_HH or the area per lipid
``draw_pmf_conditions(ax, ff)``  one force field's chain-coordinate PMFs,
                           with and without ions, for both lipids

The x axis of the first three is two lipid blocks, DOPC then DMPC, each
ordered by force field and then condition: CHARMM ions, CHARMM no ions, AMBER
ions, AMBER no ions, SLipids ions, SLipids no ions.  Each force field's pair
sits close together, and the no-ions box/bar takes a lighter tint of its force
field's colour (the same convention, and tint, as the DMPG ion-condition figure
in ``draw_dmpg_ions``).

Data layout follows the ``DMPG_JC`` precedent: the cluster-side lipyphilic
output for the no-ions runs is a *separate* set of ``lipprops_*.npz`` in
``data/noions/`` (keys ``{ff}_{lipid}_noions_eq{rep}``), with its own pymbar
``t0`` in ``data/noions/equilibrium_timeseries_stats.json`` (from
``stats.py --data-dir data/noions --lipids DOPC_noions DMPC_noions``).  The
with-ions data are the main ``data/`` files.  Each replicate is sliced to its
own production span ``[t0:]``, exactly as Figure 1 does.

Each force field's ions / no-ions pair carries a significance bracket in the
d_HH and APL panels: the hierarchical Welch test (``autocorr.compare``) on the
two systems' hierarchical CIs, Holm-adjusted across the three force fields per
lipid and property (the three brackets of one lipid block), stars from the
adjusted p (``n.s.`` drawn too).  D gets no brackets: it is one fitted scalar per
replicate with no within-run error series, so, as everywhere else in the
project, it carries only the replicate SD.  The full test table is printed.

Any panel whose inputs are absent is left empty and reported.
"""

import json
from pathlib import Path

import numpy as np

import autocorr
import figstyle
import draw_equilibrium as de
from draw_pore_energy import load_xvg
from lipiddefs import REPS
from palette import FORCEFIELDS, CMAPS

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
NOIONS_DATA = DATA / "noions"

PC_LIPIDS = ["DOPC", "DMPC"]
# (key, tick label, lipprops directory, label suffix)
CONDITIONS = [
    ("ions", "ions", DATA, ""),
    ("noions", "no ions", NOIONS_DATA, "_noions"),
]
COND_KEYS = [c[0] for c in CONDITIONS]

NOIONS_TINT = 0.42      # CMAPS position for the lighter (no-ions) shade
FF_SPACING = 0.30       # x distance between force-field pairs in a lipid block
PAIR_SPACING = 0.125    # x distance between the ions / no-ions boxes of a pair
BOX_W = 0.11
WHIS = de.WHIS
D_SCALE = de.D_SCALE
STAR_PT = 6.0


def xpos(li, fi, ci):
    """x of lipid block ``li``, force field ``fi``, condition ``ci``."""
    return li + (fi - 1) * FF_SPACING + (ci - 0.5) * PAIR_SPACING


def shade(ff, cond, colour):
    """Force-field colour: flat with ions, a lighter tint without."""
    return colour if cond == "ions" else CMAPS[ff](NOIONS_TINT)


def _cond(cond):
    return next(c for c in CONDITIONS if c[0] == cond)


def _stats(data_dir):
    path = data_dir / "equilibrium_timeseries_stats.json"
    if not path.exists():
        return None
    with open(path) as fh:
        return json.load(fh)


def load(prop):
    """``{(ff, lipid, cond): {"series": [...], "ci": ...}}`` for d_HH or APL.

    A condition with no stats JSON is skipped rather than analysed from frame
    0, which would put an equilibration transient into one column only.
    """
    stem, obs, _, scale = de.PROPS[prop]
    out = {}
    for cond, _, data_dir, suffix in CONDITIONS:
        npz_path = data_dir / f"{stem}.npz"
        stats = _stats(data_dir)
        if not npz_path.exists() or stats is None:
            print(f"  no {stem} / stats for the {cond} condition in {data_dir}")
            continue
        npz = np.load(npz_path)
        per_rep = stats["per_replicate"].get(obs, {})
        for ff, _, _ in FORCEFIELDS:
            for lipid in PC_LIPIDS:
                key = f"{ff}_{lipid}{suffix}"
                recs = {r["rep"]: r for r in per_rep.get(key, [])}
                series, t0s = [], []
                for rep in REPS:
                    name = f"{key}_eq{rep}"
                    if name not in npz:
                        continue
                    a = np.asarray(npz[name], dtype=float)
                    if a.ndim == 2:             # APL is (n_frames, n_lipids)
                        a = a.mean(axis=1)
                    series.append(a * scale)
                    t0s.append(int(recs.get(rep, {}).get("t0", 0)))
                if not series:
                    continue
                out[(ff, lipid, cond)] = {
                    "series": [s[t0:] for s, t0 in zip(series, t0s)],
                    "ci": autocorr.hierarchical_ci(series, t0s),
                }
    return out


def _categorical(ax):
    """Two lipid blocks, a dashed separator between them."""
    ax.set_xticks(range(len(PC_LIPIDS)), PC_LIPIDS)
    ax.set_xlim(-0.5, len(PC_LIPIDS) - 0.5)
    figstyle.categorical_x(ax)
    for li in range(1, len(PC_LIPIDS)):
        ax.axvline(li - 0.5, color="0.55", lw=0.7, ls=(0, (3, 2)), zorder=0)


def ions_vs_noions(data):
    """``{(lipid, ff): compare-dict + p_holm}``: the with- vs without-ions test
    per force field, Holm-adjusted across the three force fields per lipid."""
    out = {}
    for lipid in PC_LIPIDS:
        rows = []
        for ff, _, _ in FORCEFIELDS:
            a = data.get((ff, lipid, "ions"))
            b = data.get((ff, lipid, "noions"))
            if a and b:
                rows.append((ff, autocorr.compare(a["ci"], b["ci"])))
        adj = autocorr.holm([cmp["p"] for _, cmp in rows])
        for (ff, cmp), pa in zip(rows, adj):
            cmp["p_holm"] = pa
            out[(lipid, ff)] = cmp
    return out


def draw_box(ax, prop, show_ns=True, verbose=False):
    """Box-and-whisker of a per-frame property, ions / no-ions pairs per force
    field, with a significance bracket over each pair.

    Box = the per-frame distribution over the production span, pooled across
    replicates (quartile box, 5th/95th-percentile whiskers, no fliers), as in
    Figure 1.
    """
    data = load(prop)
    tops = {}
    for li, lipid in enumerate(PC_LIPIDS):
        for fi, (ff, _, colour) in enumerate(FORCEFIELDS):
            for ci, cond in enumerate(COND_KEYS):
                entry = data.get((ff, lipid, cond))
                if entry is None:
                    continue
                pooled = np.concatenate(entry["series"])
                bp = ax.boxplot([pooled], positions=[xpos(li, fi, ci)],
                                widths=BOX_W, whis=WHIS, showfliers=False,
                                patch_artist=True, manage_ticks=False)
                for patch in bp["boxes"]:
                    patch.set(facecolor=shade(ff, cond, colour),
                              edgecolor="black", linewidth=0.7)
                for part in ("whiskers", "caps"):
                    for line in bp[part]:
                        line.set(color="black", linewidth=0.7)
                for line in bp["medians"]:
                    line.set(color="black", linewidth=0.9)
                tops[(lipid, ff)] = max(tops.get((lipid, ff), -np.inf),
                                        np.percentile(pooled, WHIS[1]))
                if verbose:
                    ci_ = entry["ci"]
                    print(f"  {prop:9s} {lipid} {cond:6s} {ff:8s} "
                          f"mean {ci_['mean']:.3f} +- {ci_['ci_half']:.3f} "
                          f"({ci_['regime']})")
    tests = ions_vs_noions(data)
    ax.set_ylabel(de.PROPS[prop][2])
    _categorical(ax)
    # One bracket per pair, just above that pair's taller whisker (the pairs do
    # not overlap, so there is no ladder to stack).
    lo, hi = ax.get_ylim()
    span = hi - lo
    highest = hi
    for li, lipid in enumerate(PC_LIPIDS):
        for fi, (ff, _, _) in enumerate(FORCEFIELDS):
            cmp = tests.get((lipid, ff))
            if cmp is None:
                continue
            mark = autocorr.stars(cmp["p_holm"], ns="n.s.")
            if verbose:
                print(f"  {prop:9s} {lipid} {ff:8s} ions vs no ions  "
                      f"delta={cmp['delta']:+.3f}  dof={cmp['dof']:.1f}  "
                      f"p={cmp['p']:.2e}  p_holm={cmp['p_holm']:.2e}  {mark}")
            if mark == "n.s." and not show_ns:
                continue
            y = tops[(lipid, ff)] + 0.04 * span
            de._bracket(ax, xpos(li, fi, 0), xpos(li, fi, 1), y, 0.018 * span,
                        mark, STAR_PT)
            highest = max(highest, y + 0.10 * span)
    ax.set_ylim(lo, highest)
    return data


def draw_diffusion(ax, verbose=False):
    """Bars of D, ions / no-ions pairs per force field (+- SD across replicates).

    D is one fitted scalar per run, so -- as everywhere in the project -- the
    error is the replicate SD and there is no hierarchical test.
    """
    for li, lipid in enumerate(PC_LIPIDS):
        for ci, cond in enumerate(COND_KEYS):
            _, _, data_dir, suffix = _cond(cond)
            npz_path = data_dir / "lipprops_lateral_diffusion.npz"
            if not npz_path.exists():
                print(f"  no lateral diffusion for {cond} ({npz_path})")
                continue
            npz = np.load(npz_path)
            for fi, (ff, _, colour) in enumerate(FORCEFIELDS):
                key = f"{ff}_{lipid}{suffix}_D"
                if key not in npz:
                    continue
                d = np.asarray(npz[key], dtype=float) * D_SCALE
                x = xpos(li, fi, ci)
                e = d.std(ddof=1) if d.size > 1 else 0.0
                ax.bar([x], [d.mean()], BOX_W, color=shade(ff, cond, colour),
                       edgecolor="black", linewidth=0.7)
                ax.errorbar([x], [d.mean()], yerr=[e], fmt="none",
                            ecolor="black", elinewidth=0.7, capsize=1.2)
                if verbose:
                    print(f"  D {lipid} {cond:6s} {ff:8s} {d.mean():.2f} +- "
                          f"{e:.2f} um^2/s")
    ax.set_ylabel(r"$D$ ($\mathrm{\mu m^2\,s^{-1}}$)")
    ax.set_ylim(bottom=0)
    _categorical(ax)


# Within one force field's PMF panel the colour is fixed, so the lipid is the
# shade and the condition the line style.
LIPID_SHADE = {"DOPC": 0.90, "DMPC": 0.50}
COND_STYLE = {"ions": "-", "noions": (0, (3, 1.5))}


def _profile(lipid, cond, ff):
    """Path of one condition's full PMF, or None.

    With ions: ``data/{ff}_{lipid}_profile.xvg``.  Without: the plain name in
    ``data/noions/`` (the directory is the condition marker, as ``data_Na/``
    is for DMPG_JC), or the labelled ``{ff}_{lipid}_noions_profile.xvg``.
    """
    _, _, data_dir, suffix = _cond(cond)
    for name in dict.fromkeys([f"{ff}_{lipid}_profile.xvg",
                               f"{ff}_{lipid}{suffix}_profile.xvg"]):
        p = data_dir / name
        if p.exists():
            return p
    return None


def draw_pmf_conditions(ax, ff, ylabel=True, verbose=False):
    """One force field's min-zeroed PMFs, with and without ions, both lipids.

    Every profile that exists is drawn; a missing one is reported.  **The
    current profiles all predate the gmx wham fix** (``-auto`` dropped every
    sample at the saturated xi = 1; see ``setup_files/submit_wham.sh``), so the
    pore end of every curve is provisional until the re-run.  Returns True if
    anything was drawn.
    """
    label = dict((k, l) for k, l, _ in FORCEFIELDS)[ff]
    drawn = False
    for lipid in PC_LIPIDS:
        for cond, cond_label, _, _ in CONDITIONS:
            path = _profile(lipid, cond, ff)
            if path is None:
                if verbose:
                    print(f"  PMF {ff} {lipid} {cond}: no profile yet")
                continue
            x, y = load_xvg(path)
            ax.plot(x, y - y.min(), color=CMAPS[ff](LIPID_SHADE[lipid]),
                    ls=COND_STYLE[cond], label=f"{lipid} {cond_label}")
            drawn = True
            if verbose:
                print(f"  PMF {ff} {lipid} {cond:6s} max {np.max(y - y.min()):.1f} "
                      f"kJ/mol  ({path.relative_to(ROOT)})")
    ax.set_title(label)
    ax.set_xlim(0.0, 1.05)
    ax.set_xticks(np.arange(0.0, 1.01, 0.2))
    ax.set_xlabel(r"$\xi$")
    if ylabel:
        ax.set_ylabel(r"$\Delta G$ (kJ mol$^{-1}$)")
    if drawn:
        ax.axhline(0.0, color="0.6", lw=0.6, ls="--", zorder=0)
        ax.legend(frameon=False, loc="upper left", handlelength=2.0)
    return drawn


def legend_handles():
    """Force field x condition key for the box/bar row."""
    from matplotlib.patches import Patch
    handles, labels = [], []
    for ff, ff_label, colour in FORCEFIELDS:
        for cond, cond_label, _, _ in CONDITIONS:
            handles.append(Patch(facecolor=shade(ff, cond, colour),
                                 edgecolor="black", linewidth=0.7))
            labels.append(f"{ff_label} {cond_label}")
    return handles, labels
