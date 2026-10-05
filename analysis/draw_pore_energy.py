#!/usr/bin/env python3
"""Panel painters for the main-text **pore-energetics** figure (Figure 2).

Pure ``draw_*(ax)`` functions -- no figure creation, no saving -- so the same
panels serve an exploratory plot and ``figures/fig02_pore_energy.py``.

``draw_pmf(ax, lipid)``            the chain-coordinate PMF for one lipid, the
                                   three force fields overlaid and min-zeroed,
                                   with the five 20 ns partial profiles faint
                                   behind each full curve.
``draw_vs_barrier(ax, prop)``      an equilibrium property against that
                                   system's pore-nucleation barrier, coloured
                                   by force field and shaped by lipid.
``group_fits(prop)``               least-squares DeltaG(x) fits for AMBER and
                                   for CHARMM+SLipids separately, printed (not
                                   drawn) -- the two lines look parallel-but-
                                   offset in the d_HC panel, and this quantifies
                                   how much of that is slope vs y-offset.

The barrier is the min->max range of the full profile, and the property means
and error bars come straight from ``make_equilibrium_table.collect()`` -- so the
figure, the summary table and the standalone ``plot_*_vs_barrier.py`` scripts
cannot drift apart.  Requires numpy + matplotlib and the same ``data/`` inputs
as the summary table, plus the nine ``{ff}_{lipid}_profile.xvg``.
"""

from pathlib import Path

import numpy as np
from matplotlib.lines import Line2D

from lipiddefs import LIPIDS
from palette import FORCEFIELDS
import make_equilibrium_table as met

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

N_PARTS = 5                      # five consecutive 20 ns partial profiles
LIPID_MARKERS = {"DOPC": "o", "DMPC": "s", "DMPG": "^"}

# Properties this figure can scatter against the barrier, as keys of
# ``make_equilibrium_table.collect()``.
BARRIER_PROPS = {
    "dhc": r"$d_\mathrm{HC}$ (nm)",
    "kappa": r"$\kappa_\theta$ (kJ mol$^{-1}$ nm$^{-2}$)",
    "apl": r"APL ($\mathrm{\AA}^2$)",
    "dhh": r"$d_\mathrm{HH}$ (nm)",
    "D": r"$D$ ($\mathrm{\mu m^2\,s^{-1}}$)",
    "scd": r"$\langle -S_\mathrm{CD}\rangle$",
}


def load_xvg(path):
    """x, y columns of an xmgrace/GROMACS .xvg, skipping '#'/'@'/'&' lines."""
    x, y = [], []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line[0] in "#@&":
                continue
            cols = line.split()
            x.append(float(cols[0]))
            y.append(float(cols[1]))
    return np.array(x), np.array(y)


def barrier(ff, lipid, data=None, verbose=False):
    """Pore-nucleation barrier: the climb from the PMF basin to the pore-side top.

    **Not** ``max - min`` of the whole profile, which is what this used to be.
    The chain coordinate penalises the xi -> 0 limit (every slice must be
    lipid-filled there), so each profile *rises* toward the left edge, and the
    metastable intact-membrane state is the interior basin (argmin, at
    xi = 0.07-0.44 here, moving right as the membrane softens) -- not xi = 0.
    The barrier is therefore the highest point **on the pore side of that
    basin**, measured from it.

    For eight of the nine systems the two definitions coincide, because the
    pore peak is also the global max; they diverge exactly when the left edge
    out-tops the pore peak, which **SLipids DMPG does** (edge 59.1 vs peak
    27.7 kJ/mol). `max - min` then silently reports the edge height as a
    barrier -- the disagreement that showed up between Figure 2's PMF panels
    and its barrier scatters. That case is warned about rather than trusted.
    """
    path = (data or DATA) / f"{ff}_{lipid}_profile.xvg"
    if not path.exists():
        return None
    _, y = load_xvg(path)
    y = y - y.min()
    basin = int(np.argmin(y))
    pore_peak = float(y[basin:].max())
    if y[0] > pore_peak:
        print(f"  ! {ff}_{lipid}: the xi->0 edge ({y[0]:.1f}) out-tops the pore "
              f"peak ({pore_peak:.1f} kJ/mol); reporting the pore barrier, not "
              f"max-min")
    if verbose:
        print(f"  {lipid:5s} {ff:8s} basin at index {basin}, pore barrier = "
              f"{pore_peak:.1f} kJ/mol (max-min = {y.max():.1f})")
    return pore_peak


def partial_barriers(ff, lipid, data=None):
    """The same pore barrier from each of the five 20 ns partial profiles.

    Same basin-to-pore-side-peak definition as :func:`barrier`, applied to
    ``{ff}_{lipid}_part{1..5}_profile.xvg`` -- the faint curves behind each PMF
    in Figure 2.  Their spread is a convergence check on the full-profile
    barrier (the equilibrium table quotes it as the +/- SD); the first window
    runs systematically high (burn-in).  Returns an array, missing parts skipped.
    """
    out = []
    for part in range(1, N_PARTS + 1):
        path = (data or DATA) / f"{ff}_{lipid}_part{part}_profile.xvg"
        if not path.exists():
            continue
        _, y = load_xvg(path)
        y = y - y.min()
        out.append(float(y[int(np.argmin(y)):].max()))
    return np.asarray(out)


def draw_pmf(ax, lipid, partials=True, data=None, title=None, verbose=False):
    """Min-zeroed PMF profiles for one lipid, force fields overlaid.

    Each full curve is shifted so its own minimum is zero; the five 20 ns
    partial profiles are drawn faintly behind it as a convergence check.
    Force fields with no profile are simply absent.

    ``data`` reads the profiles from another directory (``data/data_Na`` holds
    the same systems under the previous ion condition), and ``title`` overrides
    the lipid-name panel title, which cannot tell two conditions of one lipid
    apart.
    """
    data = data or DATA
    plotted = False
    for ff, ff_label, colour in FORCEFIELDS:
        path = data / f"{ff}_{lipid}_profile.xvg"
        if not path.exists():
            continue
        plotted = True
        if partials:
            for part in range(1, N_PARTS + 1):
                ppath = data / f"{ff}_{lipid}_part{part}_profile.xvg"
                if not ppath.exists():
                    continue
                px, py = load_xvg(ppath)
                ax.plot(px, py - py.min(), color=colour, lw=0.4, alpha=0.30)
        x, y = load_xvg(path)
        ax.plot(x, y - y.min(), color=colour, label=ff_label)
        if verbose:
            b = barrier(ff, lipid, data=data)
            print(f"  {lipid:5s} {ff:8s} pore barrier = {b:.1f} kJ/mol "
                  f"(profile spans 0..{y.max() - y.min():.1f})")
    ax.axhline(0.0, color="0.6", lw=0.5, ls="--")
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel(r"$\xi$")
    ax.set_title(lipid if title is None else title)
    if not plotted:
        print(f"note: {lipid} has no profile data (blank panel)")
    return plotted


def draw_vs_barrier(ax, prop, rows=None, verbose=False):
    """One equilibrium property (x) against the pore-nucleation barrier (y).

    Colour is the force field, marker the lipid.  The x error bar is whatever
    ``collect()`` supplies for that column -- the hierarchical 95 % CI for
    APL/d_HC/d_HH/kappa, the replicate SD for D and <-S_CD>.
    """
    if rows is None:
        rows, _ = met.collect()
    for ff, ff_label, colour in FORCEFIELDS:
        for lipid in LIPIDS:
            b = barrier(ff, lipid)
            cell = rows.get((ff, lipid), {}).get(prop)
            if b is None or cell is None:
                continue
            mean, err = cell
            ax.errorbar(mean, b, xerr=err, marker=LIPID_MARKERS[lipid],
                        color=colour, mec="black", mew=0.4, ms=4.0,
                        elinewidth=0.7, capsize=1.5, linestyle="none")
            if verbose:
                print(f"  {lipid:5s} {ff:8s} {prop} = {mean:.3f}+-{err:.3f}  "
                      f"barrier = {b:.1f} kJ/mol")
    ax.set_xlabel(BARRIER_PROPS.get(prop, prop))
    ax.set_ylabel(r"$\Delta G$ (kJ mol$^{-1}$)")


# Grouping for ``group_fits``: AMBER's d_HC-vs-barrier points sit visibly off
# the CHARMM/SLipids trend, so the two are fitted separately.
FIT_GROUPS = [("CHARMM+SLipids", ["charmm", "slipids"]), ("AMBER", ["amber"])]


def _points(prop, ffs, rows):
    """([x], [barrier]) over the (ff, lipid) systems of ``ffs`` with both values."""
    xs, ys = [], []
    for ff in ffs:
        for lipid in LIPIDS:
            b = barrier(ff, lipid)
            cell = rows.get((ff, lipid), {}).get(prop)
            if b is None or cell is None:
                continue
            xs.append(cell[0])
            ys.append(b)
    return np.array(xs), np.array(ys)


def _ols(x, y):
    """Least-squares y = a*x + b with standard errors and R^2 (None if n < 2).

    SEs need at least one residual degree of freedom, so n >= 3; with n = 3
    (AMBER's three lipids) there is exactly one, and the SEs are correspondingly
    soft -- the slope *comparison* is the point, not its precision.
    """
    n = x.size
    if n < 2:
        return None
    a, b = np.polyfit(x, y, 1)
    resid = y - (a * x + b)
    sxx = float(((x - x.mean()) ** 2).sum())
    dof = n - 2
    if dof > 0 and sxx > 0:
        s2 = float((resid ** 2).sum()) / dof
        se_a = float(np.sqrt(s2 / sxx))
        se_b = float(np.sqrt(s2 * (1.0 / n + x.mean() ** 2 / sxx)))
    else:
        se_a = se_b = float("nan")
    sst = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float((resid ** 2).sum()) / sst if sst > 0 else float("nan")
    return {"n": n, "slope": float(a), "intercept": float(b), "se_slope": se_a,
            "se_intercept": se_b, "r2": r2, "x": x, "y": y}


def group_fits(prop, rows=None, groups=FIT_GROUPS, verbose=True):
    """Fit ``barrier = a * prop + b`` separately per force-field group and print
    the equations.  Returns {group_label: fit dict}; draws nothing.

    Reports each group's slope and intercept (with SEs and R^2), then the slope
    difference and -- the quantity of interest for a parallel-looking offset --
    the **vertical gap between the two lines at the mean x of all points**,
    which is what the eye reads as the offset (the raw intercept difference is
    an extrapolation to x = 0, far outside the data).
    """
    if rows is None:
        rows, _ = met.collect()
    fits = {}
    for label, ffs in groups:
        f = _ols(*_points(prop, ffs, rows))
        if f is not None:
            fits[label] = f
    if not verbose:
        return fits

    unit = BARRIER_PROPS.get(prop, prop)
    print("  separate linear fits, barrier vs %s (%s):" % (prop, unit))
    for label, f in fits.items():
        print("    %-15s n=%d  dG = %+.1f(+-%.1f) * x %+.1f(+-%.1f)   "
              "R2 = %.3f   x = [%.3f, %.3f]"
              % (label, f["n"], f["slope"], f["se_slope"], f["intercept"],
                 f["se_intercept"], f["r2"], f["x"].min(), f["x"].max()))
    if len(fits) == 2:
        (la, fa), (lb, fb) = fits.items()
        x_all = np.concatenate([fa["x"], fb["x"]])
        xm = float(x_all.mean())
        dslope = fb["slope"] - fa["slope"]
        se_dslope = float(np.sqrt(fa["se_slope"] ** 2 + fb["se_slope"] ** 2))
        gap = ((fb["slope"] * xm + fb["intercept"])
               - (fa["slope"] * xm + fa["intercept"]))
        print("    slope difference (%s - %s) = %+.1f +- %.1f kJ/mol per unit x "
              "(%+.0f %% of %s)"
              % (lb, la, dslope, se_dslope, 100.0 * dslope / fa["slope"], la))
        print("    vertical offset at x = %.3f (mean of all 9): %+.1f kJ/mol   "
              "(intercept difference %+.1f kJ/mol, an extrapolation to x = 0)"
              % (xm, gap, fb["intercept"] - fa["intercept"]))
    return fits


def lipid_handles(ms=4.0):
    """Marker key for the lipids (shape), colour-neutral."""
    return [Line2D([0], [0], marker=LIPID_MARKERS[lipid], color="0.35",
                   mec="black", mew=0.4, ls="none", ms=ms, label=lipid)
            for lipid in LIPIDS]


def ff_handles():
    """Line key for the force fields (colour)."""
    return [Line2D([0], [0], color=c, label=label)
            for _, label, c in FORCEFIELDS]
