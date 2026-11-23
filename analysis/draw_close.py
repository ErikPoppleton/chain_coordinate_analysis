#!/usr/bin/env python3
"""Panel painter for the main-text **pore-closure** figure (Figure 3).

``draw_close(ax, ff, lipid)`` overlays the ten ``close`` replicate trajectories
of one (force field, lipid) system -- the chain coordinate xi against time after
the umbrella restraint is released -- as a single-hue light->dark gradient
(``palette.CMAPS``), the same replicate colouring ``plot_close_pullx.py`` uses.

The traces are **strided while reading** (each ``*_close{n}_pullx.xvg`` holds
2.5 M points for a 5 ns run, and a 52 mm panel at 600 dpi is ~1200 px wide, so
even the default stride leaves the curve ~20x oversampled); parsing every float
would cost minutes for the 90 files.  They are drawn ``rasterized=True``, which
matplotlib inlines as base64 inside the SVG -- the figure stays monolithic and
small instead of carrying millions of vector segments.
"""

from pathlib import Path

import numpy as np

from palette import CMAPS

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

N_REP = 10           # close replicates per system
STRIDE = 100         # keep every Nth sample (2.5 M points is far more than a
                     # 1200 px panel can show)
LINEWIDTH = 0.25     # thinner than the house default so individual traces read
PS_TO_NS = 1e-3


def load_pullx(path, stride=STRIDE):
    """x (ns), y (xi) from a pullx .xvg, striding *while reading*."""
    xs, ys = [], []
    with open(path) as fh:
        for i, line in enumerate(fh):
            if not line or line[0] in "#@&\n":
                continue
            if i % stride:
                continue
            cols = line.split()
            xs.append(float(cols[0]))
            ys.append(float(cols[1]))
    return np.array(xs) * PS_TO_NS, np.array(ys)


def last_time(path, tail_bytes=4096):
    """Time (ns) of the last complete sample in a pullx .xvg, without parsing
    the file: seek to the end and walk back through the final lines until one
    parses (a file still being written can end mid-line)."""
    with open(path, "rb") as fh:
        fh.seek(0, 2)
        fh.seek(max(0, fh.tell() - tail_bytes))
        lines = fh.read().decode(errors="replace").splitlines()
    for line in reversed(lines):
        cols = line.split()
        if len(cols) < 2 or line[0] in "#@&":
            continue
        try:
            float(cols[1])
            return float(cols[0]) * PS_TO_NS
        except ValueError:
            continue
    return None


def run_length(ff, lipid, verbose=True):
    """Run length (ns) of a system's close replicates: the longest replicate's
    last time, rounded up to a whole ns.  Prints a warning when the replicates
    end at different times -- a download or run that has not finished."""
    ends = []
    for i in range(N_REP):
        path = DATA / f"{ff}_{lipid}_close{i + 1}_pullx.xvg"
        if path.exists():
            t = last_time(path)
            if t is not None:
                ends.append(t)
    if not ends:
        return None
    longest = max(ends)
    short = [t for t in ends if t < longest - 1e-6]
    if verbose and (short or len(ends) < N_REP):
        print(f"warning: {ff}_{lipid}: {len(ends) - len(short)}/{N_REP} "
              f"replicates reach {longest:g} ns; the others end at "
              f"{', '.join(f'{t:g}' for t in sorted(short)) or 'nothing (missing)'} ns"
              " -- incomplete download or run?")
    return float(np.ceil(longest - 1e-9))


def draw_close(ax, ff, lipid, stride=STRIDE, lw=LINEWIDTH, verbose=False):
    """The ten close trajectories of one system, dark (rep 1) -> light (rep 10)."""
    cmap = CMAPS[ff]
    shades = np.linspace(0.9, 0.3, N_REP)
    drawn = 0
    for i in range(N_REP):
        path = DATA / f"{ff}_{lipid}_close{i + 1}_pullx.xvg"
        if not path.exists():
            continue
        x, y = load_pullx(path, stride)
        ax.plot(x, y, color=cmap(shades[i]), lw=lw, rasterized=True)
        drawn += 1
    if verbose:
        print(f"  {lipid:5s} {ff:8s} {drawn}/{N_REP} replicates")
    elif drawn < N_REP:
        print(f"note: {ff}_{lipid} has {drawn}/{N_REP} close replicates")
    ax.set_xlim(0, None)
    return drawn


# --- closure-time CDF (the alternative Figure 3) ----------------------------
CLOSE_XI = 0.6            # a pore counts as closed once xi drops below this
CLOSE_WINDOW_PS = 100.0   # ... in a trailing moving average over this window
CLOSE_STRIDE = 100        # read every 100th 2 fs sample -> 0.2 ps spacing,
                          # 500 samples per 100 ps window
CLOSE_CACHE = DATA / "close_times.json"


def closure_time(path, xi=CLOSE_XI, window_ps=CLOSE_WINDOW_PS,
                 stride=CLOSE_STRIDE):
    """First time (ns) at which the trailing ``window_ps`` moving average of xi
    falls below ``xi``, or None if it never does.

    The average is taken over the samples kept by ``stride`` (0.2 ps apart by
    default) rather than every 2 fs step; xi is smooth on that scale, so the
    difference is negligible, and reading every line of a 50 ns file would
    cost minutes.  The time reported is the *end* of the first window whose
    mean is below the threshold, so it can be no earlier than ``window_ps``.
    """
    t, y = load_pullx(path, stride)
    if len(t) < 2:
        return None
    dt_ps = (t[1] - t[0]) / PS_TO_NS
    n = max(1, int(round(window_ps / dt_ps)))
    if len(y) < n:
        return None
    c = np.cumsum(np.insert(y, 0, 0.0))
    avg = (c[n:] - c[:-n]) / n               # avg[k] = mean of y[k : k+n]
    below = np.nonzero(avg < xi)[0]
    return float(t[below[0] + n - 1]) if below.size else None


def closure_times(ff, lipid, refresh=False, verbose=True):
    """``(times, run_ns)`` for one system: the ten replicates' closure times in
    ns (None where the pore never closed) and the run length.

    Results are cached in ``data/close_times.json`` keyed by the system and
    checked against each pullx file's size and mtime plus the closure
    parameters, so the 90 large files are only re-read when one changes.
    """
    import json
    cache = json.loads(CLOSE_CACHE.read_text()) if CLOSE_CACHE.exists() else {}
    paths = [DATA / f"{ff}_{lipid}_close{i + 1}_pullx.xvg" for i in range(N_REP)]
    sig = {"xi": CLOSE_XI, "window_ps": CLOSE_WINDOW_PS, "stride": CLOSE_STRIDE,
           "files": [[p.name, p.stat().st_size, int(p.stat().st_mtime)]
                     for p in paths if p.exists()]}
    key = f"{ff}_{lipid}"
    entry = cache.get(key)
    if refresh or not entry or entry.get("sig") != sig:
        times = [closure_time(p) if p.exists() else None for p in paths]
        entry = {"sig": sig, "times": times}
        cache[key] = entry
        CLOSE_CACHE.write_text(json.dumps(cache, indent=1))
    times = entry["times"]
    run_ns = run_length(ff, lipid, verbose=verbose)
    if verbose:
        shut = sorted(t for t in times if t is not None)
        print(f"  {lipid:5s} {ff:8s} {len(shut)}/{N_REP} closed within "
              f"{run_ns:g} ns" + (f" (first {shut[0]:.2f}, last {shut[-1]:.2f} ns)"
                                  if shut else ""))
    return times, run_ns


def draw_closure_cdf(ax, ff, lipid, colour, t_max, lw=1.0, verbose=False):
    """Cumulative number of closed pores (of ten) against time, as a step line.

    Closure is the first time the 100 ps moving average of xi drops below 0.6
    (``closure_time``).  A system whose ten pores have all closed is extended
    flat at 10 out to ``t_max`` (the longest run in the figure): closed pores do
    not reopen, so its shorter run says nothing new after that.  A system with
    pores still open stops at the end of its own run, since what the open ones
    would do later is unknown.
    """
    times, run_ns = closure_times(ff, lipid, verbose=verbose)
    shut = sorted(t for t in times if t is not None)
    end = t_max if len(shut) == N_REP else run_ns
    x = [0.0] + shut + [end]
    y = [0] + list(range(1, len(shut) + 1)) + [len(shut)]
    ax.step(x, y, where="post", color=colour, lw=lw)
    if end < t_max:
        # Mark where the data stop, so a short run is not read as "no closures
        # ever".
        ax.plot([end], [len(shut)], marker="|", color=colour, ms=4, mew=lw)
    return times, run_ns
