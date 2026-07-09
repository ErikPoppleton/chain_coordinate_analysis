#!/usr/bin/env python3
"""For every '*profile.xvg' file in the data/ folder, plot the amount of
sampling that went into the PMF (in nanoseconds) against the max-min spread of
the profile (its total energy range).

The sampling window of each profile is read straight from the `gmx wham`
command line stored in the file header (`-b`/`-e`, in ps):

    * <ff>_part{1..5}_profile.xvg   -b/-e give a 20 ns window
    * <ff>_20-100_profile.xvg       -b 20000 (no -e)  -> 80 ns
    * <ff>_profile.xvg              no -b/-e           -> full 100 ns

Colours follow the other plotting scripts: CHARMM black, AMBER red,
SLipids green.
"""

import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# Project layout: this script lives in analysis/, data in data/, output in plots/.
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
PLOTS = ROOT / "plots"

# (force-field key as it appears in the filename prefix, label, colour).
FORCEFIELDS = [
    ("charmm", "CHARMM", "black"),
    ("amber", "AMBER", "red"),
    ("slipids", "SLipids", "green"),
]


def load_y(path):
    """Return the y (energy) column of an xmgrace/GROMACS .xvg file."""
    y = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line[0] in "#@&":
                continue
            y.append(float(line.split()[1]))
    return np.array(y)


def sampling_ns(path, default_end_ps):
    """Length of the trajectory window used for this profile, in ns.

    Parsed from the `-b`/`-e` (ps) flags on the `gmx wham` command line in the
    header.  Missing -b means 0; missing -e means the full trajectory."""
    with open(path) as fh:
        header = fh.read(4000)
    b = re.search(r"-b\s+(\d+)", header)
    e = re.search(r"-e\s+(\d+)", header)
    begin = int(b.group(1)) if b else 0
    end = int(e.group(1)) if e else default_end_ps
    return (end - begin) / 1000.0


def start_ns(path):
    """Start time (-b, ps) of the trajectory window, in ns.  Missing -b = 0."""
    with open(path) as fh:
        header = fh.read(4000)
    b = re.search(r"-b\s+(\d+)", header)
    return (int(b.group(1)) if b else 0) / 1000.0


def force_field(fname):
    """Map a filename to one of the FORCEFIELDS entries, else None."""
    for key, label, color in FORCEFIELDS:
        if fname.startswith(key):
            return key, label, color
    return None


def scatter_plot(points, xlabel, outfile):
    """points: list of (x, spread, label, color, size).  Scatter, one legend
    entry per force field, in the shared colour scheme."""
    plt.figure(figsize=(6, 5))
    seen = set()
    for x, spread, label, color, size in points:
        plot_label = label if label not in seen else None
        seen.add(label)
        plt.scatter(x, spread, color=color, label=plot_label,
                    s=size, edgecolors="white", linewidths=0.5, zorder=3)
    plt.xlabel(xlabel)
    plt.ylabel(r"PMF range, max $-$ min (kJ mol$^{-1}$)")
    # Force a small, uniform marker in the legend regardless of the plotted
    # point sizes (which scale with sampling time in the start-time plot).
    leg = plt.legend()
    for handle in leg.legend_handles:
        handle.set_sizes([40])
    plt.tight_layout()
    outpath = PLOTS / outfile
    plt.savefig(outpath, dpi=300)
    print(f"wrote {outpath}")


def main():
    files = sorted(DATA.glob("*profile.xvg"))

    # Full-trajectory profiles carry no -e; use the largest -e found anywhere
    # (the parts run to 100000 ps) as the full-trajectory end time.
    ends = [int(m) for f in files
            for m in re.findall(r"-e\s+(\d+)", f.read_text()[:4000])]
    default_end_ps = max(ends) if ends else 100000

    duration_pts, start_pts = [], []
    for fname in files:
        ff = force_field(fname.name)
        if ff is None:
            continue
        key, label, color = ff
        y = load_y(fname)
        spread = y.max() - y.min()
        duration = sampling_ns(fname, default_end_ps)
        duration_pts.append((duration, spread, label, color, 40))
        # In the start-time plot, marker area scales with the ns of sampling.
        start_pts.append((start_ns(fname), spread, label, color, 4 * duration))

    scatter_plot(duration_pts, "Sampling per profile (ns)", "range_vs_time.png")
    scatter_plot(start_pts, "Start time, -b (ns)", "range_vs_starttime.png")


if __name__ == "__main__":
    main()
