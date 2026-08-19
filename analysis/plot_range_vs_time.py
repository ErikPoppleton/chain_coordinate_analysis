#!/usr/bin/env python3
"""For every '*profile.xvg' file in the data/ folder, plot the amount of
sampling that went into the PMF (in nanoseconds) against the max-min spread of
the profile (its total energy range).  One pair of figures is produced per
lipid (DOPC, DMPC, DMPG); lipids with no profile data are skipped.

The sampling window of each profile is read straight from the `gmx wham`
command line stored in the file header (`-b`/`-e`, in ps):

    * <ff>_<lipid>_part{1..5}_profile.xvg  -b/-e give a 20 ns window
    * <ff>_<lipid>_20-100_profile.xvg      -b 20000 (no -e)  -> 80 ns
    * <ff>_<lipid>_profile.xvg             no -b/-e          -> full 100 ns

Colours follow the other plotting scripts: CHARMM black, AMBER red,
SLipids green.
"""

import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from palette import FORCEFIELDS
from lipiddefs import LIPIDS

# Project layout: this script lives in analysis/, data in data/, output in plots/.
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
PLOTS = ROOT / "plots"

# Lipids to plot; profiles are named "{ff}_{lipid}_..._profile.xvg".


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


def scatter_plot(points, xlabel, title, outfile):
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
    plt.title(title)
    # Force a small, uniform marker in the legend regardless of the plotted
    # point sizes (which scale with sampling time in the start-time plot).
    leg = plt.legend()
    for handle in leg.legend_handles:
        handle.set_sizes([40])
    plt.tight_layout()
    outpath = PLOTS / outfile
    plt.savefig(outpath, dpi=300)
    plt.close()
    print(f"wrote {outpath}")


def plot_lipid(lipid):
    """Build the range-vs-time and range-vs-starttime scatters for one lipid."""
    files = sorted(DATA.glob(f"*_{lipid}_*profile.xvg"))
    if not files:
        print(f"skipped {lipid}: no profile data")
        return

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

    # Plot (and thus legend) in FORCEFIELDS order (CHARMM, AMBER, SLipids) to
    # match the other scripts; the glob above is alphabetical.  Stable sort, so
    # files within one force field keep their order.
    ff_rank = {label: i for i, (_key, label, _color) in enumerate(FORCEFIELDS)}
    duration_pts.sort(key=lambda p: ff_rank[p[2]])
    start_pts.sort(key=lambda p: ff_rank[p[2]])

    scatter_plot(duration_pts, "Sampling per profile (ns)", lipid,
                 f"range_vs_time_{lipid}.png")
    scatter_plot(start_pts, "Start time, -b (ns)", lipid,
                 f"range_vs_starttime_{lipid}.png")


def main():
    for lipid in LIPIDS:
        plot_lipid(lipid)


if __name__ == "__main__":
    main()
