#!/usr/bin/env python2
"""Stage 2 of the tilt-modulus pipeline: compute the membrane lipid **tilt
modulus** chi from the tilt-angle distribution, using the ``ost_pymodules``
``lipid_analysis`` code (Johner / Harries / Khelashvili method).

Reads the ``data/tilt_{ff}_{lipid}.{pdb,dcd}`` pair written by ``tilt_convert.py``
(topology PDB + DCD).  For each system it:

  1. loads the topology + trajectory into an OST CoordGroup;
  2. assigns each atom an element from its name (C/H/O/N/P) so the density map
     used to find the water/lipid interface is correct;
  3. runs ``AnalyzeLipidTiltAndSplay`` -- which builds the water/lipid interface,
     assigns a membrane normal to every lipid in every frame, and returns the
     per-lipid tilt angle (angle between the head->tail vector and that normal);
  4. pools all lipid tilts and fits the tilt PMF  -log[P(a)/sin(a)]  with
     ``FitTiltDistribution`` to extract chi (in units of k_B T).

Runs under the OpenStructure Python 2.7 environment, e.g.::

    ~/.local/bin/ost analysis/tilt_modulus.py data/tilt_charmm_DMPC
    /path/to/openstructure/bin/python analysis/tilt_modulus.py data/tilt_charmm_DMPC --stride 5

Writes ``data/tilt_moduli.json`` (chi, dchi, n_lipids, n_frames per system) and,
per system, the library's tilt-distribution / PMF PNGs into ``plots/tilt/``.

The head->tail and neutral-plane selections per lipid:
  * head      = phosphate (CHARMM/SLipids ``P``, AMBER ``P31``)
  * tail      = terminal acyl methyls of both chains (COM); DOPC C18, DMPx C14
  * neutral   = glycerol ``C2`` (splay center-of-mass distance reference)
"""

import os
import sys
import json
import math
import argparse

from ost import io, mol, geom, conop

# Import the img submodule explicitly: ost_pymodules' density_alg /
# trajectory_utilities do ``from ost import *`` and then use ``img.*``, which
# only resolves if the img submodule has already been imported into the ost
# package (the interactive ``ost`` startup does this; a plain python does not).
# Guarded so this module still imports on an OST built without the image (IPLT)
# libraries -- selections can be inspected, but AnalyzeLipidTiltAndSplay (which
# builds the water-density interface) will fail at call time until img exists.
try:
    import ost.img as img  # noqa: F401  (registers ost.img for density_alg's *)
    _HAVE_IMG = True
except ImportError:
    _HAVE_IMG = False


def _set_prefix_path():
    """Point OST at its shared-data prefix (holds atom_scattering_properties.txt,
    needed by EntityToDensityRosetta).  The interactive ``ost`` startup does this
    from ``$DNG_ROOT``; a plain python must do it itself, else density map
    building dies with 'missing PrefixPath'."""
    import ost
    from ost import SetPrefixPath
    root = os.getenv("DNG_ROOT") or os.getenv("OST_ROOT")
    if not root:
        d = os.path.dirname(os.path.abspath(ost.__file__))
        for _ in range(6):
            d = os.path.dirname(d)
            if os.path.isdir(os.path.join(d, "share", "openstructure")):
                root = d
                break
    if root:
        SetPrefixPath(root)


if _HAVE_IMG:
    _set_prefix_path()

OST_PYMODULES = os.path.expanduser("~/software/ost_pymodules")
sys.path.append(OST_PYMODULES)
import lipid_analysis  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
PLOTS = os.path.join(ROOT, "plots", "tilt")

WATER_NAME = {"charmm": "TIP3", "amber": "TP3", "slipids": "SOL"}

# Last acyl carbon of each chain, per lipid.  The tail reference point is the
# COM of the last THREE carbons of each chain (six atoms total) -- per Khelashvili
# 2013 SI Fig. S2 ("three terminal carbon atoms ... six cyan spheres").
TERM = {"DOPC": 18, "DMPC": 14, "DMPG": 14}


def _tail_carbons(ff, n):
    """OST aname= list for the last 3 carbons of each chain."""
    last3 = (n - 2, n - 1, n)
    if ff == "amber":
        # tilt_convert renames the 2nd tail C1{n} -> CB{n} so both tails are
        # selectable in the merged residue: tail A C1x + tail B CBx.
        return ",".join("C1%d,CB%d" % (c, c) for c in last3)
    return ",".join("C2%d,C3%d" % (c, c) for c in last3)     # sn-1 + sn-2


def selections(ff, lipid):
    """Return (rname, head_dict, tail_dict, distance_dict) for the lipid.

    Matches Khelashvili et al. 2013 (JCTC 9:3866, reference/khelashvili2013-
    calculating.pdf): for all-atom PC lipids the director t connects **the
    midpoint between the phosphate and the backbone glycerol-C2 atom** to **the
    geometric COM of the terminal carbons of the two chains**.  The tilt is the
    angle between t and the bilayer normal; chi is fit from
    PMF(a) = -kT ln[P(a)/sin a] and reported in kT/rad^2 (no area normalisation
    -- area only enters the splay/bending modulus).

    NOTE the head is the *geometric midpoint* of P and C2.  The library's
    AnalyzeCenterOfMassPos is **mass-weighted**, so ``aname=P,C2`` gives a COM
    pulled ~0.8 A toward P vs the true midpoint -- negligible in the ~18 A
    director.  The neutral-plane selection (glycerol C2) is only used for the
    splay distance."""
    n = TERM[lipid]
    tail = "aname=" + _tail_carbons(ff, n)  # COM of last 3 carbons of each chain
    if ff == "amber":
        rname = "LIP"                      # merged residue from tilt_convert.py
        head = "aname=P31,C2"              # ~midpoint of phosphate + glycerol C2
        neutral = "aname=C2"
    else:
        rname = lipid
        head = "aname=P,C2"                # ~midpoint of phosphate + glycerol C2
        neutral = "aname=C2"
    return rname, {rname: head}, {rname: tail}, {rname: neutral}


_MASS = {"C": 12.011, "H": 1.008, "O": 15.999, "N": 14.007, "P": 30.974}


def assign_elements(eh):
    """Set each atom's element (and mass) from the first alphabetic char of its
    name.  Only C/H/O/N/P are present (ions dropped in stage 1), so this is
    exact and is all ``EntityToDensityRosetta`` needs for the interface
    density.  Set directly on the atom handles, as in the library's examples."""
    for a in eh.atoms:
        nm = a.GetName().lstrip("0123456789")
        ele = nm[0].upper() if nm else "C"
        a.SetElement(ele)
        a.SetMass(_MASS.get(ele, 12.011))


def load_trajectory(basis):
    p = io.IOProfile(processor=conop.HeuristicProcessor(),
                     dialect="CHARMM", fault_tolerant=True)
    eh = io.LoadPDB(basis + ".pdb", profile=p)
    t = io.LoadCHARMMTraj(eh, basis + ".dcd", stride=1)
    return eh, t


def analyze(basis, stride=1, density_stride=10, nbins=90,
            within_size_normals=10.0, density_cutoff=0.3):
    if not _HAVE_IMG:
        raise SystemExit(
            "This OST build lacks the image (IPLT) libraries, so the water-"
            "density interface used for the membrane normals cannot be built.\n"
            "Rebuild OST with image support, or use the global-z tilt method.")
    parts = os.path.basename(basis).split("_")    # tilt_{ff}_{lipid}[_{rep}]
    ff, lipid = parts[1], parts[2]
    rep = parts[3] if len(parts) > 3 else None
    rname, head_d, tail_d, dist_d = selections(ff, lipid)
    water = WATER_NAME[ff]

    eh, t = load_trajectory(basis)
    if stride > 1:
        t = t.Filter(eh.Select(""), stride=stride)
        eh = t.GetEntity()
    assign_elements(eh)
    print("loaded %s %s: %d atoms, %d frames (stride %d)"
          % (ff, lipid, eh.GetAtomCount(), t.GetFrameCount(), stride))

    outdir = os.path.join(PLOTS, "%s_%s" % (ff, lipid))
    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    # Run the library's TILT pipeline directly -- exactly the sequence
    # AnalyzeLipidTiltAndSplay uses for tilts (build the water/lipid interface,
    # assign a normal to each lipid, compute the tilt) but WITHOUT the splay
    # step, which we don't need and which requires extra per-cell plumbing
    # (chain names, bool props) irrelevant to a single flat bilayer.
    fb = "%s_%s_" % (ff, lipid)
    t.CopyFrame(0)
    lipids = eh.Select("rname=" + rname)
    waters = eh.Select("rname=" + water)
    (_, _, b_eh) = lipid_analysis.GetBoundaryBetweenViews(
        t, waters, lipids, outdir, density_cutoff, density_stride,
        within_size_normals, fb)

    head_sele = "(rname=%s and %s)" % (rname, head_d[rname])
    sele_ev = eh.Select(head_sele, mol.MATCH_RESIDUES)
    for i, r in enumerate(sele_ev.Select("rname=" + rname).residues):
        r.SetIntProp("index", i)
    normal_dict = lipid_analysis.AssignNormalsToLipids(
        t, sele_ev, b_eh.Select(""), [rname], head_d)
    tilt_dict = lipid_analysis.AnalyzeLipidTilts(
        t, eh.Select(head_sele, mol.MATCH_RESIDUES), [rname],
        normal_dict, head_d, tail_d, None)

    tilts = tilt_dict[rname][0]                   # N_lipids x N_frames (radians)
    tilt_list = [x for row in tilts for x in row]
    n_lipids = len(tilts)
    key = "%s_%s%s" % (ff, lipid, "_" + rep if rep else "")
    chi, dchi, chi_list = lipid_analysis.FitTiltDistribution(
        tilt_list, nbins=nbins, outdir=outdir, filename_basis=key)
    print("  chi = %.3f +/- %.3f kT  (%d lipids, %d frames)"
          % (chi, dchi, n_lipids, t.GetFrameCount()))
    return {"key": key, "ff": ff, "lipid": lipid, "rep": rep,
            "chi_kT": chi, "dchi_kT": dchi, "n_lipids": n_lipids,
            "n_frames": t.GetFrameCount(), "chi_list": list(chi_list)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bases", nargs="+",
                    help="data/tilt_{ff}_{lipid} basename(s) (no extension)")
    ap.add_argument("--stride", type=int, default=1,
                    help="use every Nth trajectory frame (default 1)")
    ap.add_argument("--density-stride", type=int, default=10,
                    help="stride for the interface density (default 10)")
    ap.add_argument("--nbins", type=int, default=90)
    ap.add_argument("--within-normals", type=float, default=10.0,
                    help="patch radius (A) for interface normals (default 10)")
    ap.add_argument("--density-cutoff", type=float, default=0.3)
    args = ap.parse_args()

    if not os.path.isdir(PLOTS):
        os.makedirs(PLOTS)
    results = {}
    for basis in args.bases:
        basis = basis[:-4] if basis.endswith(".pdb") or basis.endswith(".dcd") else basis
        res = analyze(basis, args.stride, args.density_stride, args.nbins,
                      args.within_normals, args.density_cutoff)
        results[res["key"]] = res

    out = os.path.join(DATA, "tilt_moduli.json")
    existing = {}
    if os.path.isfile(out):
        with open(out) as fh:
            existing = json.load(fh)
    existing.update(results)
    with open(out, "w") as fh:
        json.dump(existing, fh, indent=2, sort_keys=True)
    print("wrote %s" % out)


if __name__ == "__main__":
    main()
