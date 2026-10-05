#!/usr/bin/env python3
"""Director-fluctuation (DF) tilt, twist and bending moduli.

Fourier-space counterpart of the two real-space tilt analyses in this
directory (ost_pymodules / ReGiS in ``tilt_modulus.py``, lipidator / ReSiS via
``run_lipidator.py``).  It follows Watson, Brandt, Welch & Brown, PRL 109,
028102 (2012), as applied atomistically by Levine et al., JACS 136, 13582
(2014), with the theory of Watson et al., JCP 135, 244701 (2011).

Method
------
For every lipid and frame a unit director ``n`` is taken from the head group
to the chain ends (pointing into the bilayer).  Per leaflet ``a`` its in-plane
components are Fourier transformed directly from the lipid positions,

    n_q^a = (sqrt(A) / N_a) * sum_i n_i,xy * exp(-i q . r_i),

which is the continuum transform ``g_q = A**-1/2 * int g(r) e^{-iqr} dr`` with
each lipid standing for an area ``A / N_a`` (no grid, so no empty-cell
choices).  The bilayer field is ``n_q = (n_q^up - n_q^down) / 2`` and it is
split into components along q (longitudinal) and perpendicular to q
(transverse).  With bilayer moduli (Watson 2012 / Levine 2014 convention)

    < |n_q^perp|^2 > = k_B T / (K_theta + K_tw q^2)
    < |n_q^par |^2 > = k_B T / (K_c q^2)              (small q)

K_theta is the q -> 0 intercept of the inverse transverse spectrum.  The
transverse spectrum needs no reference surface: the surface normal is a
gradient field and has no transverse Fourier component, and protrusion
gradients are longitudinal (Watson 2011).

Units: report K_theta in kJ mol^-1 nm^-2 (json ``K_theta_kJmol_nm2``, at
``--temperature``); fits are done in k_B T/nm^2 (K_theta) and k_B T (K_tw,
K_c), which are kept alongside.  K_theta here is the
**bilayer** modulus, i.e. twice the monolayer coefficient of
(1/2) kappa_theta |m|^2.

Normalisation check: ``--selftest`` feeds independent Gaussian per-lipid tilts
of known stiffness through the same transform and fit.

Run (conda env ``lipid64``, from the repository root)::

    PYTHONNOUSERSITE=1 conda run -n lipid64 python analysis/tilt_df.py \
        traj/charmm_DMPC.gro traj/charmm_DMPC1.xtc --lipid DMPC \
        -o plots/tilt_three_methods/df_charmm_DMPC_1

CHARMM / SLipids naming (one residue per lipid; sn-1 = C2x, sn-2 = C3x) only.
"""

from __future__ import print_function

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from lipiddefs import TAIL_NC  # noqa: E402

KB_KJ = 0.0083144626  # kJ mol^-1 K^-1


# --------------------------------------------------------------------------
# directors
# --------------------------------------------------------------------------

def director_selections(lipid, kind):
    """Atom-name lists (head, tail) for one lipid, per director definition."""
    n = TAIL_NC[lipid]
    if kind == "levine":
        # Levine 2014: midpoint of P and C2 -> midpoint of the terminal methyls
        return ["P", "C2"], ["C2%d" % n, "C3%d" % n]
    if kind == "rsf":
        # this project's ReGiS definition (tilt_modulus.py): P + glycerol C2 ->
        # last three carbons of each chain
        tail = ["C2%d" % k for k in range(n - 2, n + 1)] + \
               ["C3%d" % k for k in range(n - 2, n + 1)]
        return ["P", "C2"], tail
    raise ValueError(kind)


def build_groups(u, lipid, kind):
    """Per-lipid index arrays (n_lipids, k) for head and tail atoms."""
    head_names, tail_names = director_selections(lipid, kind)
    res = u.select_atoms("resname %s" % lipid).residues
    heads, tails = [], []
    for r in res:
        h = [r.atoms.select_atoms("name %s" % nm).indices for nm in head_names]
        t = [r.atoms.select_atoms("name %s" % nm).indices for nm in tail_names]
        if any(len(x) != 1 for x in h + t):
            raise RuntimeError("residue %s %d lacks director atoms %s / %s"
                               % (r.resname, r.resid, head_names, tail_names))
        heads.append([x[0] for x in h])
        tails.append([x[0] for x in t])
    return np.array(heads), np.array(tails)


# --------------------------------------------------------------------------
# spectra
# --------------------------------------------------------------------------

def mode_list(nmax):
    """Integer (nx, ny) modes in a half-plane, excluding q = 0."""
    modes = []
    for nx in range(0, nmax + 1):
        for ny in range(-nmax, nmax + 1):
            if nx == 0 and ny <= 0:
                continue
            modes.append((nx, ny))
    return np.array(modes, dtype=float)


def frame_spectrum(pos_xy, nxy, up, box, modes):
    """|n_par|^2 and |n_perp|^2 for one frame, every mode.

    pos_xy, nxy : (N, 2) lipid positions (nm) and director xy components
    up          : (N,) bool, upper leaflet
    box         : (Lx, Ly) in nm
    modes       : (M, 2) integer mode indices
    """
    Lx, Ly = box
    area = Lx * Ly
    q = 2.0 * np.pi * modes / np.array([Lx, Ly])          # (M, 2)
    qn = np.linalg.norm(q, axis=1)
    qhat = q / qn[:, None]
    phase = np.exp(-1j * pos_xy @ q.T)                    # (N, M)
    field = np.zeros((len(modes), 2), dtype=complex)
    for sel, sign in ((up, 1.0), (~up, -1.0)):
        n_a = sel.sum()
        # (M, 2): sum_i n_i * exp(-i q r_i)
        fa = (phase[sel].T @ nxy[sel]) * (np.sqrt(area) / n_a)
        field += 0.5 * sign * fa
    par = field[:, 0] * qhat[:, 0] + field[:, 1] * qhat[:, 1]
    perp = field[:, 1] * qhat[:, 0] - field[:, 0] * qhat[:, 1]
    return np.abs(par) ** 2, np.abs(perp) ** 2, qn


def fit_tilt(q, s_perp, w=None, qcut=2.0):
    """Weighted fit 1/<|n_perp|^2> = Kt + Ktw q^2 for q <= qcut."""
    m = q <= qcut + 1e-9
    y = 1.0 / s_perp[m]
    X = np.vstack([np.ones(m.sum()), q[m] ** 2]).T
    W = np.ones(m.sum()) if w is None else w[m]
    coef = np.linalg.lstsq(X * np.sqrt(W)[:, None], y * np.sqrt(W), rcond=None)[0]
    return coef[0], coef[1], int(m.sum())


def fit_bend(q, s_par, w=None, qcut=1.5):
    """Weighted fit 1/<|n_par|^2> = Kc q^2 (through the origin), q <= qcut."""
    m = q <= qcut + 1e-9
    y = 1.0 / s_par[m]
    x = q[m] ** 2
    W = np.ones(m.sum()) if w is None else w[m]
    return float(np.sum(W * x * y) / np.sum(W * x * x)), int(m.sum())


def shell_average(q, vals, tol=1e-3):
    """Average values over modes with the same |q| (for plotting)."""
    order = np.argsort(q)
    qs, vs = q[order], vals[order]
    out_q, out_v = [], []
    i = 0
    while i < len(qs):
        j = i
        while j < len(qs) and qs[j] - qs[i] < tol * max(qs[i], 1.0):
            j += 1
        out_q.append(qs[i:j].mean())
        out_v.append(vs[i:j].mean())
        i = j
    return np.array(out_q), np.array(out_v)


# --------------------------------------------------------------------------
# trajectory analysis
# --------------------------------------------------------------------------

def analyse(gro, xtc, lipid, kind, nmax, skip_ns, stride):
    import MDAnalysis as mda
    u = mda.Universe(gro, xtc)
    heads, tails = build_groups(u, lipid, kind)
    masses = u.atoms.masses
    hm = masses[heads] / masses[heads].sum(axis=1, keepdims=True)
    tm = masses[tails] / masses[tails].sum(axis=1, keepdims=True)
    if kind == "levine":   # plain midpoints, as in Levine 2014
        hm = np.full(heads.shape, 1.0 / heads.shape[1])
        tm = np.full(tails.shape, 1.0 / tails.shape[1])
    modes = mode_list(nmax)

    spar, sperp, boxes, times, apl = [], [], [], [], []
    for ts in u.trajectory[::stride]:
        if ts.time / 1000.0 < skip_ns:
            continue
        x = u.atoms.positions / 10.0                     # nm
        box = ts.dimensions[:3] / 10.0
        hp = np.einsum("lk,lkd->ld", hm, x[heads])
        tp = np.einsum("lk,lkd->ld", tm, x[tails])
        d = tp - hp
        d -= box * np.round(d / box)                     # lipids are whole, guard anyway
        n = d / np.linalg.norm(d, axis=1)[:, None]
        up = n[:, 2] < 0.0                               # head -> tail points down
        p, t, qn = frame_spectrum(hp[:, :2], n[:, :2], up, box[:2], modes)
        spar.append(p)
        sperp.append(t)
        boxes.append(box[:2])
        times.append(ts.time / 1000.0)
        apl.append(box[0] * box[1] / up.sum())
    boxes = np.array(boxes)
    q = 2.0 * np.pi * np.linalg.norm(modes / boxes.mean(axis=0), axis=1)
    return dict(q=q, modes=modes, spar=np.array(spar), sperp=np.array(sperp),
                boxes=boxes, times=np.array(times), apl=np.array(apl),
                n_lipids=len(heads))


def summarise(res, nblocks, qcuts_t, qcut_b, temperature):
    q, spar, sperp = res["q"], res["spar"], res["sperp"]
    nf = len(spar)
    mpar, mperp = spar.mean(axis=0), sperp.mean(axis=0)
    blocks = np.array_split(np.arange(nf), nblocks)
    bpar = np.array([spar[b].mean(axis=0) for b in blocks])
    bperp = np.array([sperp[b].mean(axis=0) for b in blocks])
    sem_perp = bperp.std(axis=0, ddof=1) / np.sqrt(nblocks)
    sem_par = bpar.std(axis=0, ddof=1) / np.sqrt(nblocks)
    # weights for fits in 1/S: var(1/S) ~ (sem/S^2)^2
    w_perp = 1.0 / (sem_perp / mperp ** 2) ** 2
    w_par = 1.0 / (sem_par / mpar ** 2) ** 2
    kT = KB_KJ * temperature
    out = {"n_frames": int(nf), "n_lipids": int(res["n_lipids"]),
           "apl_nm2": float(res["apl"].mean()),
           "box_nm": [float(v) for v in res["boxes"].mean(axis=0)],
           "q_min_nm": float(q.min()), "temperature_K": temperature,
           "nblocks": nblocks, "tilt": {}, "bend": {}}
    for qc in qcuts_t:
        kt, ktw, nm = fit_tilt(q, mperp, w_perp, qc)
        bk = np.array([fit_tilt(q, bperp[i], w_perp, qc)[:2] for i in range(nblocks)])
        out["tilt"]["qcut_%.2f" % qc] = {
            "K_theta_kT_nm2": float(kt),
            "K_theta_sem": float(bk[:, 0].std(ddof=1) / np.sqrt(nblocks)),
            "K_tw_kT": float(ktw),
            "K_tw_sem": float(bk[:, 1].std(ddof=1) / np.sqrt(nblocks)),
            "K_theta_kJmol_nm2": float(kt * kT),
            "K_theta_sem_kJmol_nm2": float(kT * bk[:, 0].std(ddof=1) / np.sqrt(nblocks)),
            "K_tw_kJmol": float(ktw * kT),
            "n_modes": nm}
    kc, nm = fit_bend(q, mpar, w_par, qcut_b)
    bkc = np.array([fit_bend(q, bpar[i], w_par, qcut_b)[0] for i in range(nblocks)])
    out["bend"]["qcut_%.2f" % qcut_b] = {
        "K_c_kT": float(kc), "K_c_sem": float(bkc.std(ddof=1) / np.sqrt(nblocks)),
        "K_c_kJmol": float(kc * kT),
        "n_modes": nm}
    return out, dict(mpar=mpar, mperp=mperp, sem_par=sem_par, sem_perp=sem_perp)


# --------------------------------------------------------------------------
# self test of the normalisation
# --------------------------------------------------------------------------

def selftest(nmax=4, nframes=4000, seed=1):
    """Independent Gaussian tilts: expect <|n_perp|^2> = <|n_par|^2> = kT/K_theta
    with K_theta = 2 kappa_mono, at every q."""
    rng = np.random.default_rng(seed)
    L, nleaf, kappa = 6.2, 64, 10.0           # nm, lipids/leaflet, kT/nm^2 (monolayer)
    a = L * L / nleaf
    sigma = np.sqrt(1.0 / (kappa * a))        # per-component, energy (1/2) kappa a t^2
    modes = mode_list(nmax)
    acc = np.zeros(len(modes))
    for _ in range(nframes):
        pos = rng.uniform(0, L, size=(2 * nleaf, 2))
        nxy = rng.normal(0, sigma, size=(2 * nleaf, 2))
        up = np.arange(2 * nleaf) < nleaf
        _, t, _ = frame_spectrum(pos, nxy, up, (L, L), modes)
        acc += t
    est = 1.0 / (acc / nframes).mean()
    print("selftest: expected K_theta(bilayer) = %.2f kT/nm^2, recovered %.2f"
          % (2 * kappa, est))
    return est


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("gro", nargs="?")
    ap.add_argument("xtc", nargs="?")
    ap.add_argument("--lipid", default="DMPC")
    ap.add_argument("--director", choices=["levine", "rsf"], default="levine")
    ap.add_argument("--nmax", type=int, default=6, help="largest mode index")
    ap.add_argument("--skip-ns", type=float, default=0.0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--nblocks", type=int, default=10)
    ap.add_argument("--qcut-tilt", type=float, nargs="+", default=[1.5, 2.1, 2.5])
    ap.add_argument("--qcut-bend", type=float, default=1.5)
    ap.add_argument("--temperature", type=float, default=310.0)
    ap.add_argument("-o", "--out", help="output basename (.npz / .json)")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    res = analyse(args.gro, args.xtc, args.lipid, args.director,
                  args.nmax, args.skip_ns, args.stride)
    summ, spec = summarise(res, args.nblocks, args.qcut_tilt, args.qcut_bend,
                           args.temperature)
    summ.update({"gro": args.gro, "xtc": args.xtc, "lipid": args.lipid,
                 "director": args.director, "skip_ns": args.skip_ns,
                 "convention": "bilayer K_theta (Watson 2012 / Levine 2014)"})
    print(json.dumps(summ, indent=2))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        np.savez(args.out + ".npz", q=res["q"], modes=res["modes"],
                 times=res["times"], boxes=res["boxes"], apl=res["apl"],
                 **spec)
        with open(args.out + ".json", "w") as fh:
            json.dump(summ, fh, indent=2)


if __name__ == "__main__":
    main()
