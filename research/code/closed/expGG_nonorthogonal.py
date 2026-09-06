"""Experiment GG -- the multi-determinant ceiling, done properly this time.

`expAA_headroom.py` expanded the exact ground state in a FIXED ORTHONORMAL determinant basis and
truncated to the k largest coefficients.  That is optimal within that basis, which is why it was
called a ceiling, and it is the LEAST COMPACT way to build a multi-determinant trial.  Real
multi-determinant trials are NON-ORTHOGONAL: a few determinants pointing in different directions,
each carrying structure a fixed-basis expansion smears across dozens of coefficients.  So
expAA's "~100 determinants for overlap 0.99, worsening with U" is an upper bound on a bad
construction, not a property of multi-determinant trials, and Part 5's closure of that route does
not stand on it.

This measures the real thing: maximise the exact overlap over k determinants AND their
coefficients, with no orthogonality imposed between them.

The coefficients are not searched, because for fixed orbitals they are closed form.  With
v_j the sector-space amplitude vector of determinant j, S_jk = v_j . v_k and b_j = v_j . g,

    max_c  |c.b| / sqrt(c^T S c)  =  sqrt(b^T S^-1 b),   attained at c ~ S^-1 b

so the search runs over the ORBITALS only and the objective is exact in them.  At k = 1 this
reduces to (v.g)^2/(v.v), which must reproduce expZ's single-determinant ceiling exactly -- that
is the gate, and it is checked rather than assumed.

S goes singular when two determinants collapse onto each other, which is the optimiser telling
you the effective k is smaller than the k you asked for.  A pseudo-inverse handles it and the
condition number is reported, because a k that silently collapsed is a k that did not help.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from model2d import Model2D
from sector_ed import ground_energy
from trial import _det_amplitudes, free_trial


def _amps(p, N, n_up, n_dn):
    a = np.linalg.qr(p[:N * n_up].reshape(N, n_up))[0]
    b = np.linalg.qr(p[N * n_up:].reshape(N, n_dn))[0]
    au, _ = _det_amplitudes(a, N, n_up)
    ad, _ = _det_amplitudes(b, N, n_dn)
    return np.kron(au, ad)


def overlap_k(p, N, n_up, n_dn, k, gn, want_cond=False):
    per = N * (n_up + n_dn)
    V = np.array([_amps(p[j * per:(j + 1) * per], N, n_up, n_dn) for j in range(k)])
    S = V @ V.T
    b = V @ gn
    val = float(b @ np.linalg.pinv(S, rcond=1e-12) @ b)
    ov = float(np.sqrt(max(val, 0.0)))
    if want_cond:
        w = np.linalg.eigvalsh(S)
        return ov, float(w.max() / max(w.min(), 1e-300))
    return ov


def best_k_dets(K, U, n_up, n_dn, g, k, n_restarts=4, seed=0):
    """As `best_k`, but returns the determinants and their optimal coefficients too.

    The coefficients are not searched: for fixed orbitals they are closed form, c ~ S^-1 b with
    S_jk = v_j . v_k and b_j = v_j . g.  NOTE these orbitals are fitted to the EXACT ground
    state, so what they measure is a CEILING -- how far the bias could fall given the best k
    determinants -- and not a method.  Producing them without knowing the answer is a separate
    problem, and the point of measuring the ceiling first is to find out whether that problem is
    worth solving.
    """
    N = K.shape[0]
    gn = g / np.linalg.norm(g)
    per = N * (n_up + n_dn)
    ov, p = _best_params(K, U, n_up, n_dn, gn, k, n_restarts, seed)
    dets = []
    for j in range(k):
        pj = p[j * per:(j + 1) * per]
        dets.append({+1: np.linalg.qr(pj[:N * n_up].reshape(N, n_up))[0],
                     -1: np.linalg.qr(pj[N * n_up:].reshape(N, n_dn))[0]})
    V = np.array([_amps(p[j * per:(j + 1) * per], N, n_up, n_dn) for j in range(k)])
    c = np.linalg.pinv(V @ V.T, rcond=1e-12) @ (V @ gn)
    return ov, dets, c


def _best_params(K, U, n_up, n_dn, gn, k, n_restarts, seed):
    N = K.shape[0]
    per = N * (n_up + n_dn)
    rng = np.random.default_rng(seed)
    pf = free_trial(K, n_up, n_dn)
    base = np.concatenate([pf[+1].ravel(), pf[-1].ravel()])
    starts = [np.concatenate([base + 0.35 * rng.standard_normal(per) for _ in range(k)])]
    starts += [rng.standard_normal(per * k) for _ in range(n_restarts)]
    best, bv = None, 0.0
    for s in starts:
        r = minimize(lambda q: -overlap_k(q, N, n_up, n_dn, k, gn), s,
                     method="L-BFGS-B", options=dict(maxiter=3000, ftol=1e-13))
        if -r.fun > bv:
            bv, best = -r.fun, r.x
    return bv, best


def best_k(K, U, n_up, n_dn, g, k, n_restarts=4, seed=0):
    N = K.shape[0]
    gn = g / np.linalg.norm(g)
    per = N * (n_up + n_dn)
    rng = np.random.default_rng(seed)
    pf = free_trial(K, n_up, n_dn)
    base = np.concatenate([pf[+1].ravel(), pf[-1].ravel()])
    starts = [np.concatenate([base + 0.35 * rng.standard_normal(per) for _ in range(k)])]
    starts += [rng.standard_normal(per * k) for _ in range(n_restarts)]
    best, bv = None, 0.0
    for s in starts:
        r = minimize(lambda p: -overlap_k(p, N, n_up, n_dn, k, gn), s,
                     method="L-BFGS-B", options=dict(maxiter=3000, ftol=1e-13))
        if -r.fun > bv:
            bv, best = -r.fun, r.x
    return bv, overlap_k(best, N, n_up, n_dn, k, gn, want_cond=True)[1]


# expAA's fixed-orthonormal-basis numbers, for the comparison this file exists to make
FIXED_BASIS = {4.0: {1: 0.94097, 2: 0.94636, 4: 0.95567, 8: 0.96772, 16: 0.97653},
               8.0: {1: 0.84366, 2: 0.85753, 4: 0.87927, 8: 0.90396, 16: 0.92816},
               12.0: {1: 0.76303, 2: 0.78336, 4: 0.81345, 8: 0.84524, 16: 0.88169}}


if __name__ == "__main__":
    Lx, Ly, nu, nd = 2, 4, 3, 3
    ks = [1, 2, 3, 4, 6]
    print("=" * 96)
    print(f"NON-ORTHOGONAL MULTI-DETERMINANT CEILING  {Lx}x{Ly} at ({nu},{nd})")
    print("k = 1 must reproduce expZ's single-determinant ceiling exactly -- that is the gate.")
    print()
    print(f"{'U':>5} {'k':>3} {'non-orth':>10} {'fixed basis':>12} {'gain':>9} "
          f"{'cond(S)':>11} {'fixed-basis k needed':>21}")
    for U in (4.0, 8.0, 12.0):
        m = Model2D(Lx=Lx, Ly=Ly, t=1.0, mu=0.0, U=U, dtau=0.05, L=1, theta=0.0)
        ex, g = ground_energy(m.K, U, nu, nd, want_vec=True)
        fb = FIXED_BASIS[U]
        for k in ks:
            ov, cond = best_k(m.K, U, nu, nd, g, k)
            same = fb.get(k)
            # how many fixed-basis determinants would be needed to match this overlap
            need = next((kk for kk in sorted(fb) if fb[kk] >= ov - 1e-9), None)
            need_s = f"{need}" if need else f">16"
            gain = f"{ov - same:+9.5f}" if same else f"{'--':>9}"
            print(f"{U:5.1f} {k:3d} {ov:10.5f} "
                  f"{(f'{same:12.5f}' if same else f'{chr(45)*2:>12}')} {gain} "
                  f"{cond:11.2e} {need_s:>21}", flush=True)
        print()
    print("If a handful of non-orthogonal determinants reaches what tens of fixed-basis ones")
    print("needed, then expAA measured a bad construction and Part 5's multi-determinant closure")
    print("does not stand -- the constrained-path bias would not be floored after all.")
