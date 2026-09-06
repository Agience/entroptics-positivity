"""Experiment Z -- the CEILING of the single-determinant trial family.

Experiment X answered "can the dial be turned by a cheap criterion" with NO: the variational
energy of the trial determinant prefers a symmetry-broken state whose overlap and whose bias are
both worse, by an order of magnitude, at U = 8 and U = 12.  Lowering <Psi_T|H|Psi_T> moves the
node AWAY from the true one, because the variational principle optimises the energy and the
constrained path only cares about the node.

That leaves a prior question, and it is the one that decides whether trial-wavefunction work is
worth doing at all:

    HOW MUCH ROOM IS THERE?  What is the smallest bias any single Slater determinant can give?

The free determinant is one point in that family.  If it is already close to the family's best,
then the bias measured in expV is near the ceiling of every single-determinant CPMC, no criterion
however clever helps, and the next lever has to be multi-determinant.  If the best determinant is
far better, the bias is not a property of the method at all and finding the knob is the whole
problem.

The best determinant is found by maximising the exact overlap |<Psi_T|Psi_0>| directly over the
orbitals -- which uses exact diagonalisation and is therefore NOT a method, it is a CEILING
measurement.  Nothing here is proposed as an algorithm; it exists to bound what an algorithm
could achieve.

Optimisation: the orbitals are carried as an unconstrained N x n matrix and orthonormalised by QR
inside the objective, so the search runs on a flat space and the Stiefel constraint is exact at
every evaluation.  Restarts from several random starts plus the free determinant, because the
overlap surface is not convex and a single start would silently report a local maximum as the
ceiling.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from model2d import Model2D
from sector_ed import ground_energy
from cpmc_fast import run
from trial import _det_amplitudes, free_trial, trial_overlap, trial_energy


def _psi_from_params(p, N, n_up, n_dn):
    a = p[:N * n_up].reshape(N, n_up)
    b = p[N * n_up:].reshape(N, n_dn)
    return {+1: np.linalg.qr(a)[0], -1: np.linalg.qr(b)[0]}


def best_determinant(K, U, n_up, n_dn, g, n_restarts=6, seed=0):
    """Maximise |<det | Psi_0>| over the orbitals.  Returns (psi_t, overlap)."""
    N = K.shape[0]
    gn = g / np.linalg.norm(g)

    def neg_overlap(p):
        pt = _psi_from_params(p, N, n_up, n_dn)
        au, _ = _det_amplitudes(pt[+1], N, n_up)
        ad, _ = _det_amplitudes(pt[-1], N, n_dn)
        v = np.kron(au, ad)
        nv = np.linalg.norm(v)
        if nv == 0:
            return 0.0
        return -abs((v / nv) @ gn)

    rng = np.random.default_rng(seed)
    pf = free_trial(K, n_up, n_dn)
    starts = [np.concatenate([pf[+1].ravel(), pf[-1].ravel()])]
    starts += [rng.standard_normal(N * (n_up + n_dn)) for _ in range(n_restarts)]
    best, bv = None, 0.0
    for s in starts:
        r = minimize(neg_overlap, s, method="L-BFGS-B",
                     options=dict(maxiter=4000, ftol=1e-14, gtol=1e-10))
        if -r.fun > bv:
            bv, best = -r.fun, r.x
    return _psi_from_params(best, N, n_up, n_dn), bv


if __name__ == "__main__":
    Lx, Ly, nu, nd = 2, 4, 3, 3
    beta, dtau = 8.0, 0.05
    print("=" * 100)
    print(f"CEILING  {Lx}x{Ly} at ({nu},{nd}), beta = {beta}, dtau = {dtau}")
    print(f"{'U':>5} {'Psi_T':>10} {'overlap':>9} {'<T|H|T>':>11} {'CPMC':>19} "
          f"{'exact':>11} {'bias':>10} {'rel':>8} {'killed':>7}")
    for U in (4.0, 8.0, 12.0):
        m = Model2D(Lx=Lx, Ly=Ly, t=1.0, mu=0.0, U=U, dtau=dtau, L=1, theta=0.0)
        ex, g = ground_energy(m.K, U, nu, nd, want_vec=True)
        pf = free_trial(m.K, nu, nd)
        pb, ovb = best_determinant(m.K, U, nu, nd, g)
        for name, pt in (("free", pf), ("best det", pb)):
            ov = trial_overlap(pt, m.K, U, nu, nd)
            ev = trial_energy(pt, m.K, U, nu, nd)
            rs = [run(m, nu, nd, beta, n_walkers=400, seed=s, psi_t=pt) for s in (1, 2, 3)]
            e = np.array([r["e"] for r in rs])
            se = e.std(ddof=1) / np.sqrt(3)
            b = e.mean() - ex
            print(f"{U:5.1f} {name:>10} {ov:9.5f} {ev:+11.5f} {e.mean():+12.5f}+-{se:.5f} "
                  f"{ex:+11.5f} {b:+10.5f} {abs(b)/abs(ex):8.5f} "
                  f"{sum(r['killed'] for r in rs):7d}", flush=True)
    print()
    print("The 'best det' row is a CEILING, not a method: its orbitals were fitted to the exact")
    print("ground state.  The gap between the two rows is the entire headroom available to any")
    print("scheme that chooses a single Slater determinant, however it chooses it.")
