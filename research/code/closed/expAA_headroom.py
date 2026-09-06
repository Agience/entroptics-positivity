"""Experiment AA -- how many determinants would it take, and is it worth building?

Experiment Z closed the single-determinant family: at 2x4 (3,3) the free determinant IS the
maximum-overlap single Slater determinant (seven random restarts from overlap ~0.002 all climb
to exactly the free value), so there is no criterion to find and no headroom to find it in.  The
bias expV measured is that family's floor, not a starting point.

The next lever is a multi-determinant trial wavefunction.  Before writing one -- which means
rebuilding the overlap and the mixed Green's function as weighted sums over determinants, and
paying k times the cost per walker -- the question is whether it would help, and that is pure
linear algebra with no Monte Carlo in it:

    expand the exact ground state in a determinant basis, sort by weight, and read off how many
    determinants a given overlap costs.

The truncation to the k largest coefficients IS the optimal k-determinant approximation in that
basis, so this is a ceiling for each k, exactly as expZ was the ceiling for k = 1.

Two bases, because the basis is the whole question:

  free orbitals      the eigenvectors of K.  What a CPMC implementation would reach for.
  natural orbitals   the eigenvectors of the exact 1-particle density matrix of Psi_0.  The most
                     compact single-particle basis there is, and unavailable in practice -- so it
                     bounds what any smarter choice of orbitals could buy.

The change of basis between determinant bases is the matrix of minors: <site occupation a |
orbital occupation b> = det V[a, b] with V the orbital coefficient matrix.  It is orthogonal
because V is, which is checked rather than assumed.
"""
from __future__ import annotations

import numpy as np
from itertools import combinations

from model2d import Model2D
from sector_ed import ground_energy, _states


def minor_matrix(V, N, n):
    """T[a, b] = det V[occ_a, orb_b] -- the determinant-basis change of basis induced by V."""
    occ = list(combinations(range(N), n))
    T = np.empty((len(occ), len(occ)))
    for a, oa in enumerate(occ):
        rows = V[list(oa), :]
        for b, ob in enumerate(occ):
            T[a, b] = np.linalg.det(rows[:, list(ob)])
    return T


def one_rdm(psi, N, n_up, n_dn):
    """The spin-up 1-RDM of the sector state, <c^dag_i c_j>, by direct second quantisation."""
    su, sd = _states(N, n_up), _states(N, n_dn)
    du, dd = len(su), len(sd)
    idx = {s: k for k, s in enumerate(su)}
    P = psi.reshape(du, dd)
    R = np.zeros((N, N))
    for b, sb in enumerate(su):
        for j in range(N):
            if not (sb >> j) & 1:
                continue
            s1 = sb & ~(1 << j)
            sj = (-1) ** bin(sb & ((1 << j) - 1)).count("1")
            for i in range(N):
                if (s1 >> i) & 1:
                    continue
                si = (-1) ** bin(s1 & ((1 << i) - 1)).count("1")
                R[i, j] += si * sj * float(P[idx[s1 | (1 << i)]] @ P[b])
    return R


def spectrum(psi, N, n_up, n_dn, V):
    """Coefficients of psi in the determinant basis built from orbitals V, sorted by weight."""
    Tu = minor_matrix(V, N, n_up)
    Td = minor_matrix(V, N, n_dn) if n_dn != n_up else Tu
    orth = float(np.abs(Tu.T @ Tu - np.eye(Tu.shape[0])).max())
    c = Tu.T @ psi.reshape(len(Tu), len(Td)) @ Td
    w = np.sort(c.ravel() ** 2)[::-1]
    return w / w.sum(), orth


def k_for(cum, target):
    return int(np.searchsorted(cum, target) + 1)


if __name__ == "__main__":
    Lx, Ly, nu, nd = 2, 4, 3, 3
    N = Lx * Ly
    print("=" * 88)
    print(f"MULTI-DETERMINANT HEADROOM  {Lx}x{Ly} at ({nu},{nd}), "
          f"sector dimension {56*56}")
    print(f"{'U':>5} {'basis':>9} {'orth err':>9} {'k=1':>8} {'k=2':>8} {'k=4':>8} "
          f"{'k=8':>8} {'k=16':>8} {'k for 0.99':>11} {'k for 0.999':>12}")
    for U in (4.0, 8.0, 12.0):
        m = Model2D(Lx=Lx, Ly=Ly, t=1.0, mu=0.0, U=U, dtau=0.05, L=1, theta=0.0)
        ex, g = ground_energy(m.K, U, nu, nd, want_vec=True)
        g = g / np.linalg.norm(g)
        Vf = np.linalg.eigh(m.K)[1]
        R = one_rdm(g, N, nu, nd)
        Vn = np.linalg.eigh(R)[1][:, ::-1]           # natural orbitals, most occupied first
        for label, V in (("free", Vf), ("natural", Vn)):
            w, orth = spectrum(g, N, nu, nd, V)
            cum = np.cumsum(w)
            ovl = lambda k: np.sqrt(cum[min(k, len(cum)) - 1])
            print(f"{U:5.1f} {label:>9} {orth:9.1e} {ovl(1):8.5f} {ovl(2):8.5f} "
                  f"{ovl(4):8.5f} {ovl(8):8.5f} {ovl(16):8.5f} "
                  f"{k_for(cum, 0.99**2):11d} {k_for(cum, 0.999**2):12d}", flush=True)
    print()
    print("Columns k=1..16 are the OVERLAP |<Psi_T|Psi_0>| of the best k-determinant trial in")
    print("that basis.  k = 1 in the free row must reproduce expZ's single-determinant ceiling.")
    print("The two right-hand columns are how many determinants an overlap target costs, which")
    print("is the cost multiplier a multi-determinant CPMC would pay per walker.")
