"""Gate: the k > 1 objects, against exact arithmetic in the sector basis.

`gate_multi.py` shows the multi-determinant walker IS the single-determinant walker at k = 1.
That gate is blind to everything k = 1 cannot exercise -- the sum over determinants in the
overlap, the overlap-weighted mixture that forms the Green's function, and the coefficients.  So
a k = 2 result cannot be read as physics on the strength of it.

Two objects, each checked against a construction that shares no code with the walker:

  OVERLAP   sum_j c_j det(P_j^up' phi^up) det(P_j^dn' phi^dn)  against  <Psi_T|phi> taken as a
            plain inner product of amplitude vectors in the 3136-dimensional sector basis.
  ENERGY    the mixed estimator built from the weighted Green's function, against
            <Psi_T|H|phi> / <Psi_T|phi> with H the sparse sector Hamiltonian.

The second is the one that matters: it is the quantity the walk reports, and it is where a wrong
mixture would show up while the overlap stayed right.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import kron as skron, identity, csr_matrix

from model2d import Model2D
from sector_ed import ground_energy, _hop_matrix, _states
from trial import _det_amplitudes
from cpmc_multi import MultiCPMC
from expGG_nonorthogonal import best_k_dets


def sector_vector(dets, coeffs, N, n_up, n_dn):
    v = np.zeros(len(_states(N, n_up)) * len(_states(N, n_dn)))
    for c, d in zip(coeffs, dets):
        au, _ = _det_amplitudes(d[+1], N, n_up)
        ad, _ = _det_amplitudes(d[-1], N, n_dn)
        v = v + c * np.kron(au, ad)
    return v


def sector_hamiltonian(m, n_up, n_dn):
    Hu, su = _hop_matrix(m.N, n_up, m.K)
    Hd, sd = _hop_matrix(m.N, n_dn, m.K)
    du, dd = len(su), len(sd)
    H = skron(Hu, identity(dd), format="csr") + skron(identity(du), Hd, format="csr")
    nu = np.array([[(s >> i) & 1 for i in range(m.N)] for s in su], float) - 0.5
    nd = np.array([[(s >> i) & 1 for i in range(m.N)] for s in sd], float) - 0.5
    dg = np.concatenate([m.U * (nd @ nu[a]) for a in range(du)])
    return H + csr_matrix((dg, (np.arange(du * dd), np.arange(du * dd))), shape=H.shape)


if __name__ == "__main__":
    Lx, Ly, nu, nd = 2, 4, 3, 3
    U = 8.0
    m = Model2D(Lx=Lx, Ly=Ly, t=1.0, mu=0.0, U=U, dtau=0.05, L=1, theta=0.0)
    ex, g = ground_energy(m.K, U, nu, nd, want_vec=True)
    H = sector_hamiltonian(m, nu, nd)
    rng = np.random.default_rng(4)

    print("=" * 92)
    print("k > 1 OBJECTS AGAINST EXACT SECTOR ARITHMETIC")
    print(f"{'k':>3} {'walker overlap':>16} {'sector overlap':>16} {'rel diff':>10} "
          f"{'walker E_mix':>14} {'sector E_mix':>14} {'rel diff':>10}")
    worst_o = worst_e = 0.0
    for k in (1, 2, 3, 4):
        _, dets, c = best_k_dets(m.K, U, nu, nd, g, k, n_restarts=2, seed=1)
        ch = MultiCPMC(m, nu, nd, dets, c, n_walkers=3, seed=2)
        # ch.c, not c: the class may flip the global sign of Psi_T (which is arbitrary) so the
        # walkers start on the positive side of the node.  Comparing against the UNflipped
        # coefficients reports a relative overlap error of exactly 2.0 -- the signature of a sign
        # flip -- while the energy stays exact, because the energy is a ratio and the sign cancels.
        psi = sector_vector(dets, ch.c, m.N, nu, nd)
        # a random walker, not the starting one, so the test is not trivial
        ch.phi = {s: np.linalg.qr(rng.standard_normal(ch.phi[s].shape))[0] for s in (+1, -1)}
        D, G = ch._dets_and_greens(ch.phi)
        wj, tot, Gm = ch._mix(D, G)

        w0 = float(tot[0])
        phi_vec = np.kron(_det_amplitudes(ch.phi[+1][0], m.N, nu)[0],
                          _det_amplitudes(ch.phi[-1][0], m.N, nd)[0])
        s0 = float(psi @ phi_vec)
        do = abs(w0 - s0) / max(abs(s0), 1e-300)

        kin = sum(float(np.sum(m.K * Gm[s][0].T)) for s in (+1, -1))
        nuj = np.einsum("jwii->jwi", G[+1])[:, 0] - 0.5
        ndj = np.einsum("jwii->jwi", G[-1])[:, 0] - 0.5
        e_walk = kin + float(m.U * (wj[:, 0] @ np.sum(nuj * ndj, axis=1)) / tot[0])
        e_sec = float(psi @ (H @ phi_vec)) / s0
        de = abs(e_walk - e_sec) / max(abs(e_sec), 1e-300)
        worst_o, worst_e = max(worst_o, do), max(worst_e, de)
        print(f"{k:3d} {w0:16.8f} {s0:16.8f} {do:10.2e} {e_walk:14.8f} {e_sec:14.8f} "
              f"{de:10.2e}")
    print()
    print(f"worst overlap disagreement {worst_o:.2e}    worst energy disagreement {worst_e:.2e}")
    print("PASS" if max(worst_o, worst_e) < 1e-8 else
          "FAIL -- the k > 1 path does not compute what it claims, and no k > 1 result may be read")
