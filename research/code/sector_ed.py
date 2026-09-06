"""Exact ground state of the Hubbard model in a FIXED particle-number sector.

The full Fock space is 4^N, which caps exact diagonalisation at 6 or 7 sites.  A canonical
calculation only ever needs one (n_up, n_dn) sector, and that has dimension
C(N, n_up) * C(N, n_dn) -- 3136 at 8 sites half filled against 65536 for the full space, and
sparse besides.  That is what makes a closed-shell test lattice with a real sign problem
reachable at all.

Basis: a pair of occupation bitmasks, one per spin.  Hopping moves one electron within its own
spin species and carries the Jordan-Wigner sign of the occupied orbitals it passes.
"""
from __future__ import annotations

import numpy as np
from itertools import combinations
from scipy.sparse import lil_matrix, kron, identity, csr_matrix
from scipy.sparse.linalg import eigsh


def _states(N, n):
    return [sum(1 << i for i in occ) for occ in combinations(range(N), n)]


def _hop_matrix(N, n, K):
    """One spin species: <a| sum_ij K_ij c^dag_i c_j |b> on the n-particle basis."""
    st = _states(N, n)
    idx = {s: k for k, s in enumerate(st)}
    H = lil_matrix((len(st), len(st)))
    for b, sb in enumerate(st):
        for j in range(N):
            if not (sb >> j) & 1:
                continue
            s1 = sb & ~(1 << j)
            sgn_j = (-1) ** bin(sb & ((1 << j) - 1)).count("1")
            for i in range(N):
                if K[i, j] == 0.0 or (s1 >> i) & 1:
                    continue
                sgn_i = (-1) ** bin(s1 & ((1 << i) - 1)).count("1")
                H[idx[s1 | (1 << i)], b] += K[i, j] * sgn_i * sgn_j
    return csr_matrix(H), st


def ground_energy(K, U, n_up, n_dn, want_vec=False):
    """Lowest eigenvalue of  sum_s K c^dag c  +  U sum_i (n_up-1/2)(n_dn-1/2)  in the sector."""
    N = K.shape[0]
    Hu, su = _hop_matrix(N, n_up, K)
    Hd, sd = _hop_matrix(N, n_dn, K)
    du, dd = len(su), len(sd)
    H = kron(Hu, identity(dd), format="csr") + kron(identity(du), Hd, format="csr")
    # the interaction is diagonal in this basis
    diag = np.empty(du * dd)
    nu = np.array([[(s >> i) & 1 for i in range(N)] for s in su], float) - 0.5
    nd = np.array([[(s >> i) & 1 for i in range(N)] for s in sd], float) - 0.5
    for a in range(du):
        diag[a * dd:(a + 1) * dd] = U * (nd @ nu[a])
    H = H + csr_matrix((diag, (np.arange(du * dd), np.arange(du * dd))), shape=H.shape)
    if H.shape[0] <= 400:
        w, v = np.linalg.eigh(H.toarray())
        return (float(w[0]), v[:, 0]) if want_vec else float(w[0])
    w, v = eigsh(H, k=1, which="SA", maxiter=20000, tol=0)
    return (float(w[0]), v[:, 0]) if want_vec else float(w[0])


def closed_shells(K, max_n=None):
    """Fillings whose free ground state is a CLOSED shell -- the free level below the Fermi
    level is fully occupied and the next is strictly higher.  An open shell makes the trial
    wavefunction ambiguous, which is not a small effect: measured on 2x2, an open shell moved
    the free-projection energy by 0.77 where a closed shell moved it by 0.003."""
    ev = np.linalg.eigvalsh(K)
    N = len(ev)
    out = []
    for n in range(1, (max_n or N) + 1):
        if n < N and ev[n] - ev[n - 1] > 1e-9:
            out.append(n)
    return out
