"""Trial wavefunctions for the constrained path, and a measure of how good each one is.

The constrained-path bias is not a property of the METHOD, it is a property of the NODE, and the
node is set by Psi_T alone.  The constraint is exact when Psi_T is the exact ground state.  So
the interesting axis is not "is CPMC biased" -- it is -- but "how does the bias move with Psi_T",
and whether it can be moved cheaply.  That makes the bias a dial rather than a tax, and it is the
one place in this whole rig where the answer is not already pinned by a proof.

Two families, both single Slater determinants so the O(N^2) rank-1 machinery is untouched:

  free   the U = 0 ground state.  Spin-symmetric, no order parameter, the standard first choice.
  UHF    unrestricted Hartree-Fock: h_sigma = K + U diag(n_{-sigma} - 1/2), solved to
         self-consistency from a staggered antiferromagnetic start.  On a bipartite lattice at
         and near half filling this is the physically right symmetry to break, and breaking it
         changes where the node sits.

The quality measure is the overlap |<Psi_T|Psi_0>| with the exact ground state, computed in the
particle-number sector.  It is reported alongside every bias so the two can be read against each
other -- a bias that tracks the overlap is the constraint behaving as advertised; a bias that
does not is a defect somewhere else.
"""
from __future__ import annotations

import numpy as np
from itertools import combinations

from sector_ed import _states, ground_energy


def free_trial(K, n_up, n_dn):
    w, v = np.linalg.eigh(K)
    return {+1: v[:, :n_up].copy(), -1: v[:, :n_dn].copy()}


def uhf_trial(K, U, n_up, n_dn, stagger, h0=1.0, iters=400, mix=0.3, tol=1e-12):
    """Self-consistent UHF from a staggered start.  Returns (psi_t, converged, m_stag)."""
    N = K.shape[0]
    nu = 0.5 + 0.5 * h0 * stagger
    nd = 0.5 - 0.5 * h0 * stagger
    conv = False
    for _ in range(iters):
        vu = np.linalg.eigh(K + U * np.diag(nd - 0.5))[1]
        vd = np.linalg.eigh(K + U * np.diag(nu - 0.5))[1]
        Pu, Pd = vu[:, :n_up], vd[:, :n_dn]
        nu2 = np.einsum("ij,ij->i", Pu, Pu)
        nd2 = np.einsum("ij,ij->i", Pd, Pd)
        d = max(np.abs(nu2 - nu).max(), np.abs(nd2 - nd).max())
        nu, nd = (1 - mix) * nu + mix * nu2, (1 - mix) * nd + mix * nd2
        if d < tol:
            conv = True
            break
    vu = np.linalg.eigh(K + U * np.diag(nd - 0.5))[1]
    vd = np.linalg.eigh(K + U * np.diag(nu - 0.5))[1]
    m_stag = float(np.mean(stagger * (nu - nd)) / 2.0)
    return {+1: vu[:, :n_up].copy(), -1: vd[:, :n_dn].copy()}, conv, m_stag


def stagger_2d(Lx, Ly):
    """(-1)^(x+y) in the site ordering model2d uses: index = x * Ly + y."""
    return np.array([(-1.0) ** (x + y) for x in range(Lx) for y in range(Ly)])


# ---- overlap of a Slater determinant with the exact sector ground state

def _det_amplitudes(P, N, n):
    """Amplitude of each occupation bitmask in the determinant with orbitals P (N x n).

    <n_1 ... n_k | det(P) > is the n x n minor of P on the occupied rows, with the sign
    convention fixed by taking the rows in increasing order -- the same order `sector_ed`
    builds its basis in, so the two are directly comparable.
    """
    st = _states(N, n)
    out = np.empty(len(st))
    for k, occ in enumerate(combinations(range(N), n)):
        out[k] = np.linalg.det(P[list(occ), :])
    return out, st


def trial_overlap(psi_t, K, U, n_up, n_dn):
    """|<Psi_T | Psi_0>| with both normalised.  1.0 means the constraint is exact."""
    N = K.shape[0]
    au, _ = _det_amplitudes(psi_t[+1], N, n_up)
    ad, _ = _det_amplitudes(psi_t[-1], N, n_dn)
    v = np.kron(au, ad)                       # sector_ed's basis is kron(up, dn), same order
    v = v / np.linalg.norm(v)
    _, g = ground_energy(K, U, n_up, n_dn, want_vec=True)
    return float(abs(v @ (g / np.linalg.norm(g))))


def trial_energy(psi_t, K, U, n_up, n_dn):
    """<Psi_T|H|Psi_T> for a single determinant -- the variational number the walk starts from."""
    e = 0.0
    G = {}
    for s, n in ((+1, n_up), (-1, n_dn)):
        P = psi_t[s]
        G[s] = P @ P.T                        # orthonormal orbitals: the density matrix
        e += float(np.sum(K * G[s].T))
    nu, nd = np.diag(G[+1]), np.diag(G[-1])
    # Wick for a single determinant: <n_up n_dn> factorises across spin species exactly
    return e + float(U * np.sum((nu - 0.5) * (nd - 0.5)))


def field_trial(K, n_up, n_dn, stagger, h):
    """One-parameter family: the ground state of K -+ (h/2) diag(stagger), spin up taking the
    minus sign.  h = 0 is the free determinant; large h is a classical Neel state.

    This is a DIAL, not a fitted constant.  Nothing here chooses an h -- the whole curve is
    reported, and the question asked of it is whether the h that a scalable criterion picks is
    the h that actually minimises the bias.  A single preferred value quoted out of this sweep
    would be a fit to exact-diagonalisation data that does not exist at the sizes that matter.
    """
    d = 0.5 * h * np.asarray(stagger, float)
    vu = np.linalg.eigh(K - np.diag(d))[1]
    vd = np.linalg.eigh(K + np.diag(d))[1]
    return {+1: vu[:, :n_up].copy(), -1: vd[:, :n_dn].copy()}
