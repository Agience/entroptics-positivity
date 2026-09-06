"""UDT-stratified products for the DQMC chain.

A naive product B(L-1)...B(0) of the slice matrices has condition number
~exp(beta * bandwidth(K)), which passes 1e16 around beta ~ 4 here and makes
(I + B)^{-1} singular in float64.  The standard cure is to carry the product in
factored form U D T (U orthogonal, D positive diagonal, T well-conditioned),
re-orthogonalising every n_stab slices, and to form the determinant and the
Green's function from the factors with the large/small split:

    I + U D T = U Db (Db^-1 U^T T^-1 + Ds) T,    Db = max(D,1), Ds = min(D,1)

    det(I + B)   = det(U) det(Db) det(M) det(T),   M = Db^-1 U^T T^-1 + Ds
    (I + B)^-1   = T^-1 M^-1 Db^-1 U^T

Batched over R chains.
"""
from __future__ import annotations

import numpy as np


def udt_product(Bl: np.ndarray, n_stab: int = 8):
    """Bl: (R, L, N, N) slice matrices, product taken as B(L-1) ... B(0).
    Returns (U, D, T) with B = U diag(D) T."""
    R, L, N, _ = Bl.shape
    U = np.broadcast_to(np.eye(N), (R, N, N)).copy()
    D = np.ones((R, N))
    T = np.broadcast_to(np.eye(N), (R, N, N)).copy()
    l = 0
    while l < L:
        blk = min(n_stab, L - l)
        M = Bl[:, l]
        for k in range(1, blk):
            M = Bl[:, l + k] @ M
        M = (M @ U) * D[:, None, :]
        Q, Rm = np.linalg.qr(M)
        d = np.abs(np.einsum('rii->ri', Rm))
        d = np.where(d > 0, d, 1e-300)
        T = ((Rm / d[:, :, None]) @ T)
        U, D = Q, d
        l += blk
    return U, D, T


def slogdet_one_plus(U, D, T):
    """sign, log|det| of (I + U diag(D) T)."""
    Db = np.maximum(D, 1.0)
    Ds = np.minimum(D, 1.0)
    Tinv = np.linalg.pinv(T)          # pinv, not inv: at unbounded (continuous) fields a QR
                                      # block can produce a near-zero diagonal and T goes singular.
    M = (np.swapaxes(U, 1, 2) @ Tinv) / Db[:, :, None] + np.einsum('ri,ij->rij', Ds, np.eye(D.shape[1]))
    s1, l1 = np.linalg.slogdet(U)
    s2, l2 = np.linalg.slogdet(T)
    s3, l3 = np.linalg.slogdet(M)
    return s1 * s2 * s3, l1 + l2 + l3 + np.log(Db).sum(1)


def inv_one_plus(U, D, T):
    """(I + U diag(D) T)^{-1}, computed from the factors."""
    Db = np.maximum(D, 1.0)
    Ds = np.minimum(D, 1.0)
    Tinv = np.linalg.pinv(T)          # pinv, not inv: at unbounded (continuous) fields a QR
                                      # block can produce a near-zero diagonal and T goes singular.
    M = (np.swapaxes(U, 1, 2) @ Tinv) / Db[:, :, None] + np.einsum('ri,ij->rij', Ds, np.eye(D.shape[1]))
    return Tinv @ np.linalg.inv(M) @ (np.swapaxes(U, 1, 2) / Db[:, :, None])


def core_matrix(U, D, T):
    """The well-conditioned N x N matrix the sign actually lives in.

        I + U diag(D) T = U Db (Db^-1 U^T T^-1 + Ds) T = U Db M T

    so  det(I + B) = det(U) det(T) det(M) * prod(Db)  with prod(Db) > 0, and therefore

        sgn det(I + B) = sgn det(U) * sgn det(T) * sgn det(M).

    `M` is bounded by construction (Db >= 1 divides the large block, Ds <= 1 is the small
    one), while `I + B` itself has condition ~exp(beta * bandwidth) and its eigenvalues stop
    meaning anything past beta ~ 4.  So M is the right frame to read: it carries the whole
    sign and it stays conditioned."""
    Db = np.maximum(D, 1.0)
    Ds = np.minimum(D, 1.0)
    Tinv = np.linalg.pinv(T)          # pinv, not inv: at unbounded (continuous) fields a QR
                                      # block can produce a near-zero diagonal and T goes singular.
    M = (np.swapaxes(U, 1, 2) @ Tinv) / Db[:, :, None] \
        + np.einsum('ri,ij->rij', Ds, np.eye(D.shape[1]))
    sU = np.linalg.slogdet(U)[0]
    sT = np.linalg.slogdet(T)[0]
    sM = np.linalg.slogdet(M)[0]
    return M, sU * sT * sM


# ── the block formulation: never form T^-1 ───────────────────────────────────
#
# `inv_one_plus` / `core_matrix` above use
#     I + UDT = U Db (Db^-1 U^H T^-1 + Ds) T
# which needs T^-1.  cond(T) reaches 2.7e10 at N=16 in this rig, and a single Green's-function
# wrap then came out wrong by 6.6e3, which voided every N >= 12 row of the severity map.
#
# The same split factors the other way and drops the inverse entirely:
#
#     I + U D T = U Db (Db^-1 U^H + Ds T)                       [ U U^H = I ]
#
# so, with  M = Db^-1 U^H + Ds T  (bounded: Db^-1 <= 1 and Ds <= 1),
#
#     (I + UDT)^-1 = M^-1 Db^-1 U^H
#     det(I + UDT) = det(U) * prod(Db) * det(M)                 [ prod(Db) > 0 ]
#
# T appears only as a multiplicand.  This is the standard block/Schur formulation.

def _core(U, D, T):
    """M = Db^-1 U^H + Ds T, and Db, for the block formulation.  Complex-safe."""
    Db = np.maximum(D, 1.0)
    Ds = np.minimum(D, 1.0)
    Uh = np.conj(np.swapaxes(U, 1, 2))
    return Uh / Db[:, :, None] + Ds[:, :, None] * T, Db


def inv_one_plus_block(U, D, T):
    """(I + U diag(D) T)^{-1} without inverting T."""
    M, Db = _core(U, D, T)
    Uh = np.conj(np.swapaxes(U, 1, 2))
    return np.linalg.solve(M, Uh / Db[:, :, None])


def slogdet_one_plus_block(U, D, T):
    """sign, log|det| of (I + U diag(D) T) without inverting T."""
    M, Db = _core(U, D, T)
    sU, _ = np.linalg.slogdet(U)
    sM, lM = np.linalg.slogdet(M)
    return sU * sM, lM + np.log(Db).sum(1)


def sign_one_plus_block(U, D, T):
    """Just the sign -- the cheap half of the above."""
    return slogdet_one_plus_block(U, D, T)[0]
