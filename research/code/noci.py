"""Non-orthogonal configuration interaction: a multi-determinant trial built WITHOUT the answer.

`expKK` measured the ceiling -- four to six non-orthogonal determinants cut the constrained-path
bias by 3 to 9x -- using determinants fitted to the exact ground state, which nobody has.  This is
the machinery for getting comparable ones without it.

Two halves, and neither needs exact diagonalisation:

  WHERE THE DETERMINANTS COME FROM.  A mean-field solution that breaks a symmetry the true ground
  state respects is wrong in a specific, repairable way: the symmetry-related copies of it are
  degenerate with it and span the space it should have occupied.  A Neel state on a bipartite
  lattice breaks translation by one site, and translating it gives the partner that restores it.
  So the generator is: solve a broken mean field once, then apply the symmetry group.  That
  naturally produces 4 to 8 determinants -- exactly the range the ceiling study found pays.

  WHERE THE COEFFICIENTS COME FROM.  Not a fit, and not a tuned parameter anywhere.  The field
  strengths that generate the mean fields are not SELECTED either -- determinants from several of
  them all go into the basis and the eigenproblem weights them, because a basis is not a fit:
  adding vectors can only lower a variational energy, so nothing is being forced toward a target.
  In a non-orthogonal basis {phi_j} the variational
  ground state solves the generalised eigenproblem

      H c = E S c,     S_jk = <phi_j|phi_k>,     H_jk = <phi_j|H|phi_k>

  whose lowest root is an upper bound on the true ground energy by the variational principle.
  That bound is the gate: an NOCI energy BELOW the exact one means the matrix elements are wrong.

Transition matrix elements between non-orthogonal Slater determinants use the transition density
matrix rho^sigma = B (A' B)^-1 A', for which

    <A|phi_k>            = det(A_up' B_up) det(A_dn' B_dn)
    <A|H|B> / <A|B>      = sum_sigma tr(K rho^sigma) + U sum_i (rho^up_ii - 1/2)(rho^dn_ii - 1/2)

The two-body term factorises across spin species HERE, unlike in `cpmc_multi.energy`, because
each side is a SINGLE determinant -- the factorisation fails only for a sum.  Getting that
backwards cost a 6-31% error there, so it is stated rather than assumed.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import eigh


def transition(A, B):
    """(overlap, rho) for one spin: rho = B (A' B)^-1 A', overlap = det(A' B)."""
    O = A.T @ B
    d = float(np.linalg.det(O))
    if abs(d) < 1e-300:
        return d, None
    return d, B @ np.linalg.solve(O, A.T)


def h_and_s(dets, K, U):
    """The k x k overlap and Hamiltonian matrices in a non-orthogonal determinant basis."""
    k = len(dets)
    S = np.zeros((k, k))
    H = np.zeros((k, k))
    for a in range(k):
        for b in range(k):
            du, ru = transition(dets[a][+1], dets[b][+1])
            dd, rd = transition(dets[a][-1], dets[b][-1])
            s = du * dd
            S[a, b] = s
            if ru is None or rd is None:
                continue
            e = float(np.sum(K * ru.T)) + float(np.sum(K * rd.T))
            e += float(U * np.sum((np.diag(ru) - 0.5) * (np.diag(rd) - 0.5)))
            H[a, b] = s * e
    return 0.5 * (H + H.T), 0.5 * (S + S.T)


def noci(dets, K, U):
    """Lowest variational state in the span of `dets`.  Returns (energy, coefficients).

    NOTHING HERE IS FITTED.  The coefficients solve H c = E S c -- they come out of the
    Hamiltonian, and the lowest root is a variational UPPER BOUND on the true ground energy.  That
    is the difference between this and the ceiling study, which optimised determinants to maximise
    overlap with an exact ground state nobody has.

    AND NOTHING HERE IS A CHOSEN THRESHOLD.  S is singular whenever two determinants are linearly
    dependent, which symmetry generation produces routinely -- some group elements leave a
    determinant unchanged.  The null directions have to be dropped, and the level at which a
    direction counts as null is not a preference: it is the resolution float64 can carry on a
    k x k Gram matrix, eps * k * lambda_max.  Below that the direction is not small, it is absent.
    """
    H, S = h_and_s(dets, K, U)
    w, V = np.linalg.eigh(S)
    rank_floor = float(np.finfo(float).eps) * len(w) * max(float(w.max()), 0.0)
    keep = w > rank_floor
    if keep.sum() == 0:
        raise RuntimeError("the determinant basis spans nothing")
    X = V[:, keep] / np.sqrt(w[keep])
    Hp = X.T @ H @ X
    e, C = np.linalg.eigh(Hp)
    c = X @ C[:, 0]
    return float(e[0]), c


# ---- symmetry generators: where the determinants come from

def site_index(Lx, Ly):
    return lambda x, y: (x % Lx) * Ly + (y % Ly)


def translation_perm(Lx, Ly, dx, dy):
    """The site permutation for translating by (dx, dy)."""
    idx = site_index(Lx, Ly)
    p = np.empty(Lx * Ly, dtype=int)
    for x in range(Lx):
        for y in range(Ly):
            p[idx(x, y)] = idx(x + dx, y + dy)
    return p


def translate(det, perm):
    """Apply a site permutation to both spin blocks of a determinant."""
    return {s: det[s][perm].copy() for s in (+1, -1)}


def spin_flip(det):
    return {+1: det[-1].copy(), -1: det[+1].copy()}


def symmetry_family(det, Lx, Ly, *, translations=True, flip=True):
    """The determinant's orbit under translations and the spin flip, duplicates included.

    Duplicates are NOT filtered here: `noci` drops the null directions of S, which is the same
    thing done in the place that can actually see linear dependence rather than guessing at it
    from the generators.
    """
    out = []
    shifts = [(dx, dy) for dx in range(Lx) for dy in range(Ly)] if translations else [(0, 0)]
    for dx, dy in shifts:
        p = translation_perm(Lx, Ly, dx, dy)
        d = translate(det, p)
        out.append(d)
        if flip:
            out.append(spin_flip(d))
    return out
