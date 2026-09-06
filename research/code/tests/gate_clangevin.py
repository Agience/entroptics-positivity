"""Exact references for the complex-Langevin gates: the truth, and the action.

This file holds ONLY what the gates measure against.  The gates themselves are `gate_main.py`;
an earlier version of this file carried a second copy of them, which is the shape the workspace
rules exist to prevent -- two drivers drift, and the one you are not running is the one that
still passes.

  exact_observables   <n> and <n_up n_dn> per site, by a full-Fock-space trace over the SAME
                      Trotter product the sampler represents.  Trotter error therefore cancels
                      between the two sides and a disagreement is the method, not the splitting.
  action              S(X) per chain, used ONLY by the finite-difference gate.  The sampler never
                      forms the action, because for a complex field only the DRIFT is
                      single-valued: ln det has a branch cut and tr(M^-1 dM) does not.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import expm

from dqmc import Model, ExactTrotter, fock_operators
from clangevin import CLangevin, run_cl


def exact_observables(m):
    """<n> per site and <n_up n_dn> per site for the Trotterised model, by full trace."""
    et = ExactTrotter(m)
    c = et.c
    dim = 2 ** (2 * m.N)
    TL = et.Tp[m.L]
    Z = float(np.trace(TL).real)
    n = [ci.T @ ci for ci in c]
    ntot = np.zeros((dim, dim))
    docc = np.zeros((dim, dim))
    for i in range(m.N):
        ntot += n[i] + n[m.N + i]
        docc += n[i] @ n[m.N + i]
    return (float(np.trace(ntot @ TL).real) / (Z * m.N),
            float(np.trace(docc @ TL).real) / (Z * m.N))


def action(ch, X):
    """S(X) per chain, shape (C,).  Used ONLY by the finite-difference gate; the sampler never
    forms the action, because only the DRIFT is single-valued for a complex field -- ln det has
    a branch cut and the gradient tr(M^-1 dM) does not."""
    m = ch.m
    X = np.atleast_3d(X) if X.ndim == 3 else X[None]
    C = X.shape[0]
    s = 0.5 * np.sum(X * X, axis=(1, 2))
    if ch.channel == "charge":
        s = s + 1j * ch.lam * np.sum(X, axis=(1, 2))
    for sigma in (+1, -1):
        d = np.exp(ch._coupling(sigma) * X)
        B = np.broadcast_to(np.eye(m.N, dtype=complex), (C, m.N, m.N)).copy()
        for l in range(m.L):
            B = (ch.expmK[None] * d[:, l][:, None, :]) @ B
        sg, ld = np.linalg.slogdet(np.eye(m.N)[None] + B)
        s = s - (np.log(sg) + ld)
    return s
