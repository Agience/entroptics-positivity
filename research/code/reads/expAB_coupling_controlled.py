"""Experiment AB -- the order parameter on a CONTROLLED axis.

The coupling read has so far been measured along beta, where the filling is fixed but everything
else moves together: the correlation time, the weight's dynamic range, the density of spectral
crossings.  A quantity that tracks the sign problem along that axis might be tracking any of them.

`expZZ` built a cleaner axis.  Tuning `mu` per `tp` against the NON-INTERACTING density -- a
closed-form function of the single-particle spectrum, carrying no sampling noise and computed
without touching the sign -- fixes every cell at `n = 1.00000` while the bipartite structure
varies.  Along that axis the sign problem switches on:

    tp        0.00      0.15      0.30      0.70
    neg frac  0.0000    0.0500    0.1850    0.0800

and the negative fraction is NOT monotone in tp, which makes it a real test: a read that merely
increases with tp would fail it, while a read that tracks the sign problem has to reproduce the
non-monotonicity.

Measured here, on the same configurations that produce the negative fraction:

  * the coupling between the two channels' Green's-function diagonals, with its own exact
    re-pairing null and no constant supplied;
  * the permuted control, which must sit at |z| ~ 1 wherever the read is meaningful.

The prediction the read has to survive is specific: `strength` should be exactly -1 at tp = 0,
where the identity is exact and the negative fraction is zero, and should depart NON-MONOTONICALLY
with tp, following 0.0500, 0.1850, 0.0800 rather than tp itself.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

import entroptics as E
from model2d import Model2D
from stable import udt_product, inv_one_plus_block, slogdet_one_plus_block
from expZZ_matched_filling import tune_mu, free_density


def channels_and_sign(m, n_draw, seed):
    """(A, B, signs) -- the two channels' diagonals per configuration, and the weight's sign."""
    lam = float(np.arccosh(np.exp(m.dtau * m.U / 2.0)))
    rng = np.random.default_rng(seed)
    A, B, S = [], [], []
    for _ in range(n_draw):
        X = rng.choice([-1.0, 1.0], size=(m.L, m.N))
        row, s = {}, 1.0
        for sigma in (+1, -1):
            d = np.exp(sigma * lam * X)
            Bl = m.expmK[None, :, :] * d[:, None, :]
            U, D, T = udt_product(Bl[None], 4)
            row[sigma] = np.real(np.diag(inv_one_plus_block(U, D, T)[0])).copy()
            sg, _ = slogdet_one_plus_block(U, D, T)
            s *= float(np.real(sg[0]))
        A.append(row[+1]); B.append(row[-1]); S.append(s)
    return np.array(A), np.array(B), np.array(S)


if __name__ == "__main__":
    Lx, Ly, U, dtau, beta = 2, 4, 4.0, 0.125, 8.0
    L = int(round(beta / dtau))
    n_draw = 400
    rng = np.random.default_rng(17)
    print("=" * 108)
    print(f"THE ORDER PARAMETER ON A CONTROLLED AXIS   {Lx}x{Ly}, U = {U}, beta = {beta}")
    print("Every row at n = 1.00000 by the free-density control; only the bipartite structure")
    print("varies.  The negative fraction is NON-MONOTONE in tp, so a read that merely grows")
    print("with tp fails this test.")
    print()
    print(f"{'tp':>5} {'mu':>8} {'n_free':>8} | {'neg frac':>9} | {'strength':>9} "
          f"{'deficit':>9} {'z':>9} {'tight':>7} | {'control z':>10}")
    rows = []
    for tp in (0.0, 0.15, 0.3, 0.7):
        mu = tune_mu(Lx, Ly, tp, beta, target=1.0)
        m = Model2D(Lx=Lx, Ly=Ly, t=1.0, tp=tp, mu=mu, U=U, dtau=dtau, L=L, theta=0.0)
        A, B, S = channels_and_sign(m, n_draw, seed=int(tp * 100) + 3)
        c = E.reads.coupling(A, B)
        cn = E.reads.coupling(A, B[rng.permutation(len(B))])
        neg = float(np.mean(S < 0))
        rows.append((tp, neg, 1.0 + float(c.strength)))
        print(f"{tp:5.2f} {mu:8.4f} {free_density(Lx,Ly,tp,beta,mu):8.5f} | {neg:9.4f} | "
              f"{c.strength:9.4f} {1+c.strength:9.4f} {c.z:9.2f} {c.tightness:7.3f} | "
              f"{cn.z:10.2f}", flush=True)

    print()
    print("=" * 108)
    neg = np.array([r[1] for r in rows]); def_ = np.array([r[2] for r in rows])
    rho, p = spearmanr(def_, neg)
    print(f"Spearman(coupling deficit, negative fraction) over the controlled axis: "
          f"{rho:+.3f}  p = {p:.4f}")
    print(f"  deficits         {np.round(def_, 4)}")
    print(f"  negative fracs   {np.round(neg, 4)}")
    print()
    print("The negative fraction is non-monotone in tp.  A read that reproduces that ordering is")
    print("tracking the sign problem; one that simply rises with tp is tracking tp.")
