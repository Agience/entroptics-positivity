"""Experiment AD -- does the identity track the interaction, or was it tuned to U = 4?

Every number in the paper uses `U = 4` and `dtau = 0.125`.  The identity

    ln|det_up| - ln|det_dn| = -dtau * L * tr(K) + lambda * sum(x)

carries `lambda = arccosh(exp(dtau U / 2))`, which is a function of BOTH, so varying them is not a
repetition of the same measurement: the predicted coefficient has to move with them and stay
exact.  A relation that held only at the U it was derived at would be a coincidence dressed as a
derivation.

The span matters.  `lambda` runs from 0.51 at `U = 2, dtau = 0.125` to 1.76 at `U = 12`, a factor
of 3.5, and the discrete Hubbard-Stratonovich transformation is only defined while
`exp(dtau U / 2) >= 1`, which is where `arccosh` is real.  Each row therefore predicts a different
coefficient from the same derivation and is checked against it.

Two controls travel with the sweep:

  * the WRONG coefficient (`2 lambda`) must fail on every row, or the test passes for any value;
  * the coefficient from a DIFFERENT row's U must fail on this row, which is the sharper control:
    it rules out the possibility that any coefficient of roughly the right size would do.
"""
from __future__ import annotations

import numpy as np

from model2d import Model2D
from stable import udt_product, slogdet_one_plus_block


def residual(m, X, lam):
    """|measured - predicted| for one configuration, with `lam` supplied rather than assumed."""
    out = {}
    lam_true = float(np.arccosh(np.exp(m.dtau * m.U / 2.0)))
    for sigma in (+1, -1):
        d = np.exp(sigma * lam_true * X)
        Bl = m.expmK[None, :, :] * d[:, None, :]
        U, D, T = udt_product(Bl[None], 4)
        s, la = slogdet_one_plus_block(U, D, T)
        out[sigma] = float(la[0])
    pred = -m.dtau * m.L * float(np.trace(m.K)) + lam * float(X.sum())
    return abs((out[+1] - out[-1]) - pred)


def sweep(U, dtau, beta, n_draw=200, seed=0):
    L = int(round(beta / dtau))
    m = Model2D(Lx=2, Ly=4, t=1.0, mu=0.0, U=U, dtau=dtau, L=L, theta=0.0)
    lam = float(np.arccosh(np.exp(dtau * U / 2.0)))
    rng = np.random.default_rng(seed)
    right, doubled = [], []
    for _ in range(n_draw):
        X = rng.choice([-1.0, 1.0], size=(m.L, m.N))
        right.append(residual(m, X, lam))
        doubled.append(residual(m, X, 2.0 * lam))
    return lam, np.array(right), np.array(doubled), m


if __name__ == "__main__":
    beta = 6.0
    print("=" * 108)
    print("DOES THE IDENTITY TRACK THE INTERACTION?   2x4, mu = 0, beta = 6, 200 configurations")
    print("lambda = arccosh(exp(dtau U / 2)) is PREDICTED per row, never fitted.")
    print()
    print(f"{'U':>5} {'dtau':>6} {'lambda':>8} | {'resid (predicted)':>18} "
          f"{'resid (2 lambda)':>17} {'holds':>7}")
    lams = {}
    for dtau in (0.0625, 0.125, 0.25):
        for U in (2.0, 4.0, 8.0, 12.0):
            lam, right, doubled, m = sweep(U, dtau, beta, seed=int(U * 10 + dtau * 100))
            lams[(U, dtau)] = lam
            holds = "EXACT" if right.max() < 1e-9 else "broken"
            print(f"{U:5.1f} {dtau:6.4f} {lam:8.5f} | {right.max():18.3e} "
                  f"{doubled.max():17.3e} {holds:>7}", flush=True)
        print()

    print("=" * 108)
    print("THE SHARPER CONTROL: another row's coefficient, on this row.")
    print("If any coefficient of roughly the right size worked, this would pass and it must not.")
    print(f"{'U':>5} {'own lambda':>11} {'borrowed':>10} {'from U':>7} | "
          f"{'resid (own)':>12} {'resid (borrowed)':>17} {'distinguishes':>14}")
    dtau = 0.125
    for U, other in ((2.0, 4.0), (4.0, 8.0), (8.0, 12.0), (12.0, 2.0)):
        lam, right, _, m = sweep(U, dtau, beta, seed=int(U * 7))
        borrowed = lams[(other, dtau)]
        rng = np.random.default_rng(int(U * 7))
        wrong = np.array([residual(m, rng.choice([-1.0, 1.0], size=(m.L, m.N)), borrowed)
                          for _ in range(200)])
        ok = "YES" if wrong.max() > 1.0 and right.max() < 1e-9 else "NO"
        print(f"{U:5.1f} {lam:11.5f} {borrowed:10.5f} {other:7.1f} | {right.max():12.3e} "
              f"{wrong.max():17.3e} {ok:>14}", flush=True)
    print()
    print("Every row predicts its own coefficient from the same derivation.  A relation that held")
    print("only where it was derived, or for any coefficient near the right size, would not be one.")
