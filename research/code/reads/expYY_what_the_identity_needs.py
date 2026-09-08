"""Experiment YY -- what the identity actually needs: bipartite structure, half filling, or both?

`expWW` verified the identity that carries positivity,

    ln|det_up(x)| - ln|det_dn(x)| = -dtau * L * tr(K) + lambda * sum(x)

to 1e-14 -- but every row where it held had BOTH a bipartite lattice AND mu = 0.  Those two are
confounded in all the data so far, so "the identity needs particle-hole symmetry" is a reading of
the result, not a measurement of it.

The 2 x 2 that separates them.  Next-nearest-neighbour hopping `tp` connects sites on the SAME
sublattice, so it destroys the bipartite structure while leaving everything else in place; mu
moves the filling.  Varying them independently gives four cells, and the identity's residual in
each says which ingredient it needs:

    tp = 0, mu = 0     bipartite, half filled     -- the verified case
    tp = 0, mu != 0    bipartite, doped
    tp != 0, mu = 0    not bipartite, mu = 0
    tp != 0, mu != 0   neither

A residual at machine precision in exactly one cell means the identity needs both.  A residual at
machine precision in two means one ingredient is doing the work alone, and the paper's account has
to name that one rather than the pair.

Reported alongside: the sign-lock fraction, since the identity's whole significance is that it
forces the two determinants to share a sign.  If a cell breaks the identity but keeps the lock,
the identity is sufficient and not necessary, which is a different claim from the one being made.
"""
from __future__ import annotations

import numpy as np

from model2d import Model2D
from stable import udt_product, slogdet_one_plus_block


def log_dets(m, X, block=4):
    lam = float(np.arccosh(np.exp(m.dtau * m.U / 2.0)))
    out = {}
    for sigma in (+1, -1):
        d = np.exp(sigma * lam * X)
        Bl = m.expmK[None, :, :] * d[:, None, :]
        U, D, T = udt_product(Bl[None], block)
        s, la = slogdet_one_plus_block(U, D, T)
        out[sigma] = (float(np.real(s[0])), float(la[0]))
    return out


def is_bipartite(m, Lx, Ly):
    """Zero hopping within a sublattice is the definition, checked rather than assumed."""
    sub = np.array([(-1.0) ** (x + y) for x in range(Lx) for y in range(Ly)])
    return float(np.abs(m.K[np.outer(sub, sub) > 0] -
                        np.diag(np.diag(m.K))[np.outer(sub, sub) > 0]).max())


if __name__ == "__main__":
    Lx, Ly, U, dtau, beta = 2, 4, 4.0, 0.125, 4.0
    L = int(round(beta / dtau))
    n_draw = 300
    lam = float(np.arccosh(np.exp(dtau * U / 2.0)))
    print("=" * 108)
    print(f"WHAT THE IDENTITY NEEDS   {Lx}x{Ly}, U = {U}, beta = {beta}, {n_draw} configurations")
    print("tp breaks the bipartite structure; mu moves the filling.  They are varied separately")
    print("because every row that verified the identity so far had both, and could not tell them")
    print("apart.")
    print()
    print(f"{'tp':>5} {'mu':>5} {'bipartite':>10} | {'residual max':>13} {'residual med':>13} "
          f"{'identity':>9} | {'signs lock':>11} {'neg frac':>9}")
    for tp in (0.0, 0.3, 0.7):
        for mu in (0.0, 0.4):
            m = Model2D(Lx=Lx, Ly=Ly, t=1.0, tp=tp, mu=mu, U=U, dtau=dtau, L=L, theta=0.0)
            bip = is_bipartite(m, Lx, Ly) < 1e-12
            rng = np.random.default_rng(int(tp * 100 + mu * 10))
            res, lock, neg = [], 0, 0
            for _ in range(n_draw):
                X = rng.choice([-1.0, 1.0], size=(m.L, m.N))
                o = log_dets(m, X)
                pred = -dtau * L * float(np.trace(m.K)) + lam * float(X.sum())
                res.append(abs((o[+1][1] - o[-1][1]) - pred))
                lock += int(o[+1][0] == o[-1][0])
                neg += int(o[+1][0] * o[-1][0] < 0)
            res = np.array(res)
            holds = "EXACT" if res.max() < 1e-10 else "broken"
            print(f"{tp:5.2f} {mu:5.2f} {str(bip):>10} | {res.max():13.4e} "
                  f"{np.median(res):13.4e} {holds:>9} | {lock/n_draw:11.4f} {neg/n_draw:9.4f}",
                  flush=True)
    print()
    print("The cell(s) reading EXACT are the ones the identity needs.  If a cell breaks the")
    print("identity and still locks the signs, the identity is sufficient but not necessary and")
    print("the account has to say so.")
