"""Experiment ZZ -- bipartite structure versus filling, at MATCHED filling.

An uncontrolled version of this comparison produced a false finding and is the reason this file
exists.  Varying `tp` (next-nearest hopping, which destroys the bipartite structure) at fixed
`mu = 0` appeared to show that breaking the bipartite structure REMOVES the sign problem: the
non-bipartite cells locked their signs perfectly at every beta while the bipartite doped cell had
a 8.5% negative fraction.

The density was never measured.  It is:

    tp = 0.0, mu = 0.0   ->   <n> = 1.00000      half filling, pinned by particle-hole symmetry
    tp = 0.3, mu = 0.0   ->   <n> = 0.75235
    tp = 0.7, mu = 0.0   ->   <n> = 0.75109

`mu = 0` is half filling only when the symmetry that pins it is present.  Break it and the same mu
lands near three-quarter filling, where the sign problem is mild for ordinary reasons.  The whole
apparent effect was a filling comparison wearing a symmetry label.

So mu is tuned per `tp` to match the filling before anything is compared.  That is a control
variable set to a target, like a temperature -- not a fit: nothing about the identity or the sign
enters the tuning, which sees only <n>.

The question, asked properly: at the SAME filling, does the bipartite structure change whether the
identity holds and whether the signs lock?
"""
from __future__ import annotations

import numpy as np

from model2d import Model2D
from stable import udt_product, inv_one_plus_block, slogdet_one_plus_block


def measure(m, n_draw, seed):
    """(mean density per site, negative fraction, identity residual) from one sample."""
    lam = float(np.arccosh(np.exp(m.dtau * m.U / 2.0)))
    rng = np.random.default_rng(seed)
    ns, neg, res = [], 0, []
    for _ in range(n_draw):
        X = rng.choice([-1.0, 1.0], size=(m.L, m.N))
        n_tot, s, lg = 0.0, 1.0, {}
        for sigma in (+1, -1):
            d = np.exp(sigma * lam * X)
            Bl = m.expmK[None, :, :] * d[:, None, :]
            U, D, T = udt_product(Bl[None], 4)
            G = inv_one_plus_block(U, D, T)[0]
            n_tot += float(np.mean(1.0 - np.real(np.diag(G))))
            sg, la = slogdet_one_plus_block(U, D, T)
            s *= float(np.real(sg[0])); lg[sigma] = float(la[0])
        pred = -m.dtau * m.L * float(np.trace(m.K)) + lam * float(X.sum())
        res.append(abs((lg[+1] - lg[-1]) - pred))
        ns.append(n_tot); neg += int(s < 0)
    return float(np.mean(ns)), neg / n_draw, np.array(res)


def free_density(Lx, Ly, tp, beta, mu):
    """Non-interacting density per site at (beta, mu): 2/N sum_k 1/(1 + exp(beta(eps_k - mu))).

    Exact and noise-free.  A first version of this control bisected on a density MEASURED from 60
    uniform field samples, which is two things wrong at once: the estimate carries sampling noise,
    so the bisection converged on a fluctuation (rows landed at <n> = 0.98 to 1.06 when they were
    all supposed to be 1.0); and an unweighted mean over uniform configurations is not the density
    anyway, while the weighted one has an effective sample size of 1 to 8 out of 400 here.

    The free density is a control variable, not a measurement of the interacting system: it is a
    known function of the single-particle spectrum, it is computed WITHOUT touching the sign, and
    it fixes the same reference point for every tp.  That is what a control has to do.
    """
    from model2d import hop_2d
    eps = np.linalg.eigvalsh(hop_2d(Lx, Ly, t=1.0, tp=tp, mu=0.0))
    return 2.0 * float(np.mean(1.0 / (1.0 + np.exp(beta * (eps - mu)))))


def tune_mu(Lx, Ly, tp, beta, target=1.0):
    """Bisect mu so the FREE density hits `target`.  Exact, deterministic, sign-blind."""
    lo, hi = -8.0, 8.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if free_density(Lx, Ly, tp, beta, mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


if __name__ == "__main__":
    Lx, Ly, U, dtau, beta = 2, 4, 4.0, 0.125, 8.0
    L = int(round(beta / dtau))
    n_draw = 400
    print("=" * 104)
    print(f"BIPARTITE OR FILLING?  AT MATCHED FILLING   {Lx}x{Ly}, U = {U}, beta = {beta}")
    print("mu is tuned per tp so every row sits at the same density.  The tuner sees only <n>.")
    print()
    print(f"{'tp':>5} {'tuned mu':>9} {'n_free':>8} {'bipartite':>10} | {'resid max':>12} "
          f"{'identity':>9} | {'signs lock':>11} {'neg frac':>9}")
    for tp in (0.0, 0.15, 0.3, 0.7):
        mu = tune_mu(Lx, Ly, tp, beta, target=1.0)
        m = Model2D(Lx=Lx, Ly=Ly, t=1.0, tp=tp, mu=mu, U=U, dtau=dtau, L=L, theta=0.0)
        n, neg, res = measure(m, n_draw, seed=int(tp * 100) + 1)
        sub = np.array([(-1.0) ** (x + y) for x in range(Lx) for y in range(Ly)])
        off = m.K - np.diag(np.diag(m.K))
        bip = float(np.abs(off[np.outer(sub, sub) > 0]).max()) < 1e-12
        holds = "EXACT" if res.max() < 1e-10 else "broken"
        nf = free_density(Lx, Ly, tp, beta, mu)
        print(f"{tp:5.2f} {mu:9.4f} {nf:8.5f} {str(bip):>10} | {res.max():12.3e} {holds:>9} | "
              f"{1-neg:11.4f} {neg:9.4f}", flush=True)
    print()
    print("Every row is at the same density, so a difference between them is the bipartite")
    print("structure and nothing else.  That is the comparison the uncontrolled version could")
    print("not make.")
