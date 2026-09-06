"""The constrained-path trade, measured: what the constraint COSTS and what it BUYS.

Two numbers decide whether CPMC answers the stated criterion ("2-D, decoupling, not necessarily
sign-free, fast").  Neither is worth anything without the other:

  COST  the constrained-path bias -- the systematic error from killing walkers at the trial node.
        It is confounded with Trotter error, which is O(dtau^2) and NOT the constraint's fault,
        so it is only meaningful after a dtau -> 0 extrapolation.  Both branches are extrapolated
        on the SAME dtau grid; free projection carries no constraint bias, so its residual is the
        size of everything else, and only CPMC-minus-free is chargeable to the constraint.

  BUY   free projection's average sign as beta grows.  Free projection is exact and carries the
        sign; CPMC clips it.  If <sgn> stays near 1 there is nothing to buy.  If it decays
        exponentially while CPMC's answer stays put, the trade is real and priced.

Runs on `cpmc_fast`, which is gated bit-identical to the scalar reference at one walker
(`gate_fast.py`, worst 5.9e-14 over four legs, negative control 3.2e+01).
"""
from __future__ import annotations

import numpy as np

from model2d import Model2D
from sector_ed import ground_energy
from cpmc_fast import run


def extrapolate(m_of, nu, nd, beta, dtaus, *, constrained, seeds=(1, 2, 3), n_walkers=400):
    """Fit E(dtau) = E0 + c dtau^2 -- the Trotter law for a symmetric splitting.

    The error on each point is the spread over independent seeds, not the within-run block
    error: at fixed dtau the walk is one correlated chain, and its internal scatter understates
    how far the answer moves when the chain is restarted.
    """
    xs, ys, es, rows = [], [], [], []
    for dt in dtaus:
        m = m_of(dt)
        r = [run(m, nu, nd, beta, constrained=constrained, n_walkers=n_walkers, seed=s)
             for s in seeds]
        e = np.array([q["e"] for q in r])
        se = float(e.std(ddof=1) / np.sqrt(len(seeds)))
        xs.append(dt ** 2); ys.append(float(e.mean())); es.append(max(se, 1e-9))
        rows.append((dt, float(e.mean()), se, float(np.mean([q["sgn"] for q in r]))))
    A = np.vstack([np.ones(len(xs)), xs]).T
    W = np.diag(1.0 / np.array(es) ** 2)
    cov = np.linalg.inv(A.T @ W @ A)
    coef = cov @ A.T @ W @ np.array(ys)
    return float(coef[0]), float(np.sqrt(cov[0, 0])), rows


if __name__ == "__main__":
    Lx, Ly, nu, nd = 2, 4, 3, 3
    dtaus = [0.10, 0.05, 0.025]
    mk = lambda U: (lambda dt: Model2D(Lx=Lx, Ly=Ly, t=1.0, mu=0.0, U=U, dtau=dt, L=1, theta=0.0))

    print("=" * 84)
    print("COST  constrained-path bias after Trotter extrapolation, beta = 8, 2x4 at (3,3)")
    print(f"{'U':>4} {'exact':>11} {'free E0':>12} {'free-ex':>9} "
          f"{'CPMC E0':>12} {'CPMC-ex':>9} {'CP bias':>9} {'rel':>8}")
    for U in (2.0, 4.0, 8.0):
        m_of = mk(U)
        ex = ground_energy(m_of(0.05).K, U, nu, nd)
        f0, fse, frows = extrapolate(m_of, nu, nd, 8.0, dtaus, constrained=False)
        c0, cse, crows = extrapolate(m_of, nu, nd, 8.0, dtaus, constrained=True)
        print(f"{U:4.1f} {ex:+11.5f} {f0:+12.5f} {f0-ex:+9.5f} "
              f"{c0:+12.5f} {c0-ex:+9.5f} {c0-f0:+9.5f} {abs(c0-f0)/abs(ex):8.5f}", flush=True)
        for (dt, y, s, sg), (_, y2, s2, _) in zip(frows, crows):
            print(f"       dtau={dt:5.3f}   free {y:+.5f}+-{s:.5f} <sgn>={sg:.4f}"
                  f"   CPMC {y2:+.5f}+-{s2:.5f}")

    print()
    print("=" * 84)
    print("BUY   free projection's average sign vs beta, against CPMC's stability.  U = 4")
    print(f"{'beta':>5} {'<sgn> free':>11} {'free E':>19} {'CPMC E':>19} {'exact':>11}")
    m = mk(4.0)(0.05)
    ex = ground_energy(m.K, 4.0, nu, nd)
    for beta in (2.0, 4.0, 8.0, 12.0, 16.0, 24.0):
        fr = [run(m, nu, nd, beta, constrained=False, n_walkers=400, seed=s) for s in (1, 2, 3)]
        cr = [run(m, nu, nd, beta, constrained=True, n_walkers=400, seed=s) for s in (1, 2, 3)]
        fe = np.array([q["e"] for q in fr]); ce = np.array([q["e"] for q in cr])
        sg = float(np.mean([q["sgn"] for q in fr]))
        print(f"{beta:5.1f} {sg:11.5f} {fe.mean():+12.5f}+-{fe.std(ddof=1)/np.sqrt(3):.5f} "
              f"{ce.mean():+12.5f}+-{ce.std(ddof=1)/np.sqrt(3):.5f} {ex:+11.5f}", flush=True)
