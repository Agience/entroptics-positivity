"""Experiment SS -- can the coupling, read where there is NO sign problem, predict one that has
not appeared yet?

This is the capability the tool is claimed for, pointed at the sign problem: a rate read off clean
data and carried outward.  `expQQ` measured, on one importance-sampled chain per row:

    beta    1 + strength     <sgn>
      1        0.0132       1.00000
      2        0.0341       0.99609
      3        0.1492       0.95781

The coupling deficit is already moving at beta = 1, where <sgn> is 1.00000 +- 0.00000 and carries
no information at all.  It reads O(1) Green's-function diagonals, so it has no dynamic-range
problem, and it is measured at |z| ~ 130 against the instrument's own exact re-pairing null.

The claim to test: the deficit's growth in beta is smooth enough that measuring it in the CHEAP,
SIGN-FREE regime predicts the average sign in the expensive regime.  If it does, the severity of a
sign problem can be known before paying for it -- which is not something the average sign can ever
do, because where <sgn> = 1 exactly it has no derivative to read.

Scored honestly: the coupling is fitted NOTHING.  The deficit is measured at small beta, its
growth rate is taken from those points alone, and the extrapolation is then compared against a
DIRECTLY MEASURED <sgn> at large beta that the extrapolation never saw.  A prediction that has
seen its target is not a prediction.
"""
from __future__ import annotations

import numpy as np

import entroptics as E
from dqmc import Model
from expQQ_coupling_vs_sign import run


if __name__ == "__main__":
    N, U, dtau = 8, 4.0, 0.125
    t2, mu = 0.7, 1.0
    betas = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0]
    print("=" * 104)
    print(f"READ IT WHERE IT IS CHEAP, PREDICT IT WHERE IT IS NOT   N = {N}, U = {U}, "
          f"t2 = {t2}, mu = {mu}")
    print("The coupling is measured on O(1) Green's-function diagonals; <sgn> needs a weight")
    print("whose log-magnitude spans orders.  Both come from the same chain at each beta.")
    print()
    print(f"{'beta':>5} | {'<sgn>':>9} {'+-':>8} {'-ln<sgn>':>9} | {'strength':>9} "
          f"{'deficit':>9} {'ln deficit':>11} {'z':>8} {'tight':>7}")
    rows = []
    for beta in betas:
        L = int(round(beta / dtau))
        m = Model(N=N, t=1.0, t2=t2, mu=mu, U=U, dtau=dtau, L=L)
        A, B, S, acc = run(m, R=64, warm=25, n_meas=40, seed=int(beta * 7))
        s = float(S.mean()); se = float(S.std(ddof=1) / np.sqrt(len(S)))
        c = E.reads.coupling(A, B)
        d = 1.0 + float(c.strength)
        rows.append(dict(beta=beta, sgn=s, se=se, d=d, z=float(c.z)))
        nl = -np.log(max(s, 1e-12))
        print(f"{beta:5.1f} | {s:9.5f} {se:8.5f} {nl:9.5f} | {c.strength:9.4f} {d:9.5f} "
              f"{np.log(max(d,1e-12)):11.4f} {c.z:8.1f} {c.tightness:7.3f}", flush=True)

    print()
    print("=" * 104)
    # the extrapolation: growth rate of the deficit taken from the SIGN-FREE rows only
    free = [r for r in rows if r["sgn"] > 0.999]
    hard = [r for r in rows if r["sgn"] <= 0.999]
    print(f"rows with no measurable sign problem (<sgn> > 0.999): {len(free)}  "
          f"-- the only rows the prediction is allowed to see")
    if len(free) >= 3 and hard:
        b = np.array([r["beta"] for r in free]); ld = np.log([r["d"] for r in free])
        g, c0 = np.polyfit(b, ld, 1)
        print(f"deficit growth from those rows alone: ln(deficit) = {c0:+.4f} {g:+.4f} * beta")
        print()
        print(f"{'beta':>5} {'deficit predicted':>18} {'deficit measured':>17} "
              f"{'ratio':>7} | {'<sgn> measured':>15}")
        for r in hard:
            pred = float(np.exp(c0 + g * r["beta"]))
            print(f"{r['beta']:5.1f} {pred:18.5f} {r['d']:17.5f} "
                  f"{pred/max(r['d'],1e-12):7.3f} | {r['sgn']:15.5f}")
        print()
        print("The deficit column is a prediction from sign-free data only.  If it tracks the")
        print("measured deficit where the sign problem is real, then the coupling's growth is")
        print("readable before the sign problem appears -- and the ratio is how well.")
    else:
        print("not enough separation between sign-free and sign-problem rows to extrapolate")
