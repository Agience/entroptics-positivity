"""Experiment PP -- does the coupling PREDICT the average sign, where the average sign cannot be
measured?

`expOO` found the structure and calibrated the instrument against a known identity.  At half
filling particle-hole symmetry gives G_up[i,i](x) + G_dn[i,i](x) = 1 configuration by
configuration, so the two centred channels are exact negatives and the coupling must read
strength = -1.  It reads **-1.0000 at every beta**, with the exact re-pairing null putting the
permuted control at |z| <= 1.7.  Doped, the identity breaks configuration-wise and the strength
degrades: -0.9845, -0.9467, -0.8035, -0.1132 across beta = 2, 4, 6, 8.

THE ASYMMETRY THAT MAKES THIS WORTH SOMETHING.  Those first two doped rows have a negative-weight
fraction of exactly 0.0000 -- there is no sign problem at all, and the average sign is 1 -- while
the coupling has already moved by 0.0155 and 0.0533.  The coupling is measuring the APPROACH to
the sign problem in a regime where the sign itself reports nothing.  And the coupling has no
dynamic-range problem: it reads Green's-function diagonals, which are O(1), while the weight's
log-magnitude spreads over 8 to 16 orders and its average is dominated by a handful of
configurations.

So the question here is direct: across a grid, does the measured coupling track the average sign?
If it does, the coupling is a proxy that stays sharp exactly where the direct estimator dies --
which is what the instrument is for.

<sgn> is computed from the SAME uniform samples by log-sum-exp, sum(s exp(l - lmax)) /
sum(exp(l - lmax)), so no importance sampling and no separate run enters; both columns are read
off one draw and the comparison is not confounded by two different samplers.
"""
from __future__ import annotations

import numpy as np

import entroptics as E
from model2d import Model2D
from expOO_signlock import sample


def mean_sign(sg, lg):
    """<sgn> = sum w / sum |w| from uniform samples, by log-sum-exp.

    The direct form overflows immediately: log|w| here spans 8 to 16 orders, so sum|w| is carried
    entirely by its largest few terms and the naive ratio is 0/0 or noise.  This is the same
    quantity, computed the only way float64 can hold it.
    """
    lm = lg.max()
    e = np.exp(lg - lm)
    return float((sg * e).sum() / e.sum()), float(e.sum() / e.max())


if __name__ == "__main__":
    Lx, Ly, U, dtau = 2, 4, 4.0, 0.125
    n_draw = 400
    rng = np.random.default_rng(11)
    print("=" * 112)
    print(f"DOES THE COUPLING PREDICT THE AVERAGE SIGN?   {Lx}x{Ly}, U = {U}, "
          f"{n_draw} uniform configurations per row")
    print("Both columns come from ONE draw, so no two-sampler confound.  'ESS' is the effective")
    print("number of configurations carrying sum|w| -- when it collapses to a few, <sgn> is a")
    print("number about those few and nothing else, while the coupling is still reading all 400.")
    print()
    print(f"{'beta':>5} {'mu':>5} | {'<sgn>':>10} {'ESS':>8} {'neg frac':>9} | "
          f"{'strength':>9} {'1+strength':>11} {'z':>8} {'resolved':>9}")
    rows = []
    for beta in (2.0, 4.0, 6.0, 8.0, 10.0):
        L = int(round(beta / dtau))
        for mu in (0.0, 0.4, 0.8):
            m = Model2D(Lx=Lx, Ly=Ly, t=1.0, mu=mu, U=U, dtau=dtau, L=L, theta=0.0)
            A, B, sg, lg = sample(m, n_draw, seed=int(beta * 100 + mu * 10))
            s, ess = mean_sign(sg, lg)
            c = E.reads.coupling(A, B)
            rows.append(dict(beta=beta, mu=mu, sgn=s, ess=ess, neg=float(np.mean(sg < 0)),
                             strength=float(c.strength), z=float(c.z), res=int(c.resolved)))
            print(f"{beta:5.1f} {mu:5.2f} | {s:10.5f} {ess:8.1f} {np.mean(sg<0):9.4f} | "
                  f"{c.strength:9.4f} {1+c.strength:11.4f} {c.z:8.2f} {c.resolved:9d}",
                  flush=True)

    print()
    print("=" * 112)
    # A HAND-CHOSEN CUT, AND THEREFORE NOT QUOTABLE. 20 is picked, not derived, and it decides
    # which rows the rank correlation below is computed on. Nothing in the paper leans on that
    # correlation and nothing should without first showing it survives moving the cut. Same
    # defect as the selection in expQQ and the verdict in expRR, marked the same way.
    live = [r for r in rows if r["ess"] > 20]
    dead = [r for r in rows if r["ess"] <= 20]
    print(f"rows where <sgn> is actually resolvable (ESS > 20): {len(live)} of {len(rows)}")
    if len(live) >= 4:
        from scipy.stats import spearmanr
        x = np.array([1 + r["strength"] for r in live])
        y = np.array([r["sgn"] for r in live])
        rho, p = spearmanr(x, y)
        print(f"Spearman(1 + strength, <sgn>) over the resolvable rows: {rho:+.3f}  p = {p:.4f}")
    if dead:
        print()
        print("Rows where <sgn> is NOT resolvable, and what the coupling still reads there:")
        for r in dead:
            print(f"  beta={r['beta']:.0f} mu={r['mu']:.1f}  ESS {r['ess']:6.1f}  "
                  f"<sgn> {r['sgn']:+.5f} (meaningless)  |  strength {r['strength']:+.4f} "
                  f"at z = {r['z']:.1f}, resolved {r['res']}")
    print()
    print("The instrument earns its place only if it stays sharp on the rows where the direct")
    print("estimator has collapsed.  A high |z| there, with the permuted control at |z| ~ 1, is")
    print("a measurement; a small |z| would mean it dies with the thing it was meant to replace.")
