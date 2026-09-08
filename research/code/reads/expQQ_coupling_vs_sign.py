"""Experiment QQ -- the coupling against the average sign, both from the SAME importance-sampled
chain.

`expPP` asked this question and could not answer it, because its <sgn> column was computed from
UNIFORMLY drawn fields.  The weight's log-magnitude spans 8 to 16 orders, so sum|w| is carried by
a handful of draws: the effective sample size came out at 1.2 to 7.6 out of 400 on every row and
the column was meaningless.  That is a design error, not a result -- the average sign has to come
from a chain that samples |w|.

Here it does.  `StableChains` is the rig's importance sampler, stabilised through UDT and gated
against brute-force enumeration of every auxiliary field to 5e-15, so its sign is the real one.
Both columns are read off the same chain at the same samples, so nothing is confounded by two
samplers.

The chain also fixes the second thing `expPP` could not control: uniform draws are not where the
physics lives, so a coupling measured on them is a coupling between two channels at typical
FIELDS rather than at typical CONFIGURATIONS of the ensemble that actually contributes.

What is being asked.  The coupling reads O(1) Green's-function diagonals and was measured at
|z| = 19 to 48 while the average sign was still exactly 1 -- it has no dynamic-range problem and
it moves early.  If it tracks <sgn> where <sgn> can be measured, it is a proxy that stays sharp
past the point where the direct estimator dies, which is the whole claim being tested.

The sign problem is turned on with next-nearest hopping and doping, which is how this rig has
produced one since the start: t2 breaks the bipartite structure that protects half filling.
"""
from __future__ import annotations

import numpy as np

import entroptics as E
from dqmc import Model
from sampler_stable import StableChains
from stable import udt_product, inv_one_plus


def channels(ch):
    """Per-spin Green's-function diagonals for every chain, at its current field.

    `StableChains.measure` returns the SPIN-AVERAGED correlator, which is exactly the object the
    coupling cannot use: averaging the two channels together destroys the relationship between
    them that is being measured.
    """
    m = ch.m
    out = {}
    for sigma in (+1, -1):
        Bl = ch._Bl(ch.S, sigma)
        U, D, T = udt_product(Bl, ch.n_stab)
        G = inv_one_plus(U, D, T)
        out[sigma] = np.einsum("rii->ri", G).real.copy()
    return out[+1], out[-1]


def run(m, R=64, warm=30, n_meas=40, seed=0):
    ch = StableChains(m, R=R, seed=seed)
    for _ in range(warm):
        ch.sweep()
    A, B, S = [], [], []
    for _ in range(n_meas):
        ch.sweep()
        a, b = channels(ch)
        A.append(a); B.append(b); S.append(ch.cur_sign.copy())
    return np.vstack(A), np.vstack(B), np.concatenate(S), ch.accepted / max(ch.proposed, 1)


if __name__ == "__main__":
    N, U, dtau = 8, 4.0, 0.125
    print("=" * 112)
    print(f"COUPLING AND AVERAGE SIGN FROM ONE IMPORTANCE-SAMPLED CHAIN   N = {N}, U = {U}")
    print("StableChains is gated against brute-force field enumeration at 5e-15.")
    print("t2 and mu turn the sign problem on; both columns come from the same samples.")
    print()
    print(f"{'beta':>5} {'t2':>5} {'mu':>5} | {'<sgn>':>9} {'+-':>8} {'n':>6} {'acc':>6} | "
          f"{'strength':>9} {'1+strength':>11} {'z':>8} {'resolved':>9} {'tight':>7}")
    rows = []
    for beta in (1.0, 2.0, 3.0):
        L = int(round(beta / dtau))
        for t2, mu in ((0.0, 0.0), (0.0, 0.6), (0.7, 0.6), (0.7, 1.0)):
            m = Model(N=N, t=1.0, t2=t2, mu=mu, U=U, dtau=dtau, L=L)
            A, B, S, acc = run(m, seed=int(beta * 10 + t2 * 10 + mu * 10))
            s, se = float(S.mean()), float(S.std(ddof=1) / np.sqrt(len(S)))
            c = E.reads.coupling(A, B)
            rows.append((beta, t2, mu, s, se, float(c.strength), float(c.z), int(c.resolved)))
            print(f"{beta:5.1f} {t2:5.2f} {mu:5.2f} | {s:9.5f} {se:8.5f} {len(S):6d} "
                  f"{acc:6.3f} | {c.strength:9.4f} {1+c.strength:11.4f} {c.z:8.2f} "
                  f"{c.resolved:9d} {c.tightness:7.3f}", flush=True)

    print()
    print("=" * 112)
    from scipy.stats import spearmanr
    sg = np.array([r[3] for r in rows]); st = np.array([1 + r[5] for r in rows])
    # A HAND-CHOSEN CUT, AND THEREFORE NOT QUOTABLE.  0.05 is picked, not derived, and it
    # decides which rows the rank correlation below is computed on.  The same defect made an
    # extrapolation elsewhere in this directory unreproducible: its cut admitted three rows on
    # one seed and two on three others, so the figure it produced existed on one draw only.
    # Nothing in the paper leans on the correlation below, and nothing should without first
    # showing it is stable across the cut.
    live = np.array([r[4] for r in rows]) < 0.05
    print(f"rows where <sgn> is resolved to better than +-0.05: {int(live.sum())} of {len(rows)}")
    if live.sum() >= 4:
        rho, p = spearmanr(st[live], sg[live])
        print(f"Spearman(1 + strength, <sgn>) = {rho:+.3f}   p = {p:.4f}")
        print("A strong NEGATIVE rank correlation would say the coupling degrades as the average")
        print("sign falls.  It is computed on rows selected by a CHOSEN cut, so it is reported")
        print("here and is not quoted in the paper.")
