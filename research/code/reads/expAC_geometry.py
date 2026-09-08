"""Experiment AC -- does any of it hold on a second lattice?

Every number in the paper's central sections comes from one 2x4 lattice, while the claims are
stated generally.  A result measured on a single geometry is a result about that geometry until
shown otherwise, and the three claims are cheap to re-measure:

  1. THE IDENTITY   ln|det_up| - ln|det_dn| = -dtau L tr(K) + lambda sum(x), at machine precision
                    on a bipartite lattice at half filling.
  2. THE LOCKSTEP   the two channels change sign on exactly the same configurations, with a
                    NON-ZERO individual flip rate so the statement is not vacuous.
  3. THE CALIBRATION  the coupling between the two channels reads exactly -1.

Geometries.  A periodic Lx x Ly lattice is bipartite when both sides are even; an odd side makes
an odd ring and destroys it.  So `2x3` and `3x4` are built-in controls -- the identity and the
lockstep must FAIL there, and a run in which they hold everywhere would mean the test is not
sensitive to the property it claims to depend on.

Half filling is `mu = 0` only where the particle-hole symmetry that pins it is present, which is
exactly the bipartite case; the odd geometries are therefore reported at `mu = 0` and read as the
controls they are, not as matched-filling comparisons.
"""
from __future__ import annotations

import numpy as np

import entroptics as E
from model2d import Model2D
from stable import udt_product, inv_one_plus_block, slogdet_one_plus_block


def read_lattice(Lx, Ly, beta, U=4.0, dtau=0.125, n_draw=300, seed=0):
    L = int(round(beta / dtau))
    m = Model2D(Lx=Lx, Ly=Ly, t=1.0, mu=0.0, U=U, dtau=dtau, L=L, theta=0.0)
    lam = float(np.arccosh(np.exp(dtau * U / 2.0)))
    rng = np.random.default_rng(seed)
    res, su, sd, A, B = [], [], [], [], []
    for _ in range(n_draw):
        X = rng.choice([-1.0, 1.0], size=(m.L, m.N))
        row, lg, sg = {}, {}, {}
        for sigma in (+1, -1):
            d = np.exp(sigma * lam * X)
            Bl = m.expmK[None, :, :] * d[:, None, :]
            Uu, D, T = udt_product(Bl[None], 4)
            row[sigma] = np.real(np.diag(inv_one_plus_block(Uu, D, T)[0])).copy()
            s, la = slogdet_one_plus_block(Uu, D, T)
            sg[sigma] = float(np.real(s[0])); lg[sigma] = float(la[0])
        pred = -dtau * L * float(np.trace(m.K)) + lam * float(X.sum())
        res.append(abs((lg[+1] - lg[-1]) - pred))
        su.append(sg[+1]); sd.append(sg[-1])
        A.append(row[+1]); B.append(row[-1])
    su, sd = np.array(su), np.array(sd)
    c = E.reads.coupling(np.array(A), np.array(B))
    return dict(N=m.N, res=np.array(res), flip=float(np.mean(su < 0)),
                agree=float(np.mean(su == sd)), neg=float(np.mean(su * sd < 0)),
                strength=float(c.strength), z=float(c.z), resolved=bool(c.resolved))


if __name__ == "__main__":
    print("=" * 112)
    print("DOES ANY OF IT HOLD ON A SECOND LATTICE?   U = 4, mu = 0, 300 configurations per row")
    print("Both sides even => bipartite.  An odd side makes an odd ring and is a CONTROL: the")
    print("identity and the lockstep must fail there.")
    print()
    print(f"{'lattice':>8} {'N':>3} {'beta':>5} {'bipartite':>10} | {'identity':>9} "
          f"{'resid max':>11} | {'flip rate':>10} {'agree':>7} {'neg':>7} | "
          f"{'strength':>9} {'z':>8}")
    for (Lx, Ly) in ((2, 4), (4, 4), (2, 6), (4, 6), (2, 3), (3, 4)):
        bip = (Lx % 2 == 0) and (Ly % 2 == 0)
        for beta in (6.0, 10.0):
            r = read_lattice(Lx, Ly, beta, seed=Lx * 100 + Ly * 10 + int(beta))
            holds = "EXACT" if r["res"].max() < 1e-9 else "broken"
            print(f"{f'{Lx}x{Ly}':>8} {r['N']:3d} {beta:5.1f} {str(bip):>10} | {holds:>9} "
                  f"{r['res'].max():11.3e} | {r['flip']:10.4f} {r['agree']:7.4f} "
                  f"{r['neg']:7.4f} | {r['strength']:9.4f} {r['z']:8.1f}", flush=True)
    print()
    print("A claim that holds on 2x4 and nowhere else is a claim about 2x4.  The odd-side rows")
    print("are what say the test can tell the difference.")
