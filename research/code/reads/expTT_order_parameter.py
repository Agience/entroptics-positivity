"""Experiment TT -- the order parameter, per CONFIGURATION.

The result so far is an ensemble statistic: across parameter points, the channel coupling departs
from -1 as the sign problem develops.  That is a correlation between two averages.  What would
make it a mechanism is a statement about INDIVIDUAL configurations, and there is a natural
candidate, because the identity that protects positivity is itself per-configuration.

On a bipartite lattice at half filling, particle-hole symmetry gives

    G_up[i,i](x) + G_dn[i,i](x) = 1        for every site i, for every configuration x

so define the residual

    r(x) = max_i | G_up[i,i](x) + G_dn[i,i](x) - 1 |

which is exactly zero when the identity holds, is O(N) to compute for one configuration, and needs
no exact answer.  The claims to test, in order of strength:

  1. r(x) = 0 identically at half filling on a bipartite lattice -- the identity, verified to
     machine precision rather than asserted.  If this fails the whole framing is wrong.
  2. Every configuration with w(x) < 0 has LARGE r(x): the residual separates the negative-weight
     configurations from the positive ones.  This is the per-configuration statement.
  3. There is a threshold in r below which no negative weight occurs -- i.e. r BOUNDS the sign.
     Reported as the measured separation: max r over positive-weight configurations against min r
     over negative-weight ones.  A gap means the residual is a sufficient condition for positivity
     on the sampled set.

Nothing here is fitted.  r is a residual of an exact identity; the separation is measured, not
chosen; and the determinants are stabilised, because a naive product fakes a sign problem.
"""
from __future__ import annotations

import numpy as np

from model2d import Model2D
from stable import udt_product, inv_one_plus_block, slogdet_one_plus_block


def per_config(m, n_draw, seed=0):
    """(r, sign, log|w|) per configuration -- all three O(N^3) from the same factorisation."""
    rng = np.random.default_rng(seed)
    lam = float(np.arccosh(np.exp(m.dtau * m.U / 2.0)))
    R, S, LW = [], [], []
    for _ in range(n_draw):
        X = rng.choice([-1.0, 1.0], size=(m.L, m.N))
        diag, sgn, lw = {}, 1.0, 0.0
        for sigma in (+1, -1):
            d = np.exp(sigma * lam * X)
            Bl = m.expmK[None, :, :] * d[:, None, :]
            U, D, T = udt_product(Bl[None], 4)
            G = inv_one_plus_block(U, D, T)[0]
            s, la = slogdet_one_plus_block(U, D, T)
            diag[sigma] = np.real(np.diag(G)).copy()
            sgn *= float(np.real(s[0]))
            lw += float(la[0])
        R.append(float(np.abs(diag[+1] + diag[-1] - 1.0).max()))
        S.append(sgn); LW.append(lw)
    return np.array(R), np.array(S), np.array(LW)


if __name__ == "__main__":
    Lx, Ly, U, dtau = 2, 4, 4.0, 0.125
    n_draw = 2000
    print("=" * 104)
    print(f"THE ORDER PARAMETER, PER CONFIGURATION   {Lx}x{Ly}, U = {U}, {n_draw} configurations")
    print("r(x) = max_i |G_up[i,i] + G_dn[i,i] - 1|, the residual of the identity that protects")
    print("positivity.  Zero when it holds.  O(N) per configuration, no exact answer needed.")
    print()

    print("CLAIM 1  the identity holds exactly at half filling on a bipartite lattice")
    print(f"{'beta':>5} {'max r over all configs':>24} {'neg frac':>9}")
    for beta in (2.0, 4.0, 6.0, 8.0):
        L = int(round(beta / dtau))
        m = Model2D(Lx=Lx, Ly=Ly, t=1.0, mu=0.0, U=U, dtau=dtau, L=L, theta=0.0)
        r, s, lw = per_config(m, 400, seed=int(beta))
        print(f"{beta:5.1f} {r.max():24.3e} {float(np.mean(s < 0)):9.4f}")

    print()
    print("CLAIMS 2 AND 3  doped: does the residual separate the negative-weight configurations?")
    print(f"{'beta':>5} {'mu':>5} {'neg':>5} {'r | w>0  (max)':>15} {'r | w<0  (min)':>15} "
          f"{'gap':>8} {'separated':>10} {'median r+':>10} {'median r-':>10}")
    for beta in (4.0, 6.0, 8.0, 10.0):
        L = int(round(beta / dtau))
        for mu in (0.4, 0.8):
            m = Model2D(Lx=Lx, Ly=Ly, t=1.0, mu=mu, U=U, dtau=dtau, L=L, theta=0.0)
            r, s, lw = per_config(m, n_draw, seed=int(beta * 10 + mu * 10))
            pos, neg = r[s > 0], r[s < 0]
            if len(neg) == 0:
                print(f"{beta:5.1f} {mu:5.2f} {0:5d} {pos.max():15.5f} {'--':>15} "
                      f"{'--':>8} {'--':>10} {np.median(pos):10.5f} {'--':>10}")
                continue
            gap = float(neg.min() - pos.max())
            sep = "YES" if gap > 0 else "overlap"
            print(f"{beta:5.1f} {mu:5.2f} {len(neg):5d} {pos.max():15.5f} {neg.min():15.5f} "
                  f"{gap:8.4f} {sep:>10} {np.median(pos):10.5f} {np.median(neg):10.5f}",
                  flush=True)
    print()
    print("A positive gap means no configuration with r below that level carried a negative")
    print("weight: on the sampled set the residual is a sufficient condition for positivity, and")
    print("the order parameter is a per-configuration statement rather than an ensemble one.")
