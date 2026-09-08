"""Experiment XX -- the residual of the exact identity, as a per-configuration order parameter.

`expWW` derived and verified the identity that carries positivity at half filling:

    ln|det_up(x)| - ln|det_dn(x)| = -dtau * L * tr(K) + lambda * sum(x)

exact to 1e-14, and decisively wrong at 2*lambda, so the coefficient is derived rather than fitted.
It explains both facts that looked contradictory: the ratio det_up/det_dn is an EXPONENTIAL, hence
positive, hence the signs lock; and its exponent is a sum over L*N field components, hence the
8-to-16-order spread in the magnitudes.

Doping breaks it, with a residual that grows in beta (7.96, 15.92, 24.73, 31.79 at beta = 2, 4, 6,
8).  That residual

    rho(x) = | ln|det_up| - ln|det_dn| + dtau * L * tr(K) - lambda * sum(x) |

is the first per-configuration quantity in this work that is the exact failure of an exact
identity rather than a proxy for one.  It costs one extra scalar over what the sampler already
computes, and nothing in it is chosen.

Two earlier candidates failed and the reason each failed is worth carrying:

  * the Green's-function residual `max_i |G_up[i,i] + G_dn[i,i] - 1|` -- a coarse summary of the
    object that carries the sign, and it did not separate;
  * the spectral distance `min_k |lambda_k(M)|` -- correct as a distance to a flip, but a distance
    is SIGN-BLIND: it cannot say which side of the boundary a configuration is on.

rho is neither: it is a signed statement about a relation, not a distance.  Whether it classifies
is the question.
"""
from __future__ import annotations

import numpy as np

from model2d import Model2D
from expWW_ratio import log_dets


def residual_and_sign(m, X, lam):
    """(rho, sign of the weight) for one configuration."""
    o = log_dets(m, X)
    su, lu = o[+1]
    sd, ld = o[-1]
    pred = -m.dtau * m.L * float(np.trace(m.K)) + lam * float(X.sum())
    return abs((lu - ld) - pred), su * sd


if __name__ == "__main__":
    Lx, Ly, U, dtau = 2, 4, 4.0, 0.125
    n_draw = 800
    lam = float(np.arccosh(np.exp(dtau * U / 2.0)))
    print("=" * 112)
    print(f"THE RESIDUAL OF THE EXACT IDENTITY   {Lx}x{Ly}, U = {U}, {n_draw} configurations")
    print("rho = |ln|det_up| - ln|det_dn| + dtau L tr(K) - lambda sum(x)|, zero when the identity")
    print("holds.  Nothing in it is chosen: the coefficient is derived and verified at 1e-14.")
    print()
    print(f"{'beta':>5} {'mu':>5} {'neg':>5} | {'rho | w>0 max':>14} {'rho | w<0 min':>14} "
          f"{'gap':>9} {'separates':>10} | {'med rho+':>10} {'med rho-':>10} {'ratio':>7}")
    for beta in (4.0, 6.0, 8.0, 10.0):
        L = int(round(beta / dtau))
        for mu in (0.4, 0.8):
            m = Model2D(Lx=Lx, Ly=Ly, t=1.0, mu=mu, U=U, dtau=dtau, L=L, theta=0.0)
            rng = np.random.default_rng(int(beta * 100 + mu * 10))
            R, S = [], []
            for _ in range(n_draw):
                X = rng.choice([-1.0, 1.0], size=(m.L, m.N))
                r, s = residual_and_sign(m, X, lam)
                R.append(r); S.append(s)
            R, S = np.array(R), np.array(S)
            pos, neg = R[S > 0], R[S < 0]
            if len(neg) == 0:
                print(f"{beta:5.1f} {mu:5.2f} {0:5d} | {pos.max():14.5f} {'--':>14} "
                      f"{'--':>9} {'--':>10} | {np.median(pos):10.5f} {'--':>10} {'--':>7}")
                continue
            gap = float(neg.min() - pos.max())
            ratio = float(np.median(neg) / max(np.median(pos), 1e-30))
            print(f"{beta:5.1f} {mu:5.2f} {len(neg):5d} | {pos.max():14.5f} {neg.min():14.5f} "
                  f"{gap:9.4f} {('YES' if gap > 0 else 'overlap'):>10} | "
                  f"{np.median(pos):10.5f} {np.median(neg):10.5f} {ratio:7.3f}", flush=True)
    print()
    print("A positive gap would mean no configuration below that residual carried a negative")
    print("weight -- the residual would be a sufficient condition for positivity, computable per")
    print("configuration from quantities the sampler already has.")
