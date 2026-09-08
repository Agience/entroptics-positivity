"""Experiment WW -- WHAT the lockstep is, exactly.

`expVV` established that at half filling the two channels' determinants change sign on exactly the
same configurations, with identical rates, up to 12.17% of configurations at beta = 12.  Part 8
established something that sits oddly beside it: the two log-magnitudes differ by 8 to 16 ORDERS.

Those two facts together rule out the obvious relation.  If `det_dn = c * det_up` for a constant
c, the magnitudes would be locked too.  So whatever relates them preserves the SIGN and leaves the
MAGNITUDE free, which means the ratio

    R(x) = det(I + B_dn(x)) / det(I + B_up(x))

is positive on every configuration and varies over orders.  A positive quantity that varies over
orders is an exponential of something, and there is only one thing in the problem with the right
shape: the field itself.  For the spin decoupling `B_dn(x) = B_up(-x)`, so the natural candidate is

    ln R(x) = -2 lambda * sum_{l,i} x_{l,i}         (up to a sign convention)

which would close the mechanism completely: the ratio is `exp(positive-or-negative real)`, hence
POSITIVE, hence the signs lock; and its exponent is a sum over L*N field components, hence the
8-to-16-order spread in magnitude.

Measured here, with nothing fitted:

  1. Is R(x) > 0 on every configuration at half filling?  (It must be, or expVV is wrong.)
  2. Is ln R(x) an exact linear function of sum x?  Reported as the residual of
     `ln R + 2 lambda sum x`, which is zero if the identity holds -- NOT a fitted slope.  The
     coefficient is predicted from the decoupling, not measured from the data.
  3. Doped, where does it break?  The same residual, which should stop being zero.

If (2) holds the sign problem's absence at half filling is a one-line identity, and its presence
under doping is the failure of that identity -- which is a far stronger statement than a
correlation between two averages.
"""
from __future__ import annotations

import numpy as np

from model2d import Model2D
from stable import udt_product, slogdet_one_plus_block


def log_dets(m, X, block=4):
    """(sign, log|det|) per spin, in the conditioned frame."""
    lam = float(np.arccosh(np.exp(m.dtau * m.U / 2.0)))
    out = {}
    for sigma in (+1, -1):
        d = np.exp(sigma * lam * X)
        Bl = m.expmK[None, :, :] * d[:, None, :]
        U, D, T = udt_product(Bl[None], block)
        s, la = slogdet_one_plus_block(U, D, T)
        out[sigma] = (float(np.real(s[0])), float(la[0]))
    return out


if __name__ == "__main__":
    Lx, Ly, U, dtau = 2, 4, 4.0, 0.125
    n_draw = 400
    print("=" * 112)
    print(f"WHAT THE LOCKSTEP IS   {Lx}x{Ly}, U = {U}, {n_draw} configurations per row")
    print("R = det_dn / det_up.  The coefficient below is PREDICTED from the decoupling")
    print("(lambda = arccosh(exp(dtau U / 2)), B_dn(x) = B_up(-x)), never fitted to the data.")
    print()
    print(f"{'beta':>5} {'mu':>5} | {'R > 0 always':>13} {'ln R spread':>12} | "
          f"{'residual max':>13} {'residual med':>13} {'holds':>7}")
    for beta in (2.0, 4.0, 6.0, 8.0, 10.0):
        L = int(round(beta / dtau))
        lam = float(np.arccosh(np.exp(dtau * U / 2.0)))
        for mu in (0.0, 0.4):
            m = Model2D(Lx=Lx, Ly=Ly, t=1.0, mu=mu, U=U, dtau=dtau, L=L, theta=0.0)
            rng = np.random.default_rng(int(beta * 100 + mu * 10))
            pos, lnR, res = 0, [], []
            for _ in range(n_draw):
                X = rng.choice([-1.0, 1.0], size=(m.L, m.N))
                o = log_dets(m, X)
                su, lu = o[+1]
                sd, ld = o[-1]
                pos += int(su * sd > 0)
                lnR.append(ld - lu)
                # the PREDICTED identity, with no free parameter
                res.append(abs((ld - lu) + 2.0 * lam * float(X.sum())))
            lnR, res = np.array(lnR), np.array(res)
            holds = "YES" if res.max() < 1e-8 else "no"
            print(f"{beta:5.1f} {mu:5.2f} | {pos/n_draw:13.4f} {lnR.std():12.4f} | "
                  f"{res.max():13.4e} {np.median(res):13.4e} {holds:>7}", flush=True)
    print()
    print("'R > 0 always' must be 1.0000 at half filling -- that IS the lockstep, restated.")
    print("If 'residual' is at machine precision there, the lockstep is a one-line identity and")
    print("the sign problem is exactly the failure of that identity under doping.")
