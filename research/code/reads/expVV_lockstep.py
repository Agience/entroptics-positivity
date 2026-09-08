"""Experiment VV -- the lock is not avoidance, it is LOCKSTEP.

`expUU` measured, in the conditioned frame and with the arithmetic gated three ways, that at half
filling the distance from a configuration's spectrum to a sign flip falls to

    beta    2      4       6        8        10
    min d   0.024  0.008   0.0013   0.0012   0.00032

while the negative-weight fraction stays exactly 0.0000 at every one of them.  Configurations come
arbitrarily close to a zero and never produce a negative weight.

So particle-hole symmetry does not hold configurations AWAY from the boundary.  It makes the two
spin channels cross it TOGETHER: when one determinant changes sign, so does the other, and the
product's sign is preserved.  That is a per-configuration, exact statement, and it is what this
file tests.

Measured per configuration, both channels separately:

  1. LOCKSTEP.  At half filling, sign(det_up) == sign(det_dn) on every configuration -- so
     individual channels DO change sign (the count of them is reported), and they never do it
     alone.  A row where the individual channels never flip at all would make the claim vacuous,
     so the flip count is what makes it a test.
  2. THE BREAK.  Doped, the two channels flip independently, and the fraction on which they
     disagree IS the negative-weight fraction.
  3. THE APPROACH.  The per-channel distances to a zero, so that "close to the boundary" is
     measured rather than inferred.

No thresholds: a sign is a determinant sign from `slogdet`, a distance is a minimum modulus, and
agreement is equality of two signs.
"""
from __future__ import annotations

import numpy as np

from model2d import Model2D
from stable import udt_product, core_matrix, slogdet_one_plus_block


def per_channel(m, X, block=4):
    """(sign_up, sign_dn, d_up, d_dn) for one configuration, in the conditioned frame."""
    lam = float(np.arccosh(np.exp(m.dtau * m.U / 2.0)))
    s, d = {}, {}
    for sigma in (+1, -1):
        e = np.exp(sigma * lam * X)
        Bl = m.expmK[None, :, :] * e[:, None, :]
        U, D, T = udt_product(Bl[None], block)
        sg, _ = slogdet_one_plus_block(U, D, T)
        s[sigma] = float(np.real(sg[0]))
        M, _ = core_matrix(U, D, T)
        d[sigma] = float(np.min(np.abs(np.linalg.eigvals(M[0]))))
    return s[+1], s[-1], d[+1], d[-1]


if __name__ == "__main__":
    Lx, Ly, U, dtau = 2, 4, 4.0, 0.125
    n_draw = 600
    print("=" * 116)
    print(f"LOCKSTEP, NOT AVOIDANCE   {Lx}x{Ly}, U = {U}, {n_draw} configurations per row")
    print("Both channels read separately.  'up flips' is the fraction with sign(det_up) < 0 --")
    print("if that is 0 the lockstep claim is vacuous, so it is reported alongside.")
    print()
    print(f"{'beta':>5} {'mu':>5} | {'up flips':>9} {'dn flips':>9} {'agree':>8} {'disagree':>9} "
          f"{'neg frac':>9} {'match':>7} | {'min d_up':>10} {'min d_dn':>10}")
    for beta in (2.0, 4.0, 6.0, 8.0, 10.0, 12.0):
        L = int(round(beta / dtau))
        for mu in (0.0, 0.4):
            m = Model2D(Lx=Lx, Ly=Ly, t=1.0, mu=mu, U=U, dtau=dtau, L=L, theta=0.0)
            rng = np.random.default_rng(int(beta * 100 + mu * 10))
            su, sd, du, dd = [], [], [], []
            for _ in range(n_draw):
                X = rng.choice([-1.0, 1.0], size=(m.L, m.N))
                a, b, c, e = per_channel(m, X)
                su.append(a); sd.append(b); du.append(c); dd.append(e)
            su, sd = np.array(su), np.array(sd)
            du, dd = np.array(du), np.array(dd)
            agree = float(np.mean(su == sd))
            neg = float(np.mean(su * sd < 0))
            match = "YES" if abs((1 - agree) - neg) < 1e-12 else "NO"
            print(f"{beta:5.1f} {mu:5.2f} | {float(np.mean(su<0)):9.4f} "
                  f"{float(np.mean(sd<0)):9.4f} {agree:8.4f} {1-agree:9.4f} {neg:9.4f} "
                  f"{match:>7} | {du.min():10.6f} {dd.min():10.6f}", flush=True)
    print()
    print("At half filling 'agree' must be 1.0000 while 'up flips' is NOT 0 -- individual channels")
    print("change sign and never do it alone.  That is the lock, and it is lockstep rather than")
    print("avoidance because the distances show the boundary IS being approached.")
    print("'disagree' and 'neg frac' must be the same number: a negative weight is exactly a")
    print("configuration on which the two channels disagree.")
