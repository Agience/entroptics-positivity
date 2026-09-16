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


def one_seed(m, seed, n_draw):
    """One independent draw of `n_draw` configurations: the per-channel flip rates and agreement.

    Split out from the reporting loop so that a row can be repeated across seeds. The seed is an
    argument rather than a function of (beta, mu), which is what it used to be -- that gave exactly
    one number per row and no way to ask whether the number was reproducible.
    """
    rng = np.random.default_rng(seed)
    su, sd, du, dd = [], [], [], []
    for _ in range(n_draw):
        X = rng.choice([-1.0, 1.0], size=(m.L, m.N))
        a, b, c, e = per_channel(m, X)
        su.append(a); sd.append(b); du.append(c); dd.append(e)
    su, sd = np.array(su), np.array(sd)
    du, dd = np.array(du), np.array(dd)
    agree = float(np.mean(su == sd))
    return dict(up=float(np.mean(su < 0)), dn=float(np.mean(sd < 0)), agree=agree,
                neg=float(np.mean(su * sd < 0)), d_up=float(du.min()), d_dn=float(dd.min()))


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    # WHY SEEDS ARE A FLAG. Two columns of this table are estimates and two are not, and that
    # distinction is the whole point of section 3: the flip rate is a proportion of 600 draws and
    # has a spread, while `agree` is 1.0000 on EVERY seed rather than averaging to it. A spread of
    # zero is only meaningful against seeds that could have produced a non-zero one, so the seeds
    # have to be real and the count has to be visible. With one seed the column cannot say anything.
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4],
                    help="independent repeats per row; the spread across them is the error")
    args = ap.parse_args()

    Lx, Ly, U, dtau = 2, 4, 4.0, 0.125
    n_draw = 600
    print("=" * 116)
    print(f"LOCKSTEP, NOT AVOIDANCE   {Lx}x{Ly}, U = {U}, {n_draw} configurations per row, "
          f"{len(args.seeds)} seeds")
    print("Both channels read separately.  'up flips' is the fraction with sign(det_up) < 0 --")
    print("if that is 0 the lockstep claim is vacuous, so it is reported alongside.")
    print("Errors are the SPREAD ACROSS SEEDS.  Where a column reads +- 0.0000 it did so on every")
    print("seed; that is the claim, and it is different from a mean that happens to land there.")
    print()
    print(f"{'beta':>5} {'mu':>5} | {'up flips':>17} {'dn flips':>9} {'agree':>17} "
          f"{'disagree':>9} {'neg frac':>9} {'match':>7} | {'min d_up':>10} {'min d_dn':>10}")
    for beta in (2.0, 4.0, 6.0, 8.0, 10.0, 12.0):
        L = int(round(beta / dtau))
        for mu in (0.0, 0.4):
            m = Model2D(Lx=Lx, Ly=Ly, t=1.0, mu=mu, U=U, dtau=dtau, L=L, theta=0.0)
            # the seed is offset per (beta, mu) so two rows never reuse one field sequence
            rows = [one_seed(m, int(beta * 100 + mu * 10) + 1000 * s, n_draw) for s in args.seeds]
            up = np.array([r["up"] for r in rows])
            dn = np.array([r["dn"] for r in rows])
            ag = np.array([r["agree"] for r in rows])
            ng = np.array([r["neg"] for r in rows])
            sd_ = (lambda v: float(v.std(ddof=1)) if len(v) > 1 else 0.0)
            match = "YES" if abs((1 - ag.mean()) - ng.mean()) < 1e-12 else "NO"
            print(f"{beta:5.1f} {mu:5.2f} | {up.mean():8.4f} +- {sd_(up):6.4f} "
                  f"{dn.mean():9.4f} {ag.mean():8.4f} +- {sd_(ag):6.4f} "
                  f"{1 - ag.mean():9.4f} {ng.mean():9.4f} "
                  f"{match:>7} | {min(r['d_up'] for r in rows):10.6f} "
                  f"{min(r['d_dn'] for r in rows):10.6f}", flush=True)
    print()
    print("At half filling 'agree' must be 1.0000 while 'up flips' is NOT 0 -- individual channels")
    print("change sign and never do it alone.  That is the lock, and it is lockstep rather than")
    print("avoidance because the distances show the boundary IS being approached.")
    print("'disagree' and 'neg frac' must be the same number: a negative weight is exactly a")
    print("configuration on which the two channels disagree.")
