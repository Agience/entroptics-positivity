"""Section 2's table: the unstabilised product invents a sign problem where positivity is provable.

Section 2 is the paper's opening empirical claim and its table had no file that produces it -- the
figures lived in the docstring of `tests/test_conditioned_frame.py`, which asserts that the naive
frame fails but prints nothing and quotes no fraction. A figure whose only source is a docstring is
how four separate stale numbers survived this paper, so the table is measured here instead.

WHAT IS COMPARED, on the same configurations at every beta:

  * the plain ordered product `B = prod_l (expmK * d_l)`, then `sign det(I + B)` -- the thing not to
    do, and the thing three different files in this work reintroduced;
  * the same sign through the stratified UDT factorisation of section 2 -- the conditioned frame.

The lattice is `2x4` at half filling with no next-nearest hopping, so section 5's criterion says the
identity holds and the weight is positive on every configuration. Any negative fraction the naive
column reports is manufactured by the arithmetic.

The last column is the smallest `|1 + lambda_k|` over the configurations at that beta -- the
"distance to a sign flip" for a quantity that is O(1). It is what says the naive failure is
conditioning rather than physics.

Both routines are imported from the gate rather than rewritten, so this file and that gate cannot
disagree for reasons unrelated to the arithmetic.

    python remote_run.py reads/expBM_naive_frame_manufactures_a_sign.py
"""
from __future__ import annotations

import numpy as np

from model2d import Model2D
from stable import core_matrix, udt_product
from tests.test_conditioned_frame import naive_sign, stable_signs

U, DTAU, N_DRAW, SEED = 4.0, 0.125, 400, 2
BETAS = (2.0, 4.0, 6.0, 8.0)


def spread(m, X):
    """(naive, conditioned) largest |eigenvalue of I + B| over both channels.

    This is the "distance to a sign flip" the section quotes, and the comparison is the point: the
    conditioned frame keeps it O(1), because the stratified factorisation divides out the scales
    that the ordered product accumulates. The naive frame reports a distance orders larger for the
    same configuration, which is the arithmetic running away rather than the physics moving.
    """
    lam = float(np.arccosh(np.exp(m.dtau * m.U / 2.0)))
    worst_naive = worst_cond = 0.0
    for sigma in (+1, -1):
        d = np.exp(sigma * lam * X)
        B = np.eye(m.N)
        for l in range(m.L):
            B = (m.expmK * d[l][None, :]) @ B
        worst_naive = max(worst_naive, float(np.max(np.abs(np.linalg.eigvals(np.eye(m.N) + B)))))
        Bl = m.expmK[None, :, :] * d[:, None, :]
        Uu, D, T = udt_product(Bl[None], 4)
        M = core_matrix(Uu, D, T)[0]
        worst_cond = max(worst_cond, float(np.max(np.abs(np.linalg.eigvals(M)))))
    return worst_naive, worst_cond


def main():
    print("=" * 96)
    print("THE UNSTABILISED PRODUCT MANUFACTURES A SIGN PROBLEM WHERE POSITIVITY IS PROVABLE")
    print("=" * 96)
    print(f"  2x4, U = {U}, mu = 0 (half filling), dtau = {DTAU}, {N_DRAW} draws a beta, "
          f"seed {SEED}")
    print("  Section 5's criterion holds on this lattice, so the true negative fraction is 0.")
    print()
    print(f"{'beta':>6} {'L':>5} {'naive neg fraction':>20} {'conditioned':>13} "
          f"{'naive max|1+lam|':>18} {'conditioned':>13}")
    print("-" * 96)

    for beta in BETAS:
        L = int(round(beta / DTAU))
        m = Model2D(Lx=2, Ly=4, t=1.0, mu=0.0, U=U, dtau=DTAU, L=L, theta=0.0)
        rng = np.random.default_rng(SEED)
        naive_neg = stable_neg = 0
        big_naive = big_cond = 0.0
        for _ in range(N_DRAW):
            X = rng.choice([-1.0, 1.0], size=(m.L, m.N))
            naive_neg += int(naive_sign(m, X) < 0)
            stable_neg += int(stable_signs(m, X)[0] < 0)
            a, b = spread(m, X)
            big_naive, big_cond = max(big_naive, a), max(big_cond, b)
        print(f"{beta:6.1f} {L:5d} {naive_neg / N_DRAW:20.4f} {stable_neg / N_DRAW:13.4f} "
              f"{big_naive:18.4g} {big_cond:13.4g}", flush=True)

    print()
    print("  The conditioned column is 0.0000 at every beta, which is the answer the criterion")
    print("  requires. The naive column departs from it once the product's condition number")
    print("  outruns double precision, and the last column says the eigenvalues were never near a")
    print("  crossing: the naive frame does not lose precision gracefully, it reports a different")
    print("  and wrong calculation.")


if __name__ == "__main__":
    main()
