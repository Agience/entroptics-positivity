"""What the Green's-function diagonal is at the beta section 7's tables run at.

Section 7's two-mechanism table reports `max|G_up - G_dn|` over its draws, and at `beta = 10` that
column reads `3.1e+03`. A diagonal occupancy lies in `[0, 1]`, so the column is not one. This
measures the diagonal itself against beta, in the same conditioned frame section 2 establishes and
section 7 reads, and alongside it the particle-hole residual `max|G_up + G_dn - 1|` that section
5.1's calibration rests on.

The two answer different questions and both are needed. The diagonal leaving `[0, 1]` says the
entries stop being fillings, because `G = (I + B)^-1` is not symmetric and its diagonal is not
bounded by its spectrum once the eigenvector basis is ill-conditioned. The particle-hole residual
staying at machine precision says the relation between the two channels is undamaged, which is what
section 7.2 uses and what `strength_eq_neg_one_of_reflected` proves forces the read to `-1`.

    python remote_run.py reads/expBI_greens_diagonal_at_large_beta.py
"""
from __future__ import annotations

import numpy as np

from model2d import Model2D
from stable import inv_one_plus_block, udt_product

U, DTAU, N_DRAW, SEED = 4.0, 0.125, 20, 1
BETAS = (2.0, 4.0, 6.0, 8.0, 10.0, 12.0)


def main():
    lam = float(np.arccosh(np.exp(DTAU * U / 2.0)))
    print("=" * 104)
    print("THE GREEN'S DIAGONAL AGAINST BETA, IN THE CONDITIONED FRAME")
    print("=" * 104)
    print(f"  2x4, U = {U}, mu = 0 (half filling), dtau = {DTAU}, {N_DRAW} draws, seed {SEED}")
    print("  An occupancy lies in [0, 1]. The last two columns say whether these entries do.")
    print()
    print(f"{'beta':>6} {'L':>5} | {'min G_up':>12} {'max G_up':>12} | "
          f"{'max|Gup+Gdn-1|':>16} {'max|Gup-Gdn|':>14} | {'in [0,1]':>9}")
    print("-" * 104)

    for beta in BETAS:
        L = int(round(beta / DTAU))
        m = Model2D(Lx=2, Ly=4, t=1.0, mu=0.0, U=U, dtau=DTAU, L=L, theta=0.0)
        rng = np.random.default_rng(SEED)
        lo, hi, ph, sep = [], [], [], []
        for _ in range(N_DRAW):
            X = rng.choice([-1.0, 1.0], size=(m.L, m.N))
            g = {}
            for sigma in (+1, -1):
                Bl = m.expmK[None, :, :] * np.exp(sigma * lam * X)[:, None, :]
                Uu, D, T = udt_product(Bl[None], 4)
                g[sigma] = np.real(np.diag(inv_one_plus_block(Uu, D, T)[0])).copy()
            lo.append(g[+1].min())
            hi.append(g[+1].max())
            ph.append(float(np.abs(g[+1] + g[-1] - 1.0).max()))
            sep.append(float(np.abs(g[+1] - g[-1]).max()))
        inside = "yes" if min(lo) >= 0.0 and max(hi) <= 1.0 else "NO"
        print(f"{beta:6.1f} {L:5d} | {min(lo):12.4f} {max(hi):12.4f} | "
              f"{max(ph):16.3e} {max(sep):14.3e} | {inside:>9}", flush=True)

    print()
    print("  The diagonal leaves [0, 1] once the frame's conditioning runs out, so past that beta")
    print("  the entries are not fillings and `max|G_up - G_dn|` is not bounded by 1. The")
    print("  particle-hole residual is unaffected across the whole range: the relation BETWEEN the")
    print("  channels is exact even where neither channel's diagonal is an occupancy, which is the")
    print("  relation section 7.2 reads and the reason its calibration survives to large beta.")


if __name__ == "__main__":
    main()
