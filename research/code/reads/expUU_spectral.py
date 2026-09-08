"""Experiment UU -- the sign is a determinant sign in the conditioned frame, and the order
parameter is the distance to a zero.

`expTT` established the particle-hole identity exactly (max residual 6.7e-15 at half filling) and
then failed to make it per-configuration: the residual of `G_up[i,i] + G_dn[i,i] = 1` has a
tendency but no separation -- positive-weight configurations reach r = 190 while negative-weight
ones go down to r = 0.22.  The Green's-function diagonal is a coarse summary of the object that
carries the sign.

The object that carries it is the SPECTRUM.  `det(I + B) = prod_k (1 + lambda_k)`, so a sign flip
is an eigenvalue crossing, and the continuous quantity preceding a flip is the distance to it.

WHICH FRAME.  Not `I + B`.  A first version formed B as the naive ordered product and read its
eigenvalues; at beta = 6 and 8 it reported negative-weight fractions of 0.0225 and 0.5075 AT HALF
FILLING, where positivity is provable, with "distances" of 56, 1117 and 31683 where the quantity is
O(1).  `stable.py` states the reason itself: `I + B` has condition ~exp(beta * bandwidth) and its
eigenvalues stop meaning anything past beta ~ 4.  `core_matrix` gives

    I + U D T = U Db M T,     M = Db^-1 U' T^-1 + Ds,     prod(Db) > 0

so the sign is carried entirely by `det(U) det(T) det(M)` and M is bounded by construction.

NOTHING HERE IS A THRESHOLD, A CONSTANT OR A FIT.

  * The sign is `slogdet`, not a count of eigenvalues below a level.  Counting real eigenvalues
    would need a tolerance to decide which are real -- a threshold -- and it is unnecessary: the
    parity of real negative eigenvalues IS the determinant's sign, exactly.
  * The order parameter is `min_k |lambda_k(M)|`, the distance from the spectrum to zero.  A
    minimum modulus needs no tolerance.  It is zero exactly at a sign flip and positive otherwise.
  * The stabilisation block is not chosen.  Every row is computed at three block sizes and the
    spread across them is reported; a row whose answer depends on the block is a row where the
    arithmetic, not the physics, is speaking.
  * The agreement column compares two INDEPENDENT routes to the sign -- the block formulation and
    the core matrix's own factored sign.  An earlier version compared a spectrum against itself,
    which agrees on garbage and did.
"""
from __future__ import annotations

import numpy as np

from model2d import Model2D
from stable import udt_product, core_matrix, slogdet_one_plus_block


def read_config(m, X, block):
    """(sign from the block form, sign from the core factorisation, distance to a zero)."""
    lam = float(np.arccosh(np.exp(m.dtau * m.U / 2.0)))
    s_block, s_core, gap = 1.0, 1.0, np.inf
    for sigma in (+1, -1):
        d = np.exp(sigma * lam * X)
        Bl = m.expmK[None, :, :] * d[:, None, :]
        U, D, T = udt_product(Bl[None], block)
        s, _ = slogdet_one_plus_block(U, D, T)
        s_block *= float(np.real(s[0]))
        M, sc = core_matrix(U, D, T)
        s_core *= float(np.real(sc[0]))
        gap = min(gap, float(np.min(np.abs(np.linalg.eigvals(M[0])))))
    return s_block, s_core, gap


if __name__ == "__main__":
    Lx, Ly, U, dtau = 2, 4, 4.0, 0.125
    n_draw = 500
    blocks = (2, 4, 8)
    print("=" * 116)
    print(f"THE SIGN IN THE CONDITIONED FRAME   {Lx}x{Ly}, U = {U}, {n_draw} configurations")
    print("Order parameter: min |eigenvalue| of the core matrix -- the distance to a sign flip.")
    print("No threshold, no constant: the sign is a determinant sign and the parameter a minimum")
    print("modulus.  Every row is computed at three stabilisation blocks and the spread reported.")
    print()
    print(f"{'beta':>5} {'mu':>5} | {'routes agree':>13} {'neg frac':>9} {'blk spread':>11} | "
          f"{'min d | w>0':>12} {'min d | w<0':>12} {'med d+':>9} {'med d-':>9} {'sep':>8}")
    for beta in (2.0, 4.0, 6.0, 8.0, 10.0):
        L = int(round(beta / dtau))
        for mu in (0.0, 0.4, 0.8):
            m = Model2D(Lx=Lx, Ly=Ly, t=1.0, mu=mu, U=U, dtau=dtau, L=L, theta=0.0)
            per_block = []
            for blk in blocks:
                rng = np.random.default_rng(int(beta * 100 + mu * 10))
                agree, S, G = 0, [], []
                for _ in range(n_draw):
                    X = rng.choice([-1.0, 1.0], size=(m.L, m.N))
                    sb, sc, g = read_config(m, X, blk)
                    agree += int(np.sign(sb) == np.sign(sc))
                    S.append(sb); G.append(g)
                per_block.append((agree / n_draw, np.array(S), np.array(G)))
            # the block the row is reported at is the middle one; the spread is the check
            negs = [float(np.mean(s < 0)) for _, s, _ in per_block]
            spread = max(negs) - min(negs)
            agree, S, G = per_block[1]
            pos, neg = G[S > 0], G[S < 0]
            c = lambda a, w: (f"{a.min():{w}.6f}" if len(a) else f"{'--':>{w}}")
            md = lambda a, w: (f"{np.median(a):{w}.6f}" if len(a) else f"{'--':>{w}}")
            sep = "--"
            if len(pos) and len(neg):
                sep = "YES" if neg.min() > pos.max() else "overlap"
            print(f"{beta:5.1f} {mu:5.2f} | {agree:13.4f} {negs[1]:9.4f} {spread:11.4f} | "
                  f"{c(pos,12)} {c(neg,12)} {md(pos,9)} {md(neg,9)} {sep:>8}", flush=True)
    print()
    print("'routes agree' must be 1.0000 -- two independent paths to the same sign.")
    print("At half filling 'neg frac' must be 0.0000 at EVERY beta; that is the row the naive")
    print("product got wrong.  'blk spread' must be 0.0000 -- a row that moves with the")
    print("stabilisation block is arithmetic, not physics.")
