"""Experiment OO -- read the SIGN LOCK with Entroptics' coupling, the instrument built for it.

Part 8 measured the structure that carries positivity, and it is a coupling structure:

    at half filling  sign(det_up) = sign(det_dn) in 300/300 configurations at every beta,
    while           log|det_dn| - log|det_up| is spread over 8 to 16 ORDERS.

Two sides, locked in one channel and independent in another.  That is what `reads.coupling`
measures, and it has never been pointed at this.  Part 1's coupling test was
`coupling(A, diag(sigma) A)` -- a frame against its own sign-flipped self -- which is why it
collapsed onto the average sign.  Coupling the UP channel to the DOWN channel is a different
question about two physically distinct objects whose relationship IS the positivity.

WHY THIS INSTRUMENT AND NOT A CORRELATION COEFFICIENT.  `coupling` standardises the measured
alignment by its EXACT null -- a uniformly random re-pairing of the two sides, each keeping its
own internal structure -- so `z` answers "are these two coupled" and not "does either have
structure", with no constant supplied and nothing fitted.  It also returns `resolved`: how many
modes the coupling actually occupies.  That is the quantity this experiment exists for.

WHAT WOULD BE ACTIONABLE.  Part 2's collapse law says no ESTIMATOR of a fixed sum beats
M<sgn>^2.  It says nothing about SPLITTING the sum.  If the lock-breaking is confined to a small
number of modes, those modes can be treated exactly and the rest sampled -- which is outside what
the collapse law forbids.  `resolved` is what says whether that number is small.

Gates, both from the instrument's own null:
  * half filling must read as coupled -- the lock is provable there;
  * a row-permuted second side must read as uncoupled, which is the negative control the exact
    null makes available for free.
"""
from __future__ import annotations

import numpy as np

import entroptics as E
from model2d import Model2D
from stable import udt_product, inv_one_plus_block, slogdet_one_plus_block


def sample(m, n_draw, seed=0):
    """Per-configuration channel frames and the weight's sign.

    A[x] and B[x] are the up and down Green's-function diagonals -- the one-body content of each
    channel at that configuration, on the same ordered axis (the configuration index), which is
    what `coupling` requires.  Stabilised, because a naive product fakes a sign problem and that
    lesson has already been relearned once in this directory.
    """
    rng = np.random.default_rng(seed)
    lam = float(np.arccosh(np.exp(m.dtau * m.U / 2.0)))
    A, B, sg, lg = [], [], [], []
    for _ in range(n_draw):
        X = rng.choice([-1.0, 1.0], size=(m.L, m.N))
        row, sgn, lsum = {}, 1.0, 0.0
        for sigma in (+1, -1):
            d = np.exp(sigma * lam * X)
            Bl = m.expmK[None, :, :] * d[:, None, :]
            U, D, T = udt_product(Bl[None], 4)
            G = inv_one_plus_block(U, D, T)[0]
            s, la = slogdet_one_plus_block(U, D, T)
            row[sigma] = np.real(np.diag(G)).copy()
            sgn *= float(np.real(s[0]))
            lsum += float(la[0])
        A.append(row[+1]); B.append(row[-1]); sg.append(sgn); lg.append(lsum)
    return np.array(A), np.array(B), np.array(sg), np.array(lg)


def report(tag, a, b, rng):
    c = E.reads.coupling(a, b)
    perm = rng.permutation(len(b))
    cn = E.reads.coupling(a, b[perm])
    return c, cn


if __name__ == "__main__":
    Lx, Ly, U, dtau = 2, 4, 4.0, 0.125
    n_draw = 400
    rng = np.random.default_rng(7)
    print("=" * 108)
    print(f"THE SIGN LOCK, READ AS A COUPLING   {Lx}x{Ly}, U = {U}, {n_draw} configurations")
    print("Sides: the up and down Green's-function diagonals, on the configuration axis.")
    print("`z` is standardised by the exact re-pairing null; the permuted row is the control.")
    print()
    print(f"{'beta':>5} {'mu':>5} {'neg frac':>9} | {'z':>9} {'strength':>9} {'tightness':>10} "
          f"{'resolved':>9} {'sign':>5} | {'z (permuted)':>13} {'resolved':>9}")
    for beta in (2.0, 4.0, 6.0, 8.0):
        L = int(round(beta / dtau))
        for mu in (0.0, 0.4):
            m = Model2D(Lx=Lx, Ly=Ly, t=1.0, mu=mu, U=U, dtau=dtau, L=L, theta=0.0)
            A, B, sg, lg = sample(m, n_draw, seed=int(beta * 10) + int(mu * 10))
            c, cn = report("", A, B, rng)
            neg = float(np.mean(sg < 0))
            print(f"{beta:5.1f} {mu:5.2f} {neg:9.4f} | {c.z:9.2f} {c.strength:9.4f} "
                  f"{c.tightness:10.4f} {c.resolved:9d} {c.sign:5.0f} | "
                  f"{cn.z:13.2f} {cn.resolved:9d}", flush=True)
    print()
    print("If `resolved` is small where the lock breaks, the breaking lives in few modes and")
    print("those modes could be treated exactly while the rest is sampled -- which is outside")
    print("what Part 2's collapse law forbids, since that law is about estimating a fixed sum")
    print("and not about splitting one.")
