"""Experiment AE -- is -1 the reading, or is SATURATION the reading?

Every measurement so far reads the coupling at the repulsive symmetric point, where it comes out
at exactly -1.  A read that only ever returns -1 where the model is sign-free is indistinguishable
from a read that has learned one number, so the question is what it does at the OTHER place this
model is provably sign-free.

There are two of them, and they are structurally opposite:

  REPULSIVE, spin channel, half filling.  `G_dn = 1 - G_up` exactly.  The two channels are exact
      NEGATIVES about 1/2, and the determinants share a sign through the identity of section 4.
      Anti-synchronised.

  ATTRACTIVE, charge channel, ANY filling.  The field couples to `n_up + n_dn - 1`, so the two
      spins see the IDENTICAL diagonal factor, `B_up = B_dn`, and the weight is `det^2 >= 0`.
      Synchronised -- and, unlike the repulsive case, doping does not touch it.

So the two mechanisms make opposite predictions for a read that is measuring synchronisation, and
the same prediction for a read that is reporting a constant.  Doping is the discriminator: it is
exactly what destroys saturation on the repulsive rows, and it must leave the attractive rows
untouched, because the attractive model stays sign-free there.

lambda is the decoupling's own constant in both channels, `arccosh(exp(dtau |U| / 2))`, with no
free parameter anywhere in the file.

EVERY ROW IS REPLICATED OVER SEEDS AND REPORTED AS A RANGE.  A first pass ran one seed per row and
printed `-0.2670` for the repulsive mu = 0.4 row; the same row over six seeds runs -0.3532 to
0.0000, a spread the size of the value.  The saturated rows do not move at all, which is what makes
the contrast readable -- but a single draw could not have told the two situations apart.
"""
from __future__ import annotations

import numpy as np

import entroptics_adapter as EA
from model2d import Model2D
from stable import udt_product, inv_one_plus_block, slogdet_one_plus_block


def read_run(Lx, Ly, beta, U, mu, channel, dtau=0.125, n_draw=300, seed=0):
    """One row: the two channels' Green's diagonals, the weight's sign, and the coupling.

    `channel` selects which exact rewriting of the interaction is decoupled -- 'spin' couples the
    field to `n_up - n_dn`, so the two spins get opposite diagonal factors; 'charge' couples it to
    `n_up + n_dn - 1`, so they get the same one.  Both are exact identities on the four states of
    a site, so this is not two approximations being compared.
    """
    L = int(round(beta / dtau))
    m = Model2D(Lx=Lx, Ly=Ly, t=1.0, mu=mu, U=abs(U), dtau=dtau, L=L, theta=0.0)
    lam = float(np.arccosh(np.exp(dtau * abs(U) / 2.0)))
    rng = np.random.default_rng(seed)
    A, B, S, sep = [], [], [], []
    for _ in range(n_draw):
        X = rng.choice([-1.0, 1.0], size=(m.L, m.N))
        row, s = {}, 1.0
        for sigma in (+1, -1):
            # spin: the two channels see  +lam x  and  -lam x.  charge: both see  +lam x.
            e = (sigma if channel == "spin" else +1) * lam
            d = np.exp(e * X)
            Bl = m.expmK[None, :, :] * d[:, None, :]
            Uu, D, T = udt_product(Bl[None], 4)
            row[sigma] = np.real(np.diag(inv_one_plus_block(Uu, D, T)[0])).copy()
            sg, _ = slogdet_one_plus_block(Uu, D, T)
            s *= float(np.real(sg[0]))
        A.append(row[+1]); B.append(row[-1]); S.append(s)
        sep.append(float(np.abs(row[+1] - row[-1]).max()))
    A, B, S = np.array(A), np.array(B), np.array(S)
    c = EA.channel_alignment(A, B)
    # NO FILLING COLUMN.  These are uniform draws over the field, not importance samples, so no
    # expectation value estimated from them means anything -- the same effective-sample collapse
    # that makes the average sign unmeasurable this way.  `mu` is an input and needs no estimator.
    return dict(neg=float(np.mean(S < 0)), strength=float(c.strength), z=float(c.z),
                resolved=bool(c.resolved), sep=float(np.max(sep)))


if __name__ == "__main__":
    print("=" * 116)
    print("TWO SIGN-FREE MECHANISMS, OPPOSITE IN STRUCTURE.  2x4, |U| = 4, dtau = 0.125,")
    print("300 configurations per row, SIX SEEDS per row reported as a range.  Rows are labelled")
    print("by their INPUTS; see the note below on why no filling is estimated here.")
    print()
    print(f"{'U':>5} {'channel':>8} {'mu':>5} {'lattice':>8} {'beta':>5} | "
          f"{'neg sign, over seeds':>21} | {'strength, over seeds':>20} {'|s| = 1':>8} | "
          f"{'max |Gup-Gdn|':>12}")

    SEEDS = (1, 7, 23, 45, 93, 101)
    rows = []
    for beta in (6.0, 10.0):
        for mu in (0.0, 0.4, 0.8):
            rows.append((2, 4, beta, +4.0, mu, "spin"))
        for mu in (0.0, 0.4, 0.8):
            rows.append((2, 4, beta, -4.0, mu, "charge"))
    rows.append((2, 3, 10.0, +4.0, 0.0, "spin"))
    rows.append((3, 4, 10.0, +4.0, 0.0, "spin"))

    for (Lx, Ly, beta, U, mu, ch) in rows:
        rs = [read_run(Lx, Ly, beta, U, mu, ch, seed=s) for s in SEEDS]
        st = [r["strength"] for r in rs]
        neg = [r["neg"] for r in rs]
        sep = max(r["sep"] for r in rs)
        sat = "YES" if all(r["resolved"] and abs(abs(r["strength"]) - 1.0) < 1e-3 for r in rs)             else "no"
        print(f"{U:5.1f} {ch:>8} {mu:5.2f} {f'{Lx}x{Ly}':>8} {beta:5.1f} | "
              f"{min(neg):.4f}-{max(neg):.4f} | {min(st):8.4f} to {max(st):8.4f} {sat:>8} | "
              f"{sep:12.3e}", flush=True)
    print()
    print("READ THE LAST COLUMN BEFORE THE STRENGTH.  On the attractive rows the two channels are")
    print("BIT-IDENTICAL, so the read is being handed the same array twice: +1 is the SIGN of the")
    print("reading and carries nothing about the instrument's discrimination.  What is load-bearing")
    print("is the CONTRAST -- doping is what desaturates the repulsive rows, and it leaves the")
    print("attractive ones exactly where they were, because the attractive model stays sign-free.")
    print()
    print("Note also that saturation and a zero measured negative fraction are NOT equivalent.  The")
    print("repulsive mu = 0.8, beta = 6 row runs a negative fraction of 0.0000 to 0.0100 while the")
    print("read runs -0.87 to -0.54: the departure from saturation exceeds the negative fraction on")
    print("every seed, including the seed where the negative fraction is zero.  That is the section")
    print("5 result -- the read departs before the average sign does -- not a contradiction of it.")
    print()
    print("The ranges earn their place on the LAST TWO ROWS.  A non-bipartite lattice has a seed")
    print("with no negative weight in 300 draws; reported alone it would look like a counterexample.")
