"""Experiment AT -- the screen's balance read, given the system's OWN zero.

`Screen.register` takes a lens's own laws: `entry`, `inverse`, `energy` and `zero`.  Its
documentation is explicit about what happens if they are withheld -- the side inherits the
library's derived defaults, "the right laws for a system that has no others and THE WRONG ONES FOR
A SYSTEM THAT DOES".

This system has its own zero.  At half filling on a bipartite lattice the two channels satisfy
`G_up[i,i] + G_dn[i,i] = 1` configuration by configuration, so each channel balances at **1/2** --
the particle-hole point.  The library's default zero is the column mean, which is the sample's own
centre and therefore says nothing about where the system balances.

WITH THE DEFAULT ZERO THERE IS NO READ.  The default zero IS the column mean, so the residual
scored against it is identically zero and the pvalue is exactly 1 whatever the data -- an identity,
not a measurement.  The control column is printed to show that the separation belongs to the zero.

READ THE PVALUE, NOT THE BOOLEAN.  `closed` is a per-run decision at the reader's level and
fluctuates: on a sign-free ring it fires 9 times in 12.  The pvalue behind it is the quantity, and
its median across independent runs separates with no overlap.  That is the same discipline the rest
of this paper uses for every sampled figure.

SCOPE.  This separates REAL-weight sign problems at fixed filling.  It does not see a phase
problem: the route B cases of section 5, whose channels are exactly related and whose weights carry
a phase, read closed.  That is consistent with section 7 -- the relation between the channels does
not carry the phase, and this read is a statement about each channel against its own zero.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import expm

import entroptics_adapter as EA
from stable import udt_product, inv_one_plus_block, slogdet_one_plus_block

PARTICLE_HOLE_ZERO = 0.5


def channels(Kmat, seed, beta=8.0, U=4.0, dtau=0.125, n=300):
    """The two channels' Green's diagonals, and the weight's mean-phase deficit."""
    L = int(round(beta / dtau)); N = Kmat.shape[0]
    eK = expm(-dtau * np.asarray(Kmat, dtype=complex))
    lam = float(np.arccosh(np.exp(dtau * U / 2.0)))
    rng = np.random.default_rng(seed)
    A, B, W = [], [], []
    for _ in range(n):
        X = rng.choice([-1.0, 1.0], size=(L, N))
        g, d_ = {}, {}
        for sg in (+1, -1):
            d = np.exp(sg * lam * X)
            Bl = eK[None, :, :] * d[:, None, :]
            Uu, D, T = udt_product(Bl[None], 4)
            g[sg] = np.diag(inv_one_plus_block(Uu, D, T)[0]).copy()
            s, _ = slogdet_one_plus_block(Uu, D, T)
            d_[sg] = complex(s[0])
        A.append(g[+1]); B.append(g[-1]); W.append(d_[+1] * d_[-1])
    return np.array(A), np.array(B), 1.0 - float(abs(np.asarray(W).mean()))


def balance_pvalue(A, B, own_zero=True):
    """The smallest side pvalue from `Screen.balance`, and the boolean it implies.

    `own_zero=False` withholds the law and lets the library default to the column mean, which is
    the control: it is what makes the system's own zero the thing being tested.
    """
    s = EA.balance_at_own_zero()
    kw = {}
    if own_zero:
        kw["zero"] = lambda x: np.full(np.asarray(x).shape[1],
                                       PARTICLE_HOLE_ZERO + 0j)
    s.register("up", entry=lambda x: np.asarray(x), **kw)
    s.register("dn", entry=lambda x: np.asarray(x), **kw)
    s.place("up", A); s.place("dn", B)
    b = s.balance()
    return float(min(b.pvalue.values())), bool(all(b.closed.values()))


if __name__ == "__main__":
    from reads.expAO_spectral_criterion import build, ring, tri_ladder

    SEEDS = (5, 11, 23, 41, 67, 83, 97, 109, 127, 151, 173, 199)
    cases = [("2x4 clean", build(2, 4, 0, 0, 0), "no"),
             ("ring 6 flux pi/4", ring(6, np.pi / 4), "no"),
             ("ring 8 flux pi/4", ring(8, np.pi / 4), "no"),
             ("staggered h = 0.2", build(2, 4, 0, 0, 0.2), "yes"),
             ("staggered h = 0.6", build(2, 4, 0, 0, 0.6), "yes"),
             ("ring 5 flux pi/2", ring(5, np.pi / 2), "phase"),
             ("tri ladder pi/2", tri_ladder(8, np.pi / 2), "phase")]

    print("=" * 104)
    print(f"BALANCE AT THE SYSTEM'S OWN ZERO ({PARTICLE_HOLE_ZERO}), all rows at half filling.")
    print(f"{len(SEEDS)} independent runs per row; the median is the quantity, not any one run.")
    print()
    print(f"{'case':>20} {'sign problem':>13} | {'deficit':>8} | "
          f"{'OWN zero: median p':>19} {'closed':>8} | {'DEFAULT zero: median p':>23}")
    for name, K, sp in cases:
        Kc = np.asarray(K, dtype=complex)
        own, dflt, cl, ds = [], [], [], []
        for sd in SEEDS:
            A, B, d = channels(Kc, sd)
            ds.append(d)
            p, c = balance_pvalue(A, B, True); own.append(p); cl.append(c)
            dflt.append(balance_pvalue(A, B, False)[0])
        print(f"{name:>20} {sp:>13} | {np.mean(ds):8.4f} | {np.median(own):19.5f} "
              f"{sum(cl):3d}/{len(SEEDS):<3d} | {np.median(dflt):23.5f}", flush=True)
    print()
    print("The DEFAULT column is an identity, not a measurement: the default zero IS the column")
    print("mean, so the residual scored against it is identically zero and the pvalue is exactly")
    print("1 whatever the data. Withholding the system's law does not weaken the read, it removes")
    print("it -- so the separation in the OWN column belongs entirely to the zero.")
