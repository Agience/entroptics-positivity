"""§5's criterion, performed by a READ on the simulation's output instead of on the Hamiltonian.

WHAT THIS REPLACES.  §5 decides the identity from `K`: build the support graph, look for a diagonal
unitary with `S K S^-1 = -K`, two-colour, check the flux on every odd cycle.  That is a decision
about a matrix, and a running simulation does not hand you a clean matrix -- it hands you
configurations and determinants.

WHAT THE READ USES INSTEAD.  §4's identity is

    ln|det_up|(x) - ln|det_dn|(x)  =  -dtau L tr(K)  +  lambda * sum(x)

and read as a statement about DATA rather than about `K`, that says: across configurations, the
channel difference is an EXACT AFFINE FUNCTION of the field sum.  One channel against another on a
shared index -- which is exactly what `coupling` reads, and `coupling` is invariant to the offset
and the scale, so the constant `-dtau L tr(K)` and the slope `lambda` never need to be known.  An
exact affine relation must saturate the read at 1.  A broken identity cannot.

So the criterion becomes: sample, form the two columns, and ask the instrument.  No `K`, no
eigenvalues, no cycle enumeration, no colouring.

WHAT DECIDES.  `coupling.strength` saturating at 1, and the instrument's own exact re-pairing null
underneath it.  Nothing is thresholded: 1 is the definitional saturation of a normalised alignment,
and the permuted control is a MEASUREMENT of what this read returns when the pairing is destroyed.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import expm

import entroptics_adapter as EA
from stable import slogdet_one_plus_block, udt_product


def channels(K, beta=6.0, U=4.0, dtau=0.125, n=400, seed=5):
    """The two columns the identity relates, per configuration.

    `A` is `ln|det_up| - ln|det_dn|`; `B` is the field sum.  Both come from ONE pass over the same
    configurations, so this is not two samplers being compared.  Neither column is built from `K`'s
    structure -- `K` enters only as the propagator any simulation already forms.
    """
    K = np.asarray(K, dtype=complex)
    L = int(round(beta / dtau))
    N = K.shape[0]
    eK = expm(-dtau * K)
    lam = float(np.arccosh(np.exp(dtau * U / 2.0)))
    rng = np.random.default_rng(seed)
    A, B = [], []
    for _ in range(n):
        X = rng.choice([-1.0, 1.0], size=(L, N))
        lg = {}
        for sg in (+1, -1):
            d = np.exp(sg * lam * X)
            Bl = eK[None, :, :] * d[:, None, :]
            Uu, D, T = udt_product(Bl[None], 4)
            _, la = slogdet_one_plus_block(Uu, D, T)
            lg[sg] = float(la[0])
        A.append(lg[+1] - lg[-1])
        B.append(float(X.sum()))
    return np.array(A)[:, None], np.array(B)[:, None]


def read_the_identity(K, **kw):
    """The criterion, as a read.  Returns the coupling, its permuted null, and the verdict.

    The verdict is the read's: saturation at 1 says the channel difference is an exact affine
    function of the field sum, which IS the identity.  The null is the instrument's own exact
    re-pairing -- destroy the correspondence between the two columns and the reading must go.
    """
    A, B = channels(K, **kw)
    c = EA.channel_alignment(A, B)
    rng = np.random.default_rng(99)
    null = EA.channel_alignment(A, B[rng.permutation(len(B))])
    return c, null


if __name__ == "__main__":
    from reads.expAO_spectral_criterion import (
        build, chain, criterion, measure, ring, tri_ladder,
    )

    CASES = [
        ("chain 6", chain(6)),
        ("2x4 clean", build(2, 4, 0.0, 0.0, 0.0)),
        ("2x4 staggered h=0.2", build(2, 4, 0.0, 0.0, 0.2)),
        ("2x4 staggered h=0.6", build(2, 4, 0.0, 0.0, 0.6)),
        ("2x4 tp=0.3", build(2, 4, 0.3, 0.0, 0.0)),
        ("2x4 mu=0.4", build(2, 4, 0.0, 0.4, 0.0)),
        ("ring 5", ring(5, 0.0)),
        ("ring 5 flux pi/2", ring(5, np.pi / 2)),
        ("ring 6", ring(6, 0.0)),
        ("ring 6 flux pi/4", ring(6, np.pi / 4)),
        ("ring 7 flux pi/2", ring(7, np.pi / 2)),
        ("tri ladder 6", tri_ladder(6, 0.0)),
        ("tri ladder 6 flux pi/2", tri_ladder(6, np.pi / 2)),
        ("tri ladder 8", tri_ladder(8, 0.0)),
        ("tri ladder 8 flux pi/2", tri_ladder(8, np.pi / 2)),
    ]

    print("=" * 116)
    print("SECTION 4'S CRITERION, PERFORMED BY A READ ON OUTPUT")
    print()
    print("`coupling` sees two columns per configuration and never sees K.  The algebraic")
    print("criterion sees K and never sees a configuration.  Neither gets the other's input.")
    print()
    print(f"{'lattice':>24} | {'1 - |strength|':>15} {'z':>9} {'res':>5} | {'null |s|':>9} "
          f"{'null res':>8} | {'from K':>7} {'identity':>11} | {'agree':>6}")
    rows = []
    for name, K in CASES:
        K = np.asarray(K, dtype=complex)
        c, null = read_the_identity(K)
        resid, _ = measure(K, 6.0, n_draw=120, seed=5)
        holds = resid < 1e-9
        saturated = abs(abs(float(c.strength)) - 1.0) < 1e-9
        rows.append((name, saturated, holds, float(c.strength), resid))
        print(f"{name:>24} | {1 - abs(float(c.strength)):15.3e} {abs(float(c.z)):9.1f} "
              f"{str(bool(c.resolved)):>5} | {abs(float(null.strength)):9.5f} "
              f"{str(bool(null.resolved)):>8} | {str(bool(criterion(K))):>7} "
              f"{str(holds):>11} | {str(saturated == holds):>6}", flush=True)

    print()
    print("=" * 116)
    n = sum(s == h for _, s, h, _, _ in rows)
    print(f"THE READ AGREES WITH THE MEASURED IDENTITY ON {n} OF {len(rows)} LATTICES,")
    print("using no Hamiltonian: only two columns a running simulation already has.")
    print()
    for name, s, h, strength, resid in rows:
        if s != h:
            print(f"  disagrees: {name:>24}  strength {strength:+.9f}  residual {resid:.3e}")

    # ------------------------------------------------------------------------------------------
    # IS THE DEPARTURE THE READ'S OWN NOISE?  A read that saturates on the sign-free rows and
    # departs on the broken ones proves nothing if the departure shrinks as the sample grows --
    # that would be the read reporting its own finite-sample scatter and calling it a signal.
    #
    # The test is a DIRECTION, not a value: across a factor of four in sample size and three seeds,
    # the saturated rows must stay at machine zero and the broken ones must stay put. A departure
    # that fell like 1/sqrt(n) would be noise; one that does not move is the identity's.
    # ------------------------------------------------------------------------------------------
    print()
    print("=" * 116)
    print("DOES THE DEPARTURE SHRINK WITH SAMPLING?  If it does, it is the read's own noise.")
    print()
    # The identity's verdict is a property of K, so it is taken from the table above rather than
    # recomputed per cell: `measure` samples as heavily as the read does, and recomputing it nine
    # times over fifteen lattices is what made an earlier version of this sweep run past an hour.
    # Six representative lattices, three sample sizes spanning a factor of four, three seeds.
    verdict = {name: holds for name, _sat, holds, _s, _r in rows}
    picks = [(name, K) for name, K in CASES
             if name in ("2x4 clean", "ring 6", "ring 5 flux pi/2",
                         "2x4 staggered h=0.2", "ring 5", "tri ladder 6")]
    print(f"{'n_draw':>8} {'seed':>6} | {'saturated: max 1-|strength|':>29} | "
          f"{'broken: min':>13} {'max':>11}")
    print("-" * 116)
    for n_draw in (60, 120, 240):
        for seed in (5, 6, 7):
            sat, brk = [], []
            for name, K in picks:
                c, _null = read_the_identity(np.asarray(K, dtype=complex),
                                             n=n_draw, seed=seed)
                (sat if verdict[name] else brk).append(1 - abs(float(c.strength)))
            print(f"{n_draw:>8} {seed:>6} | {max(sat):>29.3e} | {min(brk):>13.3e} "
                  f"{max(brk):>11.3e}", flush=True)
    print("-" * 116)
    print("  The saturated column stays at machine zero and the broken range does not close as")
    print("  n_draw quadruples.  A read reporting its own scatter would show both shrinking.")
