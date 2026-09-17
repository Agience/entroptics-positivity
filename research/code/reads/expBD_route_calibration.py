"""Experiment BD -- the calibration table of section 7, printed.

WHY THIS FILE EXISTS.  Section 7 quotes a table of nine lattices with columns `route A`,
`route B`, `max|G_up + G_dn - 1|` and `strength`, and attributes it to
`reads/expAO_spectral_criterion.py`.  That file supplies the criterion -- `route_A`, `route_B`,
`ring`, `tri_ladder`, `measure` -- but its own `__main__` prints a different table entirely (the
spectral-asymmetry comparison on a 2x4).  The measurements behind the quoted table live in
`tests/test_spectral_criterion.py`, where they are ASSERTED as bounds and never printed.

So the numbers were reproducible in the sense that a gate holds them, and not reproducible in the
sense the paper's own measurement note requires: nothing printed them.  This does.

WHAT IT SHOWS.  Section 5 gives two routes to the identity.  Route A is `S K S^-1 = -K`; route B
is `S K S^-1 = -conj(K)`.  Both give the section-4 identity; only route A leaves the weights real.
Only route B
forces the particle-hole relation `G_dn = 1 - G_up` configuration by configuration, and only route
B therefore calibrates the coupling to exactly `-1`.

That separation is the point, and it is why section 7 attributes the calibration to route B rather
than to the identity: on `ring 6` and `ring 8` at flux `pi/4` the identity still holds and the
lattice is still sign-free, but the two channels are no longer exact negatives about `1/2` and the
read comes back close to `-1` without reaching it.  Positivity, the identity and the exact `-1`
coincide everywhere else in sections 2 to 4 and come apart here.
"""
from __future__ import annotations

import numpy as np

from scipy.linalg import expm                                     # noqa: E402

import entroptics_adapter as EA                                            # noqa: E402
from stable import udt_product, inv_one_plus_block                # noqa: E402
from reads.expAO_spectral_criterion import (                      # noqa: E402
    build, measure, ring, route_A, route_B, tri_ladder, triangular,
)

U, DTAU, BETA, N_DRAW, SEED = 4.0, 0.125, 8.0, 250, 5


def channels(K, beta=BETA, n=N_DRAW, seed=SEED):
    """The two channels' Green's-function diagonals, complex where K is.

    This is `tests/test_spectral_criterion._channels` verbatim in construction, deliberately: the
    point of this file is to PRINT what that gate asserts, so reproducing its arithmetic by a
    different route would make the two disagree for reasons that have nothing to do with the
    physics.  `lambda` is the Hirsch constant of section 4, not a free parameter.
    """
    K = np.asarray(K, dtype=complex)
    N = K.shape[0]
    L = int(round(beta / DTAU))
    eK = expm(-DTAU * K)
    lam = float(np.arccosh(np.exp(DTAU * U / 2.0)))
    rng = np.random.default_rng(seed)
    A, B = [], []
    for _ in range(n):
        X = rng.choice([-1.0, 1.0], size=(L, N))
        g = {}
        for sg in (+1, -1):
            d = np.exp(sg * lam * X)
            Bl = eK[None, :, :] * d[:, None, :]
            Uu, D, T = udt_product(Bl[None], 4)
            g[sg] = np.diag(inv_one_plus_block(Uu, D, T)[0]).copy()
        A.append(g[+1])
        B.append(g[-1])
    return np.array(A), np.array(B)


CASES = [
    ("2x4 real",                    build(2, 4, 0.0, 0.0, 0.0)),
    ("ring 8, real",                ring(8, 0.0)),
    ("ring 6, flux pi",             ring(6, np.pi)),
    ("ring 5, flux pi/2",           ring(5, np.pi / 2)),
    ("ring 7, flux pi/2",           ring(7, np.pi / 2)),
    ("triangular 4x4, flux pi/2",   triangular(4, 4, np.pi / 2)),
    ("ring 6, flux pi/4",           ring(6, np.pi / 4)),
    ("ring 6, flux pi/3",           ring(6, np.pi / 3)),
    ("ring 8, flux pi/4",           ring(8, np.pi / 4)),
]

if __name__ == "__main__":
    print("=" * 104)
    print(f"THE CALIBRATION COMES FROM ROUTE B, NOT FROM THE IDENTITY."
          f"   U = {U}, beta = {BETA}, {N_DRAW} configurations per row")
    print("Both routes give the section-4 identity; only route A leaves the weights real.")
    print("Only route B forces")
    print("G_dn = 1 - G_up configuration by configuration, and only route B calibrates the read.")
    print("=" * 104)
    print(f"{'K':<28}{'route A':>9}{'route B':>9}{'max|G_up+G_dn-1|':>20}"
          f"{'strength':>12}{'identity resid':>17}")
    print("-" * 104)
    for name, K in CASES:
        Kc = np.asarray(K, dtype=complex)
        a, b = bool(route_A(Kc)), bool(route_B(Kc))
        A, B = channels(Kc)
        ph = float(np.abs(A + B - 1.0).max())
        c = EA.channel_alignment(A, B)
        resid, _neg = measure(Kc, BETA, n_draw=100, seed=SEED)
        print(f"{name:<28}{('yes' if a else 'no'):>9}{('yes' if b else 'no'):>9}"
              f"{ph:>20.1e}{float(c.strength):>12.4f}{resid:>17.1e}")
    print("-" * 104)
    print()
    print("  Where route B is open the particle-hole relation holds to machine precision and the")
    print("  read is exactly -1.  Where only route A is open the lattice is STILL sign-free and")
    print("  the identity STILL holds -- the last column -- but the channels are no longer exact")
    print("  negatives about 1/2 and the read stops short of -1.  That is the separation, and it")
    print("  is why the calibration is attributed to route B rather than to the identity.")

    # ------------------------------------------------------------------------------------------
    # Section 7.1's second collision, which was quoted against `expAO` and printed by nothing.
    #
    # It is the same construction as the table above with a different question asked of it. There
    # the columns answer "which route calibrates the read"; here they answer "can a read of the
    # RELATION between the channels see a phase problem", and the answer is that it cannot: the
    # relation is exact on every row while the sign deficit runs from 0 to about 0.5.
    #
    # NO SECOND READ IS CARRIED HERE. The paper used to quote a `Screen match` column beside the
    # coupling, and nothing could produce it: `Screen.couple` takes the NAMES of lenses registered
    # on a screen, not two arrays, so the values had no call behind them. The blindness does not
    # need a second read to be visible -- the relation being exact on every row while the deficit
    # spans zero to about one half is the whole statement -- so the column is gone rather than
    # reconstructed.
    # ------------------------------------------------------------------------------------------
    COLLISION = [
        ("2x4 real, half filled",        build(2, 4, 0.0, 0.0, 0.0)),
        ("ring 5, flux pi/2",            ring(5, np.pi / 2)),
        ("ring 7, flux pi/2",            ring(7, np.pi / 2)),
        ("triangular ladder, flux pi/2", tri_ladder(6, np.pi / 2)),
    ]
    print()
    print("=" * 104)
    print(f"THE CHANNELS EXACTLY RELATED, AND THE SIGN BEHAVIOUR NOT."
          f"   beta = {BETA}, {N_DRAW} configurations per row")
    print("Route B restores the identity on an odd cycle and leaves a phase behind.  If a read of")
    print("the RELATION could see that phase, these rows would not all saturate.")
    print("=" * 104)
    print(f"{'K':<32}{'relation resid':>16}{'sign deficit':>14}{'coupling':>11}")
    print("-" * 104)
    for name, K in COLLISION:
        Kc = np.asarray(K, dtype=complex)
        A, B = channels(Kc)
        rel = float(np.abs(A + B - 1.0).max())
        _resid, deficit = measure(Kc, BETA, n_draw=N_DRAW, seed=SEED)
        c = EA.channel_alignment(A, B)
        print(f"{name:<32}{rel:>16.1e}{deficit:>14.5f}{float(c.strength):>11.4f}")
    print("-" * 104)
    print()
    print("  The relation is exact on every row and the read returns the same saturated value on")
    print("  every row, while the sign deficit spans zero to about one half.  No read of the")
    print("  relation between the two channels can see a phase problem, because the relation does")
    print("  not carry it: G_dn = 1 - G_up holds whether the weight is positive or spread around")
    print("  the circle.")
