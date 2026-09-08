"""Can the instrument choose the model order for a transfer-operator identification?

WHY THIS IS THE ONE WORTH ASKING.  §9's classification closes three rows and leaves one: identifying
an operator from a correlation sequence aggregates no signed configuration and compares no channels,
so neither the Kish ceiling nor the common-mode blindness applies to it.  The paper is explicit that
this is where to look and not a result -- what it costs to identify such an operator from data a
sign problem permits is not settled there.

The prize is the bottleneck itself.  A correlation sequence `C(tau) = sum_i c_i lambda_i^tau` is
measured at SHORT tau, where the sign is still mild and sampling is cheap.  The eigenvalues it
identifies fix the operator, and the operator gives the long-time behaviour that would otherwise be
sampled at large beta at a cost of `1/<sgn>^2`.

WHAT BLOCKED IT.  Model order -- how many `lambda_i` to keep.  Eight approaches failed, and the
reason they failed is instructive: every standard answer is a CHOSEN NUMBER.  An information
criterion picks a penalty; a singular-value cut picks a level; a stability window picks a width.
Under this repo's rules none of them is admissible, and picking one to make the answer come out is
precisely the defect the rest of this work has been removing.

WHAT IS DIFFERENT.  `spectral_optics` reports `resolved_modes` -- how many modes stand above a floor
the instrument DERIVES from the data rather than one a caller supplies.  That is model-order
selection performed by the read.

THE ANSWER, MEASURED 2026-09-08: IT DOES NOT, AND THE REASON IS STRUCTURAL RATHER THAN A FLOOR SET
WRONG.  `resolved_modes` returns 1 for every true order above 1, on three separate framings and at
every noise level including none:

    Hankel matrix of one sequence            -> 1 on k = 1, 2, 3, even at zero noise
    replicate sequences, one per bin         -> 0 on every k (correct: the modes are in the MEAN,
                                                and a correlation read centres it away)
    many correlators sharing one spectrum    -> 1 on k = 1, 2, 3, 4

The Hankel matrix is genuinely rank 3 where it should be -- `numpy.linalg.matrix_rank` says 3, with
singular values `6.02`, `0.727`, `0.0339` and then `1e-16`.  So the rank is there and the read does
not report it.  `spectral_optics` reads a UNIT-DIAGONAL CORRELATION MATRIX, and for a correlation
sequence every lag column is a decaying exponential of the same modes; normalising each lag to unit
variance makes them near-perfectly correlated with one another, so the correlation matrix is near
rank one whatever the true order.  Measured on the multi-operator frame: `top_share = 0.9949`.

MODEL ORDER FOR THIS OBJECT IS NOT A CORRELATION-RANK QUESTION.  It lives in the relative SCALE
structure across lags, which is exactly what correlation normalisation removes.  That is why a
ninth variance-based criterion would not have worked either, and it is worth having established
before writing one.  §9's fourth row is not closed by this: the physics argument for it stands, and
what is now measured is that this instrument family does not answer it and precisely why.

THE REST OF THIS FILE ANSWERS THAT ON DATA WHOSE ANSWER IS KNOWN.  A synthetic sequence
built from `k` exponentials with known decay rates and known noise, handed to the instrument as a
Hankel matrix.  If `resolved_modes` does not recover `k` here -- where there is no sign problem, no
Trotter error and no autocorrelation -- then it will not recover it on real data either, and the
route closes for a reason that is nothing to do with the sign problem.  Establishing that first is
the point: it is cheap, and it fails fast.
"""
from __future__ import annotations

import numpy as np

import entroptics as E


def hankel(c, rows=None):
    """The Hankel matrix of a sequence -- the object whose rank IS the number of exponentials.

    For `C(tau) = sum_i a_i lam_i^tau` the Hankel matrix `H[i, j] = C(i + j)` has rank exactly `k`
    in exact arithmetic, whatever the `a_i` and `lam_i` are. That is the identity the whole route
    rests on, so the order question is a RANK question, which is what the instrument reads.
    """
    c = np.asarray(c, float)
    n = len(c)
    r = rows if rows is not None else n // 2
    return np.array([c[i:i + (n - r + 1)] for i in range(r)])


def sequence(lams, amps, n_tau, noise, seed):
    """`C(tau) = sum_i a_i lam_i^tau` plus independent Gaussian noise of a known size."""
    rng = np.random.default_rng(seed)
    tau = np.arange(n_tau)
    c = np.sum([a * l ** tau for l, a in zip(lams, amps)], axis=0)
    return c + noise * rng.standard_normal(n_tau)


def order_read_by_the_instrument(c, rows=None):
    """`resolved_modes` on the sequence's Hankel matrix -- the order, chosen by the read."""
    H = hankel(c, rows)
    return int(E.reads.spectral_optics(H).resolved_modes), H


if __name__ == "__main__":
    TRUTH = [
        ("1 mode", [0.70], [1.0]),
        ("2 modes, well separated", [0.85, 0.40], [1.0, 0.8]),
        ("3 modes, well separated", [0.90, 0.55, 0.20], [1.0, 0.7, 0.5]),
        ("2 modes, close", [0.85, 0.75], [1.0, 0.8]),
        ("3 modes, one weak", [0.90, 0.55, 0.20], [1.0, 0.7, 0.02]),
    ]
    NOISE = [0.0, 1e-8, 1e-4, 1e-2]

    print("=" * 108)
    print("DOES THE INSTRUMENT RECOVER A KNOWN MODEL ORDER?  Synthetic sums of exponentials,")
    print("64 lags, Hankel matrix, order = spectral_optics(...).resolved_modes.")
    print()
    print("No order is chosen anywhere: the floor is derived by the read from the data it is given.")
    print()
    hdr = "  ".join(f"{f'noise {v:g}':>13}" for v in NOISE)
    print(f"{'truth':>26} {'k':>3} | {hdr}")
    for name, lams, amps in TRUTH:
        k = len(lams)
        cells = []
        for v in NOISE:
            got = [order_read_by_the_instrument(sequence(lams, amps, 64, v, s))[0]
                   for s in range(5)]
            agree = sum(g == k for g in got)
            cells.append(f"{str(sorted(set(got))):>9} {agree}/5")
        print(f"{name:>26} {k:3d} | " + "  ".join(f"{c:>13}" for c in cells), flush=True)

    print()
    print("=" * 108)
    print("A column that returns k on every seed is the instrument doing order selection.")
    print("A column that does not is this route closing for a reason unrelated to the sign problem,")
    print("which is worth knowing in one minute rather than after a ninth approach.")
