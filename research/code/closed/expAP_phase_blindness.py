"""Experiment AP -- a magnitude read cannot see a sign, for any exponent.

Section 7 reads a COUPLING between two channels rather than a magnitude of either. This file is
why: every read that depends on the data only through its magnitudes is blind to the sign, and the
blindness is exact rather than a matter of resolution.

The construction is one line. Take any frame `A` and flip the sign of whole rows,
`B = diag(sigma) A` with `sigma` in {-1, +1}. Then

    |B_tj| = |A_tj|   cell for cell

so `|A|^q` and `|B|^q` agree for EVERY exponent `q`, and any functional of the magnitudes agrees
with them. The singular spectrum is blind too, because `diag(sigma)` is orthogonal and singular
values are invariant under an orthogonal factor.

So the frames are identical to a magnitude read and maximally different in sign -- and the sign of
the determinantal weight is exactly what a sign problem is made of. Only a read that compares two
sides can see it, which is what section 7 uses.

The test carries its own positive control: the coupling BETWEEN the two frames does see the flip,
so the blindness demonstrated is a property of magnitude reads and not of the frames being
indistinguishable.
"""
from __future__ import annotations

import numpy as np

import entroptics_adapter as EA


def flipped(A, sigma):
    """B = diag(sigma) A -- whole rows negated, magnitudes untouched."""
    return sigma[:, None] * A


def magnitude_moments(A, qs=(0.5, 1.0, 1.5, 2.0, 3.0)):
    """sum |A|^q for a range of exponents: the family every power-marginal read lives in."""
    M = np.abs(A)
    return {q: float((M ** q).sum()) for q in qs}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    T, N = 400, 8
    A = rng.standard_normal((T, N))

    half = np.where(rng.random(T) < 0.5, -1.0, 1.0)
    allf = -np.ones(T)
    frames = {"A (original)": A,
              "B (half the rows flipped)": flipped(A, half),
              "C (every row flipped)": flipped(A, allf)}

    print("=" * 96)
    print("A MAGNITUDE READ CANNOT SEE A SIGN, FOR ANY EXPONENT.")
    print(f"T = {T}, N = {N}. B and C differ from A only by row sign flips.")
    print()
    print(f"{'frame':>28} | {'max |diff of |X||':>18} | " +
          "  ".join(f"sum|X|^{q}" for q in (0.5, 1.0, 2.0, 3.0)))
    base = magnitude_moments(A)
    for name, X in frames.items():
        cell = float(np.abs(np.abs(X) - np.abs(A)).max())
        mom = magnitude_moments(X)
        vals = "  ".join(f"{mom[q]:9.4f}" for q in (0.5, 1.0, 2.0, 3.0))
        print(f"{name:>28} | {cell:18.3e} | {vals}", flush=True)

    print()
    worst = max(abs(magnitude_moments(X)[q] - base[q])
                for X in frames.values() for q in base)
    print(f"worst disagreement across every frame and exponent: {worst:.3e}")

    print()
    print("The singular spectrum is blind for the same reason -- diag(sigma) is orthogonal:")
    for name, X in frames.items():
        sv = np.linalg.svd(X, compute_uv=False)
        print(f"  {name:>28}  max |s(X) - s(A)| = "
              f"{float(np.abs(sv - np.linalg.svd(A, compute_uv=False)).max()):.3e}", flush=True)

    print()
    print("THE POSITIVE CONTROL.  A read that compares two sides DOES see the flip, so the")
    print("blindness above is a property of magnitude reads and not of these frames:")
    for name, X in frames.items():
        c = EA.channel_alignment(A, X)
        print(f"  coupling(A, {name:>28}) = {c.strength:+.4f}   z = {c.z:8.1f}", flush=True)
    print()
    print("That is why section 7 reads a coupling between the two channels and not a magnitude")
    print("of either: the sign of a determinantal weight lives in the relation, not in the sizes.")
