"""Experiment AU -- telling a sign problem from a phase problem, and when a rotation removes one.

`concentration` reports two distinct things about a cloud of unit vectors, and the distinction is
the whole content here:

    resultant = |mean row|   DIRECTIONAL -- the von Mises-Fisher sufficient statistic.  An
                             antipodal cloud reads ~0.  On the weight's unit-modulus frame this
                             IS |<w/|w|>|, by definition rather than by measurement.
    focus     = sigma1^2/M   AXIAL -- the power fraction on the leading principal axis.  An
                             antipodal cloud still reads ~1.

A REAL sign problem is an antipodal cloud: the weights sit at +-1, on one line through the origin.
A PHASE problem is spread around the circle.  So the pair separates the two failure modes, which
matter differently -- section 4's criterion speaks to the first, and its route B produces the
second.

WHAT `focus = 1` ACTUALLY CERTIFIES.  Rank one in the (Re, Im) plane means every weight lies on a
single line through the origin, which is exactly the statement that ONE GLOBAL PHASE ROTATION makes
them all real.  That is actionable rather than descriptive: the rotation is recoverable from the
cloud's own leading direction, no angle is chosen, and applying it returns a real sign problem with
the same |<sgn>| as before.  Where `focus < 1` no rotation helps, and the residual imaginary part
after de-rotation stays at 1.

Measured: real and artificially rotated rows read `focus = 1.0000` and de-rotate to a maximum
imaginary part of 1e-16, recovering |<sgn>| exactly; genuine phase rows read 0.57 to 0.83 and
de-rotate to 1.0.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import expm

import entroptics as E
from stable import udt_product, slogdet_one_plus_block


def unit_weights(Kmat, seed=5, beta=8.0, U=4.0, dtau=0.125, n=400):
    """The configuration weights, on the unit circle: magnitudes divided out."""
    L = int(round(beta / dtau)); N = Kmat.shape[0]
    eK = expm(-dtau * np.asarray(Kmat, dtype=complex))
    lam = float(np.arccosh(np.exp(dtau * U / 2.0)))
    rng = np.random.default_rng(seed)
    W = []
    for _ in range(n):
        X = rng.choice([-1.0, 1.0], size=(L, N)); d_ = {}
        for sg in (+1, -1):
            d = np.exp(sg * lam * X)
            Bl = eK[None, :, :] * d[:, None, :]
            Uu, D, T = udt_product(Bl[None], 4)
            s, _ = slogdet_one_plus_block(Uu, D, T)
            d_[sg] = complex(s[0])
        W.append(d_[+1] * d_[-1])
    W = np.array(W)
    return W / np.abs(W)


def axial_and_directional(u):
    """(resultant, focus) of the weight cloud, read on the (Re, Im) plane."""
    c = E.reads.concentration(np.stack([u.real, u.imag], axis=1))
    return float(c.resultant), float(c.focus)


def derotate(u):
    """Rotate onto the cloud's OWN leading axis. The angle is read, never chosen."""
    X = np.stack([u.real, u.imag], axis=1)
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    theta = float(np.arctan2(Vt[0, 1], Vt[0, 0]))
    return u * np.exp(-1j * theta), theta


if __name__ == "__main__":
    from reads.expAO_spectral_criterion import build, ring, tri_ladder, triangular

    print("=" * 100)
    print("AXIAL AGAINST DIRECTIONAL: which failure mode is it, and can a rotation remove it?")
    print()
    print(f"{'case':>24} | {'resultant':>10} {'focus':>8} | {'max|Im|':>9} "
          f"{'after de-rotation':>18} | {'verdict':>18}")
    rows = [("2x4 clean", build(2, 4, 0, 0, 0), 0.0),
            ("staggered h = 0.2", build(2, 4, 0, 0, 0.2), 0.0),
            ("staggered h = 0.6", build(2, 4, 0, 0, 0.6), 0.0),
            ("doped mu = 0.4", build(2, 4, 0, 0.4, 0), 0.0),
            ("staggered, rotated 0.7", build(2, 4, 0, 0, 0.6), 0.7),
            ("doped, rotated 1.9", build(2, 4, 0, 0.4, 0), 1.9),
            ("ring 6 flux pi/4", ring(6, np.pi / 4), 0.0),
            ("ring 5 flux pi/2", ring(5, np.pi / 2), 0.0),
            ("tri ladder flux pi/2", tri_ladder(8, np.pi / 2), 0.0),
            ("triangular 3x3 pi/2", triangular(3, 3, np.pi / 2), 0.0)]
    for name, K, rot in rows:
        u = unit_weights(np.asarray(K, dtype=complex)) * np.exp(1j * rot)
        r, f = axial_and_directional(u)
        v, _ = derotate(u)
        after = float(np.abs(v.imag).max())
        # `0.999` is a CHOSEN cut and it decides a printed LABEL only -- `resultant`, `focus` and
        # the de-rotated residual are the measured columns and none of them passes through it.
        # Nothing in the paper is quoted from this word. Marked, not removed: the table needs a
        # verdict column to be readable, and there is no cut-free way to write one.
        verdict = ("sign-free" if r > 0.999 else
                   ("rotatable -> SIGN" if after < 1e-9 else "genuine PHASE"))
        print(f"{name:>24} | {r:10.5f} {f:8.4f} | {float(np.abs(u.imag).max()):9.4f} "
              f"{after:18.2e} | {verdict:>18}", flush=True)
    print()
    print("`resultant` is |<w/|w|>| by construction on this frame -- arithmetic, not a finding.")
    print("`focus` is the finding: rank one in the plane means a single global rotation makes")
    print("every weight real, and the rows above show it doing so to 1e-16 or failing at 1.0.")
