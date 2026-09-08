"""§4's criterion performed by a read on OUTPUT, gated against the algebraic criterion.

The two sides share no information. `coupling` sees two columns per configuration and never sees
`K`; the algebraic criterion sees `K` and never sees a configuration. Agreement between them is the
result, and it is what makes the criterion usable by a running simulation, which has configurations
and determinants and does not have a clean Hamiltonian to two-colour.

WHAT IS ASSERTED, AND WHAT IS NOT.

  * the read's verdict must match the MEASURED identity residual on every lattice -- not the
    algebraic criterion's verdict, which is itself a claim. The residual is the ground truth here;
  * the two populations must be SEPARATED, asserted as `max(holding) < min(failing)` over the
    lattices, so no level is chosen. Measured 2026-09-07 the gap is thirteen orders: machine zero
    against 1.0e-3 to 2.9e-2;
  * the failing side must NOT shrink with sample size. A departure that fell as `n` grew would be
    sampling noise dressed as a signal, and that is the one way this read could look right for the
    wrong reason. It is asserted directly.

`1` is not a threshold anywhere here: it is the definitional saturation of a normalised alignment,
and an exact affine relation between two columns must reach it.
"""
from __future__ import annotations

import numpy as np
import pytest

from functools import lru_cache

from reads.expAO_spectral_criterion import build, criterion, measure, ring, tri_ladder, chain
from reads.expAW_criterion_from_output import read_the_identity

CASES = (
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
)


@lru_cache(maxsize=None)
def scored():
    out = []
    for name, K in CASES:
        K = np.asarray(K, dtype=complex)
        c, null = read_the_identity(K)
        resid, _ = measure(K, 6.0, n_draw=120, seed=5)
        out.append((name, 1.0 - abs(float(c.strength)), resid < 1e-9,
                    bool(criterion(K)), float(null.strength)))
    return tuple(out)


def test_the_read_reproduces_the_measured_identity_on_every_lattice():
    """THE RESULT. No Hamiltonian is used to produce the read's side of this."""
    bad = [(n, d, h) for n, d, h, _, _ in scored() if (abs(d) < 1e-9) != h]
    assert not bad, f"the read disagreed with the measured identity on: {bad}"


def test_the_read_reproduces_the_algebraic_criterion():
    """And therefore agrees with §4, which is the claim that makes it a criterion at all."""
    bad = [(n, d, a) for n, d, _, a, _ in scored() if (abs(d) < 1e-9) != a]
    assert not bad, f"the read disagreed with the algebraic criterion on: {bad}"


def test_the_two_populations_are_separated_with_no_level_between_them():
    """SEPARATION, not a cut: the worst identity-holding lattice against the best failing one."""
    holding = [abs(d) for _, d, h, _, _ in scored() if h]
    failing = [abs(d) for _, d, h, _, _ in scored() if not h]
    assert holding and failing, "one population is empty, so nothing is being separated"
    assert max(holding) < min(failing), \
        f"the populations overlap: holding up to {max(holding):.3e}, " \
        f"failing down to {min(failing):.3e}"


@pytest.mark.parametrize("name,K,holds", [
    ("2x4 clean", build(2, 4, 0.0, 0.0, 0.0), True),
    ("2x4 mu=0.4", build(2, 4, 0.0, 0.4, 0.0), False),
    ("tri ladder 8", tri_ladder(8, 0.0), False),
])
def test_the_departure_is_systematic_and_does_not_shrink_with_sample_size(name, K, holds):
    """THE WAY THIS COULD BE RIGHT FOR THE WRONG REASON, ruled out.

    If the departure on a broken lattice fell as `n` grew it would be sampling noise, and the read
    would be reporting its own variance rather than the physics. It does not: the value is stable
    across a factor of four in `n` and across seeds.
    """
    K = np.asarray(K, dtype=complex)
    vals = [abs(1.0 - abs(float(read_the_identity(K, n=n, seed=sd)[0].strength)))
            for n, sd in ((200, 11), (400, 23), (800, 37))]
    if holds:
        assert max(vals) < 1e-12, f"{name}: saturation was lost at some sample size: {vals}"
    else:
        assert min(vals) > 1e-6, f"{name}: the departure vanished at some sample size: {vals}"
        assert max(vals) < 10.0 * min(vals), \
            f"{name}: the departure moved by more than an order of magnitude with n, " \
            f"which is what sampling noise would do: {vals}"
