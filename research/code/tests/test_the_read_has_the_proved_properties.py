"""The bridge: does the read we actually call have the properties Lean proves of it?

`research/lean/SignProblem/Alignment.lean` proves three things about the centred, normalised
alignment, and section 11 cites them as guarantees about `coupling.strength`:

    strength_affine_invariant    invariant to an offset and a positive scale
    abs_strength_eq_one_iff      saturation at 1 is EXACTLY an affine relation
    deficit_mem_Icc              the deficit cannot leave [0, 2]

Those are theorems about a mathematical object. This file asks whether that object is the thing
`entroptics_adapter.channel_alignment` returns -- because a proof about `r` and an implementation
that computes something slightly different would be two correct halves and one wrong whole, which is
the failure mode neither half can reveal on its own. The Lean cannot check the library and the
library cannot check the Lean; only this can.

WHAT WOULD BREAK IT, concretely. If the read did not centre, `strength_affine_invariant` would fail
on the offset and the criterion of section 6 would need `tr(K)`. If it normalised by something
other than the two norms, the bound would fail and section 7's `[0, 2]` claim with it. If it used a
rank-reduced or shrunk estimate, saturation would stop being exact on affine data and the build
check would lose its verdict. Each of those is a plausible implementation and each is caught here.

NEGATIVE CONTROLS. Every property is paired with a case that must NOT have it: data that is not
affinely related must not saturate, and a read that saturated on everything would pass the positive
half of these tests unchanged. That is the same discipline as the rest of the gates here.
"""
from __future__ import annotations

import numpy as np
import pytest

import entroptics_adapter as EA

#: DERIVED, not chosen: the identity being checked is exact in real arithmetic, so the only budget
#: is float64 accumulation over a few hundred terms. Anything at 1e-12 is arithmetic; anything above
#: it is a different quantity being computed.
EXACT = 1e-12


def _columns(seed, n=400):
    """Two unrelated columns, shaped as the reads take them: one row per configuration."""
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((n, 1))
    b = rng.standard_normal((n, 1))
    return a, b


def _strength(a, b):
    return float(EA.channel_alignment(a, b).strength)


# ── strength_affine_invariant ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("p,q", [(0.0, 1.0), (5.0, 1.0), (-3.25, 1.0),
                                 (0.0, 7.5), (12.0, 0.125), (-4.0, 3.0)])
def test_the_read_is_invariant_to_an_offset_and_a_positive_scale(p, q):
    """`Alignment.strength_affine_invariant`, on the read itself.

    This is the property section 6 depends on: section 4's identity carries an offset
    `-dtau L tr(K)` and a slope `lambda`, and the criterion runs on output precisely because
    neither reaches the read. A read that failed this would need the Hamiltonian.
    """
    a, b = _columns(0)
    assert _strength(q * a + p, b) == pytest.approx(_strength(a, b), abs=EXACT)


def test_the_invariance_check_can_fail():
    """THE NEGATIVE CONTROL for the invariance tests.

    An UNCENTRED cosine is the obvious wrong implementation, and it is not offset-invariant. If the
    comparison above could not tell the two apart it would pass whatever the read did, so the
    difference is exhibited here on the same data.
    """
    a, b = _columns(0)

    def uncentred(x, y):
        x, y = x.ravel(), y.ravel()
        return float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y)))

    assert abs(uncentred(a + 5.0, b) - uncentred(a, b)) > 1e-3, \
        "the uncentred read is offset-invariant here, so this control shows nothing"


def test_a_negative_scale_flips_the_sign_and_not_the_magnitude():
    """The sign convention, which matters because section 7 reads `-1` and not `+1`.

    Lean states the invariance for `q > 0`; for `q < 0` the magnitude is preserved and the sign
    turns over. Asserted rather than assumed, because the calibration of section 7.2 is a SIGNED
    claim -- the channels being exact negatives is what makes the read `-1`.
    """
    a, b = _columns(1)
    assert _strength(-2.0 * a, b) == pytest.approx(-_strength(a, b), abs=EXACT)


# ── abs_strength_le_one and deficit_mem_Icc ──────────────────────────────────────────────────

@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_the_read_is_bounded_and_so_is_the_deficit(seed):
    """`Alignment.abs_strength_le_one` and `Alignment.deficit_mem_Icc`.

    The `[0, 2]` range is what section 7.2 quotes for the deficit. It needs no hypothesis in
    Lean and none here.
    """
    a, b = _columns(seed)
    s = _strength(a, b)
    assert -1.0 - EXACT <= s <= 1.0 + EXACT
    assert 0.0 - EXACT <= 1.0 + s <= 2.0 + EXACT


# ── abs_strength_eq_one_iff ──────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("r,offset", [(1.0, 0.0), (2.5, 0.0), (-0.75, 0.0),
                                      (1.0, 9.0), (-3.0, -2.5)])
def test_an_exact_affine_relation_saturates_the_read(r, offset):
    """`Alignment.abs_strength_eq_one_of_affine`, which is section 6's criterion.

    Section 4's identity says the channel difference is an exact affine function of the field sum.
    If that holds, the read MUST saturate -- so a rig that returned anything else here would be
    computing something other than the alignment the criterion is stated for.
    """
    a, _ = _columns(2)
    b = r * a + offset
    assert abs(_strength(a, b)) == pytest.approx(1.0, abs=EXACT)


@pytest.mark.parametrize("c", [1.0, 0.0, 2.5])
def test_the_particle_hole_calibration_is_exactly_minus_one(c):
    """`Alignment.strength_eq_neg_one_of_reflected`, which is section 7.2's calibration.

    Particle-hole symmetry gives `G_dn = 1 - G_up` configuration by configuration -- an affine
    relation with slope `-1` -- so the read cannot come back at anything but `-1`. Section 7 reports
    `-1.0000` as a measurement; this is the check that no other value was available to it, and it is
    the reason the calibration needs no constant supplied.
    """
    a, _ = _columns(4)
    assert _strength(a, c - a) == pytest.approx(-1.0, abs=EXACT)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_unrelated_columns_do_not_saturate(seed):
    """THE NEGATIVE CONTROL for saturation, and the half the build check of section 6 uses.

    `no_affine_relation_of_abs_strength_ne_one` says a read short of 1 rules an affine relation out.
    That is only useful if the read actually falls short on data with no such relation -- a read
    that saturated on everything would pass every positive test above and detect nothing.
    """
    a, b = _columns(seed)
    assert abs(_strength(a, b)) < 0.5, \
        "unrelated columns saturated the read, so saturation carries no information"


def test_a_broken_affine_relation_stops_saturating_in_proportion():
    """Between the two: perturbing an exact relation must move the read off 1 monotonically.

    Section 5's "no tolerance and no onset" claim is about the identity residual; this is its
    counterpart at the read, and it is what makes the deficit a continuous order parameter rather
    than a yes/no. A read that snapped between 1 and 0 would satisfy both tests above and still be
    useless as the order parameter section 7 reports.
    """
    a, _ = _columns(3)
    rng = np.random.default_rng(99)
    noise = rng.standard_normal(a.shape)
    deficits = []
    for eps in (0.0, 1e-3, 1e-2, 1e-1):
        b = 2.0 * a + eps * noise
        deficits.append(1.0 - abs(_strength(a, b)))
    assert deficits[0] == pytest.approx(0.0, abs=EXACT)
    assert all(x < y for x, y in zip(deficits, deficits[1:])), \
        f"the deficit is not monotone in the perturbation: {deficits}"
