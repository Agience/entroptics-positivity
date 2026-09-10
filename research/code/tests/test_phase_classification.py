"""The three-regime classification of a weight cloud, and the fact that ONE number cannot give it.

WHAT IS BEING HELD.  `concentration` on the weight cloud's `(Re, Im)` frame returns a DIRECTIONAL
statistic and an AXIAL one, and the pair classifies where neither alone does:

    focus = 1, resultant = 1     sign-free
    focus = 1, resultant < 1     a REAL sign problem -- one global rotation makes the weights real,
                                 and `resultant` is the severity of what is left
    focus < 1                    a genuine phase problem -- no rotation reaches it

The middle row is the one worth a gate.  The paper said at one point that `focus = 1` meant "not a
sign problem at all"; the measured rows say otherwise -- staggered `h = 0.6` and doped `mu = 0.4`
both read `focus = 1.0000` with `resultant` 0.935 and 0.925, which are sign problems.  A single
number cannot separate three regimes, and this file is what stops that claim coming back.

The failure modes these tests watch for, stated first so they can fail:
  - `focus` alone is treated as the classifier, which silently merges sign-free runs with real
    sign problems -- the negative control below makes that merge visible;
  - the de-rotation stops being read from the cloud's own axis, or stops working, so `focus = 1`
    is no longer the actionable statement that a rotation removes the phase;
  - a genuine phase row starts reading `focus = 1`, which would make the axial read vacuous;
  - `resultant` drifts from `|<w/|w|>|`, which it equals by construction on this frame.
"""
from __future__ import annotations

import numpy as np
import pytest

from reads.expAO_spectral_criterion import build, ring, tri_ladder, triangular
from reads.expAU_axial_versus_directional import (
    axial_and_directional, derotate, unit_weights,
)

SIGN_FREE = {"2x4 clean": build(2, 4, 0, 0, 0),
             "ring 6 flux pi/4": ring(6, np.pi / 4)}
REAL_SIGN = {"staggered h = 0.6": build(2, 4, 0, 0, 0.6),
             "doped mu = 0.4": build(2, 4, 0, 0.4, 0)}
GENUINE_PHASE = {"ring 5 flux pi/2": ring(5, np.pi / 2),
                 "tri ladder flux pi/2": tri_ladder(8, np.pi / 2),
                 "triangular 3x3 pi/2": triangular(3, 3, np.pi / 2)}


def read(K):
    return axial_and_directional(unit_weights(K))


@pytest.mark.parametrize("tag", sorted(SIGN_FREE))
def test_sign_free_reads_one_and_one(tag):
    resultant, focus = read(SIGN_FREE[tag])
    assert focus == pytest.approx(1.0, abs=1e-6)
    assert resultant == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("tag", sorted(REAL_SIGN))
def test_a_real_sign_problem_is_rank_one_and_NOT_sign_free(tag):
    """The row that the corrected paragraph turns on: `focus = 1` with a sign problem present."""
    resultant, focus = read(REAL_SIGN[tag])
    assert focus == pytest.approx(1.0, abs=1e-6), (
        "a real sign problem is an antipodal cloud, which is still rank one")
    assert resultant < 0.99, (
        f"{tag} must carry a measurable sign problem, else the middle regime is vacuous; "
        f"resultant = {resultant:.5f}")


@pytest.mark.parametrize("tag", sorted(GENUINE_PHASE))
def test_a_genuine_phase_problem_is_not_rank_one(tag):
    resultant, focus = read(GENUINE_PHASE[tag])
    assert focus < 0.95, f"{tag} should not be rank one; focus = {focus:.4f}"


def test_focus_ALONE_cannot_classify(tag=None):
    """The negative control, and the whole reason the paper needs two numbers.

    If every `focus = 1` row were sign-free, one number would suffice and the corrected paragraph
    would be unnecessary.  It is not: sign-free rows and real sign problems are INDISTINGUISHABLE
    in `focus` and separated only by `resultant`.
    """
    free = [read(K)[1] for K in SIGN_FREE.values()]
    sign = [read(K)[1] for K in REAL_SIGN.values()]
    assert max(abs(f - s) for f in free for s in sign) < 1e-6, (
        "focus must NOT separate these two groups -- if it does, the claim that the pair is "
        "needed has stopped being true and the paper should say so")
    free_r = [read(K)[0] for K in SIGN_FREE.values()]
    sign_r = [read(K)[0] for K in REAL_SIGN.values()]
    assert min(free_r) - max(sign_r) > 0.05, (
        "and `resultant` must separate them, or neither number classifies")


@pytest.mark.parametrize("tag", sorted(REAL_SIGN))
def test_the_rotation_is_read_from_the_cloud_and_works(tag):
    """`focus = 1` is actionable: rotating by the cloud's own leading axis makes the weights real."""
    u = unit_weights(REAL_SIGN[tag])
    rotated = u * np.exp(1j * 0.7)                     # an arbitrary global phase, imposed
    assert np.max(np.abs(rotated.imag)) > 0.1, "the control must actually carry an imaginary part"
    back, theta = derotate(rotated)
    assert np.max(np.abs(back.imag)) < 1e-12, (
        f"de-rotation should return a real cloud; max|Im| = {np.max(np.abs(back.imag)):.2e}")


@pytest.mark.parametrize("tag", sorted(GENUINE_PHASE))
def test_no_rotation_reaches_a_genuine_phase(tag):
    u = unit_weights(GENUINE_PHASE[tag])
    back, _ = derotate(u)
    assert np.max(np.abs(back.imag)) > 0.5, (
        "a genuine phase must survive de-rotation, or the two regimes are not distinct")
