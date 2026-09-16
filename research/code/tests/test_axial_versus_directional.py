"""§7's two-mode diagnosis, gated: which failure mode, and whether a rotation removes it.

`concentration` reports a directional statistic and an axial one. On the weight's unit-modulus
frame `resultant` IS `|<w/|w|>|` by construction -- arithmetic, asserted here as such so it is never
quoted as a measurement. `focus` is the finding: rank one in the (Re, Im) plane means one global
rotation makes every weight real.

The rotated rows are the load-bearing ones. A row that is real to begin with proves nothing about
detection; a row given a genuine global phase, whose imaginary part is large, must still read
`focus = 1` and must de-rotate to machine zero.
"""
from __future__ import annotations

import numpy as np
import pytest

from reads.expAO_spectral_criterion import build, ring, tri_ladder
from reads.expAU_axial_versus_directional import (
    axial_and_directional, derotate, unit_weights)


def test_the_resultant_is_the_mean_phase_by_construction():
    """Arithmetic, pinned as arithmetic so it is not reported as a finding."""
    u = unit_weights(np.asarray(tri_ladder(8, np.pi / 2), dtype=complex))
    r, _ = axial_and_directional(u)
    assert r == pytest.approx(float(abs(u.mean())), abs=1e-9)


@pytest.mark.parametrize("rot", [0.0, 0.7, 1.9])
def test_a_rotatable_phase_reads_focus_one_and_de_rotates(rot):
    """THE LOAD-BEARING CASE at rot != 0: a real global phase, large imaginary part, still rank 1."""
    base = unit_weights(np.asarray(build(2, 4, 0, 0, 0.6), dtype=complex))
    u = base * np.exp(1j * rot)
    _, f = axial_and_directional(u)
    assert f == pytest.approx(1.0, abs=1e-6), f"rank one not detected at rot = {rot}: {f}"
    if rot:
        # Against the same weights UNROTATED, measured here. `> 0.5` was a size chosen for 'a
        # phase to remove'; the rotation is the only thing this row adds, so it is the reference.
        assert float(np.abs(u.imag).max()) > float(np.abs(base.imag).max()), \
            "this row carries no phase to remove"
    v, _ = derotate(u)
    assert float(np.abs(v.imag).max()) < 1e-9, "de-rotation did not make the weights real"
    # A global phase CANCELS in a mean of unit weights, so de-rotation must return the magnitude
    # the rotated weights already had -- exactly. `approx(0.935, abs=0.02)` pinned that magnitude
    # as a number measured once; this asserts the invariance that produces it.
    assert abs(float(v.real.mean())) == pytest.approx(abs(complex(u.mean())), abs=1e-9)


@pytest.mark.parametrize("K", [ring(5, np.pi / 2), tri_ladder(8, np.pi / 2)])
def test_a_genuine_phase_is_not_rotatable(K):
    """THE CONTROL, against the rotatable rows rather than against a chosen level.

    `focus` on a genuine phase must sit clearly below what a rank-one cloud reads, and the
    rank-one value is not assumed -- it is measured on a rotatable row in the same call.
    """
    rank_one = axial_and_directional(
        unit_weights(np.asarray(build(2, 4, 0, 0, 0.6), dtype=complex)) * np.exp(1j * 0.7))[1]
    u = unit_weights(np.asarray(K, dtype=complex))
    _, f = axial_and_directional(u)
    assert f < rank_one - 0.1,         f"a genuine phase row is not separated from rank one: {f:.4f} against {rank_one:.4f}"

    # and the residual after de-rotation, against what a rotatable row leaves
    v, _ = derotate(u)
    rotatable_residual = float(np.abs(derotate(
        unit_weights(np.asarray(build(2, 4, 0, 0, 0.6), dtype=complex))
        * np.exp(1j * 0.7))[0].imag).max())
    assert float(np.abs(v.imag).max()) > 1e6 * max(rotatable_residual, 1e-15),         "de-rotation removed a phase it should not have"


@pytest.mark.parametrize("rot", [0.3, 0.7, 1.0, 1.3])
def test_a_global_phase_cancels_and_costs_nothing(rot):
    """The correct estimator never sees a global phase, because it cancels in the ratio.

    `<O> = sum(O w)/sum(w)` multiplies numerator and denominator alike, so `|<w>|` is invariant.
    Pinned so the read is never sold as a cost saving: there is nothing to save.
    """
    base = unit_weights(np.asarray(build(2, 4, 0, 0, 0.6), dtype=complex))
    u = base * np.exp(1j * rot)
    assert abs(complex(u.mean())) == pytest.approx(abs(complex(base.mean())), abs=1e-9),         "a global phase moved the correct estimator; it must cancel"


@pytest.mark.parametrize("rot", [0.7, 1.0, 1.3])
def test_only_the_real_part_estimator_is_fooled(rot):
    """What `focus` is actually worth: it says when `mean(Re w)` has stopped being the right read.

    That estimator is correct for real weights and is the natural thing to carry over to complex
    ones, where it reads cos(theta) too small.
    """
    base = unit_weights(np.asarray(build(2, 4, 0, 0, 0.6), dtype=complex))
    u = base * np.exp(1j * rot)
    naive = abs(float(u.real.mean()))
    correct = abs(complex(u.mean()))
    assert naive < correct, f"the real-part estimator was not depressed at rot = {rot}"
    _, f = axial_and_directional(u)
    assert f == pytest.approx(1.0, abs=1e-6), "focus should still say the phase is global"


def test_de_rotation_manufactures_no_gain_where_the_phase_is_genuine():
    """THE CONTROL. A correction that improved every row would be an artefact."""
    u = unit_weights(np.asarray(ring(5, np.pi / 2), dtype=complex))
    naive = abs(float(u.real.mean()))
    v, _ = derotate(u)
    assert abs(float(v.real.mean())) == pytest.approx(naive, rel=0.01),         "de-rotation produced a gain on a genuinely complex row"
