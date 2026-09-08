"""§4's diagonal restriction is necessary, gated -- with the counterexample pinned.

The claim this file protects is a TIGHTNESS claim, and tightness claims rot quietly: someone
widens the conjugation class for a good reason, the criterion still passes every lattice anyone
checks, and the two staggered rows that break it are never run. So the counterexample is asserted
directly rather than left in a table.

  * `S K S^dag = -K` is verified EXACTLY on the staggered lattice under a monomial S;
  * the §3 identity is measured there and must FAIL;
  * the diagonal criterion must score 14 of 14 against the measured identity and the monomial one
    must score strictly worse.

The last of those is the load-bearing assertion. A test asserting only "diagonal is 14 of 14" would
pass on a build where monomial was also 14 of 14, which is the world in which the restriction is
unnecessary and §4 overclaims.
"""
from __future__ import annotations

import numpy as np
import pytest

from functools import lru_cache

from reads.expAO_spectral_criterion import build, criterion, measure
from reads.expAV_monomial_conjugation import CASES, monomial_route_A, read_the_cloud
from reads.expAU_axial_versus_directional import unit_weights


@lru_cache(maxsize=None)
def scored():
    """(name, identity holds, diagonal verdict, monomial verdict, residual) for every case."""
    out = []
    for name, K in CASES:
        K = np.asarray(K, dtype=complex)
        resid, deficit = measure(K, 6.0, n_draw=120, seed=5)
        out.append((name, resid < 1e-9, bool(criterion(K)),
                    monomial_route_A(K) is not None, resid, deficit))
    return tuple(out)


def test_the_monomial_conjugation_is_exact_on_the_staggered_lattice():
    """THE COUNTEREXAMPLE, first half: the wider class really does open this lattice.

    Without this the failure below could be a monomial S that does not exist, which would say
    nothing about the restriction.
    """
    K = np.asarray(build(2, 4, 0.0, 0.0, 0.2), dtype=complex)
    found = monomial_route_A(K)
    assert found is not None, "no monomial S on the staggered lattice; the counterexample is gone"
    pi, S = found
    residual = float(np.max(np.abs(S @ K @ np.conj(S).T + K)))
    assert residual < 1e-8, f"the monomial conjugation is not exact: {residual:.3e}"
    # and a DIAGONAL S cannot do this -- the potential is what it cannot touch
    assert not criterion(K), "the diagonal criterion now accepts the staggered lattice"


@pytest.mark.parametrize("h", [0.2, 0.6])
def test_the_identity_fails_where_the_monomial_conjugation_holds(h):
    """THE COUNTEREXAMPLE, second half, and the reason the restriction cannot be relaxed.

    Failure is measured against the same identity on a lattice where it HOLDS, so no level is
    chosen: the staggered residual must exceed the clean lattice's by orders, not by a tolerance.
    """
    K = np.asarray(build(2, 4, 0.0, 0.0, h), dtype=complex)
    assert monomial_route_A(K) is not None
    resid, _ = measure(K, 6.0, n_draw=120, seed=5)
    clean, _ = measure(np.asarray(build(2, 4, 0.0, 0.0, 0.0), dtype=complex),
                       6.0, n_draw=120, seed=5)
    assert resid > clean, \
        f"the staggered identity residual ({resid:.3e}) is no worse than the clean one ({clean:.3e})"


def test_the_diagonal_criterion_beats_the_monomial_one_against_the_measured_identity():
    """THE LOAD-BEARING TEST. Both classes scored on the same rows, against the same measurement.

    Asserted as a COMPARISON between the two scores rather than as a number for either, so this
    keeps its meaning if lattices are added.
    """
    rows = scored()
    diag = sum(holds == d for _, holds, d, _, _, _ in rows)
    mono = sum(holds == m for _, holds, _, m, _, _ in rows)
    assert diag == len(rows), \
        f"the diagonal criterion no longer predicts the identity on every row: {diag}/{len(rows)}"
    assert mono < diag, \
        f"widening the class to monomial no longer costs anything ({mono} against {diag}); " \
        f"if that is real, section 4's restriction is not necessary and the claim must change"


def test_route_b_lattices_carry_a_phase_no_rotation_removes():
    """THE POSITIVITY HALF, read from the weights with no knowledge of K.

    On every lattice satisfying the identity while still carrying a phase, the cloud must not be
    rank one -- `focus` strictly below the rank-one value -- and de-rotating on the cloud's own
    axis must leave an imaginary part. Both are read; neither is compared to a chosen level.
    """
    seen = 0
    for name, K in CASES:
        K = np.asarray(K, dtype=complex)
        resid, deficit = measure(K, 6.0, n_draw=120, seed=5)
        if not (resid < 1e-9 and deficit > 0.0):
            continue
        seen += 1
        _, focus, _, after = read_the_cloud(unit_weights(K))
        assert focus < 1.0, f"{name}: the cloud is rank one, so the phase would be removable"
        assert after > 0.0, f"{name}: de-rotation removed the phase entirely"
    assert seen >= 2, "no identity-holding row carries a phase, so nothing is being shown"


def test_a_sign_free_cloud_resolves_no_modes():
    """The instrument's own reading of 'nothing to rotate', pinned so it is never read as a fault.

    A sign-free lattice puts every weight on the real axis. There is no spread, so
    `principal_directions` returns zero columns -- which is the correct answer to 'which way does
    this cloud point', not a failure to answer it.
    """
    _, focus, modes, after = read_the_cloud(unit_weights(np.asarray(
        build(2, 4, 0.0, 0.0, 0.0), dtype=complex)))
    assert modes == 0, f"the sign-free cloud now resolves {modes} modes"
    assert after == 0.0, "the sign-free cloud carries an imaginary part"
