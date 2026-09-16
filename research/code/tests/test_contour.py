"""§9.4, gated: a contour shift does not mitigate the two-dimensional phase problem.

Run at a cheaper point than the paper's table -- `beta = 3` rather than `6` -- because the scan is
the most expensive thing in this repo. What is preserved is the structure that makes the claim mean
anything:

  * the undeformed point must HAVE a phase problem. At `mu = 0` this lattice is half filled on a
    bipartite graph, the sign-free point of §5, where the undeformed phase is 1.0000 and every
    shift can only spoil it -- a scan there returns a null result for a reason that has nothing to
    do with contour deformation. That trap is asserted against directly.
  * the large-amplitude point is the one that matters. A thimble need not lie near the real axis,
    so a second regime at large shift is what would falsify the reading.
"""
from __future__ import annotations

import pytest

from functools import lru_cache

from closed.expAR_contour_2d import phase_at as _phase_at
from model2d import Model2D


@lru_cache(maxsize=None)
def _cached(mu, L, c, kind, n_meas, warm):
    return _phase_at(_model(mu, L), c, kind, n_meas=n_meas, warm=warm)


def phase_at(m, c, kind, *, n_meas, warm):
    """Memoised: the undeformed baseline is asked for by two tests and is the expensive half.

    Keyed on the model's parameters rather than the object, so the same scan is run once.
    """
    return _cached(m._cache_mu, m.L, c, kind, n_meas, warm)


def _model(mu, L=15):
    m = Model2D(Lx=4, Ly=4, t=1.0, mu=mu, U=6.0, dtau=0.2, L=L, theta=0.0)
    m._cache_mu = mu
    return m


def test_the_undeformed_point_has_a_phase_problem_to_fix():
    """Without this the scan below is a null result about nothing."""
    base, se, _ = phase_at(_model(1.0), 0.0, "uniform", n_meas=120, warm=60)
    # Against the SAME lattice at its sign-free point, resolved by the two scans' own error bars,
    # rather than against a chosen number. The mu = 0 scan is the one the third test runs, so this
    # costs nothing; `< 0.95` was a guess at how far below 1 counts as a phase problem.
    free, free_se, _ = phase_at(_model(0.0), 0.0, "uniform", n_meas=100, warm=50)
    assert base + se < free - free_se, \
        f"no phase problem to mitigate: {base:.4f} +- {se:.4f} against the sign-free " \
        f"{free:.4f} +- {free_se:.4f}"
    assert base > 0.0, "the phase has collapsed entirely; pick a milder point"


def test_a_large_shift_does_not_help():
    """The load-bearing amplitude: a second regime far from the real axis would show here."""
    m = _model(1.0)
    base, _, _ = phase_at(m, 0.0, "uniform", n_meas=120, warm=60)
    val, _, _ = phase_at(m, 4.0, "uniform", n_meas=120, warm=60)
    assert val < base, f"the shift improved the phase: {val:.4f} against {base:.4f}"


def test_the_sign_free_point_is_useless_for_this_test_and_is_known_to_be():
    """The trap this experiment was written to avoid, pinned so it cannot be walked into again.

    At mu = 0 the lattice is at §5's sign-free point: the undeformed phase is 1 and any scan there
    shows every shift 'doing nothing' for a trivial reason.
    """
    base, _, _ = phase_at(_model(0.0), 0.0, "uniform", n_meas=100, warm=50)
    assert base == pytest.approx(1.0, abs=1e-9), \
        f"mu = 0 no longer sits at the sign-free point ({base:.4f}); re-examine the case choice"
