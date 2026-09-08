"""§4's CONTROL, gated: bipartite structure versus filling, compared at the SAME filling.

This is the control rather than the result, and it is the one that decides whether §4 says anything
at all.  Varying `tp` destroys the bipartite structure, and it also moves the band, so `mu = 0`
stops being half filling: `tp = 0` sits at `<n> = 1.00000` and `tp = 0.7` at `<n> = 0.75`.  Compared
without matching the filling, the same data reads as "breaking the bipartite structure REMOVES the
sign problem", which is a filling comparison wearing a symmetry label.

`mu` is therefore tuned per `tp` against the NON-INTERACTING density -- a closed-form function of
the single-particle spectrum, carrying no sampling noise, and seeing neither the identity nor the
sign.  That is a control variable set to a target, like a temperature, and not a fit.

The tests below assert, at matched filling:

  * the tuner recovers `mu = 0` for the bipartite lattice WITHOUT being told half filling lies
    there -- the control validating itself;
  * bipartite  =>  the identity is exact and no weight is negative;
  * non-bipartite  =>  the identity is broken and weights ARE negative.

The last is what makes the first two mean anything: a suite that only checked the bipartite side
would pass on a build that had stopped being sensitive to the structure at all.
"""
from __future__ import annotations

import numpy as np
import pytest

from model2d import Model2D
from reads.expZZ_matched_filling import free_density, measure, tune_mu

LX, LY, BETA, U, DTAU = 2, 4, 8.0, 4.0, 0.125
BIPARTITE, BROKEN = 0.0, [0.3, 0.7]


def at(tp, n_draw=200, seed=0):
    mu = tune_mu(LX, LY, tp, BETA)
    m = Model2D(Lx=LX, Ly=LY, t=1.0, tp=tp, mu=mu, U=U, dtau=DTAU,
                L=int(round(BETA / DTAU)), theta=0.0)
    n, neg, res = measure(m, n_draw, seed)
    return mu, n, neg, res


def test_the_tuner_recovers_half_filling_on_the_bipartite_lattice():
    """The control validating itself: nothing tells it that mu = 0 is half filling here."""
    mu = tune_mu(LX, LY, BIPARTITE, BETA)
    assert mu == pytest.approx(0.0, abs=1e-3), f"the tuner put half filling at mu = {mu:.5f}"


@pytest.mark.parametrize("tp", [BIPARTITE] + BROKEN)
def test_every_row_is_compared_at_the_same_filling(tp):
    """Without this the comparison below is between fillings, not between structures."""
    mu = tune_mu(LX, LY, tp, BETA)
    assert free_density(LX, LY, tp, BETA, mu) == pytest.approx(1.0, abs=1e-3)


def test_untuned_mu_is_not_half_filling_once_the_structure_is_broken():
    """Why the tuning is needed at all, asserted rather than asserted-about."""
    bip = free_density(LX, LY, BIPARTITE, BETA, 0.0)
    assert bip == pytest.approx(1.0, abs=1e-3)
    # Against the bipartite lattice measured on the line above, which mu = 0 does put at half
    # filling. `> 0.1` was a distance from half chosen by hand; this is the comparison it meant.
    for tp in BROKEN:
        n = free_density(LX, LY, tp, BETA, 0.0)
        assert abs(n - 1.0) > abs(bip - 1.0), \
            f"tp = {tp} at mu = 0 sits at <n> = {n:.5f}, as close to half as the bipartite " \
            f"lattice's {bip:.5f}"


def test_at_matched_filling_the_bipartite_lattice_is_exact_and_sign_free():
    _, _, neg, res = at(BIPARTITE, seed=1)
    assert res.max() < 1e-9, f"the identity failed at {res.max():.2e}"
    assert neg == 0.0, "a negative weight appeared on the bipartite lattice"


@pytest.mark.parametrize("tp", BROKEN)
def test_at_matched_filling_breaking_the_structure_breaks_both(tp):
    """THE CONTROL. Same filling, structure removed: identity gone and a sign problem present."""
    _, _, neg, res = at(tp, seed=2)
    assert res.max() > 1.0, f"the identity survived at tp = {tp}: {res.max():.2e}"
    assert neg > 0.0, f"no sign problem appeared at tp = {tp}, so nothing is controlled for"
