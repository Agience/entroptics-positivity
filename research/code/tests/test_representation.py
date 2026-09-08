"""The read is a property of a (Hamiltonian, decoupling) pair, pinned as a gate.

The decoupling family in `model2d` splits `U = (1-theta)U + theta U` and decouples each piece in
its own channel.  Both rewritings are exact identities on the four states of a site, so every theta
describes the SAME PHYSICS -- that is the model's own gate, checked by quadrature there.

At half filling the whole family is also sign-free: weights real and positive at every theta, the
charge channel included.  So along that axis the model is fixed AND its sign structure is fixed,
and the read still runs from -1.0000 to +1.0000.  These tests pin that, because it is the reason
every claim in sections 2 to 5 is a claim about the spin decoupling named there and not about the
Hubbard model.

They also pin the negative direction: doping the same family produces genuinely complex weights,
and at theta = 1 the read is saturated at exactly +1 while half the draws have negative real part.
"""
from __future__ import annotations

import numpy as np
import pytest

from functools import lru_cache

from reads.expAI_phase_is_not_a_parity import sweep as _sweep


@lru_cache(maxsize=None)
def sweep(theta, beta, mu, seed):
    """Memoised: several tests here sweep the same theta grid at the same seeds."""
    return _sweep(theta, beta, mu, seed=seed)

SEEDS = (1, 7, 23, 45)


@pytest.mark.parametrize("theta", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_the_half_filled_family_is_sign_free_at_every_theta(theta):
    """Not assumed: the charge channel is sign-free here too, and that is why the axis works."""
    for seed in SEEDS:
        r = sweep(theta, 4.0, 0.0, seed)
        assert r["negre"] == 0.0, f"a negative real part appeared at theta = {theta}"
        assert r["imagshare"] < 1e-9, f"the weight was not real at theta = {theta}"


def test_the_read_spans_the_full_range_while_the_physics_holds_still():
    """THE RESULT. Same Hamiltonian, no sign problem anywhere, and the read runs -1 to +1."""
    ends = {}
    for theta in (0.0, 1.0):
        vals = [sweep(theta, 4.0, 0.0, s) for s in SEEDS]
        assert all(v["negre"] == 0.0 for v in vals), "this axis must be sign-free throughout"
        ends[theta] = [v["s"] for v in vals]
    assert all(s == pytest.approx(-1.0, abs=1e-4) for s in ends[0.0]), ends[0.0]
    assert all(s == pytest.approx(+1.0, abs=1e-4) for s in ends[1.0]), ends[1.0]

    middle = [sweep(0.5, 4.0, 0.0, s)["s"] for s in SEEDS]
    # 'Departed from both ends' stated against the two ends measured just above, not against a
    # half-way level: the interior must sit nearer the origin than it does to either endpoint.
    lo, hi = float(np.mean(ends[0.0])), float(np.mean(ends[1.0]))
    assert all(abs(s) < min(abs(s - lo), abs(s - hi)) for s in middle), \
        f"the interior did not depart from both ends {lo:.4f}/{hi:.4f}: {middle}"


@pytest.mark.parametrize("mu", [0.4, 0.8])
def test_doping_the_family_produces_genuinely_complex_weights(mu):
    """The complex-weight case the parity argument of §7 does not reach.

    This test asserted `imagshare > 1.0` while `imagshare` was `|Im w| / |Re w|` -- a ratio that
    can exceed 1 only because it is unbounded, which is the defect that statistic has.  It now
    reads a share bounded by 1, so the threshold that used to pass is now unreachable, and the
    claim is carried by the phase fraction instead.
    """
    # Compared to the SAME family at half filling, which is the control this claim needs. The
    # control is EXACTLY zero on two of these three -- so the earlier `10x` and `1e6x` margins
    # were multiplying zero, padding chosen against nothing. What the control being exactly zero
    # supports is the strict statement, and that is what is asserted.
    for seed in SEEDS:
        doped = sweep(1.0, 4.0, mu, seed)
        free = sweep(1.0, 4.0, 0.0, seed)
        assert free["negre"] == 0.0, \
            f"the half-filled control is no longer exactly clean: {free}"
        assert doped["imagshare"] > free["imagshare"], \
            "the doped weight is not measurably more imaginary than the half-filled one"
        assert doped["negre"] > free["negre"], \
            f"the doped negative fraction does not stand out: {doped['negre']} vs {free['negre']}"


@pytest.mark.parametrize("mu", [0.4, 0.8])
def test_the_read_saturates_on_the_worst_complex_row(mu):
    """AND THE READ DOES NOT REACH THEM EITHER -- the collision of §5, at theta = 1."""
    for seed in SEEDS:
        r = sweep(1.0, 4.0, mu, seed)
        # 'The worst complex row' against the two poles this quantity has: the sign-free control,
        # exactly 0, and a coin flip at 1/2, which is what a fraction of negative real parts is at
        # chance. The row must sit nearer chance than nearer the clean control. `> 0.3` was neither.
        free = sweep(1.0, 4.0, 0.0, seed)["negre"]
        assert abs(r["negre"] - 0.5) < abs(r["negre"] - free), \
            f"this row is nearer the sign-free control than chance: {r['negre']:.4f}"
        assert r["s"] == pytest.approx(1.0, abs=1e-4), \
            f"expected the degenerate saturation at theta = 1, got {r['s']:.4f}"


def test_the_effective_sample_size_refuses_a_reweighted_quantity():
    """The guard on the estimator: prior draws cannot support a reweighted expectation here.

    Pinned so no later draft quotes a mean sign or mean phase from this experiment.

    THE ONE CHOSEN LEVEL LEFT IN THIS SUITE, AND IT IS NAMED RATHER THAN DRESSED UP. `0.1 * n` is
    a chosen efficiency cut. Everything else in these files was rewritten as a comparison between
    quantities measured in the same call; this one has no such reference, because the physics
    supplies no level at which reweighting stops being supported -- it degrades continuously.

    What IS measured, 2026-09-07, is that the premise the docstring implies is wrong. `ess` here is
    `(sum |w|)^2 / sum |w|^2`, a MAGNITUDE statistic, and it reads 2.0-14.7 out of 400 across the
    whole grid -- INCLUDING at mu = 0, where every weight is real and positive and reweighting is
    perfectly valid. So the low ESS is the determinant magnitudes' dynamic range, not the sign
    problem, and this gate does not measure what its name suggests. It is kept because the
    conclusion it protects is still correct -- an ESS of 2 supports no reweighted mean -- and it
    is flagged here so no later draft cites it as evidence about the sign.
    """
    for mu in (0.0, 0.8):
        for theta in (0.0, 0.5, 1.0):
            r = sweep(theta, 4.0, mu, 1)
            assert r["ess"] < 0.1 * r["n"], \
                f"ESS reached {r['ess']:.1f}/{r['n']} -- re-examine whether reweighting is now valid"


@pytest.mark.parametrize("mu", [0.4, 0.8])
def test_the_doped_weights_carry_a_genuine_phase(mu):
    """A phase problem, not a sign problem -- and measured with a bounded statistic.

    `|Im w| / |Re w|` was used here once and is unbounded: it diverges wherever `Re w` passes near
    zero, which is what a weight crossing between signs does, so it ordered the rows backwards.
    `|Im w| / |w|` is a share, and the phase fraction is what distinguishes a genuine phase from a
    real weight of either sign.
    """
    for seed in SEEDS:
        r = sweep(0.5, 4.0, mu, seed)
        # `|Im w|/|w|` is a SHARE, so it has two poles that are not chosen: 0, which the
        # half-filled control reads to 3e-15, and 1, the definitional maximum. The row must sit
        # past the midpoint of its own two poles. `0.85` and `0.4` were guesses at where that lies.
        free = sweep(0.5, 4.0, 0.0, seed)
        assert free["imagshare"] < 1e-9, f"the control carries a phase: {free['imagshare']:.3e}"
        assert r["imagshare"] > (free["imagshare"] + 1.0) / 2.0, \
            f"|Im w|/|w| did not pass the midpoint of its own poles: {r['imagshare']:.4f}"


@pytest.mark.parametrize("theta", [0.0, 0.5, 1.0])
def test_the_half_filled_weights_carry_no_phase_at_any_theta(theta):
    """The control for the row above: at half filling every weight is real and positive."""
    for seed in SEEDS:
        r = sweep(theta, 4.0, 0.0, seed)
        assert r["imagshare"] < 1e-9, f"a phase appeared at half filling: {r['imagshare']:.3e}"
        assert 1.0 - r["absmean"] < 1e-9, \
            f"the mean phase left the real axis at half filling: {1.0 - r['absmean']:.3e}"


def test_the_lockstep_is_absent_in_the_interior_of_the_family():
    """THE MECHANISM. Sign-free throughout, but the lockstep only exists at theta = 0.

    At theta = 1 the agreement is trivial -- lam_s = 0 makes the two determinants the same number
    -- so the test asserts the interior is at chance and BOTH ends are at 1, which is the shape
    that distinguishes "the lockstep is the mechanism" from "the lockstep is always there".
    """
    from reads.expAJ_how_positivity_arises import measure
    ends = [measure(th, 4.0, 0.0, seed=1) for th in (0.0, 1.0)]
    mid = [measure(th, 4.0, 0.0, seed=1) for th in (0.25, 0.5, 0.75)]
    assert all(r["neg"] == 0.0 for r in ends + mid), "this axis must be sign-free throughout"
    assert all(r["lock"] == 1.0 for r in ends), [r["lock"] for r in ends]
    # Against the two poles the line above establishes: locked, which the ends read at exactly 1,
    # and chance, which for an agreement fraction is 1/2. The interior must be nearer chance than
    # nearer lockstep. `0.4 < lock < 0.6` was a window drawn around chance by hand.
    locked = float(np.mean([r["lock"] for r in ends]))
    assert all(abs(r["lock"] - 0.5) < abs(r["lock"] - locked) for r in mid), \
        f"the interior was not at chance: {[round(r['lock'], 4) for r in mid]}"


# ── §8.3: the scalar decoupling family, and which member wins ───────────────

def test_both_rewritings_of_the_interaction_are_exact_at_every_mixing():
    """The algebra §8.3 rests on, checked by quadrature before any lattice."""
    from model2d import single_site_identity
    for theta in (0.0, 0.25, 0.5, 0.75, 1.0):
        worst = max(abs(q - e) / abs(e) for _, _, e, q in single_site_identity(4.0, 0.2, theta))
        assert worst < 1e-12, f"the decoupling identity failed at theta = {theta}: {worst:.2e}"


def test_the_spin_channel_wins_monotonically_with_no_interior_optimum():
    """§8.3's sweep: the family is one parameter wide and its best member is an endpoint.

    Asserted as an ORDERING across theta rather than on any single value, and over seeds, because
    the negative fraction is an estimate. An interior optimum would show as a violation of the
    ordering, which is the thing worth ruling out.
    """
    means, scatters = [], []
    for theta in (0.0, 0.25, 0.5, 0.75, 1.0):
        negs = [sweep(theta, 4.0, 0.8, s)["negre"] for s in SEEDS]
        means.append(float(np.mean(negs)))
        scatters.append(float(max(negs) - min(negs)))
    assert means == sorted(means), f"not monotone in theta: {means}"
    # The endpoints are separated by more than the read scatters across seeds at any theta --
    # both measured here. `< 0.01` and `> 0.4` were two levels pinned on one draw each.
    assert means[-1] - means[0] > max(scatters), \
        f"the endpoints differ by {means[-1] - means[0]:.5f}, no more than the seed scatter " \
        f"({max(scatters):.5f}): {means}"
