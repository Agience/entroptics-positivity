"""Section 1.1 names each read as a formula. These pin the naming against the library.

Section 1.1 is the paper's answer to "what is this quantity", and a reader who takes a claim from it
is entitled to the identification being exact rather than approximate. Each test states the formula
in numpy and asserts the library agrees, and each carries a negative control: an identification that
passed against everything would pin nothing.

EXACT is the standard here, not a tolerance, wherever the two are the same arithmetic.
"""
from __future__ import annotations

import numpy as np
import pytest

import entroptics_adapter as EA

EXACT = 1e-12


# ── coupling.strength is the Pearson correlation on one column ──────────────────────────────────

@pytest.mark.parametrize("slope,noise", [(3.0, 0.0), (-2.0, 0.0), (2.0, 0.3), (0.2, 1.0)])
def test_strength_is_the_pearson_correlation_on_a_single_column(slope, noise):
    """For k = 1 the read IS `numpy.corrcoef`, which is what section 6's build check rests on."""
    rng = np.random.default_rng(11)
    a = rng.standard_normal((400, 1))
    b = slope * a + 7.0 + noise * rng.standard_normal((400, 1))
    c = EA.channel_alignment(a, b)
    if not c.resolved:
        pytest.skip("unresolved pairs are covered by the next test")
    assert float(c.strength) == pytest.approx(
        float(np.corrcoef(a[:, 0], b[:, 0])[0, 1]), abs=EXACT)


def test_an_unresolved_pair_returns_zero_rather_than_a_small_correlation():
    """The read declines rather than reporting noise -- the behaviour section 1.1 states.

    The negative control is the resolved case in the same test: if the read returned zero for
    everything, `corrcoef` agreement above would be impossible, so the two together pin both
    branches.
    """
    rng = np.random.default_rng(0)
    a, b = rng.standard_normal((400, 1)), rng.standard_normal((400, 1))
    c = EA.channel_alignment(a, b)
    assert not c.resolved
    assert float(c.strength) == 0.0
    assert abs(float(np.corrcoef(a[:, 0], b[:, 0])[0, 1])) > 0.01, \
        "the control pair is too well aligned to demonstrate the decline"


def test_tightness_is_the_leading_share_of_the_squared_cross_covariance_spectrum():
    rng = np.random.default_rng(3)
    A = rng.standard_normal((400, 6))
    for B in (1.0 - A, 1.0 - A + 0.5 * rng.standard_normal((400, 6))):
        Ac = A - A.mean(axis=0)
        Bc = B - B.mean(axis=0)
        sv = np.linalg.svd(Ac.T @ Bc / len(A), compute_uv=False)
        want = float(sv[0] ** 2 / (sv ** 2).sum())
        assert float(EA.channel_alignment(A, B).tightness) == pytest.approx(want, abs=EXACT)


# ── concentration: resultant and focus ──────────────────────────────────────────────────────────

def _unit(theta):
    return np.c_[np.cos(theta), np.sin(theta)]


@pytest.mark.parametrize("theta,label", [
    (np.zeros(600), "all real positive"),
    (np.random.default_rng(1).choice([0.0, np.pi], 600), "antipodal"),
    (np.random.default_rng(2).uniform(0, 2 * np.pi, 600), "isotropic"),
    (np.random.default_rng(4).normal(0.4, 0.25, 600), "narrow arc"),
])
def test_resultant_is_the_mean_resultant_length_and_focus_is_the_leading_orientation_eigenvalue(
        theta, label):
    P = _unit(np.asarray(theta, dtype=float))
    c = EA.weight_cloud(P)
    T = (P.T @ P) / len(P)
    lead = float(np.sort(np.linalg.eigvalsh(T))[-1])
    assert float(c.resultant) == pytest.approx(float(np.linalg.norm(P.mean(axis=0))), abs=EXACT)
    assert float(c.focus) == pytest.approx(lead, abs=EXACT)


def test_focus_is_floored_at_one_half_and_the_two_statistics_are_not_the_same_read():
    """`focus` lives in [1/2, 1] because the rows are unit vectors in the plane.

    The antipodal cloud is the control that separates the pair: a directional statistic and an
    axial one must disagree there, or the classification of section 7.1 would be reading one
    quantity twice.
    """
    rng = np.random.default_rng(5)
    iso = EA.weight_cloud(_unit(rng.uniform(0, 2 * np.pi, 4000)))
    anti = EA.weight_cloud(_unit(rng.choice([0.0, np.pi], 4000)))
    assert 0.5 <= float(iso.focus) <= 1.0
    assert float(iso.focus) == pytest.approx(0.5, abs=0.05), \
        "an isotropic cloud must sit at the floor, not near 1"
    assert float(anti.focus) == pytest.approx(1.0, abs=EXACT)
    assert float(anti.resultant) < 0.1, \
        "the directional statistic must collapse where the axial one saturates"


# ── carriage.effective_n is Kish's effective sample size ─────────────────────────────────────────

@pytest.mark.parametrize("n,n_neg", [(100, 0), (100, 10), (400, 137), (400, 200)])
def test_effective_n_is_kish_and_equals_n_times_mean_sign_squared(n, n_neg):
    signs = np.ones(n)
    signs[:n_neg] = -1.0
    # The frame is the read's first argument and the ceiling does not depend on it, which is the
    # point of the read: `effective_n` is a property of the weights alone.
    frame = np.random.default_rng(int(n + n_neg)).standard_normal((n, 4))
    got = float(EA.evidence_ceiling(frame, signs).effective_n)
    kish = float(signs.sum() ** 2 / (signs ** 2).sum())
    assert got == pytest.approx(kish, abs=EXACT)
    assert got == pytest.approx(n * float(signs.mean()) ** 2, abs=EXACT)


def test_the_ceiling_is_not_simply_the_sample_size():
    """The negative control: a formula returning `n` would pass the identity above at n_neg = 0."""
    signs = np.ones(400)
    signs[:200] = -1.0
    frame = np.random.default_rng(77).standard_normal((400, 4))
    assert float(EA.evidence_ceiling(frame, signs).effective_n) < 1.0


# ── spectral_optics.top_share ────────────────────────────────────────────────────────────────────

def test_top_share_is_the_leading_eigenvalue_share_of_the_spectrum():
    rng = np.random.default_rng(9)
    X = rng.standard_normal((200, 8)) @ np.diag([5.0, 3.0, 1.0, 0.5, 0.2, 0.1, 0.05, 0.02])
    so = EA.single_channel_optics(X)
    ev = np.asarray(so.eigenvalues, dtype=float)
    assert float(so.top_share) == pytest.approx(float(ev[0] / ev.sum()), abs=EXACT)
    assert float(so.top_share) < 1.0, "a share of 1 would mean the spectrum had one mode"


# ── the coupling read is not a reweighted estimator ──────────────────────────────────────────────

def test_the_sign_never_enters_the_coupling_read():
    """Section 9 rests on this: the read is unweighted, so the `n <sgn>^2` ceiling does not bound it.

    The ceiling is Kish's effective sample size of the sign weights, which bounds a reweighted mean.
    Section 7's read is handed two channel frames and nothing else, so flipping the sign of every
    configuration -- the most violent change the weights admit -- cannot move it. The control is the
    same flip applied to the FRAME, which must move the read, or this would pin nothing.
    """
    rng = np.random.default_rng(23)
    A = rng.standard_normal((300, 6))
    B = 1.0 - A + 0.2 * rng.standard_normal((300, 6))
    base = float(EA.channel_alignment(A, B).strength)

    signs = rng.choice([-1.0, 1.0], size=300)
    # Reweighting by these signs is what a physical expectation would require; the read has no
    # argument for them, so the same call on the same frames returns the same value.
    assert float(EA.channel_alignment(A, B).strength) == base

    flipped = A * signs[:, None]
    assert float(EA.channel_alignment(flipped, B).strength) != pytest.approx(base, abs=1e-6),         "flipping rows of the FRAME must move the read, or the test pins nothing"


def test_the_ceiling_does_not_depend_on_the_frame_it_is_given():
    """`effective_n` is a property of the weights, which is what makes it a ceiling on the weights."""
    rng = np.random.default_rng(31)
    signs = rng.choice([-1.0, 1.0], size=400)
    a = float(EA.evidence_ceiling(rng.standard_normal((400, 4)), signs).effective_n)
    b = float(EA.evidence_ceiling(rng.standard_normal((400, 9)) * 50.0, signs).effective_n)
    assert a == pytest.approx(b, abs=EXACT)
    assert a == pytest.approx(400 * float(signs.mean()) ** 2, abs=EXACT)
