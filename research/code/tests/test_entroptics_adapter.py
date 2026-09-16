"""The adapter is a NAMING layer. This gate is what makes that a checkable claim rather than an
intention.

`entroptics_adapter` exists so that every Entroptics read in this repository is reached under the
paper's name for it, through one module that also holds the version pin and the two cautions that
cost real time. The risk of such a module is that it quietly becomes a second implementation -- a
centring here, a reshape there -- and then the paper's numbers depend on code that no section
describes and no gate covers.

So every read is asserted to BE the library call it names, value for value, on inputs shaped like
the ones the experiments use. Where the adapter does do something -- stacking a complex vector into
its `(Re, Im)` frame -- the test performs that step by hand and requires the results to agree, so
the convenience is pinned to one spelling rather than trusted.

THE NEGATIVE CONTROL, which is what makes this able to fail: `test_the_comparison_can_detect_a_
divergent_adapter` calls the library with a DIFFERENT argument and requires the comparison to
notice. Without it a test that compared a value with itself would pass unchanged, and every
assertion here would be vacuous.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import entroptics as E                                             # noqa: E402
import entroptics_adapter as A                                     # noqa: E402
from entroptics.dynamics import dynamics as _dynamics              # noqa: E402
from entroptics.environment import to_numpy                        # noqa: E402


def _frames(seed=0, n=120, k=6):
    """Two frames of the shape the channel reads take: one row per configuration."""
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((n, k))
    b = 1.0 - a + 0.05 * rng.standard_normal((n, k))    # the particle-hole relation, perturbed
    return a, b


def _cloud(seed=1, n=400, theta=0.4):
    """A complex weight cloud carrying a global phase, as section 7.1 reads them."""
    rng = np.random.default_rng(seed)
    s = rng.choice([1.0, -1.0], size=n, p=[0.8, 0.2])
    return (s * rng.gamma(4.0, 1.0, size=n)) * np.exp(1j * theta)


# ---------------------------------------------------------------------------------------------
# the version pin
# ---------------------------------------------------------------------------------------------

def test_the_pin_comes_from_requirements_and_is_met():
    """The number is read from the file pip installs from, not written here as well."""
    req = (Path(__file__).resolve().parents[2] / "requirements.txt").read_text(encoding="utf-8")
    assert f"entroptics>={A.REQUIRED_VERSION}" in req or \
           f"entroptics=={A.REQUIRED_VERSION}" in req, \
        "the adapter's pin is not the one in requirements.txt"
    assert A._version_tuple(A.INSTALLED_VERSION) >= A._version_tuple(A.REQUIRED_VERSION)


def test_the_pin_would_refuse_an_older_reader():
    """The check is able to fire -- a comparison that always passes is not a check.

    The refusal itself cannot be provoked without uninstalling the library, so what is asserted is
    the comparison the refusal is made of, at a version below the pin.
    """
    assert A._version_tuple("0.2.1") < A._version_tuple(A.REQUIRED_VERSION)
    assert A._version_tuple(A.REQUIRED_VERSION) >= A._version_tuple("0.2.3"), \
        "the paper's numbers were read through 0.2.3; the pin must not fall below it"


# ---------------------------------------------------------------------------------------------
# each read IS the library call it names
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [0, 1, 2])
def test_channel_alignment_is_the_coupling_read(seed):
    a, b = _frames(seed)
    got, want = A.channel_alignment(a, b), E.reads.coupling(a, b)
    assert float(got.strength) == float(want.strength)
    assert float(got.z) == float(want.z)
    assert bool(got.resolved) == bool(want.resolved)


def test_the_comparison_can_detect_a_divergent_adapter():
    """THE NEGATIVE CONTROL. Compare against a different argument and the equality must fail.

    Without this every assertion above could be comparing a value with itself.
    """
    a, b = _frames(0)
    mismatched = E.reads.coupling(a, b[::-1])
    assert float(A.channel_alignment(a, b).strength) != float(mismatched.strength), \
        "the comparison cannot tell two different reads apart, so it proves nothing"


@pytest.mark.parametrize("theta", [0.0, 0.4, 1.3])
def test_weight_cloud_is_concentration_on_the_re_im_frame(theta):
    w = _cloud(theta=theta)
    by_hand = E.reads.concentration(np.stack([w.real, w.imag], axis=1))
    got = A.weight_cloud(w)
    assert float(got.resultant) == float(by_hand.resultant)
    assert float(got.focus) == float(by_hand.focus)


def test_weight_cloud_passes_an_already_stacked_frame_through_untouched():
    """A caller that has already built the frame must get the same answer as one that has not."""
    w = _cloud()
    X = np.stack([w.real, w.imag], axis=1)
    assert float(A.weight_cloud(X).focus) == float(A.weight_cloud(w).focus)


def test_cloud_axes_is_principal_directions():
    w = _cloud()
    X = np.stack([w.real, w.imag], axis=1)
    got, want = A.cloud_axes(w), E.reads.principal_directions(X)
    assert np.asarray(got).shape == np.asarray(want).shape
    assert np.array_equal(np.asarray(got), np.asarray(want))


def test_cloud_axes_returns_no_column_on_a_cloud_with_no_axis():
    """The parameter-free 'there is none' the adapter's docstring claims, exercised.

    A real-weight cloud has its imaginary part identically zero, so there is no axis in the plane
    to report. A hand-rolled PCA would return one anyway.
    """
    rng = np.random.default_rng(3)
    w = rng.choice([1.0, -1.0], size=300) * rng.gamma(4.0, 1.0, size=300)
    assert np.asarray(A.cloud_axes(w.astype(complex))).shape[1] <= 1


@pytest.mark.parametrize("n,p", [(400, 0.75), (2000, 0.55)])
def test_evidence_ceiling_is_carriage_and_is_exactly_n_mean_sign_squared(n, p):
    rng = np.random.default_rng(int(1000 * p) + n)
    w = rng.choice([1.0, -1.0], size=n, p=[p, 1 - p])
    X = rng.standard_normal((n, 8))
    got, want = A.evidence_ceiling(X, w), E.carriage(X, w)
    assert float(got.effective_n) == float(want.effective_n)
    # and it IS the ceiling section 9 derives, not merely the same as the library's number
    assert float(got.effective_n) == pytest.approx(n * float(w.mean()) ** 2, rel=1e-12)


def test_autocorrelation_time_is_reconstruct_decay_and_not_the_raw_operator():
    """The read must be the CONNECTED one, and the two must be distinguishable on this input.

    `rates().dominant` reads the raw operator, where the constant function is an eigenmode with
    |mu| = 1. On a sign sequence with a large mean it returns that mode instead of the decay. The
    second assertion is what makes the first one worth making.
    """
    rng = np.random.default_rng(7)
    s = np.where(rng.random(3000) < 0.9, 1.0, -1.0)
    for i in range(1, s.size):                      # a chain, not independent draws
        if rng.random() < 0.3:
            s[i] = s[i - 1]
    got = A.autocorrelation_time(s, 1000)
    want = np.asarray(to_numpy(_dynamics(s.reshape(-1, 1)).reconstruct_decay(1000)), float)
    assert np.array_equal(got, want)

    # the trap, exhibited. `rates()` reads the RAW operator, which is not translation-invariant: on
    # a sequence with mean 0.81 the constant component dominates and the reported decay rate is
    # pulled down, so the tau it implies is far too long. Centring the same sequence -- which is
    # what `reconstruct_decay` does by reading the CONNECTED operator -- more than quadruples the
    # rate. If these agreed, the caution the adapter documents would not be showable here.
    raw_rate = float(np.asarray(_dynamics(s.reshape(-1, 1)).rates().dominant).ravel()[0])
    centred_rate = float(np.asarray(
        _dynamics((s - s.mean()).reshape(-1, 1)).rates().dominant).ravel()[0])
    assert centred_rate > 2.0 * raw_rate, \
        (f"the raw and centred operators agree here ({raw_rate:.3f} vs {centred_rate:.3f}), so "
         f"this input cannot show the trap")
    assert 1.0 / raw_rate > 2.0 * (0.5 + float(got[1:].sum())), \
        "the raw operator's implied tau is not longer than the connected read's on this input"


def test_single_channel_optics_is_spectral_optics():
    rng = np.random.default_rng(5)
    data = rng.standard_normal((200, 12))
    got, want = A.single_channel_optics(data), E.reads.spectral_optics(data)
    assert int(got.resolved_modes) == int(want.resolved_modes)
    assert float(got.phase) == float(want.phase)


def test_denoise_is_aperture_extract_and_returns_W_s_own_units():
    rng = np.random.default_rng(11)
    W = rng.standard_normal((150, 10))
    clean, info = A.denoise(W)
    clean_ref, info_ref = E.Aperture(W).extract()
    assert np.array_equal(np.asarray(clean), np.asarray(clean_ref))
    # No rescaling, checked on a frame with nothing resolvable: there the shrunk projection is
    # empty, so `clean.mean(axis=0)` is the centre. On a frame carrying structure the two
    # differ, which is section 9.7's result.
    assert np.allclose(np.asarray(clean).mean(axis=0), np.asarray(info["centre"]), atol=1e-9)
    assert np.allclose(np.asarray(info["centre"]), np.asarray(info_ref["centre"]))


def test_balance_at_own_zero_is_a_screen():
    assert isinstance(A.balance_at_own_zero(), type(E.Screen()))
