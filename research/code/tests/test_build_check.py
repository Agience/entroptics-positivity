"""The build check's gate -- the read catches a model-implementation mismatch the sign cannot see.

WHAT IS BEING HELD.  A lattice built with a periodic wrap where open was intended, on an ODD
chain, is a non-bipartite model.  At `beta = 2` it produces no negative weight at all, so every
standard health check passes, and the section 4 criterion computed on the INTENDED lattice says
`sign-free` because it reads the model rather than the code.  `coupling` on two logged columns
fires anyway, and needs neither `lambda` nor `tr(K)` to do it.

THE NEGATIVE CONTROLS, which are what make this gate able to fail:

  - the SAME lattice built correctly must stay saturated at machine zero at the same beta and
    seed, or the read is firing on beta rather than on the bug;
  - the sign must be genuinely silent on the bugged run -- if a negative weight appears, the
    experiment is outside the window it claims and the practitioner did not need this read;
  - the read must NOT detect a wrong decoupling constant.  That is a documented scope limit and
    a consequence of scale-invariance, so a gate that let it pass silently would be concealing
    the boundary the experiment is honest about.  Here it is asserted.
"""
from __future__ import annotations

import numpy as np
import pytest

from reads.expAO_spectral_criterion import chain, criterion
from reads.expAZ_build_check import departure, run

BETA, SEED, N = 2.0, 5, 7


@pytest.fixture(scope="module")
def lattices():
    return chain(N, periodic=False), chain(N, periodic=True)


def test_the_oracle_is_silent_on_the_intended_model(lattices):
    """The premise: a practitioner checking their INTENDED lattice is told it is sign-free."""
    K_ok, K_bug = lattices
    assert bool(criterion(K_ok)) is True
    assert bool(criterion(K_bug)) is False, (
        "the as-built lattice must actually be outside the sign-free class, or there is no bug "
        "to detect and the whole experiment is vacuous")


def test_the_sign_is_silent_in_the_claimed_window(lattices):
    """If a negative weight appears, the read is not needed and the claim is overstated."""
    _, K_bug = lattices
    _, _, sg = run(K_built=K_bug, beta=BETA, seed=SEED)
    assert float((sg < 0).mean()) == 0.0


def test_the_read_fires_on_the_bug(lattices):
    _, K_bug = lattices
    A, B, _ = run(K_built=K_bug, beta=BETA, seed=SEED)
    dep, _, resolved = departure(A, B)
    assert resolved
    assert dep > 1e-6, f"the read must depart from saturation on a non-bipartite build; got {dep:.2e}"


def test_the_correctly_built_control_stays_saturated(lattices):
    """Same beta, same seed, same everything but the bug."""
    K_ok, _ = lattices
    A, B, _ = run(K_built=K_ok, beta=BETA, seed=SEED)
    dep, _, resolved = departure(A, B)
    assert resolved
    assert abs(dep) < 1e-12, f"a correct build must saturate; got {dep:.2e}"


def test_the_bug_stands_clear_of_the_control_by_orders(lattices):
    K_ok, K_bug = lattices
    Ab, Bb, _ = run(K_built=K_bug, beta=BETA, seed=SEED)
    Ac, Bc, _ = run(K_built=K_ok, beta=BETA, seed=SEED)
    bug, _, _ = departure(Ab, Bb)
    ctrl, _, _ = departure(Ac, Bc)
    assert bug > 1e6 * max(abs(ctrl), 1e-16)


def test_the_departure_is_reproducible_across_seeds(lattices):
    """A read that is not reproducible is not a check -- the spread must be well under the signal."""
    _, K_bug = lattices
    deps = []
    for seed in (5, 11, 23, 37):
        A, B, _ = run(K_built=K_bug, beta=BETA, seed=seed)
        d, _, _ = departure(A, B)
        deps.append(d)
    deps = np.asarray(deps)
    assert deps.mean() > 5 * deps.std()


def test_a_wrong_lambda_is_NOT_detected(lattices):
    """The documented scope limit, asserted so it cannot quietly stop being true.

    A wrong decoupling constant rescales an affine relation and leaves it affine.  The read is
    scale-invariant, which is why it needs no `lambda` and why it cannot see a wrong one.  If this
    ever starts failing, the read has changed and the experiment's scope paragraph is stale.
    """
    K_ok, _ = lattices
    lam_gauss = float(np.sqrt(0.125 * 4.0))
    A, B, sg = run(K_built=K_ok, beta=BETA, seed=SEED, lam_built=lam_gauss)
    dep, _, resolved = departure(A, B)
    assert resolved
    assert abs(dep) < 1e-12, (
        f"the read is documented as blind to a wrong lambda; it returned {dep:.2e}. "
        f"Either the read changed or expAZ's scope paragraph is now wrong.")
    assert float((sg < 0).mean()) == 0.0, "and the sign is silent here too, which is the danger"
