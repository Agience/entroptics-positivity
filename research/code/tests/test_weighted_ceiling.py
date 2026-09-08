"""§5's ceiling, gated: every weighted read is capped at n<sgn>^2.

The identity is exact, so it is asserted exactly rather than to a tolerance. What makes it an
argument about ALL weighted reads rather than about `carriage` is that `effective_n` is a property
of the weights alone -- the frames are irrelevant, and the third test pins that: changing the
frames entirely must not move it.
"""
from __future__ import annotations

import numpy as np
import pytest

import entroptics as E
from closed.expAS_weighted_reads_are_capped import kish, sign_weights


@pytest.mark.parametrize("n", [400, 2000])
@pytest.mark.parametrize("p", [0.999, 0.75, 0.55, 0.505])
def test_the_effective_sample_size_is_exactly_n_times_mean_sign_squared(n, p):
    w = sign_weights(n, p, seed=int(1000 * p) + n)
    X = np.random.default_rng(0).standard_normal((n, 8))
    assert E.carriage(X, w).effective_n == pytest.approx(n * float(w.mean()) ** 2, rel=1e-12)
    assert kish(w) == pytest.approx(n * float(w.mean()) ** 2, rel=1e-12)


def test_the_ceiling_does_not_depend_on_the_frames():
    """THE LOAD-BEARING PART. A ceiling set by the weights binds every aggregation of them.

    If `effective_n` moved with the frames it would be a statement about one read; it does not,
    which is why the bound applies to the coupling of §5 as much as to `carriage` itself.
    """
    w = sign_weights(600, 0.6, seed=7)
    base = None
    for seed, shape in ((1, (600, 8)), (2, (600, 32)), (3, (600, 3))):
        X = np.random.default_rng(seed).standard_normal(shape)
        v = E.carriage(X, w).effective_n
        if base is None:
            base = v
        assert v == pytest.approx(base, rel=1e-12), "the ceiling moved with the frames"


def test_the_ceiling_collapses_as_the_sign_problem_worsens():
    """Without this the identity could hold trivially at <sgn> ~ 1 and say nothing."""
    n = 2000
    ess, sgn = [], []
    for p in (0.99, 0.8, 0.6, 0.51):
        w = sign_weights(n, p, seed=int(1000 * p))
        ess.append(E.carriage(np.random.default_rng(0).standard_normal((n, 8)), w).effective_n)
        sgn.append(float(np.mean(np.sign(w))))
    assert ess == sorted(ess, reverse=True), f"not monotone in the sign: {ess}"
    # The size of the collapse is not a chosen fraction: this file's own identity says the ceiling
    # is n * <sgn>^2, so the ratio between the ends is fixed by the two average signs MEASURED on
    # those same weights. `< 0.05 * ess[0]` was a level that happened to sit above it.
    assert ess[-1] / ess[0] == pytest.approx((sgn[-1] / sgn[0]) ** 2, rel=1e-12), \
        f"the ceiling did not collapse as n<sgn>^2 requires: ess {ess}, <sgn> {sgn}"
