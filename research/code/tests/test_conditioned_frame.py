"""The negative control for every number in this repository.

The sign of `det(I + B)` has to be read in a frame that stays conditioned. `I + B` does not: its
condition number grows like `exp(beta * bandwidth)`, and past `beta ~ 4` its eigenvalues stop
meaning anything. That is not a hypothetical -- a version of the spectral analysis that formed B
as the naive ordered product reported a negative-weight fraction of **0.5075 at half filling on a
bipartite lattice**, where positivity is provable, with "distances to a sign flip" of 56, 1117 and
31683 for a quantity that is O(1).

So these tests assert both directions:

  * the conditioned frame gives positivity at half filling at every beta, and two independent
    routes to the sign agree on every configuration;
  * the NAIVE frame fails there. A test suite that only checked the good path would pass equally
    on a build that had silently reverted to the bad one.

This is the third time the naive product was reintroduced in this work, in three different files.
It is not a mistake anyone makes once.
"""
from __future__ import annotations

import numpy as np
import pytest

from model2d import Model2D
from stable import udt_product, core_matrix, slogdet_one_plus_block


def naive_sign(m, X):
    """The sign of det(I+B) with B formed as the plain ordered product -- the thing not to do."""
    lam = float(np.arccosh(np.exp(m.dtau * m.U / 2.0)))
    s = 1.0
    for sigma in (+1, -1):
        d = np.exp(sigma * lam * X)
        B = np.eye(m.N)
        for l in range(m.L):
            B = (m.expmK * d[l][None, :]) @ B
        s *= float(np.sign(np.linalg.det(np.eye(m.N) + B)))
    return s


def stable_signs(m, X, block=4):
    """(sign from the block formulation, sign from the core factorisation) -- two routes."""
    lam = float(np.arccosh(np.exp(m.dtau * m.U / 2.0)))
    sb, sc = 1.0, 1.0
    for sigma in (+1, -1):
        d = np.exp(sigma * lam * X)
        Bl = m.expmK[None, :, :] * d[:, None, :]
        U, D, T = udt_product(Bl[None], block)
        s, _ = slogdet_one_plus_block(U, D, T)
        sb *= float(np.real(s[0]))
        _, s_core = core_matrix(U, D, T)
        sc *= float(np.real(s_core[0]))
    return sb, sc


@pytest.mark.parametrize("beta", [2.0, 6.0, 10.0])
def test_conditioned_frame_gives_positivity_at_half_filling(beta):
    m = Model2D(Lx=2, Ly=4, t=1.0, mu=0.0, U=4.0, dtau=0.125,
                L=int(round(beta / 0.125)), theta=0.0)
    rng = np.random.default_rng(int(beta))
    neg = 0
    for _ in range(200):
        X = rng.choice([-1.0, 1.0], size=(m.L, m.N))
        sb, _ = stable_signs(m, X)
        neg += int(sb < 0)
    assert neg == 0, f"{neg} negative weights at half filling, where positivity is provable"


@pytest.mark.parametrize("beta", [2.0, 6.0, 10.0])
def test_two_routes_to_the_sign_agree(beta):
    """The block formulation and the core factorisation share no arithmetic past the UDT."""
    m = Model2D(Lx=2, Ly=4, t=1.0, mu=0.4, U=4.0, dtau=0.125,
                L=int(round(beta / 0.125)), theta=0.0)
    rng = np.random.default_rng(int(beta) + 11)
    for _ in range(200):
        X = rng.choice([-1.0, 1.0], size=(m.L, m.N))
        sb, sc = stable_signs(m, X)
        assert sb == sc


def test_the_answer_does_not_move_with_the_stabilisation_block():
    """A row that depends on the block is arithmetic, not physics.

    The claim is an EQUALITY BETWEEN BLOCKS, so no reference number is needed and none is used.
    A first version of this test asserted the count equalled a hardcoded 19 -- a number chosen
    rather than derived, which is the thing this work is not allowed to do, and which failed
    immediately because the true count is 9.  The sign of every individual configuration is
    compared, not just the total, so two blocks cannot agree by cancellation.
    """
    m = Model2D(Lx=2, Ly=4, t=1.0, mu=0.4, U=4.0, dtau=0.125, L=80, theta=0.0)
    per_block = {}
    for block in (2, 4, 8):
        rng = np.random.default_rng(5)
        signs = []
        for _ in range(200):
            X = rng.choice([-1.0, 1.0], size=(m.L, m.N))
            signs.append(stable_signs(m, X, block)[0])
        per_block[block] = np.array(signs)
    ref = per_block[2]
    assert float(np.mean(ref < 0)) > 0.0, "no negative weights here, so nothing is being compared"
    for block, s in per_block.items():
        assert np.array_equal(s, ref), f"block {block} disagreed with block 2 configuration-wise"


def test_the_naive_frame_fails_and_that_is_the_point():
    """THE NEGATIVE CONTROL. The naive product must report a sign problem where there is none.

    Without this, a build that reverted to the unstabilised product would pass every other test in
    this file by simply being a different -- and wrong -- calculation.
    """
    m = Model2D(Lx=2, Ly=4, t=1.0, mu=0.0, U=4.0, dtau=0.125, L=64, theta=0.0)
    rng = np.random.default_rng(2)
    naive_neg = stable_neg = 0
    for _ in range(200):
        X = rng.choice([-1.0, 1.0], size=(m.L, m.N))
        naive_neg += int(naive_sign(m, X) < 0)
        stable_neg += int(stable_signs(m, X)[0] < 0)
    assert stable_neg == 0
    assert naive_neg > 0, ("the naive product did not fail here, so this test is not the control "
                           "it claims to be -- raise beta until it does")
