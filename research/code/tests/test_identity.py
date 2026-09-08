"""The identity that carries positivity, as a gate.

At half filling on a bipartite lattice the two spin channels' determinants are related exactly:

    ln|det(I + B_up(x))| - ln|det(I + B_dn(x))| = -dtau * L * tr(K) + lambda * sum(x)

with `lambda = arccosh(exp(dtau U / 2))` the decoupling's own constant, not a fitted one. The
coefficient is DERIVED: `det(B_up)/det(B_dn) = exp(2 lambda sum x)` because the two channels'
diagonal factors are inverses, and `det(I+B) = det(B) det(I+B^-1)` carries one factor of it into
the ratio of the full determinants.

This single relation explains two facts that look contradictory on their own:

  * the ratio det_up/det_dn is an EXPONENTIAL, hence positive, hence the two determinants always
    share a sign -- which is the lockstep, and hence the absence of a sign problem;
  * its exponent is a sum over L*N field components, hence the ratio's magnitude ranges over 8 to
    16 orders -- which is why the two determinants' magnitudes look unrelated.

The tests below check the identity, check that the WRONG coefficient fails (so the test is not
passing on something vacuous), and check that doping breaks it.
"""
from __future__ import annotations

import numpy as np
import pytest

from model2d import Model2D
from stable import udt_product, slogdet_one_plus_block


def log_dets(m, X, block=4):
    """(sign, log|det(I + B_sigma)|) per spin, in the conditioned frame."""
    lam = float(np.arccosh(np.exp(m.dtau * m.U / 2.0)))
    out = {}
    for sigma in (+1, -1):
        d = np.exp(sigma * lam * X)
        Bl = m.expmK[None, :, :] * d[:, None, :]
        U, D, T = udt_product(Bl[None], block)
        s, la = slogdet_one_plus_block(U, D, T)
        out[sigma] = (float(np.real(s[0])), float(la[0]))
    return out


def residuals(m, n, seed, coefficient):
    lam = float(np.arccosh(np.exp(m.dtau * m.U / 2.0)))
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        X = rng.choice([-1.0, 1.0], size=(m.L, m.N))
        o = log_dets(m, X)
        pred = -m.dtau * m.L * float(np.trace(m.K)) + coefficient * lam * float(X.sum())
        out.append(abs((o[+1][1] - o[-1][1]) - pred))
    return np.array(out)


@pytest.mark.parametrize("beta", [2.0, 4.0, 6.0, 8.0])
def test_the_identity_is_exact_at_half_filling(beta):
    m = Model2D(Lx=2, Ly=4, t=1.0, mu=0.0, U=4.0, dtau=0.125,
                L=int(round(beta / 0.125)), theta=0.0)
    r = residuals(m, 120, seed=int(beta), coefficient=1.0)
    assert r.max() < 1e-10, f"the identity failed at {r.max():.2e}"


@pytest.mark.parametrize("beta", [2.0, 6.0])
def test_the_wrong_coefficient_fails(beta):
    """THE NEGATIVE CONTROL. A test that passes for any coefficient is testing nothing.

    The derivation gives lambda; twice lambda is the coefficient one gets by taking the ratio of
    det(B) directly instead of routing it through det(I+B) = det(B) det(I+B^-1). It must fail.
    """
    m = Model2D(Lx=2, Ly=4, t=1.0, mu=0.0, U=4.0, dtau=0.125,
                L=int(round(beta / 0.125)), theta=0.0)
    r = residuals(m, 120, seed=int(beta), coefficient=2.0)
    assert r.max() > 1.0, "the wrong coefficient did not fail, so the test is vacuous"


def test_the_identity_implies_the_signs_lock():
    """Where the identity holds the two determinants share a sign, because exp() is positive."""
    m = Model2D(Lx=2, Ly=4, t=1.0, mu=0.0, U=4.0, dtau=0.125, L=64, theta=0.0)
    rng = np.random.default_rng(9)
    flips = 0
    for _ in range(200):
        X = rng.choice([-1.0, 1.0], size=(m.L, m.N))
        o = log_dets(m, X)
        assert o[+1][0] == o[-1][0], "the identity holds here but the signs differed"
        flips += int(o[+1][0] < 0)
    assert flips > 0, "no determinant ever changed sign, so the lock is vacuous here"


@pytest.mark.parametrize("mu", [0.4, 0.8])
def test_doping_breaks_the_identity(mu):
    """The relation is not a tautology of the algebra -- it depends on the filling."""
    m = Model2D(Lx=2, Ly=4, t=1.0, mu=mu, U=4.0, dtau=0.125, L=48, theta=0.0)
    r = residuals(m, 120, seed=int(mu * 10), coefficient=1.0)
    assert r.max() > 1.0, "doping did not break the identity"


@pytest.mark.parametrize("U,dtau", [(2.0, 0.0625), (4.0, 0.125), (8.0, 0.125), (12.0, 0.0625)])
def test_the_identity_tracks_the_interaction(U, dtau):
    """The coefficient is a function of U and dtau, so each row predicts a different one.

    `lambda` spans 0.357 to 2.180 across the sweep this is drawn from -- a factor of 6.1 -- and a
    relation exact only at the U it was derived at would be a coincidence rather than a
    derivation.
    """
    m = Model2D(Lx=2, Ly=4, t=1.0, mu=0.0, U=U, dtau=dtau,
                L=int(round(6.0 / dtau)), theta=0.0)
    r = residuals(m, 80, seed=int(U * 10 + dtau * 100), coefficient=1.0)
    assert r.max() < 1e-9, f"identity failed at U={U}, dtau={dtau}: {r.max():.2e}"


def test_a_neighbouring_coefficient_fails():
    """THE SHARPER CONTROL: not just the wrong coefficient, but a nearly-right one.

    `lambda(U=4) = 0.7369` against `lambda(U=8) = 1.0850` at dtau = 0.125 -- a 47% difference. If
    any coefficient of roughly the right size satisfied the relation, this would pass, and it must
    not: the identity resolves that difference by fourteen orders.
    """
    m = Model2D(Lx=2, Ly=4, t=1.0, mu=0.0, U=4.0, dtau=0.125, L=48, theta=0.0)
    own = float(np.arccosh(np.exp(0.125 * 4.0 / 2.0)))
    neighbour = float(np.arccosh(np.exp(0.125 * 8.0 / 2.0)))
    good = residuals(m, 80, seed=3, coefficient=1.0)
    bad = residuals(m, 80, seed=3, coefficient=neighbour / own)
    assert good.max() < 1e-9
    assert bad.max() > 1.0, "a neighbouring coefficient also satisfied the relation"
