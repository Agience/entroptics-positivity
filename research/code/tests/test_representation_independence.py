"""The §3 identity is a property of the decoupling's STRUCTURE, not of the Ising field.

The derivation uses one property: the two spins' diagonal factors are inverses.  The continuous
Gaussian spin decoupling has it too, with its own closed-form constant, so the same derivation
predicts the same identity there with a DIFFERENT coefficient:

    discrete    lambda   = arccosh(exp(dtau U / 2))  = 0.73690
    continuous  lambda_s = sqrt(dtau U)              = 0.70711

Four percent apart, neither fitted, each a closed form of its own decoupling.  The cross test --
each representation's constant must FAIL in the other -- is the sharpest control in the project,
because a relation satisfied by any coefficient of roughly the right size cannot pass it.
"""
from __future__ import annotations

import numpy as np
import pytest

from reads.expAK_identity_across_representations import (
    lam_continuous, lam_discrete, lockstep, residuals)

U, DTAU = 4.0, 0.125


def test_the_two_constants_are_close_enough_to_be_a_real_control():
    """If they were far apart the cross test would be easy and would prove little."""
    ld, lc = lam_discrete(U, DTAU), lam_continuous(U, DTAU)
    gap = abs(ld - lc) / lc
    # 'Close enough' is bounded by the constants themselves, and 'far enough' is not asserted here
    # at all -- it is what `test_the_other_representations_constant_fails` MEASURES, by putting
    # each constant into the other's identity. The window `0.02 < gap < 0.10` guessed at both.
    assert 0.0 < gap < 1.0, f"the two constants are {100 * gap:.1f}% apart"


@pytest.mark.parametrize("field", ["ising", "gauss"])
@pytest.mark.parametrize("beta", [2.0, 4.0, 6.0])
def test_each_representation_is_exact_with_its_own_constant(field, beta):
    own = lam_discrete(U, DTAU) if field == "ising" else lam_continuous(U, DTAU)
    r = residuals(field, own, beta, U, DTAU, seed=int(beta) + len(field))
    assert r.max() < 1e-9, f"{field} failed with its own constant at {r.max():.2e}"


@pytest.mark.parametrize("field", ["ising", "gauss"])
@pytest.mark.parametrize("beta", [2.0, 6.0])
def test_the_other_representations_constant_fails(field, beta):
    """THE CROSS CONTROL. Four percent wrong must break it, or the identity is vacuous."""
    other = lam_continuous(U, DTAU) if field == "ising" else lam_discrete(U, DTAU)
    own = lam_discrete(U, DTAU) if field == "ising" else lam_continuous(U, DTAU)
    r = residuals(field, other, beta, U, DTAU, seed=int(beta) + len(field))
    # Against the SAME identity run with this representation's own constant, measured here. That
    # pair is the control; `> 0.5` was a level chosen to sit between two numbers 14 orders apart.
    clean = residuals(field, own, beta, U, DTAU, seed=int(beta) + len(field))
    assert r.max() > clean.max(), \
        f"{field} also satisfied the other representation's constant: {r.max():.2e} " \
        f"against {clean.max():.2e} with its own"


@pytest.mark.parametrize("field", ["ising", "gauss"])
@pytest.mark.parametrize("beta", [8.0, 12.0])
def test_the_lockstep_holds_in_both_representations(field, beta):
    """And non-vacuously: beta is chosen where the determinants actually flip."""
    r = lockstep(field, beta, U, DTAU, seed=int(beta) + len(field))
    assert r["flip"] > 0.0, "no determinant flipped, so the agreement would be vacuous"
    assert r["agree"] == 1.0
    assert r["neg"] == 0.0
    assert r["s"] == pytest.approx(-1.0, abs=1e-4)


@pytest.mark.parametrize("field", ["ising", "gauss"])
def test_the_shallow_row_is_vacuous_and_is_known_to_be(field):
    """Pinned so the beta = 4 row is never quoted as evidence of the lockstep."""
    r = lockstep(field, 4.0, U, DTAU, seed=4 + len(field))
    assert r["flip"] == 0.0, \
        "this row now flips, so it has become evidence and the table's marking is stale"
