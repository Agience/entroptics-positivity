"""`n_stab` is a hand-picked constant that every sign measurement rests on.  Is it safe?

`n_stab` -- how many slice matrices are multiplied before the product is re-orthogonalised -- is
chosen by hand, and this repository chooses FOUR different values for it: 8 in `fastdqmc`, 1 in
`fastdqmc2d`, 5 in `sampler_stable`, 8 in `stable.udt_product`'s default.  `StableChains` is the
rig behind the sign measurements and it takes 5.

The constant costs in both directions.  Too small and every extra QR is wasted work.  Too large
and the accumulated product's singular values spread past what double precision holds, the small
directions are lost into rounding, and the determinant -- so the SIGN -- comes back wrong with no
exception raised and no warning printed.

WHAT IS ALREADY COVERED, so that what this adds is clear.  `validate.py` gates `StableChains`
against brute-force enumeration of every auxiliary field at 5e-15 -- the stronger check, where it
applies, but enumeration is affordable only at small `L`, which is where stabilisation is least
stressed.  `test_conditioned_frame.test_the_answer_does_not_move_with_the_stabilisation_block`
already compares the SIGN configuration-wise across blocks 2, 4 and 8 at `beta = 10`, deeper than
anything here.  Between them the sign is well covered and this file does not claim otherwise.

WHAT THIS ADDS, three things.  A reference of `n_stab = 1`, re-orthogonalising at every slice,
rather than the smallest block in the comparison set.  The log WEIGHT as well as the sign, since a
magnitude wrong in the third decimal corrupts the importance weights while every sign still agrees.
And a NEGATIVE CONTROL: the existing check asserts that three block sizes agree without ever
showing that the comparison could detect them disagreeing, so a blind comparison would pass it
unchanged.  `test_the_check_can_detect_a_bad_constant` removes the stabilisation entirely and
requires the discrepancy to appear.

MEASURED in the exploratory sweep (24 fields per beta; the tests below use 16,
N = 8, U = 4, t2 = 0.7, mu = 1.0):

    beta   L   cond(D)     n=5             n=20        n=40
    4.0    32  2.9e+12     1.1e-12 / 0     6.7e-10     4.0e-06
    5.0    40  3.6e+15     6.8e-13 / 0     1.0e-09     2.6e-03
    6.0    48  2.4e+19     2.8e-12 / 0     4.5e-09     1.2e-03

so 5 is safe with room to spare, and the constant is genuinely two-sided: by 40 the log weight is
wrong in the third decimal, and the largest safe value falls as beta rises.  `cond(D)` at
beta = 6 is 2.4e+19, past 1/eps ~ 4.5e+15 -- the UDT factorisation carries that because separating
the scales is what it is for, which is the thing being confirmed.
"""
from __future__ import annotations

import pathlib as _pathlib
import sys as _sys

_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from dqmc import Model
from sampler_stable import StableChains
from stable import udt_product, slogdet_one_plus

N, U, DTAU, T2, MU = 8, 4.0, 0.125, 0.7, 1.0
R = 16
PRODUCTION_NSTAB = 5           # what StableChains uses


def _weights(m, S, n_stab):
    ch = StableChains(m, R=S.shape[0], seed=0, n_stab=n_stab)
    sign = np.ones(S.shape[0])
    logabs = np.zeros(S.shape[0])
    for sigma in (+1, -1):
        Uf, D, T = udt_product(ch._Bl(S, sigma), n_stab)
        sg, ld = slogdet_one_plus(Uf, D, T)
        sign *= sg
        logabs += ld
    return sign, logabs


def _fields(beta):
    L = int(round(beta / DTAU))
    m = Model(N=N, t=1.0, t2=T2, mu=MU, U=U, dtau=DTAU, L=L)
    g = np.random.default_rng(7)
    return m, g.choice([-1.0, 1.0], size=(R, L, N))


@pytest.mark.parametrize("beta", [4.0, 5.0])
def test_production_nstab_matches_every_slice_stabilisation(beta):
    """At the deepest beta the paper uses, n_stab = 5 must agree with n_stab = 1."""
    m, S = _fields(beta)
    s1, l1 = _weights(m, S, 1)
    s5, l5 = _weights(m, S, PRODUCTION_NSTAB)
    assert (s5 == s1).all(), f"sign flips at beta={beta}: {(s5 != s1).sum()} of {R}"
    assert np.max(np.abs(l5 - l1)) < 1e-8, f"log weight moved by {np.max(np.abs(l5 - l1)):.2e}"


def test_the_check_can_detect_a_bad_constant():
    """The negative control, without which the test above proves nothing.

    A test that only ever passes cannot distinguish a safe constant from a blind check.  Push
    `n_stab` to the whole imaginary-time extent -- one QR for the entire product, i.e. no
    stabilisation at all -- and the discrepancy must become large.  If this does NOT fail, the
    comparison is not sensitive to stabilisation and the passing tests above are worthless.
    """
    beta = 5.0
    m, S = _fields(beta)
    _s1, l1 = _weights(m, S, 1)
    L = int(round(beta / DTAU))
    _sbad, lbad = _weights(m, S, L)
    err = float(np.max(np.abs(lbad - l1)))
    assert err > 1e-6, (
        f"no stabilisation at all moved the log weight by only {err:.2e}; the check is not "
        f"sensitive to n_stab, so the passing cases prove nothing")


def test_the_safe_ceiling_falls_as_beta_rises():
    """The constant is two-sided AND its safe value moves, which is why it cannot be a constant.

    A value that were safe everywhere would make the choice uninteresting.  The error at a fixed
    large `n_stab` must grow with beta -- more slices in one unstabilised block, more dynamic
    range lost.
    """
    errs = {}
    for beta in (3.0, 5.0):
        m, S = _fields(beta)
        _s1, l1 = _weights(m, S, 1)
        _sk, lk = _weights(m, S, 40)
        errs[beta] = float(np.max(np.abs(lk - l1)))
    assert errs[5.0] > errs[3.0], f"error did not grow with beta: {errs}"
