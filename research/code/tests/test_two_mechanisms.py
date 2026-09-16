"""The reading is a synchronisation, and its SIGN is set by the mechanism.

This model is provably sign-free in two structurally opposite places, and the same read is put to
both:

  REPULSIVE U, spin channel, half filling.  `G_dn = 1 - G_up` exactly.  Anti-synchronised, and the
      coupling reads -1.  Doping destroys it.
  ATTRACTIVE U, charge channel, ANY filling.  The field couples to `n_up + n_dn - 1`, so both
      spins see the identical diagonal factor and the weight is `det^2`.  Synchronised, the
      coupling reads +1, and doping does NOT destroy it.

WHAT IS AND IS NOT LOAD-BEARING HERE.  On the attractive rows the two channels come out
bit-identical, so `coupling(A, A)` returning +1 carries the sign of the reading and nothing about
the instrument's discrimination -- `test_the_attractive_channels_are_bit_identical` pins that
degeneracy explicitly rather than letting a later reader mistake the +1 for evidence.  The
load-bearing test is `test_doping_separates_the_two_mechanisms`, which asserts the two rows
TOGETHER: the same doping that desaturates the repulsive read must leave the attractive one
saturated.  A suite asserting only one side would pass on a build that had stopped being sensitive
to the mechanism at all.

EVERY UNSATURATED ROW IS ASSERTED OVER SEEDS.  The doped repulsive read is genuinely seed-variable
-- `-0.3532` to `0.0000` at mu = 0.4 over six seeds, a spread the size of the value -- so a gate
pinned on one draw would be pinning a fluctuation.  The saturated rows do not move at all, and
that asymmetry is the result.
"""
from __future__ import annotations

import numpy as np
import pytest

import entroptics_adapter as EA
from model2d import Model2D
from stable import udt_product, inv_one_plus_block, slogdet_one_plus_block


SEEDS = (1, 7, 23, 45, 93, 101)


def run(mu, channel, beta, U=4.0, Lx=2, Ly=4, dtau=0.125, n_draw=300, seed=0):
    """`channel` picks which exact rewriting of the interaction is decoupled.

    Both are identities on the four states of a site, so this is not two approximations being
    compared: 'spin' gives the two spins opposite diagonal factors, 'charge' gives them the same
    one.  `lambda` is the decoupling's own constant in both, with nothing fitted.
    """
    L = int(round(beta / dtau))
    m = Model2D(Lx=Lx, Ly=Ly, t=1.0, mu=mu, U=abs(U), dtau=dtau, L=L, theta=0.0)
    lam = float(np.arccosh(np.exp(dtau * abs(U) / 2.0)))
    rng = np.random.default_rng(seed)
    A, B, S, sep = [], [], [], []
    for _ in range(n_draw):
        X = rng.choice([-1.0, 1.0], size=(m.L, m.N))
        row, s = {}, 1.0
        for sigma in (+1, -1):
            d = np.exp((sigma if channel == "spin" else +1) * lam * X)
            Bl = m.expmK[None, :, :] * d[:, None, :]
            Uu, D, T = udt_product(Bl[None], 4)
            row[sigma] = np.real(np.diag(inv_one_plus_block(Uu, D, T)[0])).copy()
            sg, _ = slogdet_one_plus_block(Uu, D, T)
            s *= float(np.real(sg[0]))
        A.append(row[+1]); B.append(row[-1]); S.append(s)
        sep.append(float(np.abs(row[+1] - row[-1]).max()))
    A, B = np.array(A), np.array(B)
    c = EA.channel_alignment(A, B)
    return dict(neg=float(np.mean(np.array(S) < 0)), strength=float(c.strength),
                z=float(c.z), resolved=bool(c.resolved), sep=float(np.max(sep)))


@pytest.mark.parametrize("mu", [0.0, 0.4, 0.8])
def test_the_attractive_charge_channel_is_sign_free_at_every_filling(mu):
    """The second mechanism, and the read's sign there is the OPPOSITE of the first's."""
    for seed in SEEDS:
        r = run(mu, "charge", beta=10.0, seed=seed)
        assert r["neg"] == 0.0, "the attractive charge decoupling produced a negative weight"
        assert r["resolved"]
        assert r["strength"] == pytest.approx(+1.0, abs=1e-4)


@pytest.mark.parametrize("mu", [0.0, 0.8])
def test_the_attractive_channels_are_bit_identical(mu):
    """THE DEGENERACY, pinned so the +1 above is never mistaken for evidence.

    The field couples to the charge, which is spin-independent, so `B_up = B_dn` as matrices and
    the two Green's functions agree to the last bit.  The read is being handed the same array
    twice.  Exact equality, not a tolerance: anything else would mean the two channels are being
    built by different code paths.
    """
    for seed in SEEDS:
        r = run(mu, "charge", beta=10.0, seed=seed)
        assert r["sep"] == 0.0, f"the two channels differed by {r['sep']:.3e}"


def test_doping_separates_the_two_mechanisms():
    """THE LOAD-BEARING TEST. The same doping, the two mechanisms, asserted together.

    Doping is what takes the repulsive read apart, and the attractive model stays sign-free under
    it. If the read were tracking filling, or distance from a particle-hole symmetric point, both
    sides would move. Only one does.
    """
    att_departure = {}
    for mu in (0.0, 0.4, 0.8):
        att = run(mu, "charge", beta=10.0, seed=int(mu * 100) + 11)
        att_departure[mu] = 1.0 - abs(att["strength"])
        assert att["strength"] == pytest.approx(+1.0, abs=1e-4), \
            f"doping desaturated the attractive read at mu = {mu}"

    half = run(0.0, "spin", beta=10.0, seed=13)
    assert half["strength"] == pytest.approx(-1.0, abs=1e-4)
    rep_at_half = 1.0 - abs(half["strength"])

    # No chosen tolerance. Departure from saturation is compared against the two rows this test
    # has already measured: the SAME model undoped, and the attractive mechanism under the SAME
    # doping. Both references are the instrument reading a saturated row, so they are its floor.
    # A number like `< 0.9` would have been a guess at where that floor sits.
    for mu in (0.4, 0.8):
        rep = run(mu, "spin", beta=10.0, seed=int(mu * 100) + 13)
        departure = 1.0 - abs(rep["strength"])
        assert departure > rep_at_half, \
            f"doping moved the repulsive read no further from saturation than half filling did " \
            f"at mu = {mu}: {departure:.4e} against {rep_at_half:.4e}"
        assert departure > att_departure[mu], \
            f"the same doping moved both mechanisms at mu = {mu}: repulsive {departure:.4e}, " \
            f"attractive {att_departure[mu]:.4e}"


def test_saturation_and_a_zero_negative_fraction_are_not_equivalent():
    """A row with no negative weight in 300 draws that is nonetheless not saturated.

    This is section 7's result -- the read departs before the average sign does -- and it is
    asserted here so a later draft cannot quietly upgrade saturation into an iff with sign-freedom.
    """
    rs = [run(0.8, "spin", beta=6.0, seed=s) for s in SEEDS]
    # No threshold: the two quantities are compared to EACH OTHER, seed by seed.
    for seed, r in zip(SEEDS, rs):
        assert r["resolved"]
        departure = abs(abs(r["strength"]) - 1.0)
        assert departure > r["neg"],             f"seed {seed}: departure {departure:.4f} did not exceed neg {r['neg']:.4f}"
    assert any(r["neg"] == 0.0 for r in rs),         "no seed had a zero negative fraction, so the point is not being made here"
    # Against a row that IS saturated -- the half-filled spin read, measured here. `> 0.05` was a
    # distance from saturation chosen rather than found.
    saturated = abs(abs(run(0.0, "spin", beta=6.0, seed=SEEDS[0])["strength"]) - 1.0)
    assert all(abs(abs(r["strength"]) - 1.0) > saturated for r in rs), \
        f"a seed saturated, so this row is not the unsaturated-but-sign-free case " \
        f"(saturated reference departs by {saturated:.2e})"
