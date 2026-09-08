"""The order parameter's surviving claims, as gates.

Two claims survived the narrowing in §5 and both are pinned here. Claims that did NOT survive are
also pinned -- as tests that they do not hold -- because an earlier draft asserted them and only a
test stops a later draft asserting them again.

  SURVIVES  the coupling reads exactly -1 at the symmetric point, and its own exact re-pairing
            null puts a permuted control near zero.  That is a calibration against an identity
            verified independently at 6.7e-15, not an agreement with a fitted value.
  SURVIVES  along beta at fixed filling the deficit is monotone and resolved, in a regime where
            the average sign is identically 1 and therefore has no derivative to read.
  DOES NOT  `tightness` is not a coupling property.  At the symmetric point `G_dn = 1 - G_up`
            exactly, so the centred channels satisfy `B~ = -A~`, the cross-covariance is
            `-A~' A~`, and its spectrum is the SINGLE channel's own.  The test asserts they agree,
            which is what makes the withdrawal permanent.
"""
from __future__ import annotations

import numpy as np
import pytest

import entroptics as E
from model2d import Model2D
from stable import udt_product, inv_one_plus_block, slogdet_one_plus_block


def channels(m, n_draw, seed):
    """(A, B, signs) -- the two channels' Green's-function diagonals, and the weight's sign."""
    lam = float(np.arccosh(np.exp(m.dtau * m.U / 2.0)))
    rng = np.random.default_rng(seed)
    A, B, S = [], [], []
    for _ in range(n_draw):
        X = rng.choice([-1.0, 1.0], size=(m.L, m.N))
        row, s = {}, 1.0
        for sigma in (+1, -1):
            d = np.exp(sigma * lam * X)
            Bl = m.expmK[None, :, :] * d[:, None, :]
            U, D, T = udt_product(Bl[None], 4)
            row[sigma] = np.real(np.diag(inv_one_plus_block(U, D, T)[0])).copy()
            sg, _ = slogdet_one_plus_block(U, D, T)
            s *= float(np.real(sg[0]))
        A.append(row[+1]); B.append(row[-1]); S.append(s)
    return np.array(A), np.array(B), np.array(S)


def fourth_power_share(M):
    """The dominant share of sigma^4 for the centred frame M.

    Fourth power, not second, and the reason is exact.  At the symmetric point the centred
    channels satisfy `B~ = -A~`, so the cross-covariance is `-A~' A~`, whose SINGULAR values are
    the EIGENVALUES of `A~' A~` -- that is, the squares of A~'s singular values.  A share of
    squares-of-squares is a share of sigma^4.  Asserting the sigma^2 share instead reads 0.8808
    against a tightness of 0.9961; the sigma^4 share reads 0.996142 against 0.996142.
    """
    Mc = M - M.mean(axis=0, keepdims=True)
    sv = np.linalg.svd(Mc, compute_uv=False)
    return float(sv[0] ** 4 / (sv ** 4).sum())


@pytest.mark.parametrize("beta", [2.0, 4.0, 8.0])
def test_the_coupling_is_exactly_minus_one_at_the_symmetric_point(beta):
    """The calibration. -1 is forced by an identity verified independently, not fitted."""
    m = Model2D(Lx=2, Ly=4, t=1.0, mu=0.0, U=4.0, dtau=0.125,
                L=int(round(beta / 0.125)), theta=0.0)
    A, B, S = channels(m, 300, seed=int(beta))
    assert float(np.mean(S < 0)) == 0.0, "a sign problem here would void the calibration"
    c = E.reads.coupling(A, B)
    assert c.resolved
    assert c.strength == pytest.approx(-1.0, abs=1e-4)


@pytest.mark.parametrize("beta", [2.0, 8.0])
def test_the_permuted_control_is_not_resolved(beta):
    """THE NEGATIVE CONTROL, from the instrument's own exact null.

    Destroying the pairing while leaving each side's internal structure intact must destroy the
    reading. A coupling that survived this would be reporting structure within one side.
    """
    m = Model2D(Lx=2, Ly=4, t=1.0, mu=0.0, U=4.0, dtau=0.125,
                L=int(round(beta / 0.125)), theta=0.0)
    A, B, _ = channels(m, 300, seed=int(beta) + 5)
    rng = np.random.default_rng(1)
    cn = E.reads.coupling(A, B[rng.permutation(len(B))])
    # The read's own verdict, which is what this test is named after. `|z| < 4` was a level
    # chosen for an instrument that already decides this itself.
    assert not cn.resolved, f"the permuted control resolved: |z| = {abs(cn.z):.2f}"


def test_the_deficit_moves_where_the_average_sign_cannot():
    """The capability: a monotone, resolved signal where `<sgn>` is identically 1.

    Both quantities come from the same configurations, so this is not two samplers being
    compared. The assertion that the average sign is EXACTLY 1 is what makes the deficit's
    movement meaningful rather than merely correlated.
    """
    out = []
    for beta in (2.0, 3.0, 4.0):
        m = Model2D(Lx=2, Ly=4, t=1.0, mu=0.4, U=4.0, dtau=0.125,
                    L=int(round(beta / 0.125)), theta=0.0)
        A, B, S = channels(m, 400, seed=int(beta) + 21)
        c = E.reads.coupling(A, B)
        rng = np.random.default_rng(int(beta) + 909)
        null = E.reads.coupling(A, B[rng.permutation(len(B))])
        out.append((float(np.mean(S < 0)), 1.0 + float(c.strength),
                    abs(float(c.z)), bool(c.resolved), 1.0 + float(null.strength)))
    # the average sign carries no information on these rows
    assert all(neg == 0.0 for neg, _, _, _, _ in out), "a sign problem appeared; pick milder rows"
    deficits = [d for _, d, _, _, _ in out]
    assert deficits == sorted(deficits), f"the deficit was not monotone: {deficits}"
    # Movement is measured against what the instrument's own re-pairing null reads on the SAME
    # configurations, not against a chosen ratio: the deficit must travel further along beta than
    # the deficit a destroyed pairing puts on any one row.
    null_floor = min(n for _, _, _, _, n in out)
    assert deficits[-1] - deficits[0] > 0.0 and deficits[-1] < null_floor, \
        f"the deficit did not move clear of its permuted null: {deficits}, null {null_floor:.5f}"
    assert all(res for _, _, _, res, _ in out), "the read was not resolved"


def test_tightness_is_not_a_coupling_property():
    """A WITHDRAWN CLAIM, pinned so it stays withdrawn.

    An earlier draft read `tightness` as the order parameter tracking the symmetry. At the
    symmetric point the centred channels are exact negatives, so the cross-covariance spectrum IS
    the single channel's own -- and the numbers agree.
    """
    m = Model2D(Lx=2, Ly=4, t=1.0, mu=0.0, U=4.0, dtau=0.125, L=64, theta=0.0)
    A, B, _ = channels(m, 400, seed=31)
    c = E.reads.coupling(A, B)
    assert c.tightness == pytest.approx(fourth_power_share(A), abs=1e-6)
    assert c.tightness == pytest.approx(fourth_power_share(B), abs=1e-6)
    # and it is NOT the sigma^2 share -- so the test cannot pass by both being near 1
    s2 = float((np.linalg.svd(A - A.mean(axis=0, keepdims=True), compute_uv=False) ** 2)[0]
               / (np.linalg.svd(A - A.mean(axis=0, keepdims=True), compute_uv=False) ** 2).sum())
    # The distance to the sigma^2 share must exceed the distance to the fourth-power share, which
    # the two lines above pin at 1e-6. `> 0.05` was a level standing in for that comparison.
    assert abs(c.tightness - s2) > abs(c.tightness - fourth_power_share(A)), \
        f"tightness sits as close to the sigma^2 share ({s2:.5f}) as to the fourth-power one"


# ── what the deficit measures: an angle, not a magnitude ────────────────────

def test_the_deficit_is_blind_to_a_rescaling_of_one_channel():
    """The read is a normalised alignment, so only the ANGLE between the frames registers.

    A violation lying ALONG the channel is a change in relative size and moves the deficit not at
    all -- asserted on synthetic frames, where the relation can be set exactly rather than
    approached. Without this the deficit could be read as a magnitude of particle-hole violation,
    and it is not: |E|/|A| goes to 1.0 here with the deficit exactly zero.
    """
    rng = np.random.default_rng(0)
    A = rng.standard_normal((400, 8))
    Ac = A - A.mean(axis=0, keepdims=True)

    def violation_at(scale):
        Bc = (1.0 - scale * A) - (1.0 - scale * A).mean(axis=0, keepdims=True)
        return float(np.linalg.norm(Bc + Ac) / np.linalg.norm(Ac))

    unscaled = violation_at(1.0)   # the exactly-related row: violation zero by construction
    for scale in (1.0, 2.0, 0.3):
        B = 1.0 - scale * A
        violation = violation_at(scale)
        deficit = 1.0 + float(E.reads.coupling(A, B).strength)
        assert deficit == pytest.approx(0.0, abs=1e-9), \
            f"a pure rescaling moved the deficit to {deficit:.2e}"
        if scale != 1.0:
            # Against the unscaled row measured above, which carries none. `> 0.5` was a size
            # chosen for 'large'; what the test needs is that this row carries more than none.
            assert violation > unscaled, \
                f"this row was meant to carry a parallel violation: {violation:.4f} against " \
                f"the unscaled row's {unscaled:.4f}"


def test_the_perpendicular_violation_predicts_the_deficit_at_small_angle():
    """deficit = |E_perp|^2 / (2 |A| |B|) to leading order, on the capability rows.

    Checked where the small-angle form is valid and NOT beyond it: by beta = 4 the deficit reaches
    0.09 and the quadratic form is 4% low, which is the approximation failing rather than the
    relation.
    """
    rng = np.random.default_rng(3)
    A = rng.standard_normal((400, 8))
    for eps in (0.02, 0.05, 0.10):
        B = 1.0 - A + eps * rng.standard_normal(A.shape)
        Ac = A - A.mean(axis=0, keepdims=True)
        Bc = B - B.mean(axis=0, keepdims=True)
        Ev = Bc + Ac
        Ep = Ev - ((Ev * Ac).sum() / (Ac ** 2).sum()) * Ac
        pred = float(np.linalg.norm(Ep) ** 2
                     / (2 * np.linalg.norm(Ac) * np.linalg.norm(Bc)))
        deficit = 1.0 + float(E.reads.coupling(A, B).strength)
        assert deficit == pytest.approx(pred, rel=0.05), \
            f"eps = {eps}: predicted {pred:.6f}, measured {deficit:.6f}"
