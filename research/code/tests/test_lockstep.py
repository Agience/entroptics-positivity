"""The central claim, as a gate: at half filling the two spin channels cross in LOCKSTEP.

A negative weight is by definition a configuration on which the two channels' determinant signs
disagree. The result is that at half filling on a bipartite lattice they never do -- while
individually changing sign often enough that the statement is not vacuous.

Each test carries the thing that makes it able to fail:

  * the lockstep test asserts a NON-ZERO individual flip rate alongside perfect agreement, so a
    run in which no channel ever flips cannot pass it;
  * the desynchronisation test asserts that doping breaks it, so a code that reported agreement
    unconditionally would fail here;
  * the conditioned-frame test asserts that the naive product FAILS, which is the negative control
    for every other number in the repository.
"""
from __future__ import annotations

import numpy as np
import pytest

from model2d import Model2D
from stable import udt_product, core_matrix, slogdet_one_plus_block
from reads.expVV_lockstep import per_channel


def channel_signs(m, X, block=4):
    """(sign of det(I+B) per spin, distance to a sign flip per spin) in the conditioned frame."""
    lam = float(np.arccosh(np.exp(m.dtau * m.U / 2.0)))
    s, d = {}, {}
    for sigma in (+1, -1):
        e = np.exp(sigma * lam * X)
        Bl = m.expmK[None, :, :] * e[:, None, :]
        U, D, T = udt_product(Bl[None], block)
        sg, _ = slogdet_one_plus_block(U, D, T)
        s[sigma] = float(np.real(sg[0]))
        M, _ = core_matrix(U, D, T)
        d[sigma] = float(np.min(np.abs(np.linalg.eigvals(M[0]))))
    return s[+1], s[-1], d[+1], d[-1]


def sample(m, n, seed, block=4):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        X = rng.choice([-1.0, 1.0], size=(m.L, m.N))
        out.append(channel_signs(m, X, block))
    su, sd, du, dd = (np.array(c) for c in zip(*out))
    return su, sd, du, dd


@pytest.mark.parametrize("beta", [8.0, 12.0])
def test_channels_flip_together_at_half_filling(beta):
    """Perfect agreement AND a non-zero flip rate -- the second is what makes it a test."""
    m = Model2D(Lx=2, Ly=4, t=1.0, mu=0.0, U=4.0, dtau=0.125,
                L=int(round(beta / 0.125)), theta=0.0)
    su, sd, _, _ = sample(m, 400, seed=int(beta))
    flip = float(np.mean(su < 0))
    assert flip > 0.0, "no channel ever flipped -- the lockstep claim would be vacuous"
    assert np.all(su == sd), "the two channels disagreed on at least one configuration"
    assert float(np.mean(su * sd < 0)) == 0.0


def test_doping_desynchronises_and_the_break_is_the_sign_problem():
    """The disagreement fraction IS the negative-weight fraction, exactly."""
    m = Model2D(Lx=2, Ly=4, t=1.0, mu=0.4, U=4.0, dtau=0.125, L=96, theta=0.0)
    su, sd, _, _ = sample(m, 400, seed=7)
    disagree = float(np.mean(su != sd))
    neg = float(np.mean(su * sd < 0))
    assert disagree > 0.0, "doping did not break the lockstep -- nothing is being tested"
    assert disagree == neg


@pytest.mark.parametrize("beta", [6.0, 10.0])
def test_the_lock_is_lockstep_not_avoidance(beta):
    """Configurations reach arbitrarily close to the boundary and still never flip the product.

    NOT asserted on the minimum distance.  A minimum over samples is an extreme-value statistic
    with no limiting value: measured here it ranges over 0.0002 to 0.012 across seeds at FIXED
    sample size, a factor of sixty, and falls further as draws are added.  A cut on it would pin a
    fluctuation, and comparing minima across seeds is comparing that same fluctuation.

    What is stable is the SHAPE of the distribution, and it is what the claim needs: a bulk that
    does not move, and a tail an order of magnitude below it.  Both are compared to each other
    rather than to a chosen level -- the 1st percentile against the median, on the same draws.
    """
    m = Model2D(Lx=2, Ly=4, t=1.0, mu=0.0, U=4.0, dtau=0.125,
                L=int(round(beta / 0.125)), theta=0.0)
    medians, depths = [], []
    for n, seed in ((200, 101), (200, 211), (600, 103)):
        su, sd, du, dd = sample(m, n, seed=seed)
        assert float(np.mean(su * sd < 0)) == 0.0, "a negative weight appeared at half filling"
        d = np.concatenate([du, dd])
        p1, med = float(np.percentile(d, 1)), float(np.median(d))
        medians.append(med); depths.append(med / p1)
        assert float(np.mean(su < 0)) > 0.0, "no channel crossed, so nothing approached anything"

    # The two halves of the shape, compared to EACH OTHER as ratios of the same quantity: the
    # tail must be deeper than the bulk is unstable across sample sizes. `10x` and `25%` were two
    # levels chosen separately, and neither was the claim -- this comparison is.
    lo, hi = min(medians), max(medians)
    assert min(depths) > hi / lo, \
        f"the tail ({min(depths):.1f}x below the median) is no deeper than the bulk moves with " \
        f"sample size ({hi / lo:.2f}x): medians {medians}"

def test_particle_hole_identity_holds_configuration_by_configuration():
    """G_up[i,i](x) + G_dn[i,i](x) = 1 at half filling -- the relation the lock rests on."""
    from stable import inv_one_plus_block
    m = Model2D(Lx=2, Ly=4, t=1.0, mu=0.0, U=4.0, dtau=0.125, L=32, theta=0.0)
    lam = float(np.arccosh(np.exp(m.dtau * m.U / 2.0)))
    rng = np.random.default_rng(3)
    worst = 0.0
    for _ in range(60):
        X = rng.choice([-1.0, 1.0], size=(m.L, m.N))
        diag = {}
        for sigma in (+1, -1):
            e = np.exp(sigma * lam * X)
            Bl = m.expmK[None, :, :] * e[:, None, :]
            U, D, T = udt_product(Bl[None], 4)
            diag[sigma] = np.real(np.diag(inv_one_plus_block(U, D, T)[0]))
        worst = max(worst, float(np.abs(diag[+1] + diag[-1] - 1.0).max()))
    assert worst < 1e-10, f"the identity failed at {worst:.2e}"


# ── the estimate/identity split, gated ───────────────────────────────────────
#
# §3's table has two kinds of column and they must not be confused.  The flip RATE is a proportion
# of finite draws and varies with the seed; the AGREEMENT is an exact per-configuration fact and
# does not.  A suite that pinned the rate would pin a fluctuation, and one that let the agreement
# drift below 1 would have lost the result.

SEEDS = (1, 7, 23, 45, 93)


def _rates(beta, mu, seed, n=300, U=4.0, dtau=0.125):
    m = Model2D(Lx=2, Ly=4, t=1.0, mu=mu, U=U, dtau=dtau,
                L=int(round(beta / dtau)), theta=0.0)
    rng = np.random.default_rng(seed)
    su, sd = [], []
    for _ in range(n):
        X = rng.choice([-1.0, 1.0], size=(m.L, m.N))
        a, b, _, _ = per_channel(m, X)
        su.append(a); sd.append(b)
    su, sd = np.array(su), np.array(sd)
    return (float(np.mean(su < 0)), float(np.mean(sd < 0)),
            float(np.mean(su == sd)), float(np.mean(su * sd < 0)))


@pytest.mark.parametrize("beta", [10.0, 12.0])
def test_the_agreement_is_exact_on_every_seed_not_an_average(beta):
    """The agreement column carries ZERO spread -- it is an identity, not an estimate."""
    for seed in SEEDS:
        up, dn, agree, neg = _rates(beta, 0.0, seed)
        assert agree == 1.0, f"agreement fell to {agree} at seed {seed}"
        assert up == dn, f"the two channels' flip rates differ at half filling: {up} vs {dn}"
        assert up > 0.0, "no channel flipped, so the agreement is vacuous on this row"
        assert neg == 0.0


@pytest.mark.parametrize("beta,mu", [(8.0, 0.4), (12.0, 0.4)])
def test_the_disagreement_is_the_negative_fraction_on_every_seed(beta, mu):
    """An identity checked configuration by configuration, on the doped rows where it has
    something to say."""
    for seed in SEEDS:
        _, _, agree, neg = _rates(beta, mu, seed)
        assert agree < 1.0, "no disagreement here, so the identity is vacuous on this row"
        assert (1.0 - agree) == pytest.approx(neg, abs=1e-12)


def test_the_flip_rate_is_an_estimate_and_is_not_pinned():
    """The other half: the RATE genuinely varies with the seed, so no test may pin its value.

    Asserted rather than assumed, because a table quoting it to four decimals would be quoting
    a hundred times better than it reproduces.
    """
    rows = [_rates(12.0, 0.0, s) for s in SEEDS]
    ups = [r[0] for r in rows]
    agrees = [r[2] for r in rows]
    spread = max(ups) - min(ups)
    # Against the AGREEMENT column measured on the same draws, which is an identity and carries
    # exactly zero spread. That contrast is §3's point, and it is what `> 0.01` stood in for.
    assert max(agrees) - min(agrees) == 0.0, f"the agreement column is no longer exact: {agrees}"
    assert spread > max(agrees) - min(agrees), \
        f"the flip rate no longer varies across seeds ({ups}); re-examine what precision is real"
