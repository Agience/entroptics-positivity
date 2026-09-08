"""§5's balance read, gated: the separation belongs to the system's own zero.

Three assertions, and the second is the one that makes the first mean anything:

  * with the particle-hole zero, the median pvalue separates sign-free from sign-problem at FIXED
    FILLING -- every row at `<n> = 1.00000`, so doping is not the variable;
  * with the library's DEFAULT zero the read is blind, median 1.00000 on every row. Without this
    the separation could belong to the read rather than to the law it was given;
  * the phase rows are asserted to be MISSED, so a later draft cannot claim this sees a phase
    problem. It does not, and §5 says why.

The pvalue is the quantity, not the boolean `closed`, which is a per-run decision and fluctuates.
"""
from __future__ import annotations

import numpy as np
import pytest

from reads.expAO_spectral_criterion import build, ring, tri_ladder
from reads.expAT_balance_at_the_systems_own_zero import balance_pvalue, channels

SEEDS = (5, 11, 23, 41, 67, 83)

SIGN_FREE = (build(2, 4, 0, 0, 0), ring(6, np.pi / 4))
SIGN_PROBLEM = (build(2, 4, 0, 0, 0.2), build(2, 4, 0, 0, 0.6))


_SEEN = {}


def per_seed_p(K, own_zero):
    """Memoised on the matrix's own bytes: the same rows are asked for by several tests here, and
    the comparisons below need the per-seed values, not only their median."""
    A0 = np.asarray(K, dtype=complex)
    key = (A0.shape, A0.tobytes(), bool(own_zero))
    if key not in _SEEN:
        ps = []
        for sd in SEEDS:
            A, B, _ = channels(A0, sd)
            ps.append(balance_pvalue(A, B, own_zero)[0])
        _SEEN[key] = ps
    return _SEEN[key]


def median_p(K, own_zero):
    return float(np.median(per_seed_p(K, own_zero)))


def test_the_two_populations_do_not_overlap_at_the_systems_own_zero():
    """SEPARATION, not a cut. The sign-free rows and the sign-problem rows are compared to EACH
    OTHER: the worst sign-free row must score above the best sign-problem row, with a gap.

    Every row sits at <n> = 1.00000, so filling is not the variable. No level is chosen here --
    the claim is that the two populations are ordered and separated, which is what a detector has
    to do before any level is applied to it.
    """
    free = [median_p(K, True) for K in SIGN_FREE]
    prob = [median_p(K, True) for K in SIGN_PROBLEM]
    assert min(free) > max(prob), \
        f"the populations overlap: sign-free {free}, sign-problem {prob}"
    # The gap is required to exceed the spread WITHIN either population -- both measured on these
    # same rows. A factor like `5x` would have been a margin chosen rather than found.
    within = max(max(free) - min(free), max(prob) - min(prob))
    assert min(free) - max(prob) > within, \
        f"the gap between the populations ({min(free) - max(prob):.5f}) is no larger than the " \
        f"spread inside one of them ({within:.5f}): sign-free {free}, sign-problem {prob}"


def test_the_default_zero_is_blind():
    """THE CONTROL. Withhold the system's law and the read separates nothing.

    Blindness is stated as a COMPARISON between the two zeros on the same two lattices, which is
    what 'blind' means. `> 0.5` was a level chosen for one side of that comparison.
    """
    clean, staggered = SIGN_FREE[0], SIGN_PROBLEM[1]
    own = median_p(clean, True) - median_p(staggered, True)
    default = abs(median_p(clean, False) - median_p(staggered, False))
    assert own > default, \
        f"the default zero separates these rows as well as the system's own does " \
        f"({default:.5f} against {own:.5f}); the control has stopped controlling"


@pytest.mark.parametrize("name,K", [
    ("ring 5 flux pi/2", ring(5, np.pi / 2)),
    ("tri ladder pi/2", tri_ladder(8, np.pi / 2)),
])
def test_a_phase_problem_is_missed_and_is_known_to_be(name, K):
    """Pinned so this is never read as seeing a phase problem. It does not.

    'Missed' is stated against the sign-problem population measured above: a row this read caught
    would score with them. `> 0.10` was a level; this is a comparison.
    """
    caught = max(median_p(P, True) for P in SIGN_PROBLEM)
    assert median_p(K, True) > caught, \
        f"{name}: this row now flags -- it scores {median_p(K, True):.5f} against the caught " \
        f"population's {caught:.5f}; re-examine §5's scope paragraph"


def test_the_read_is_not_a_per_column_deviation_test():
    """`balance` is correlation-aware; the obvious alternative is not, and they disagree.

    A per-column z on the mean against 1/2 orders two rows one way and `balance` orders them the
    other: similar z (2.85 against 2.30) with pvalues a factor of thirty apart. The channels are
    strongly correlated, so the effective degrees of freedom are far below the column count.

    Pinned so the read is never described as a t-test on the channel mean.
    """
    import numpy as np
    from reads.expAO_spectral_criterion import build, tri_ladder

    def naive_z(A):
        m = A.mean(axis=0) - 0.5
        se = A.std(axis=0, ddof=1) / np.sqrt(A.shape[0])
        return float(np.max(np.abs(m) / np.maximum(np.abs(se), 1e-300)))

    zs, ps, z_spread = {}, {}, {}
    for tag, K in (("staggered", build(2, 4, 0, 0, 0.2)),
                   ("ladder", tri_ladder(8, np.pi / 2))):
        z_ = [naive_z(channels(np.asarray(K, dtype=complex), sd)[0]) for sd in SEEDS]
        ps[tag] = per_seed_p(K, True)
        zs[tag] = float(np.median(z_))
        z_spread[tag] = float(max(z_) - min(z_))

    med = {k: float(np.median(v)) for k, v in ps.items()}

    # 'The naive z agrees' is asserted against the naive z's OWN scatter across these seeds.
    assert abs(zs["staggered"] - zs["ladder"]) < max(z_spread.values()), \
        f"the naive z no longer agrees on these rows: {zs}, seed spread {z_spread}"

    # THE CONTENT, AND IT NEEDS NO NUMBER: the two tests order the same two rows in OPPOSITE
    # directions. The naive z puts the staggered row higher; `balance` puts the ladder higher.
    assert zs["staggered"] > zs["ladder"], f"the naive z no longer orders these rows: {zs}"
    assert med["ladder"] > med["staggered"], \
        f"balance no longer orders these rows the other way: {med}"

    # THE LIMITATION, PINNED. The separation is a MEDIAN effect and it does NOT survive seed by
    # seed: measured 2026-09-07, the ladder's worst seed (0.00172) falls below the staggered row's
    # best (0.07191), and the pairing holds on 5 of these 6 seeds. An earlier version of this test
    # asserted `ps["ladder"] > 10 * ps["staggered"]` on the medians, which passed and read as a
    # clean split; per-seed non-overlap is what a clean split would mean, and it is false here.
    assert min(ps["ladder"]) < max(ps["staggered"]), \
        f"the populations no longer overlap seed by seed -- if that is now real, this limitation " \
        f"and the paper's paragraph on it should be re-measured: {ps}"
