"""§5's CAPABILITY claim, gated in both representations.

The claim that would be USED is not that the coupling reads -1 at the symmetric point; it is that
along beta at fixed filling the deficit from saturation moves, monotonically and resolved, where
the average sign is identically 1 and has no derivative to read.  It had only ever been measured
with the discrete Ising field.

These tests assert the SHAPE in both the Ising and the continuous Gaussian spin decoupling.  They
deliberately do NOT assert numerical agreement between the two: the field distributions are
different measures, and a test demanding agreement would be asserting something false about two
different ensembles even though they happen to agree to about 10% here.

The last test pins the LIMITATION the cross-check exposed: the deficit stops being reproducible
across seeds at the onset of the sign problem, which is the boundary of the regime the claim is
made in.
"""
from __future__ import annotations

import numpy as np
import pytest

from functools import lru_cache

from reads.expAL_capability_across_representations import read_axis as _read_axis


@lru_cache(maxsize=None)
def read_axis(field, beta, mu, seed):
    """Memoised: four tests ask for the same (field, beta, seed) rows.

    The chains are identical -- same seeds, same parameters -- so regenerating them per test
    measured nothing extra and cost the suite real time.
    """
    return _read_axis(field, beta, mu, seed=seed)

SEEDS = (1, 7, 23, 45)
MU = 0.4
CAPABILITY_BETAS = (1.0, 1.5, 2.0, 3.0)


@pytest.mark.parametrize("field", ["ising", "gauss"])
def test_the_deficit_is_monotone_and_resolved_where_the_sign_is_identically_one(field):
    """Resolution is the instrument's own verdict; how strongly it resolves is set by the
    instrument's own null. `z > 20` was a level chosen by hand. The permuted re-pairing is a
    MEASUREMENT, and every row is asked to beat the loudest reading that null produced anywhere
    on this axis -- and the movement along beta is asked to beat the seed scatter it is made of.
    """
    means, scatters = [], []
    for beta in CAPABILITY_BETAS:
        rs = [read_axis(field, beta, MU, s) for s in SEEDS]
        assert all(r["neg"] == 0.0 for r in rs), \
            f"a negative weight appeared at beta = {beta}; this row is not the capability regime"
        assert all(r["res"] for r in rs), f"not resolved at beta = {beta}"
        loudest_null = max(r["nullz"] for r in rs)
        assert all(r["z"] > loudest_null for r in rs), \
            f"{field}: a row at beta = {beta} did not beat its own permuted null " \
            f"({min(r['z'] for r in rs):.2f} against {loudest_null:.2f})"
        d = [r["deficit"] for r in rs]
        means.append(float(np.mean(d)))
        scatters.append(float(max(d) - min(d)))
    assert means == sorted(means), f"{field}: the deficit was not monotone: {means}"
    assert means[-1] - means[0] > max(scatters), \
        f"{field}: the deficit moved {means[-1] - means[0]:.5f}, no more than it scatters " \
        f"across seeds ({max(scatters):.5f}): {means}"


@pytest.mark.parametrize("field", ["ising", "gauss"])
def test_the_permuted_null_is_separated_from_the_signal_in_both(field):
    """The instrument's own exact re-pairing null: destroy the pairing, lose the reading.

    The control is a SEPARATION, and is asserted as one. Two things this deliberately does not do:

      * it does not assert `nullz < 5`, or any other level. A level here would be a number picked
        to clear the largest null reading that had been seen, which is fitting the gate to the
        data it is meant to check;
      * it does not assert that the null never resolves. `resolved` is a per-run decision with a
        per-run false-positive rate, and over these 16 rows per field it fires on the null twice
        out of 32 -- at |z| = 2.25 and 3.36, against a signal that never drops below 46. Demanding
        it never fire would be asserting the instrument has no false positives, which is not true
        and is not what the capability claim rests on.

    What the claim rests on is the gap, and the gap is what is measured here.
    """
    quiet, loud = [], []
    for beta in CAPABILITY_BETAS:
        rs = [read_axis(field, beta, MU, s) for s in SEEDS]
        loud.extend(r["z"] for r in rs)
        quiet.extend(r["nullz"] for r in rs)
    assert min(loud) > max(quiet), \
        f"{field}: the permuted control reached the signal -- null {max(quiet):.2f} against " \
        f"the weakest reading {min(loud):.2f}"


@pytest.mark.parametrize("field", ["ising", "gauss"])
def test_the_deficit_is_reproducible_across_seeds_in_that_regime(field):
    """Reproducibility is the uncertainty estimate here, so it is asserted rather than assumed."""
    # The scatter is only a problem if it is comparable to the SIGNAL, so it is compared to the
    # movement across beta rather than to a percentage chosen in advance.
    means, scatters = [], []
    for beta in CAPABILITY_BETAS:
        d = np.array([read_axis(field, beta, MU, s)["deficit"] for s in SEEDS])
        means.append(float(d.mean())); scatters.append(float(d.max() - d.min()))
    movement = means[-1] - means[0]
    assert movement > max(scatters), \
        f"{field}: scatter {max(scatters):.5f} is not small against movement {movement:.5f}"


def test_the_deficit_stops_being_reproducible_at_the_onset():
    """THE LIMITATION. Pinned so the claim is never extended past the regime it holds in.

    At beta = 4 the Ising negative fraction leaves zero and the deficit spans a factor of ten
    across the same four seeds. A single-seed measurement could not have shown this, and an
    earlier draft's extrapolation along beta was made without it.
    """
    rs = [read_axis("ising", 4.0, MU, s) for s in SEEDS]
    assert max(r["neg"] for r in rs) > 0.0, \
        "no sign problem at beta = 4 any more; re-locate the onset before trusting this bound"
    d = [r["deficit"] for r in rs]
    onset = (max(d) - min(d)) / float(np.mean(d))
    # Against the relative spread the SAME read shows INSIDE the regime the claim is made in,
    # measured here on rows this file has already drawn, rather than a level picked in advance.
    inside = []
    for beta in CAPABILITY_BETAS:
        c = [read_axis("ising", beta, MU, s)["deficit"] for s in SEEDS]
        inside.append((max(c) - min(c)) / float(np.mean(c)))
    assert onset > max(inside), \
        f"the deficit stayed as reproducible at the onset ({onset:.3f}) as inside the regime " \
        f"({max(inside):.3f}): {d}"
