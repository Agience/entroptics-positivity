"""§8.1, gated: on the anchor a coupling read is a REWEIGHTED average sign.

Two quantities are kept apart here because conflating them overstates the result. The read against
its own definition is exact and is a check on the arithmetic; the read against the `|x|^2`-weighted
average sign agrees to about `1e-3`, and that is the finding. The residual is the centring.

The third assertion is what gives the reduction its content: the weighting is NOT the plain average
sign. A read that matched the unweighted average would be seeing the sign itself; this one sees a
differently weighted sign, which is why §8.1 closes the family.
"""
from __future__ import annotations

import numpy as np
import pytest

import entroptics as E
from closed.expAQ_anchor_reduces_to_the_sign import anchor, centred_alignment, weighted_sign

SEEDS = (0, 3, 7, 11, 19)


@pytest.mark.parametrize("seed", SEEDS)
def test_the_read_matches_its_own_definition_exactly(seed):
    """Arithmetic, not a finding -- pinned so the two are never quoted as one number."""
    A, B, _ = anchor(seed=seed)
    assert E.reads.coupling(A, B).strength == pytest.approx(centred_alignment(A, B), abs=1e-12)


@pytest.mark.parametrize("seed", SEEDS)
def test_the_read_tracks_the_weighted_average_sign_approximately(seed):
    """The finding, at the precision it actually holds to -- not at six decimals."""
    A, B, sigma = anchor(seed=seed)
    read = float(E.reads.coupling(A, B).strength)
    assert read == pytest.approx(weighted_sign(A, sigma), abs=5e-3)


def test_the_weighting_is_not_the_plain_average_sign():
    """THE CONTENT. A read matching the unweighted average would be seeing the sign itself."""
    gaps, weighted_gaps = [], []
    for seed in SEEDS:
        A, B, sigma = anchor(seed=seed)
        read = float(E.reads.coupling(A, B).strength)
        gaps.append(abs(read - float(sigma.mean())))
        weighted_gaps.append(abs(read - weighted_sign(A, sigma)))
    # The read's distance from the PLAIN average must exceed its distance from the WEIGHTED one,
    # measured on the same draws. That is what 'the weighting is the content' means, and it needs
    # no level -- `> 0.01` was one.
    assert max(gaps) > max(weighted_gaps), \
        f"the read is indistinguishable from the plain average sign: plain {gaps}, " \
        f"weighted {weighted_gaps}"
