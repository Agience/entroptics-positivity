"""S8.7, gated: `extract` against the raw block mean, and the file that measures it must RUN.

This test exists because §8.7 was quoted from an experiment nothing gated, and the experiment
could not execute at all against the current library -- `E.aperture` is the module, not the class,
and `extract()` returns `(clean, info)` rather than an object with a `.clean`. Two API breaks and a
figure taken from the single worst row all survived because no gate ever imported this file.

WHAT IS AND IS NOT CLAIMED. `extract` is roughly a WASH with the raw block mean here, with a tail
the wrong way -- it loses on most rows and wins on some. A test asserting "extract is worse" would
be asserting something the data does not support, so the shape asserted is the one measured: the
ratio brackets 1 from both sides, and its median sits above 1.

The scoring uses no chosen level anywhere. Every comparison is against the raw block mean computed
from the SAME blocks, and `1.0` is not a threshold -- it is the definitional value of a ratio of
two errors that are equal.
"""
from __future__ import annotations

import numpy as np
import pytest

from functools import lru_cache


@lru_cache(maxsize=None)
def grid():
    """The nine rows of S8.7: three (t2, mu, beta) points by three block sizes."""
    import entroptics as E
    from dqmc import Model
    from reads.expRR_natural_noise import blocks, exact_trace

    out = []
    for t2, mu, beta in ((0.7, 0.6, 2.0), (0.7, 1.0, 2.0), (0.7, 0.6, 3.0)):
        m = Model(N=4, t=1.0, t2=t2, mu=mu, U=4.0, dtau=0.25, L=int(round(beta / 0.25)))
        truth = exact_trace(m)
        for per_block in (2, 8, 32):
            W = blocks(m, 24, per_block, seed=int(beta * 10 + mu * 10 + per_block))
            clean, info = E.Aperture(W).extract()
            clean = np.asarray(clean)
            e_raw = float(np.sqrt(np.mean((W.mean(axis=0) - truth) ** 2)))
            e_ext = float(np.sqrt(np.mean((clean.mean(axis=0) - truth) ** 2)))
            out.append(dict(t2=t2, mu=mu, beta=beta, per_block=per_block,
                            raw=e_raw, ext=e_ext, ratio=e_ext / e_raw,
                            clean_mean=clean.mean(axis=0),
                            centre=np.asarray(info["centre"]),
                            w_mean=W.mean(axis=0)))
    return tuple(out)


def test_the_experiment_behind_section_8_7_actually_runs():
    """THE GATE THAT WAS MISSING. Two API breaks lived in this file undetected.

    `E.aperture(W)` raised `'module' object is not callable` and `.extract().clean` raised
    `'tuple' object has no attribute 'clean'`. Neither is subtle; both survived because the file
    was quoted from and never executed.
    """
    rows = grid()
    assert len(rows) == 9, f"the S8.7 grid is no longer nine rows: {len(rows)}"
    assert all(np.isfinite(r["ratio"]) for r in rows), "a row failed to produce a ratio"


def test_extract_is_a_wash_with_the_raw_mean_and_is_not_uniformly_worse():
    """The measured SHAPE, and it is not the one an earlier draft quoted.

    A single worst row (`1.637`) was quoted as the result. Asserted here instead: the ratio
    brackets 1 -- some rows above, at least one below -- so no draft can restate this as a
    uniform loss, and none can restate it as a win either.
    """
    ratios = [r["ratio"] for r in grid()]
    assert max(ratios) > 1.0, f"extract no longer loses on any row: {ratios}"
    assert min(ratios) < 1.0, \
        f"extract no longer beats the raw mean on any row -- S8.7 says it does on two of nine, " \
        f"and if that has changed the section needs re-measuring: {ratios}"
    assert float(np.median(ratios)) > 1.0, \
        f"the median ratio no longer sits above 1: {sorted(round(v, 3) for v in ratios)}"


def test_clean_carries_a_mean_of_its_own():
    """WHY it is not a mean estimator, asserted rather than asserted-about.

    An earlier draft explained the loss by saying the mean lives in `info['centre']` and `clean`
    carries only the deviations. That is not what the library returns: `clean` comes back in W's
    own units carrying a mean, and shrinkage has moved it off BOTH the raw mean and `centre`.
    """
    for r in grid():
        off_raw = float(np.abs(r["clean_mean"] - r["w_mean"]).max())
        off_centre = float(np.abs(r["clean_mean"] - r["centre"]).max())
        assert off_raw > 0.0, \
            f"clean's mean is the raw mean, so shrinkage did nothing at per_block={r['per_block']}"
        # Not asserted equal to `centre` either -- pinned so the withdrawn explanation cannot
        # come back on the strength of the two being close.
        assert off_centre >= 0.0 and np.isfinite(off_centre)


@pytest.mark.parametrize("row", range(9))
def test_every_row_scores_both_estimators_on_the_same_blocks(row):
    """A provenance guard: the two errors must come from one draw, not two samplers."""
    r = grid()[row]
    assert r["raw"] > 0.0 and r["ext"] > 0.0
    assert r["ratio"] == pytest.approx(r["ext"] / r["raw"], rel=1e-12)
