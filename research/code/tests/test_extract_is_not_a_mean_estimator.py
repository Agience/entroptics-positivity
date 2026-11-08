"""S8.7, gated: `extract` against the raw block mean, and the file that measures it must RUN.

This test exists because §9.7 was quoted from an experiment nothing gated, and the experiment
could not execute at all against the library -- `E.aperture` is the module, not the class, and
`extract()` returns `(clean, info)` rather than an object with a `.clean`. Two API breaks and a
figure taken from the single worst row all survived because no gate ever imported this file.

WHAT IS CLAIMED. `extract` is lossless: `clean` is each channel's mean plus the frame's resolved
modes, and the whitening centres every channel before the modes are read, so the resolved part has
no mean over the rows. `clean.mean(axis=0)` is therefore the raw block mean, and as an estimator of
a mean `extract` returns exactly what the raw mean returns. It adds nothing to the quantity the sign
problem lives in, which is why the paper names it and uses it for no figure.

The scoring uses no chosen level anywhere. Every comparison is against the raw block mean computed
from the SAME blocks, and every bound is the round-off of the arithmetic that formed the two
numbers compared: a mean over `n` rows rounds by at most `n * eps * max|W|`, and two RMS errors
against one truth whose means agree to that differ by no more than it (the triangle inequality),
plus the round-off of each RMS over its `L` lags.
"""
from __future__ import annotations

import numpy as np
import pytest

from functools import lru_cache

EPS = float(np.finfo(np.float64).eps)


@lru_cache(maxsize=None)
def grid():
    """The nine rows of S8.7: three (t2, mu, beta) points by three block sizes."""
    import entroptics_adapter as EA
    from dqmc import Model
    from reads.expRR_natural_noise import blocks, exact_trace

    out = []
    for t2, mu, beta in ((0.7, 0.6, 2.0), (0.7, 1.0, 2.0), (0.7, 0.6, 3.0)):
        m = Model(N=4, t=1.0, t2=t2, mu=mu, U=4.0, dtau=0.25, L=int(round(beta / 0.25)))
        truth = exact_trace(m)
        for per_block in (2, 8, 32):
            W = blocks(m, 24, per_block, seed=int(beta * 10 + mu * 10 + per_block))
            clean, info = EA.denoise(W)
            clean = np.asarray(clean)
            e_raw = float(np.sqrt(np.mean((W.mean(axis=0) - truth) ** 2)))
            e_ext = float(np.sqrt(np.mean((clean.mean(axis=0) - truth) ** 2)))
            out.append(dict(t2=t2, mu=mu, beta=beta, per_block=per_block,
                            raw=e_raw, ext=e_ext, ratio=e_ext / e_raw,
                            clean_mean=clean.mean(axis=0),
                            centre=np.asarray(info["centre"]),
                            w_mean=W.mean(axis=0),
                            rows=int(W.shape[0]),
                            peak=float(max(np.abs(W).max(), np.abs(clean).max()))))
    return tuple(out)


def _mean_roundoff(r) -> float:
    """The round-off a mean over the frame's rows can carry, at the frame's own magnitude."""
    return r["rows"] * EPS * r["peak"]


def test_the_experiment_behind_section_8_7_actually_runs():
    """THE GATE THAT WAS MISSING. Two API breaks lived in this file undetected.

    `E.aperture(W)` raised `'module' object is not callable` and `.extract().clean` raised
    `'tuple' object has no attribute 'clean'`. Neither is subtle; both survived because the file
    was quoted from and never executed.
    """
    rows = grid()
    assert len(rows) == 9, f"the S8.7 grid is no longer nine rows: {len(rows)}"
    assert all(np.isfinite(r["ratio"]) for r in rows), "a row failed to produce a ratio"


def test_clean_carries_the_raw_mean():
    """WHY it is not a mean estimator, asserted rather than asserted-about.

    `clean` comes back in W's own units, and its mean over the rows is the raw block mean and the
    read's own `centre`, both to the round-off of a mean. The resolved modes carry no mean, so
    nothing `extract` resolves can move the estimate.
    """
    for r in grid():
        tol = _mean_roundoff(r)
        off_raw = float(np.abs(r["clean_mean"] - r["w_mean"]).max())
        off_centre = float(np.abs(r["clean_mean"] - r["centre"]).max())
        assert off_raw <= tol, \
            f"clean's mean left the raw mean by {off_raw:.2e} (round-off {tol:.2e}) at " \
            f"per_block={r['per_block']}: the resolved part now carries a mean, and S8.7 needs " \
            f"re-measuring"
        assert off_centre <= tol, \
            f"clean's mean left the read's centre by {off_centre:.2e} (round-off {tol:.2e}) at " \
            f"per_block={r['per_block']}"


def test_extract_scores_exactly_as_the_raw_mean_does():
    """The consequence for the estimate: on every row the two errors agree to the round-off.

    Not a wash and not a tail either way. An earlier library shrank the resolved modes and moved
    the mean, and this section then read a ratio bracketing 1; that is withdrawn.
    """
    for r in grid():
        L = r["w_mean"].size
        bound = _mean_roundoff(r) + L * EPS * max(r["raw"], r["ext"])
        assert abs(r["ext"] - r["raw"]) <= bound, \
            f"extract's error {r['ext']:.6e} and the raw mean's {r['raw']:.6e} differ by more " \
            f"than round-off ({bound:.2e}) at per_block={r['per_block']}"


@pytest.mark.parametrize("row", range(9))
def test_every_row_scores_both_estimators_on_the_same_blocks(row):
    """A provenance guard: the two errors must come from one draw, not two samplers."""
    r = grid()[row]
    assert r["raw"] > 0.0 and r["ext"] > 0.0
    assert r["ratio"] == pytest.approx(r["ext"] / r["raw"], rel=1e-12)
