"""§8.6's fitted ceiling, gated against a RECORDED reference rather than a live sweep.

WHY A REFERENCE AND NOT A MEASUREMENT.  The sweep costs 73 minutes and there is no cheap version
of it: measured 2026-09-07, dropping to `n_walkers = 120` grows the error bar at `U = 4, k = 1`
from `0.00028` to `0.00877` -- a factor of 31 for a 3.3x reduction in walkers, where square-root
scaling predicts 1.8 -- and pushes the `U = 4, k = 4` bias across zero, which makes every ratio
built on it meaningless.  The estimate of the error is heavy-tailed at low statistics, which is the
same behaviour §8.5 measures in complex Langevin.  A gate that takes 73 minutes is a gate that
stops being run, and "quoted from an experiment nobody runs" is the defect that put wrong figures
in §§8.6 and 8.7 in the first place.

So the expensive measurement is recorded in `reference/multidet_ceiling.json` with its parameters
and its date, and these tests are fast checks against it.  Regenerating the reference is a
deliberate act: `python -m tests.test_multidet_ceiling` from `research/code`.

WHAT THESE TESTS PROTECT.

  * `expLL_unfitted_trial.CEILING`, a hardcoded copy of these numbers used as a baseline, must
    still agree with the reference -- it drifted once and nothing noticed;
  * §8.6's quoted range `1.7x` to `10.4x` must be what the reference says;
  * the direction of the `k = 2` move must be what the reference resolves, and ONLY where the
    seed spread resolves it. An earlier draft read three reversals off point estimates and stated
    all three; one is 0.68 errors from nothing and another is 0.64.

Every comparison is interval disjointness -- two measurements are distinguishable when their error
bars do not overlap.  That is a property of the intervals, not a multiplier applied to them, so
there is no sigma level and no p anywhere in this file.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

from closed.expLL_unfitted_trial import CEILING

REFERENCE = Path(__file__).resolve().parent.parent / "reference" / "multidet_ceiling.json"
COUPLINGS = (4.0, 8.0, 12.0)
KS = (1, 2, 4, 6)


def ref():
    with REFERENCE.open(encoding="utf-8") as fh:
        return json.load(fh)


def bias(U, k):
    return ref()["bias"][str(U)][str(k)][0]


def sigma(U, k):
    return ref()["bias"][str(U)][str(k)][1]


def interval(U, k):
    b, e = abs(bias(U, k)), sigma(U, k)
    return b - e, b + e


def resolved_below(U, k_low, k_high):
    """Is k_low's bias resolvably smaller than k_high's?  Disjoint intervals, no level."""
    return interval(U, k_low)[1] < interval(U, k_high)[0]


def test_the_reference_records_what_it_was_measured_at():
    """A reference without provenance is a magic table. This is what stops it becoming one."""
    r = ref()
    p = r["parameters"]
    assert r["measured_on"] and r["measured_by"], "the reference does not say where it came from"
    assert p["n_walkers"] == 400 and p["n_meas"] == 150 and p["seeds"] == [1, 2, 3], \
        f"the reference's parameters are not the ones these tests assume: {p}"

    # THE PARAMETERS MUST BE THE ONES THE EXPERIMENT ACTUALLY USES, read out of its source.
    # The reference was seeded with `n_restarts: 2` while `expKK` uses 4, and a regeneration
    # honoured the recorded value and silently produced a different measurement: k = 1 reproduced
    # to the digit and all nine k > 1 entries moved. A provenance block that disagrees with the
    # code it claims to record is worse than none, so the agreement is checked rather than typed.
    src = (Path(__file__).resolve().parent.parent / "closed"
           / "expKK_multidet_bias.py").read_text(encoding="utf-8")
    m = re.search(r"best_k_dets\([^)]*n_restarts\s*=\s*(\d+)", src)
    assert m, "cannot find expKK's best_k_dets call to check n_restarts against"
    assert int(m.group(1)) == p["n_restarts"], \
        f"the reference records n_restarts={p['n_restarts']} but expKK uses {m.group(1)}; " \
        f"the fit is non-convex and restarted from random points, so these must agree"
    assert set(r["bias"]) == {str(U) for U in COUPLINGS}
    for U in COUPLINGS:
        assert set(r["bias"][str(U)]) == {str(k) for k in KS}


@pytest.mark.parametrize("U", COUPLINGS)
def test_expll_reads_the_ceiling_from_the_reference_and_does_not_copy_it(U):
    """There must be ONE source for these numbers.

    `expLL.CEILING` was a hand-copy and it drifted: on 2026-09-07 three of its twelve entries sat
    outside a fresh measurement's seed spread, `U = 12, k = 4` reading `0.08077` against
    `0.08356 +- 0.00262`. Only the `k = 1` row still matched -- and that was the row another
    column in the same file was dividing by, so the drift was invisible where it was worst.

    Exact equality is the assertion, because a loaded value and its source are the same number.
    A tolerance here would permit a copy to reappear.
    """
    for k in KS:
        assert CEILING[U][k] == bias(U, k), \
            f"U={U}, k={k}: expLL is not reading the reference -- {CEILING[U][k]} " \
            f"against {bias(U, k)}. If it has been copied back, delete the copy."


@pytest.mark.parametrize("U", COUPLINGS)
def test_four_to_six_determinants_beat_one(U):
    """§8.6's claim, at the k values it names, resolved by disjoint error bars."""
    for k in (4, 6):
        assert resolved_below(U, k, 1), \
            f"U={U}: k={k} does not beat the single determinant with disjoint error bars " \
            f"({abs(bias(U, k)):.5f} +- {sigma(U, k):.5f} against " \
            f"{abs(bias(U, 1)):.5f} +- {sigma(U, 1):.5f})"


def test_the_second_determinant_moves_both_ways_across_couplings():
    """The structure §8.6 reports, and only where the seed spread resolves it.

    On point estimates this table shows three reversals. Two of them -- `U = 8` at 0.68 errors and
    `U = 4`'s k=6 against k=4 at 0.64 -- are consistent with nothing, and an earlier draft stated
    them as fact. Disjointness is what separates the one real reversal from the two apparent ones.
    """
    better = [U for U in COUPLINGS if resolved_below(U, 2, 1)]
    worse = [U for U in COUPLINGS if resolved_below(U, 1, 2)]
    assert better, f"the second determinant no longer resolvably helps anywhere: {better}, {worse}"
    assert worse, f"the second determinant no longer resolvably hurts anywhere: {better}, {worse}"


def test_adding_a_sixth_determinant_to_four_settles_nothing():
    """PINNED, and only because two independent fits agree on it.

    ONLY WHAT SURVIVES A REFIT IS ASSERTED HERE. An earlier version of this test also pinned
    `U = 8`'s second determinant as unresolved, which is what the first fit showed. The second fit
    resolves it as worse -- intervals `[0.04976, 0.06022]` against `[0.06571, 0.06761]` -- because
    the fit-to-fit scatter is larger than the walker seed spread the intervals are built from. So
    that pin was itself a fluctuation, and it is gone.

    `U = 4`'s six-against-four gap is unresolved on BOTH fits, which is what makes it assertable.
    """
    assert not resolved_below(4.0, 6, 4) and not resolved_below(4.0, 4, 6), \
        "U=4's k=6 against k=4 is now resolved on this fit; check a second fit before restating " \
        "§8.6, because the fit-to-fit scatter exceeds the seed spread these intervals use"


def test_the_quoted_range_is_the_reference_range():
    """§8.6 quotes `1.7x` to `10.4x`. Both ends come from here, and the spread is real."""
    def band(U, k):
        r = abs(bias(U, 1)) / abs(bias(U, k))
        rel = float(np.hypot(sigma(U, 1) / abs(bias(U, 1)), sigma(U, k) / abs(bias(U, k))))
        return r, r * (1.0 - rel), r * (1.0 + rel)

    rows = {(U, k): band(U, k) for U in COUPLINGS for k in (4, 6)}
    assert min(v[0] for v in rows.values()) > 1.0, \
        f"a four-or-six determinant trial lost to one: { {p: round(v[0], 2) for p, v in rows.items()} }"

    # THE ENDS ARE NOT PINNED TO NUMBERS, and an earlier version of this test pinned them to
    # `1.74 +- 0.05` and `10.43 +- 0.2`. Those were one fit's values. `best_k_dets` is bit-exact
    # WITHIN a process and irreproducible ACROSS processes at ~1e-7 in overlap -- measured
    # 2026-09-08, and pinning BLAS to one thread only halves that, so it is not merely threading.
    # A 1e-7 shift in a non-convex optimum moves the bias by more than the seed spread: three
    # independent runs of U = 8, k = 4 give 0.02912, 0.03001 and 0.03251 against a quoted +-0.0027.
    # So the high end genuinely scatters -- 10.43 and 9.00 on two full runs -- and pinning it
    # would be pinning a fluctuation.
    lo = min(rows, key=lambda p: rows[p][0])
    hi = max(rows, key=lambda p: rows[p][0])
    assert rows[lo][2] < rows[hi][1], \
        f"the widest and narrowest reductions overlap, so this is one number and not a range: " \
        f"{lo} {tuple(round(v, 2) for v in rows[lo])} against " \
        f"{hi} {tuple(round(v, 2) for v in rows[hi])}"


FITS = REFERENCE.parent / "ceiling_fits.npz"


def _fits(p, force=False):
    """The fitted determinants, cached on disk and keyed by what they depend on.

    PROFILED 2026-09-07 at U = 8: `best_k_dets` costs 4.7s, 119.4s, 320.5s and 453.1s at k = 1, 2,
    4, 6 -- 898 seconds -- against 194 seconds for all three walker runs at every k. THE FIT IS
    82% OF THE SWEEP. That is why reducing walkers 8.3x only made the sweep 1.9x faster.

    It is also entirely cacheable: `best_k_dets(K, U, n_up, n_dn, g, k, n_restarts, seed)` takes
    no walkers, no measurement count and no measurement seed. It is deterministic and fixed by the
    experiment's own definition, and it was being recomputed twelve times per sweep and discarded.

    The cache stores the parameters it was built under and refuses itself if they have moved,
    because a silent stale cache is the defect this whole file exists to have removed.
    """
    from model2d import Model2D
    from sector_ed import ground_energy
    from closed.expGG_nonorthogonal import best_k_dets

    key = json.dumps({q: p[q] for q in ("dtau", "n_restarts", "fit_seed")}, sort_keys=True)
    if FITS.exists() and not force:
        blob = np.load(FITS, allow_pickle=True)
        if str(blob["key"]) == key:
            return blob["fits"].item(), blob["exact"].item()
        print(f"fit cache was built under {blob['key']}, not {key} -- refitting")

    fits, exact = {}, {}
    for U in COUPLINGS:
        m = Model2D(Lx=2, Ly=4, t=1.0, mu=0.0, U=U, dtau=p["dtau"], L=1, theta=0.0)
        ex, g = ground_energy(m.K, U, 3, 3, want_vec=True)
        exact[U] = float(ex)
        for k in KS:
            _, dets, c = best_k_dets(m.K, U, 3, 3, g, k,
                                     n_restarts=p["n_restarts"], seed=p["fit_seed"])
            fits[(U, k)] = (dets, c)
            print(f"  fitted U={U} k={k}", flush=True)
    np.savez(FITS, key=key, fits=np.array(fits, dtype=object),
             exact=np.array(exact, dtype=object))
    return fits, exact


if __name__ == "__main__":
    # THE EXPENSIVE MEASUREMENT, run deliberately.
    #
    # 73 minutes cold. With the determinant fits cached it is the sampling cost alone, which the
    # profile puts near 13 -- the fits are 82% of a cold run and depend on nothing the sweep
    # varies. Pass `--refit` to rebuild them.
    import sys
    import time
    from model2d import Model2D
    from cpmc_multi import run_multi

    p = ref()["parameters"]
    t0 = time.time()
    fits, exact_e = _fits(p, force="--refit" in sys.argv)
    t_fit = time.time() - t0
    print(f"fits ready in {t_fit / 60:.1f} min\n")

    out, exact = {}, {}
    for U in COUPLINGS:
        m = Model2D(Lx=2, Ly=4, t=1.0, mu=0.0, U=U, dtau=p["dtau"], L=1, theta=0.0)
        ex = exact_e[U]
        exact[str(U)] = round(float(ex), 5)
        row = {}
        for k in KS:
            dets, c = fits[(U, k)]
            e = np.array([run_multi(m, 3, 3, dets, c, p["beta"], n_walkers=p["n_walkers"],
                                    seed=s, n_meas=p["n_meas"])["e"] for s in p["seeds"]])
            row[str(k)] = [round(float(e.mean()) - float(ex), 5),
                           round(float(e.std(ddof=1) / np.sqrt(len(e))), 5)]
            print(f"U={U} k={k}: {row[str(k)]}", flush=True)
        out[str(U)] = row

    doc = ref()
    doc["bias"], doc["exact"] = out, exact
    doc["measured_on"] = time.strftime("%Y-%m-%d")
    REFERENCE.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {REFERENCE} in {(time.time() - t0) / 60:.1f} min "
          f"({t_fit / 60:.1f} fitting, {(time.time() - t0 - t_fit) / 60:.1f} sampling)")
