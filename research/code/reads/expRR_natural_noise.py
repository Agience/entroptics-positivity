"""Experiment RR -- give the instrument a NATURAL noise distribution, then read.

Part 1 scored `Aperture.extract` on the sign-corrected correlator and recorded "a constant factor,
and a bias floor that makes it worse than raw at high statistics".  That was applied to data whose
noise is not natural.  The sign-corrected estimator is a RATIO,

    <O> = <O sgn> / <sgn>

and the noise on a ratio of two noisy quantities is skewed and heavy-tailed once <sgn> is small --
not the well-behaved floor that Marchenko-Pastur and Tracy-Widom are built against.  The reads
were calibrated for a natural noise distribution and handed a pathological one, and the poor
result was then written down as a property of the reads.

So the noise is made natural here, and CHECKED rather than assumed:

  * samples are grouped into blocks, and each block's estimate is a mean over many samples, so
    the central limit theorem is what shapes the block-to-block spread;
  * the block distribution's skew and excess kurtosis are MEASURED and reported, and the block
    size is grown until they are small.  A row where they are not small is a row where the floor
    has no business being applied, and it is marked rather than quietly used;
  * only then is the aperture read, against its own floor.

The comparison is the one that matters and Part 1's answer is the thing being re-tested: does
`extract` beat the raw block mean, scored against exact diagonalisation of the same Trotter
product, as a function of the number of blocks?  "Better at low statistics, worse at high" was
the old answer; if the noise being unnatural was the cause, the shape should change.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import skew, kurtosis

import entroptics_adapter as EA
from dqmc import Model, ExactTrotter
from sampler_stable import StableChains


def blocks(m, n_block, per_block, R=32, warm=25, seed=0):
    """(n_block, L) signed-corrected correlator estimates, one row per block.

    Each row is a mean over `per_block * R` samples, so the row-to-row spread is a CLT object
    rather than the raw heavy-tailed ratio.
    """
    ch = StableChains(m, R=R, seed=seed)
    for _ in range(warm):
        ch.sweep()
    rows = []
    for _ in range(n_block):
        num = np.zeros(m.L)
        den = 0.0
        for _ in range(per_block):
            ch.sweep()
            sgn, G = ch.measure()
            g = np.einsum("rlii->rl", G) / m.N          # trace of the correlator per slice
            num += (sgn[:, None] * g).sum(axis=0)
            den += sgn.sum()
        rows.append(num / den)
    return np.array(rows)


def exact_trace(m):
    et = ExactTrotter(m)
    P = et.propagator()
    return np.einsum("lii->l", P) / m.N


#: Collected across every row so the two comparisons section 9.7 quotes are printed once, as
#: ranges, rather than asserted in prose.
CENTRE_GAP: list[float] = []
FRAME_GAP: list[float] = []


if __name__ == "__main__":
    N, U, dtau = 4, 4.0, 0.25
    print("=" * 108)
    print(f"A NATURAL NOISE DISTRIBUTION, THEN THE READ   N = {N}, U = {U}, dtau = {dtau}")
    print("Block estimates, with the block distribution's shape MEASURED before any floor is")
    print("applied.  |skew| and |excess kurtosis| near 0 is the CLT having taken hold.")
    print()
    for t2, mu, beta in ((0.7, 0.6, 2.0), (0.7, 1.0, 2.0), (0.7, 0.6, 3.0)):
        L = int(round(beta / dtau))
        m = Model(N=N, t=1.0, t2=t2, mu=mu, U=U, dtau=dtau, L=L)
        truth = exact_trace(m)
        print(f"t2 = {t2}, mu = {mu}, beta = {beta}   exact trace G(0) = {truth[0]:.5f}")
        print(f"{'per block':>10} {'blocks':>7} {'|skew|':>8} {'|ex kurt|':>10} {'natural':>8} "
              f"{'raw err':>10} {'extract err':>12} {'ratio':>7}")
        for per_block in (2, 8, 32):
            W = blocks(m, 24, per_block, seed=int(beta * 10 + mu * 10 + per_block))
            sk = float(np.abs(skew(W, axis=0)).mean())
            ku = float(np.abs(kurtosis(W, axis=0)).mean())
            # CHOSEN CONSTANTS, so this column is descriptive and NOT QUOTABLE: 0.5 and 1.0
            # are picked rather than derived, and they decide a yes/no verdict.  The skew and
            # kurtosis themselves are measurements and are printed beside it; read those.
            natural = "yes" if (sk < 0.5 and ku < 1.0) else "NO"
            raw = W.mean(axis=0)
            # Reached through the adapter, which is also the answer to how this line used to be
            # wrong: it named the entroptics MODULE where the class was meant and raised
            # `'module' object is not callable`, so this file could not run at all -- found
            # 2026-09-07, while re-measuring the figure 8.7 quotes from it. Nothing gates this
            # experiment, which is why it went unnoticed. One named entry point cannot be
            # misspelled that way.
            clean, info = EA.denoise(W)
            clean = np.asarray(clean)
            # `extract` returns `(clean, info)` with `clean` in W's own units, so it is used as
            # returned.
            rec = clean
            # `clean.mean(axis=0)` matches neither the frame mean nor the read's own centre on
            # these rows. `clean` is the centre plus a shrunk projection onto the resolved modes,
            # and that projection carries a mean of its own wherever there is structure to resolve.
            # On a frame with nothing resolvable the projection is empty and `clean.mean` is the
            # centre to 3e-16, so a check on Gaussian frames reads the two as equal and cannot see
            # this. These frames carry a correlator. Both gaps are collected, so section 9.7's
            # claim is a measurement on the data it is about.
            CENTRE_GAP.append(float(np.abs(rec.mean(axis=0) - np.asarray(info["centre"])).max()))
            FRAME_GAP.append(float(np.abs(rec.mean(axis=0) - W.mean(axis=0)).max()))
            e_raw = float(np.sqrt(np.mean((raw - truth) ** 2)))
            e_ext = float(np.sqrt(np.mean((rec.mean(axis=0) - truth) ** 2)))
            print(f"{per_block:10d} {24:7d} {sk:8.3f} {ku:10.3f} {natural:>8} "
                  f"{e_raw:10.5f} {e_ext:12.5f} {e_ext/max(e_raw,1e-30):7.3f}", flush=True)
        print()
    print()
    print("=" * 108)
    print("WHAT `extract` RETURNS, AND WHAT IT DOES NOT")
    print("=" * 108)
    print(f"  rows                                   {len(CENTRE_GAP)}")
    print(f"  |clean.mean(axis=0) - info['centre']|   {min(CENTRE_GAP):.1e} to {max(CENTRE_GAP):.1e}")
    print(f"  |clean.mean(axis=0) - W.mean(axis=0)|   {min(FRAME_GAP):.1e} to {max(FRAME_GAP):.1e}")
    print()
    print("  `clean` is neither: it is the centre plus a SHRUNK projection onto the resolved")
    print("  modes, and that projection carries a mean of its own wherever there is structure to")
    print("  resolve.  On a frame with nothing resolvable the projection is empty and the first")
    print("  row above would read 3e-16 -- which is what a check on Gaussian noise returns, and")
    print("  why that check cannot stand in for this one.")
    print()
    print("  Optimal singular-value shrinkage is the right operation for recovering a low-rank")
    print("  signal and the wrong one for an unbiased mean: it shrinks the mean-carrying mode")
    print("  along with everything else.  That is why section 9.7 reports `extract` as the wrong")
    print("  tool for this problem -- not because it misreports itself, but because the quantity")
    print("  it faithfully reports is not the one a sign problem lives in.")
    print()
    print("'ratio' below 1 is the read beating the raw block mean.  Part 1 found a constant")
    print("factor that turned into a LOSS at high statistics; if the unnatural noise was the")
    print("cause, the rows marked natural should not show that turn.")
