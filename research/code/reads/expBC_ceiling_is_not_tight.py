"""Experiment BC -- section 9's ceiling is a SOUND bound and not a TIGHT one, and the slack is
the chain's own autocorrelation.  The quantity that closes it is not the sign's.

WHAT SECTION 8 SAYS, AND WHY IT STAYS TRUE.  Correct importance sampling draws at `|w|`, so a
configuration enters any weighted average carrying only its sign, and Kish's effective sample
size of those weights is exactly `n <sgn>^2`.  `carriage` returns it as `effective_n` at a ratio
of 1.0000.  Every use of that number in this paper is as an UPPER bound -- "capped at" -- and an
upper bound it remains.  Nothing below weakens the argument that no weighted read can be a
severity meter; it strengthens it, because the true evidence is smaller still.

WHAT IT DOES NOT SAY.  Kish's formula assumes INDEPENDENT draws.  A DQMC run is a Markov chain, so
the evidence it actually carries is

    n <sgn>^2 / (2 tau_int)

and nothing in this paper computes `tau_int`.  Anyone reading the ceiling as the evidence a run
has, or converting it into an error bar, is off by that factor.

WHICH tau.  Not the sign's.  The standard answer is the right one: there is no single
autocorrelation time for a chain, every estimator has its own, set by its influence function.
The quantity a practitioner reports is the reweighted ratio `<O s> / <s>`, whose linearisation is

    z_t = (O_t s_t - r s_t) / <s>,      r = <O s> / <s> at the pooled value

and `z` relaxes more slowly than `s` does.  Reaching for the sign's own tau is the natural move,
because the sign is what the problem is named after, and it leaves a third of the gap.

IT IS NOT A SIGN-PROBLEM EFFECT, and that has to be said or the number reads as more than it is.
At the same beta with the sign problem switched off (`--t2 0.0`, giving <sgn> = 1.00000 exactly,
so the ceiling reads the whole sample) the slack is 3.66x, against 4.15x with it on -- a ratio of
1.13x.  On that row the sign has no variance and therefore no tau to correct by, and the slack is
there anyway.

BETA IS THE OTHER AXIS, and `--beta` scans it (one seed a point):

    beta      <sgn>      2 tau_infl
    2         0.99792    2.60x
    3         0.95590    3.56x
    4         0.83056    4.24x
    5         0.65417    3.61x

<sgn> falls monotonically, by a third across the scan; the slack does not follow it -- it rises to
beta = 4 and comes back down while <sgn> keeps falling.  A quantity the sign problem controlled
could not do that.  What beta does do is lengthen the imaginary-time extent and slow the chain.

NONE OF THE MACHINERY IS NEW, and the correction is worth having anyway.  That Kish's formula
assumes independent draws is textbook, and so is the handling of a derived quantity: Wolff's Gamma
method (Comput. Phys. Commun. 156, 143, 2004) propagates a function of primary observables by
projecting their fluctuations through its derivatives, pi_F = sum_a (dF/da) pi_a, which for
F = <O s>/<s> is exactly the z_t above, and it ships in the standard packages.  What is measured
here is an applied correction to THIS paper's bound -- section 9 states n <sgn>^2 without the
autocorrelation factor standard error analysis would supply, and on this rig that factor is
4.15x -- not a new way of obtaining it.

MEASURED HERE, at N = 8, U = 4, beta = 4, t2 = 0.7, mu = 1.0 (a real sign problem, <sgn> ~ 0.83),
against a reference built from the scatter of 128 independent replicas -- i.e. the four-seed run,
`--seeds 31 32 33 34`.  The one-seed default reproduces the same picture more coarsely (32
replicas: tau_sign 0.68, tau_infl 2.12, ceiling 2.05x too small, corrected to 1.00x, and the
control tight at 1.00x), which is enough to see the effect but not to quote:

    tau of the sign                 1.106 +- 0.140
    tau of the influence function   2.074       per diagonal 1.50 +- 0.15 to 3.18 +- 0.43
    ceiling as stated               overstates the evidence by 4.15x (3.0x to 6.4x per diagonal)
    error bar implied by it         1.98x too small
      corrected by the sign's tau   1.33x -- the natural move, and it leaves a third of the gap
      corrected by tau_infl         0.97x -- the account closes
    residual thermalisation         first-half / second-half scatter 1.07 (0.93 to 1.17)

THE CONTROL IS THE POINT.  The same sweeps, drawn i.i.d. with replacement from the pooled
ensemble, have the same marginal and therefore the same `<sgn>` and the same `n <sgn>^2`, with the
dependence between consecutive entries removed.  Measured: `tau_infl` falls from 2.074 to 0.536
and the sign's from 1.106 to 0.463 -- both the independent value of 1/2 -- and the ceiling goes
from 1.98x too small to 1.02x, i.e. tight.  So the slack is the autocorrelation and not the
weights, the observable, or the estimator's nonlinearity.

What the control must NOT be is a permutation of each chain's sweeps in time.  That leaves the
chain's mean exactly where it was, so the scatter of chain means -- the reference, and also the
estimator's measured error -- is invariant by construction, and the comparison reports that
nothing changed whatever the truth is.  It was tried first and it says nothing.

WHERE ENTROPTICS ENTERS.  `EA.autocorrelation_time` -- the adapter's name for
`dynamics(z).reconstruct_decay()` -- gives tau from ONE chain with no
window, truncation lag or multiplier: C(tau) is rebuilt from the operator's modal powers and
eigenvalues and summed over all lags, so the summation limit is not a window (pushing it from
1000 to 8000 moves the answer by 0.0e+00, asserted below).  Two cautions, both measured:

  * it must be `reconstruct_decay`, which reads the CONNECTED operator.  `rates().dominant` reads
    the raw one, where the constant function is an eigenmode with |mu| = 1; on a sign sequence
    with mean 0.83 that returns the DC mode and overstates tau by more than an order of magnitude.
  * from a scalar frame the fit is a single exponential, and this chain's influence function has a
    slower tail than that, so the one-chain read understates tau by ~1.5x.  It closes the gap from
    1.98x to ~1.21x, not to 1.00x.  The remaining slack needs the replicas.

RUNTIME.  Measured, not estimated: one seed, the default, is about seven minutes of sampling
plus a minute of reads.  The four seeds behind the 128-replica figures quoted above are about
thirty and are run with `--seeds 31 32 33 34`.  The beta scan is four one-seed runs, so roughly
another thirty; it is four separate invocations rather than a loop, because each point is a
different model and nothing is shared between them.
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from dqmc import Model                                            # noqa: E402
import entroptics_adapter as EA                          # noqa: E402

from reads.expQQ_coupling_vs_sign import run                      # noqa: E402

N, U, DTAU = 8, 4.0, 0.125
T2, MU, BETA = 0.7, 1.0, 4.0
R, WARM, NMEAS = 32, 20, 180


def tau_entroptics(x, max_lag=4000):
    """tau_int from the chain's own one-step operator.  No window, lag or multiplier."""
    x = np.asarray(x, float)
    if np.std(x) == 0:
        return np.nan
    if float(EA.dynamics(x).forgetting()["margin"]) >= 1.0:
        return np.nan                       # a non-decaying mode: tau_int does not exist
    C = EA.autocorrelation_time(x, max_lag)
    return 0.5 + float(C[1:].sum()) if np.isfinite(C).all() else np.nan


def tau_reference(Z):
    """Window-free, from the scatter of per-chain means across INDEPENDENT replicas.

    Var(mean of one chain of length T) = 2 tau c0 / T.  This needs every replica, which is what
    a production run does not have; it is ground truth here only because it was paid for.
    """
    T, Rr = Z.shape
    c0 = float(np.mean([np.var(Z[:, r]) for r in range(Rr)]))
    if c0 <= 0:
        return np.nan, np.nan
    means = Z.mean(axis=0)
    tau = T * float(np.var(means, ddof=1)) / (2 * c0)
    g = np.random.default_rng(0)
    bs = [T * float(np.var(means[g.integers(0, Rr, Rr)], ddof=1)) / (2 * c0) for _ in range(600)]
    return tau, float(np.std(bs))


def collect(seeds):
    Ss, As = [], []
    L = int(round(BETA / DTAU))
    for s in seeds:
        m = Model(N=N, t=1.0, t2=T2, mu=MU, U=U, dtau=DTAU, L=L)
        t0 = time.time()
        A, _B, S, acc = run(m, R=R, warm=WARM, n_meas=NMEAS, seed=s)
        Ss.append(np.asarray(S, float).reshape(NMEAS, R))
        As.append(np.asarray(A, float).reshape(NMEAS, R, -1))
        print(f"  seed {s}: <sgn> = {Ss[-1].mean():+.5f}  acc = {acc:.3f}  "
              f"{time.time() - t0:.0f}s", flush=True)
    return np.hstack(Ss), np.concatenate(As, axis=1)


def report(S, A, tag):
    T, Rr = S.shape
    Nobs = A.shape[2]
    sgn = float(S.mean())
    sbar = sgn
    ratio = [(A[:, :, i] * S).mean() / sbar for i in range(Nobs)]

    def influence(r, i):
        d_t = S[:, r]
        return (A[:, r, i] * d_t - ratio[i] * d_t) / sbar

    # tau of the SIGN, and tau of each observable's influence function.
    #
    # THE PER-OBSERVABLE VALUES ARE THE RESULT, NOT AN INTERMEDIATE.  There is one tau per
    # estimator, set by its influence function, and this lattice's Green's-function diagonals do
    # not share it -- so the RANGE is what can be reported and a mean alone would read as though a
    # single number existed.  They are read off the same chains and are not independent of one
    # another, so the mean carries no 1/sqrt(Nobs) and is quoted as a typical value only.
    tau_sgn_ref, se_s = tau_reference(S)
    tau_inf_refs, tau_inf_ses, tau_inf_ents = [], [], []
    for i in range(Nobs):
        Z = np.column_stack([influence(r, i) for r in range(Rr)])
        t_i, se_i = tau_reference(Z)
        tau_inf_refs.append(t_i)
        tau_inf_ses.append(se_i)
        tau_inf_ents.append(float(np.nanmean([tau_entroptics(Z[:, r]) for r in range(Rr)])))
    tau_inf = float(np.mean(tau_inf_refs))
    tau_inf_ent = float(np.mean(tau_inf_ents))
    # nan-aware: `tau_reference` returns nan on a degenerate observable (zero variance), and a
    # plain argmin would then report THAT one as the extreme and print a nan as the range.
    lo, hi = int(np.nanargmin(tau_inf_refs)), int(np.nanargmax(tau_inf_refs))

    # The estimator's TRUE error: the scatter of independent replicas' reweighted estimates.
    est = (A * S[:, :, None]).mean(axis=0) / S.mean(axis=0)[:, None]      # (Rr, Nobs)
    true_se = est.std(axis=0, ddof=1)
    # What the ceiling implies for that error bar, from ONE replica.
    implied = []
    for i in range(Nobs):
        v = [np.sqrt(A[:, r, i].var(ddof=1) / max(T * S[:, r].mean() ** 2, 1e-12))
             for r in range(Rr)]
        implied.append(float(np.mean(v)))
    factor = float(np.median([true_se[i] / implied[i] for i in range(Nobs)]))

    n = T * Rr
    kish = n * sgn * sgn
    print()
    print(f"  {tag}")
    print(f"    <sgn>                                  {sgn:+.5f}")
    print(f"    tau of the sign                        {tau_sgn_ref:6.3f} +- {se_s:.3f}")
    print(f"    tau of the influence function          {tau_inf:6.3f}   "
          f"(entroptics, one chain: {tau_inf_ent:.3f})")
    print(f"      per diagonal, {Nobs} of them             "
          f"{tau_inf_refs[lo]:.2f} +- {tau_inf_ses[lo]:.2f} to "
          f"{tau_inf_refs[hi]:.2f} +- {tau_inf_ses[hi]:.2f}")
    print(f"    n <sgn>^2  (the ceiling as stated)     {kish:9.1f}")
    print(f"    n <sgn>^2 / (2 tau_infl)               {kish / (2 * tau_inf):9.1f}   "
          f"-- overstated by {2 * tau_inf:.2f}x")
    print(f"      per diagonal                         "
          f"{2 * tau_inf_refs[lo]:.1f}x to {2 * tau_inf_refs[hi]:.1f}x, "
          f"mean {2 * tau_inf:.2f}x")
    print(f"    error bar implied by the ceiling       {factor:.2f}x too small")
    # A sign-free run has `<sgn> = 1` on every sweep, so the sign has no variance and therefore no
    # autocorrelation time -- `tau_reference` returns nan, correctly. Printing "nanx" would invite
    # that to be read as a failed computation rather than as the absence of a quantity, and the
    # absence is the point: with the sign problem switched off there is no sign tau to correct BY,
    # and the slack is still there.
    if np.isfinite(tau_sgn_ref):
        print(f"      corrected by the SIGN's tau          "
              f"{factor / np.sqrt(2 * tau_sgn_ref):.2f}x")
    else:
        print("      corrected by the SIGN's tau          -- the sign never varies, so it has none")
    print(f"      corrected by the INFLUENCE tau       {factor / np.sqrt(2 * tau_inf):.2f}x")
    return factor, tau_inf


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[31])
    # The slack is NOT a sign-problem effect, and `--t2 0.0` is how a reader checks that: same
    # beta, same imaginary-time extent, same lattice, sign problem off.  It gives <sgn> = 1
    # exactly -- the ceiling reads the whole sample -- and the slack is still there.
    ap.add_argument("--t2", type=float, default=T2,
                    help="0.0 switches the sign problem off at fixed beta (the control)")
    # beta is the OTHER axis the slack could be attributed to, and it has to be movable for the
    # claim "this is ordinary autocorrelation" to be checkable: raising beta lengthens the
    # imaginary-time extent and slows the chain whatever the sign is doing, so tau rising with
    # beta is not evidence of a sign-problem effect. It was a module constant, which made the
    # scan quoted in section 9 impossible to reproduce from this file.
    ap.add_argument("--beta", type=float, default=BETA,
                    help="inverse temperature; scan it to separate chain slowing from the sign")
    args = ap.parse_args()
    T2 = args.t2
    BETA = args.beta

    print("=" * 96)
    print("EXPERIMENT BC -- IS SECTION 8's CEILING TIGHT?")
    print(f"N = {N}, U = {U}, dtau = {DTAU}, beta = {BETA}, t2 = {T2}, mu = {MU}")
    print(f"{len(args.seeds)} seed(s) x {R} replicas x {NMEAS} sweeps")
    print("=" * 96)
    S, A = collect(args.seeds)

    print()
    print("-" * 96)
    f_chain, tau_chain = report(S, A, "THE CHAIN AS SAMPLED")

    # THE CONTROL.  Note what it must NOT be: permuting each chain's sweeps in time leaves that
    # chain's mean exactly where it was, so the scatter of chain means -- which is both the
    # reference and the estimator's measured error -- is invariant by construction, and the
    # comparison reports that nothing changed no matter what is true.  That is a defect in the
    # control, not a result.
    #
    # To break the dependence in the SAMPLING DISTRIBUTION rather than in one realisation, build
    # pseudo-replicas by drawing sweeps i.i.d. with replacement from the pooled ensemble.  The
    # marginal is preserved, so <sgn> and n <sgn>^2 are preserved, and consecutive entries are now
    # independent.  If the ceiling is tight here and slack on the real chain, the slack is the
    # autocorrelation and nothing else.
    g = np.random.default_rng(12345)
    flatS = S.reshape(-1)
    flatA = A.reshape(-1, A.shape[2])
    pick = g.integers(0, flatS.size, size=S.shape)
    print()
    print("-" * 96)
    f_sh, tau_sh = report(flatS[pick], flatA[pick],
                          "CONTROL: THE SAME SWEEPS, RESAMPLED i.i.d.")

    print()
    print("=" * 96)
    print(f"  as sampled: the ceiling's error bar is {f_chain:.2f}x too small, tau_infl "
          f"= {tau_chain:.2f}")
    print(f"  i.i.d.:     {f_sh:.2f}x, tau_infl = {tau_sh:.2f}")
    if f_sh < 1.25 and f_chain > 1.4:
        print("  The ceiling is TIGHT for independent draws and NOT tight for the chain that")
        print("  produced them.  The slack is the autocorrelation and nothing else -- same")
        print("  sweeps, same marginal, same <sgn>, same n <sgn>^2; only the dependence between")
        print("  consecutive entries is gone.")
    else:
        print("  The control does not separate the two, so the gap is not autocorrelation alone")
        print("  and nothing here should be read as if it were.")

    print()
    print("  `max_lag` IS NOT A WINDOW -- a window changes the answer as it moves:")
    z0 = (A[:, 0, 0] * S[:, 0] - ((A[:, :, 0] * S).mean() / S.mean()) * S[:, 0]) / S.mean()
    a1k, a8k = tau_entroptics(z0, 1000), tau_entroptics(z0, 8000)
    print(f"    tau at max_lag = 1000: {a1k:.6f}")
    print(f"    tau at max_lag = 8000: {a8k:.6f}     difference {abs(a1k - a8k):.1e}")

    # RESIDUAL THERMALISATION, EXCLUDED SEPARATELY.  A chain still relaxing towards its stationary
    # distribution would inflate the across-replica scatter -- and would do it through the same
    # channel as autocorrelation, so the control above cannot tell them apart.  It is a different
    # question and gets a different measurement: if the chains were still drifting, the scatter of
    # the FIRST half of each chain would exceed the scatter of the SECOND. Reported as their ratio,
    # because a ratio near one is the whole statement and no threshold is needed to read it.
    print()
    print("  RESIDUAL THERMALISATION, as the ratio of across-replica scatter, first half to second:")
    half = S.shape[0] // 2
    ratios = []
    for i in range(A.shape[2]):
        f = (A[:half] * S[:half, :, None]).mean(axis=0)[:, i] / S[:half].mean(axis=0)
        b = (A[half:] * S[half:, :, None]).mean(axis=0)[:, i] / S[half:].mean(axis=0)
        sd_b = float(b.std(ddof=1))
        if sd_b > 0:
            ratios.append(float(f.std(ddof=1)) / sd_b)
    r_med = float(np.median(ratios))
    print(f"    median over {len(ratios)} diagonals: {r_med:.2f}     "
          f"(range {min(ratios):.2f} to {max(ratios):.2f})")
    print("    A chain still thermalising would show the first half SCATTERING MORE; a ratio")
    print("    near one says the inflation above is autocorrelation and not residual drift.")
