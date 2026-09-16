"""Experiment BB -- is the deficit a CONVERGED estimate?  The precondition for section 7's limit.

THE RESULT, three seeds a cell -- `SEEDS = (0, 1, 2)`, which is what this file runs and therefore
what the paper quotes:

    row            R=32 warm=15 n=20      R=64 warm=30 n=32      verdict
    sign-free      spread 7.6%            spread 3.3%            shrinks 2.3x -- converged
    sign problem   spread 3.3%            spread 33.7%           grows 10x -- NOT converged

The central value moves as well on the second row -- 0.12081 to 0.20154 -- while `<sgn>` stays
near 0.83.  A converged estimator cannot do either.

WHY IT MATTERS.  Section 7 claims the coupling deficit detects ONSET and not SEVERITY, and rests
that limit on exactly this difference: "the deficit is a converged estimate on the sign-free rows,
and on the deepest row its error does not shrink with sampling in the way a converged estimate's
must."  That is the accuracy boundary of the whole reading -- it is what stops the deficit being
sold as a severity meter -- and it was the least-checked load-bearing claim in the paper, because
`expAM_deficit_does_not_determine_the_sign.py` and `expAN`'s deep rows both run past ten minutes.

WHAT THIS ADDS OVER expAM.  Nothing in kind: the diagnostic is the same one, from section 9.  What
it adds is that it FINISHES.  The question is a DIRECTION -- does the spread shrink when sampling
increases -- and a direction does not need the paper's precision, so one row of each kind at two
sampling levels settles it inside a budget a reader will actually spend.  A claim nobody can check
in ten minutes is a claim most readers will not check.

The diagnostic itself: an estimator whose quoted error GROWS as sampling increases is not
converged, whatever its central value is doing.  Reading a deficit as a severity requires it to be
converged, so this is the precondition rather than a refinement.
"""
from __future__ import annotations

import time

import numpy as np

import entroptics_adapter as EA                                            # noqa: E402
from dqmc import Model                                            # noqa: E402
from reads.expQQ_coupling_vs_sign import run                      # noqa: E402

N, U, DTAU = 8, 4.0, 0.125
SEEDS = (0, 1, 2)

#  label,                t2,   mu,  beta
ROWS = [
    ("sign-free (t2=0)",  0.0, 1.0, 3.0),
    ("sign problem",      0.7, 1.0, 4.0),
]
COSTS = [
    ("R=32 warm=15 n=20", dict(R=32, warm=15, n_meas=20)),
    ("R=64 warm=30 n=32", dict(R=64, warm=30, n_meas=32)),
]

if __name__ == "__main__":
    print("DOES THE DEFICIT CONVERGE?  spread across seeds, as sampling increases.")
    print(f"N = {N}, U = {U}, dtau = {DTAU}, {len(SEEDS)} seeds per cell.")
    print("An estimator whose seed spread does not SHRINK with more sampling is not converged.")
    print()
    print(f"{'row':<20}{'sampling':<20}{'deficit':>10}{'spread':>10}{'rel':>8}{'<sgn>':>10}{'secs':>8}")
    print("-" * 86)

    table = {}
    for label, t2, mu, beta in ROWS:
        for cost_tag, kw in COSTS:
            t0 = time.perf_counter()
            defs, sgns = [], []
            for s in SEEDS:
                m = Model(N=N, t=1.0, t2=t2, mu=mu, U=U, dtau=DTAU, L=int(round(beta / DTAU)))
                A, B, S, _acc = run(m, seed=s, **kw)
                c = EA.channel_alignment(A, B)
                defs.append(1.0 - abs(c.strength))
                sgns.append(float(np.mean(S)))
            d = np.array(defs)
            dt = time.perf_counter() - t0
            table[(label, cost_tag)] = (d.mean(), d.std(), float(np.mean(sgns)))
            rel = d.std() / max(abs(d.mean()), 1e-30)
            print(f"{label:<20}{cost_tag:<20}{d.mean():>10.5f}{d.std():>10.5f}"
                  f"{rel:>7.1%}{float(np.mean(sgns)):>10.5f}{dt:>8.1f}", flush=True)

    print("-" * 86)
    print()
    for label, _t2, _mu, _b in ROWS:
        lo = table[(label, COSTS[0][0])]
        hi = table[(label, COSTS[1][0])]
        rel_lo = lo[1] / max(abs(lo[0]), 1e-30)
        rel_hi = hi[1] / max(abs(hi[0]), 1e-30)
        verdict = ("error SHRINKS with sampling -- consistent with a converged estimate"
                   if rel_hi < rel_lo else
                   "error does NOT shrink with sampling -- NOT a converged estimate")
        print(f"  {label:<20} relative spread {rel_lo:.1%} -> {rel_hi:.1%}   {verdict}")
    print()
    print("  The paper claims the deficit is converged where the sign is free and not converged on")
    print("  its deepest row, and rests 'onset, not severity' on exactly that difference.")
