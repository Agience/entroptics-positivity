"""Experiment AM -- how the coupling deficit behaves as sampling effort increases.

Section 5 reads the deficit as an ONSET detector: it moves while `<sgn>` is still identically 1.
Whether it could also be read as a SEVERITY -- how bad the sign problem will get -- turns on whether
the deficit is a converged estimate on rows that HAVE a sign problem.  This file measures that
directly, by holding the system fixed and increasing the sampling.

Two systems, one with no sign problem and one with a real one, each at four seeds and three
sampling costs.  What is reported is the deficit's central value and its error at each cost.

The diagnostic is the one already used in section 8: an estimator whose quoted error GROWS as
sampling increases is not converged, whatever its central value is doing.  A well-behaved estimator
cannot do that.  Reading a deficit as a severity requires it to be converged, so this measurement
is the precondition rather than a refinement.
"""
from __future__ import annotations

import numpy as np

import entroptics as E
from dqmc import Model
from reads.expQQ_coupling_vs_sign import run

U, DTAU, N = 4.0, 0.125, 8

#  label,           t2,  mu,  beta
SIGN_FREE = ("no sign problem", 0.0, 1.0, 5.0)
SIGN_PROB = ("sign problem", 0.7, 0.5, 3.0)

COSTS = (("R=32 warm=15 n=20", dict(R=32, warm=15, n_meas=20)),
         ("R=64 warm=25 n=20", dict(R=64, warm=25, n_meas=20)),
         ("R=64 warm=25 n=40", dict(R=64, warm=25, n_meas=40)))


def measure(t2, mu, beta, seeds, R=64, warm=25, n_meas=40):
    """Deficit and <sgn> over seeds, from the SAME chains -- not two separate samplers."""
    ds, ss = [], []
    for seed in seeds:
        m = Model(N=N, t=1.0, t2=t2, mu=mu, U=U, dtau=DTAU, L=int(round(beta / DTAU)))
        A, B, S, _ = run(m, R=R, warm=warm, n_meas=n_meas, seed=seed)
        c = E.reads.coupling(A, B)
        ds.append(1.0 + float(c.strength)); ss.append(float(S.mean()))
    return np.array(ds), np.array(ss)


def sem(a):
    return float(a.std(ddof=1) / np.sqrt(len(a)))


if __name__ == "__main__":
    SEEDS = (3, 11, 19, 29)
    print("=" * 100)
    print("DOES THE DEFICIT CONVERGE?   N = 8, U = 4, dtau = 0.125, four seeds per cell.")
    print("An error that grows with sampling means the estimate is not converged.")
    print()
    print(f"{'system':>17} {'sampling':>18} | {'deficit':>9} {'+- sem':>9} | {'<sgn>':>9}")
    for label, t2, mu, beta in (SIGN_FREE, SIGN_PROB):
        for name, kw in COSTS:
            d, s = measure(t2, mu, beta, SEEDS, **kw)
            print(f"{label:>17} {name:>18} | {d.mean():9.5f} {sem(d):9.5f} | "
                  f"{s.mean():9.5f}", flush=True)
        print()
    print("Read the sign-problem block down the page.  Where the deficit is being asked to")
    print("describe a sign problem, both its central value and its error move with sampling;")
    print("on the sign-free system the same columns hold still.  A quantity still moving when")
    print("more samples are added is measuring the sampling, not the system, which is why")
    print("section 5 claims onset and not severity.")
