"""Experiment CC -- is complex Langevin WRONG here, or just noisy, and does the drift know?

The charge-channel gate left the question open by exactly the wrong margin.  Doped, the field
complexifies as the method requires, the maximum drift reaches 1e4 where the natural scale is
O(1), and the double occupancy sits 1.1 and 2.2 sigma below exact.  Two sigma on a hand-picked
point is not a result -- it is the shape a result would have.

So one point is measured properly: mu = 1.0, t2 = 0, U = 4, the most discrepant of the three,
with four times the chains and both step sizes.  If the discrepancy holds at four or more sigma
while the spin-channel control on the SAME model stays exact, then complex Langevin converges to
the wrong answer here, which is its documented failure and not an implementation defect -- gates
1 and 2 are what license that reading.

And the second question, which is the only reason this family was opened at all: the failure is
DYNAMICAL, so a read of the trajectory might see it where the average sign cannot.  The Aarts
criterion is that the distribution of the drift magnitude must fall off faster than any power.
Measured here as the tail of |K|: the ratio of high quantiles to the median, and the slope of
log P(|K| > u) against log u, which is flat-ish for a power law and steepening for an
exponential.  The control run must look different from the failing one, or the read is worthless.
"""
from __future__ import annotations

import numpy as np

from dqmc import Model
from clangevin import run_cl
from gate_clangevin import exact_observables


def tail_report(d):
    """Quantile ratios and a log-log tail slope for the drift magnitude."""
    d = np.sort(np.abs(d))
    q = lambda p: float(np.quantile(d, p))
    med = max(q(0.5), 1e-30)
    hi = d[d > q(0.90)]
    if len(hi) > 20:
        u = np.log(hi)
        p = np.log(1.0 - np.linspace(0.90, 1.0, len(hi), endpoint=False))
        slope = float(np.polyfit(u, p, 1)[0])
    else:
        slope = float("nan")
    return dict(med=med, r99=q(0.99) / med, r999=q(0.999) / med,
                mx=float(d[-1]) / med, slope=slope)


if __name__ == "__main__":
    print("=" * 100)
    print("ONE POINT, PROPERLY  mu = 1.0, t2 = 0, U = 4, N = 4, L = 10 (beta = 1)")
    m = Model(N=4, t=1.0, t2=0.0, mu=1.0, U=4.0, dtau=0.1, L=10)
    ne, de = exact_observables(m)
    print(f"  exact: n = {ne:.6f}   d = {de:.6f}")
    print()
    print(f"{'channel':>8} {'eps':>8} {'chains':>7} {'n':>21} {'d':>12} {'+-':>8} "
          f"{'d-exact':>9} {'z':>6} {'|Im|':>7} {'maxdrift':>9}")
    runs = {}
    for chan in ("spin", "charge"):
        for eps in (2e-3, 1e-3):
            r = run_cl(m, t_therm=8.0, t_meas=40.0, n_meas=200, eps=eps, seed=1,
                       adaptive=True, channel=chan, chains=512, collect_drift=True)
            z = abs(r["docc"].real - de) / max(r["docc_se"].real, 1e-12)
            runs[(chan, eps)] = r
            print(f"{chan:>8} {eps:8.1e} {r['chains']:7d} {r['n'].real:+12.6f}"
                  f"{r['n'].imag:+8.5f}i {r['docc'].real:12.6f} {r['docc_se'].real:8.5f} "
                  f"{r['docc'].real-de:+9.5f} {z:6.2f} {r['im_frac']:7.4f} "
                  f"{r['max_drift']:9.2e}", flush=True)

    print()
    print("=" * 100)
    print("THE DRIFT TAIL  (Aarts: P(|K| > u) must fall off faster than any power)")
    print(f"{'channel':>8} {'eps':>8} {'median |K|':>11} {'q99/med':>9} {'q999/med':>9} "
          f"{'max/med':>10} {'loglog slope':>13}")
    for key, r in runs.items():
        if r["drifts"] is None:
            continue
        t = tail_report(r["drifts"])
        print(f"{key[0]:>8} {key[1]:8.1e} {t['med']:11.5f} {t['r99']:9.2f} "
              f"{t['r999']:9.2f} {t['mx']:10.2f} {t['slope']:13.3f}")
    print()
    print("A steep (large negative) log-log slope means the tail is dying fast, which is what the")
    print("criterion asks for.  A shallow one is a power law and is the documented warning.  The")
    print("spin row is the control: same model, same code, no complexification, exact answer.")
