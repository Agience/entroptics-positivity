"""Experiment JJ -- is the quoted error bar reproducible across seeds, or does one runaway set it?

Three explanations for complex Langevin's behaviour here have been offered in this rig and two are
already dead:

  "the error bar does not converge"   -- withdrawn.  Bootstrapping the per-chain values inside one
                                        run gives se(C) ~ C^-0.47 to C^-0.51 on every column.
  "the adaptive constant suppressed   -- withdrawn.  Measured, adaptation costs a factor of 2 in
   the variance / the runs never         Langevin time, not 500, and the runs covered 39 to 57
   equilibrated"                         integrated autocorrelation times.

What is left unexplained is a factor of 24: at 96 chains one run quoted +-0.00025 while at 24
chains another quoted +-0.012, where chain count alone allows a factor of 2.  Same code path, same
nominal settings, different seed.

The candidate that survives both retractions is that the per-chain standard deviation is ITSELF a
heavy-tailed estimate -- whether a particular run happens to contain a runaway chain swings the
quoted error by an order of magnitude, while the scaling within any one run stays textbook.  Those
two statements are compatible and together they would explain everything seen.

So this measures the estimator of the error rather than the error: the same point, many seeds,
and the SPREAD OF THE QUOTED se.  A well-behaved estimator returns nearly the same se every time.
If se ranges over an order of magnitude across seeds, then no single run's z means anything --
which would retract the z values in Parts 6 and 7 for a reason that has nothing to do with any
constant, and would be a statement about complex Langevin rather than about this rig.

The spin channel runs at every seed as the control: it has no complexification and no runaways, so
its se must be stable, and if it is not then the instability is the harness and not the method.
"""
from __future__ import annotations

import numpy as np

from dqmc import Model
from clangevin import run_cl
from gate_clangevin import exact_observables


if __name__ == "__main__":
    mu, U = 0.6, 4.0
    m = Model(N=4, t=1.0, t2=0.0, mu=mu, U=U, dtau=0.1, L=10)
    ne, de = exact_observables(m)
    seeds = list(range(1, 13))
    print("=" * 100)
    print(f"IS THE QUOTED ERROR REPRODUCIBLE?   mu = {mu}, U = {U}, exact n = {ne:.6f}")
    print(f"12 seeds, 48 chains, eps = 1e-3, adaptation OFF (no invented constant anywhere)")
    print()
    print(f"{'chan':>7} {'seed':>5} {'n':>10} {'se':>9} {'z':>7} {'max|K|':>10} "
          f"{'top chain share':>16}")
    res = {}
    for chan in ("charge", "spin"):
        rows = []
        for s in seeds:
            r = run_cl(m, t_therm=8.0, t_meas=40.0, n_meas=120, eps=1e-3, seed=s,
                       adaptive=False, channel=chan, chains=48)
            v = np.real(r["per_chain_n"])
            dev = np.abs(v - np.median(v))
            top = float(dev.max() ** 2 / max((dev ** 2).sum(), 1e-300))
            se = float(np.real(r["n_se"]))
            z = abs(r["n"].real - ne) / max(se, 1e-12)
            rows.append((s, r["n"].real, se, z, r["max_drift"], top))
            print(f"{chan:>7} {s:5d} {r['n'].real:10.5f} {se:9.5f} {z:7.1f} "
                  f"{r['max_drift']:10.2e} {top:16.4f}", flush=True)
        res[chan] = rows
        print()

    print("=" * 100)
    print("THE ESTIMATOR OF THE ERROR, ACROSS SEEDS")
    print(f"{'chan':>7} {'mean n':>10} {'sd of n':>9} {'min se':>9} {'max se':>9} "
          f"{'se range':>9} {'min z':>7} {'max z':>7}")
    for chan, rows in res.items():
        n = np.array([r[1] for r in rows]); se = np.array([r[2] for r in rows])
        z = np.array([r[3] for r in rows])
        print(f"{chan:>7} {n.mean():10.5f} {n.std(ddof=1):9.5f} {se.min():9.5f} "
              f"{se.max():9.5f} {se.max()/max(se.min(),1e-30):9.1f} {z.min():7.1f} "
              f"{z.max():7.1f}")
    print()
    print("'sd of n' across seeds is the error bar the runs SHOULD be quoting.  Compare it to the")
    print("range of quoted se: if the quoted values swing over an order of magnitude around it,")
    print("the per-run error bar is not a measurement and no single-run z can be trusted.")
