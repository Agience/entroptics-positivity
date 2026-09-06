"""Gate: the vectorised walker must BE the reference walker, not merely agree with it on average.

At one walker the vectorised code draws exactly one uniform per site, in the same order the
scalar reference does, so the two trajectories are the same trajectory and every step's energy
must match to machine precision.  An "agrees within error bars" check would pass on a rewrite
that quietly samples a different distribution, which is exactly the class of bug this rig has
already produced once.

Four legs, because the two switches are independent: constrained and free projection, each with
and without periodic reorthonormalisation.  Population control is excluded from the identity
check on purpose -- it consumes a random draw and resamples, so it cannot be step-aligned; it is
checked separately as a statistical no-op.
"""
from __future__ import annotations

import numpy as np

from model2d import Model2D
from sector_ed import ground_energy
from cpmc import CPMC
from expU_cpmc import Free
from cpmc_fast import FastCPMC, run


def trajectory(cls_or_flag, m, nu, nd, steps, ortho, seed=7):
    if isinstance(cls_or_flag, bool):
        ch = FastCPMC(m, nu, nd, n_walkers=1, seed=seed, constrained=cls_or_flag)
    else:
        ch = cls_or_flag(m, nu, nd, n_walkers=1, seed=seed)
    out = []
    for t in range(steps):
        ch.step()
        if ortho and (t + 1) % 5 == 0:
            ch.orthonormalise()
        out.append(ch.energy())
    return np.array(out)


if __name__ == "__main__":
    m = Model2D(Lx=2, Ly=4, t=1.0, mu=0.0, U=2.0, dtau=0.05, L=1, theta=0.0)
    nu = nd = 3
    steps = 60

    print("IDENTITY  one walker, same seed, step-by-step energies")
    print(f"{'branch':>14} {'ortho':>6} {'max |diff|':>13} {'last scalar':>14} {'last fast':>14}")
    worst = 0.0
    for name, ref, flag in (("constrained", CPMC, True), ("free projection", Free, False)):
        for ortho in (False, True):
            a = trajectory(ref, m, nu, nd, steps, ortho)
            b = trajectory(flag, m, nu, nd, steps, ortho)
            d = float(np.abs(a - b).max())
            worst = max(worst, d)
            print(f"{name:>14} {str(ortho):>6} {d:13.3e} {a[-1]:+14.9f} {b[-1]:+14.9f}")
    print(f"\nworst disagreement over all four legs: {worst:.3e}")
    print("PASS" if worst < 1e-9 else "FAIL -- the rewrite is not the same algorithm")

    print()
    print("NEGATIVE CONTROL  the gate must be able to fail")
    class Wrong(FastCPMC):
        def _green(self, ph, s):                      # the finite-temperature convention
            return np.eye(ph.shape[1])[None] - super()._green(ph, s)
    a = trajectory(CPMC, m, nu, nd, steps, False)
    ch = Wrong(m, nu, nd, n_walkers=1, seed=7)
    b = []
    for t in range(steps):
        ch.step(); b.append(ch.energy())
    print(f"  with G -> I - G the same comparison gives {np.abs(a-np.array(b)).max():.3e}")

    print()
    print("U = 0 EXACTNESS  and the constraint must kill nothing")
    m0 = Model2D(Lx=2, Ly=4, t=1.0, mu=0.0, U=0.0, dtau=0.05, L=1, theta=0.0)
    ex0 = ground_energy(m0.K, 0.0, nu, nd)
    for flag, label in ((True, "constrained"), (False, "free")):
        r = run(m0, nu, nd, 6.0, constrained=flag, n_walkers=200, seed=1, n_meas=100)
        print(f"  {label:>12} {r['e']:+.12f}  exact {ex0:+.12f}  "
              f"diff {r['e']-ex0:+.2e}  killed {r['killed']}")

    print()
    print("POPULATION CONTROL is a statistical no-op on the constrained branch")
    m4 = Model2D(Lx=2, Ly=4, t=1.0, mu=0.0, U=4.0, dtau=0.05, L=1, theta=0.0)
    ex4 = ground_energy(m4.K, 4.0, nu, nd)
    for pop, lbl in ((10, "every 10"), (10**9, "never")):
        rs = [run(m4, nu, nd, 8.0, n_walkers=300, seed=s, n_meas=200, pop_every=pop)
              for s in (1, 2, 3)]
        e = np.array([r["e"] for r in rs])
        print(f"  pop control {lbl:>8}: {e.mean():+.5f} +- {e.std(ddof=1)/np.sqrt(3):.5f}"
              f"   exact {ex4:+.5f}")

    print()
    print("WEIGHT RANGE  the arithmetic must not decide the experiment")
    print("  the constrained weight is a product of one factor per (site, step); at beta = 8,")
    print("  dtau = 0.1 that is ~3000 factors, and without renormalisation it underflows to 0,")
    print("  after which the comb divides by a zero total and the population reads as extinct.")
    print(f"{'U':>5} {'dtau':>6} {'beta':>5} {'CPMC':>12} {'exact':>11} {'killed':>7} {'extinct':>8}")
    for U in (4.0, 8.0, 12.0):
        for dt in (0.10, 0.05):
            mm = Model2D(Lx=2, Ly=4, t=1.0, mu=0.0, U=U, dtau=dt, L=1, theta=0.0)
            exx = ground_energy(mm.K, U, nu, nd)
            r = run(mm, nu, nd, 8.0, n_walkers=400, seed=1, n_meas=200)
            print(f"{U:5.1f} {dt:6.3f} {8.0:5.1f} {r['e']:+12.5f} {exx:+11.5f} "
                  f"{r['killed']:7d} {str(r['extinct_at']):>8}")
