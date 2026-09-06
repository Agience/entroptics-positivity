"""The complex-Langevin gates, on the batched sampler.

  1  DRIFT     the analytic drift against a central finite difference of the complexified
               action, both channels, with the wrong-convention negative control beside it.
  2  CONTROL   the SPIN channel at half filling with no next-nearest hopping: particle-hole
               symmetric, positive weights, nothing for the method to do.  Complex Langevin must
               reduce to ordinary Langevin and reproduce exact diagonalisation, and the residual
               must extrapolate linearly to zero in the Langevin step.  A method that is wrong
               HERE is broken rather than unreliable.
  3  THE REAL  the CHARGE channel, where the action is genuinely complex.  The field must LEAVE
     QUESTION the real axis -- |Im y| = 0 would mean this is not complex Langevin at all -- and
               then the question is whether the trajectory finds the exact answer or converges
               somewhere else, which is this method's documented and unsolved failure.

The reference is `ExactTrotter`: a trace over the full Fock space of the SAME Trotter product the
sampler represents, so Trotter error cancels between the two sides and a disagreement is the
method rather than the discretisation.
"""
from __future__ import annotations

import numpy as np

from dqmc import Model
from clangevin import CLangevin, run_cl
from gate_clangevin import exact_observables, action

CHAINS = 128
T_THERM, T_MEAS = 8.0, 40.0


def fd_gate():
    print("=" * 96)
    print("GATE 1  analytic drift against a central finite difference of the action, both channels")
    for chan in ("spin", "charge"):
        m = Model(N=4, t=1.0, t2=0.7, mu=0.6, U=4.0, dtau=0.1, L=6)
        ch = CLangevin(m, seed=3, channel=chan, chains=3)
        rng = np.random.default_rng(11)
        ch.X = (rng.standard_normal((3, m.L, m.N))
                + 1j * 0.3 * rng.standard_normal((3, m.L, m.N)))
        k = ch.drift()
        h, worst = 1e-6, 0.0
        for c in (0, 2):
            for l in (0, 3, m.L - 1):
                for i in (0, m.N - 1):
                    Xp = ch.X.copy(); Xp[c, l, i] += h
                    Xm = ch.X.copy(); Xm[c, l, i] -= h
                    worst = max(worst, abs(-(action(ch, Xp)[c] - action(ch, Xm)[c])
                                           / (2 * h) - k[c, l, i]))
        bad = -ch.X + sum(ch._coupling(s) * np.einsum("clii->cli", ch.greens()[s])
                          for s in (+1, -1))
        print(f"  {chan:>7}  worst {worst:.2e}  {'PASS' if worst < 1e-5 else 'FAIL'}"
              f"   negative control (G_ii for 1 - G_ii) {abs(bad[0,0,0]-k[0,0,0]):.2e}")


def extrap(rows):
    x = np.array([r[0] for r in rows]); y = np.array([r[1] for r in rows])
    w = np.array([max(r[2], 1e-9) for r in rows])
    A = np.vstack([np.ones(len(x)), x]).T
    W = np.diag(1.0 / w ** 2)
    cov = np.linalg.inv(A.T @ W @ A)
    c = cov @ A.T @ W @ y
    return float(c[0]), float(np.sqrt(cov[0, 0]))


def control_gate():
    print()
    print("=" * 96)
    print("GATE 2  SPIN channel, half filling, t2 = 0: sign-free, so it must be EXACT")
    print(f"{'U':>5} {'eps':>9} {'n exact':>9} {'n CL':>18} {'d exact':>9} "
          f"{'d CL':>10} {'+-':>8} {'d-exact':>9} {'|Im x|':>7}")
    for U in (2.0, 4.0):
        m = Model(N=4, t=1.0, t2=0.0, mu=0.0, U=U, dtau=0.1, L=10)
        ne, de = exact_observables(m)
        rows = []
        for eps in (4e-3, 2e-3, 1e-3):
            r = run_cl(m, t_therm=T_THERM, t_meas=T_MEAS, n_meas=200, eps=eps, seed=1,
                       adaptive=False, channel="spin", chains=CHAINS)
            rows.append((eps, r["docc"].real, r["docc_se"].real))
            print(f"{U:5.1f} {eps:9.2e} {ne:9.5f} {r['n'].real:+11.5f}{r['n'].imag:+6.3f}i "
                  f"{de:9.5f} {r['docc'].real:10.5f} {r['docc_se'].real:8.5f} "
                  f"{r['docc'].real-de:+9.5f} {r['im_frac']:7.4f}", flush=True)
        v, s = extrap(rows)
        z = abs(v - de) / max(s, 1e-12)
        print(f"      eps -> 0 : {v:.5f} +- {s:.5f}   exact {de:.5f}   "
              f"diff {v-de:+.2e}   z = {z:.2f}", flush=True)


def charge_gate():
    print()
    print("=" * 96)
    print("GATE 3  CHARGE channel: the action is complex, so the field must leave the real axis")
    print(f"{'mu':>5} {'t2':>5} {'U':>5} | {'n exact':>9} | {'n CL':>18} | {'d exact':>9} | "
          f"{'d CL':>18} | {'+-':>7} | {'|Im y|':>7} | {'maxdrift':>9}")
    for mu, t2, U in ((0.0, 0.0, 2.0), (0.6, 0.7, 4.0), (1.0, 0.0, 4.0)):
        m = Model(N=4, t=1.0, t2=t2, mu=mu, U=U, dtau=0.1, L=10)
        ne, de = exact_observables(m)
        r = run_cl(m, t_therm=T_THERM, t_meas=T_MEAS, n_meas=200, eps=1e-3, seed=1,
                   adaptive=True, channel="charge", chains=CHAINS)
        print(f"{mu:5.2f} {t2:5.2f} {U:5.1f} | {ne:9.5f} | {r['n'].real:+9.5f}"
              f"{r['n'].imag:+8.5f}i | {de:9.5f} | {r['docc'].real:+9.5f}"
              f"{r['docc'].imag:+8.5f}i | {r['docc_se'].real:7.5f} | "
              f"{r['im_frac']:7.4f} | {r['max_drift']:9.2e}", flush=True)
    print()
    print("|Im y| = 0 would mean the field never complexified and the run is not this method.")
    print("A converged imaginary part in an observable that must be real, or a real part missing")
    print("exact by many error bars, is the documented failure -- gates 1 and 2 are what")
    print("separate that from an implementation defect.")


if __name__ == "__main__":
    fd_gate()
    control_gate()
    charge_gate()
