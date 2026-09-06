"""Gate: at k = 1 the multi-determinant walker must BE the single-determinant walker.

With one determinant and coefficient 1 the sum collapses, the spins factorise again, and the
ratio `sum_j w_j (1 + d_up G_j^up)(1 + d_dn G_j^dn) / sum_j w_j` must reduce exactly to
`1 + d G[i,i]` per spin.  Both classes draw one uniform per site in the same order, so at one
walker with the same seed the two must produce the SAME trajectory, not merely agree on average.

The negative control reinstates the finite-temperature Green's convention -- the bug that cost
this rig a day -- so the gate is shown to be able to fail.
"""
import numpy as np
from model2d import Model2D
from sector_ed import ground_energy
from cpmc_fast import FastCPMC
from cpmc_multi import MultiCPMC
from trial import free_trial

if __name__ == "__main__":
    m = Model2D(Lx=2, Ly=4, t=1.0, mu=0.0, U=4.0, dtau=0.05, L=1, theta=0.0)
    nu = nd = 3
    pf = free_trial(m.K, nu, nd)
    steps = 40
    print("IDENTITY  one walker, same seed, k = 1 against the single-determinant walker")
    print(f"{'branch':>14} {'max |diff|':>13} {'last single':>14} {'last multi':>14}")
    worst = 0.0
    for name, con in (("constrained", True), ("free projection", False)):
        a = FastCPMC(m, nu, nd, n_walkers=1, seed=7, constrained=con, psi_t=pf)
        b = MultiCPMC(m, nu, nd, [pf], [1.0], n_walkers=1, seed=7, constrained=con)
        ea, eb = [], []
        for _ in range(steps):
            a.step(); b.step()
            ea.append(a.energy()); eb.append(b.energy())
        d = float(np.abs(np.array(ea) - np.array(eb)).max())
        worst = max(worst, d)
        print(f"{name:>14} {d:13.3e} {ea[-1]:+14.9f} {eb[-1]:+14.9f}")
    print(f"\nworst over both branches: {worst:.3e}   {'PASS' if worst < 1e-9 else 'FAIL'}")

    print()
    print("NEGATIVE CONTROL  the gate must be able to fail")
    class Wrong(MultiCPMC):
        def _dets_and_greens(self, ph):
            D, G = super()._dets_and_greens(ph)
            I = np.eye(ph[+1].shape[1])
            return D, {s: I[None, None] - G[s] for s in (+1, -1)}
    a = FastCPMC(m, nu, nd, n_walkers=1, seed=7, psi_t=pf)
    b = Wrong(m, nu, nd, [pf], [1.0], n_walkers=1, seed=7)
    ea, eb = [], []
    for _ in range(steps):
        a.step(); b.step()
        ea.append(a.energy()); eb.append(b.energy())
    print(f"  with G -> I - G the same comparison gives "
          f"{float(np.abs(np.array(ea)-np.array(eb)).max()):.3e}")

    print()
    print("U = 0 EXACTNESS  and the constraint must kill nothing")
    m0 = Model2D(Lx=2, Ly=4, t=1.0, mu=0.0, U=0.0, dtau=0.05, L=1, theta=0.0)
    ex0 = ground_energy(m0.K, 0.0, nu, nd)
    p0 = free_trial(m0.K, nu, nd)
    ch = MultiCPMC(m0, nu, nd, [p0], [1.0], n_walkers=200, seed=1)
    es = []
    for t in range(220):
        ch.step()
        if (t + 1) % 5 == 0: ch.orthonormalise()
        if (t + 1) % 10 == 0: ch.population_control(200)
        if t >= 120: es.append(ch.energy())
    print(f"  k=1  {np.mean(es):+.12f}  exact {ex0:+.12f}  diff {np.mean(es)-ex0:+.2e}  "
          f"killed {ch.killed}")
