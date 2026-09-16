"""Experiment U -- the constrained-path bias, measured where the sign problem is real.

2x4 at (3,3): eight sites, six electrons, doped off half filling, and a CLOSED shell so the
trial wavefunction is well defined -- which matters by a factor of 300 (measured on 2x2: an
open shell moved the free-projection energy by 0.77 where a closed shell moved it by 0.003).
Sector dimension 3136, so the exact ground state is available.

Reported side by side:
  free projection   carries the sign; exact in principle, and its average sign shows the cost
  constrained path  kills walkers that cross the node; NO sign problem, and a bias instead
against the exact ground state.  The question is not whether CPMC is biased -- it is -- but
whether the bias is small enough to be worth the exponential it removes.
"""
import numpy as np, time, json
from model2d import Model2D
from sector_ed import ground_energy
from cpmc import CPMC


class Free(CPMC):
    """Same walker; the sign is carried instead of clipped."""
    def step(self):
        m = self.m; alive = []
        for k, phi in enumerate(self.walk):
            ph = {s: self.expK_half @ phi[s] for s in (+1, -1)}
            G = {s: self._green(ph[s], s) for s in (+1, -1)}
            weight = self.w[k]
            for i in range(m.N):
                rs = {s: {x: 1.0 + (np.exp(s*self.lam*x)-1.0)*G[s][i, i]
                          for x in (1.0, -1.0)} for s in (+1, -1)}
                rp = rs[+1][1.0]*rs[-1][1.0]; rm = rs[+1][-1.0]*rs[-1][-1.0]
                tot = abs(rp) + abs(rm)
                if tot <= 0: break
                weight *= 0.5*tot
                x = 1.0 if self.rng.random() < abs(rp)/tot else -1.0
                weight *= np.sign(rp if x > 0 else rm)
                for s in (+1, -1):
                    d = np.exp(s*self.lam*x)-1.0; R = rs[s][x]
                    ph[s][i, :] *= (1.0+d)
                    col = -G[s][:, i].copy(); col[i] += 1.0
                    G[s] = G[s] + (d/R)*np.outer(col, G[s][i, :])
            ph = {s: self.expK_half @ ph[s] for s in (+1, -1)}
            alive.append((ph, weight))
        self.walk = [a[0] for a in alive]; self.w = np.array([a[1] for a in alive])
        # Rescale by the mean magnitude.  Both estimators here are RATIOS of weight sums, so a
        # common factor cancels exactly; without it the weights overflow float64 well before
        # beta = 16 and the sign decay cannot be measured at all.
        s = np.abs(self.w).mean()
        if s > 0: self.w = self.w / s
        return len(alive)
    def population_control(self, target):
        pass                                    # signed weights: comb resampling is invalid
    def mean_sign(self):
        return float(np.sum(self.w) / np.sum(np.abs(self.w)))


def run(cls, m, nup, ndn, steps, nw, seed, meas_from):
    ch = cls(m, nup, ndn, n_walkers=nw, seed=seed)
    es, sg = [], []
    for t in range(steps):
        ch.step()
        if (t+1) % 5 == 0: ch.orthonormalise()
        if (t+1) % 10 == 0: ch.population_control(nw)
        if t >= meas_from:
            es.append(ch.energy()); sg.append(float(np.sum(ch.w)/np.sum(np.abs(ch.w))))
    return np.array(es), np.array(sg), ch.killed


if __name__ == "__main__":
    Lx, Ly, nup, ndn = 2, 4, 3, 3
    steps, nw, mfrom = 140, 240, 100
    rows = []
    print(f"{Lx}x{Ly} lattice, (n_up,n_dn)=({nup},{ndn}) -- doped, closed shell, "
          f"sector dim {3136}")
    print(f"\n{'U':>5} {'exact GS':>11} {'free proj':>20} {'<sgn> free':>11} "
          f"{'CPMC':>19} {'CP bias':>9} {'rel':>7} {'s':>5}")
    for U in (2.0, 4.0, 6.0, 8.0):
        m = Model2D(Lx=Lx, Ly=Ly, t=1.0, mu=0.0, U=U, dtau=0.05, L=1, theta=0.0)
        ex = ground_energy(m.K, U, nup, ndn)
        t0 = time.time()
        ef, sf, _ = run(Free, m, nup, ndn, steps, nw, 4, mfrom)
        ec, sc, killed = run(CPMC, m, nup, ndn, steps, nw, 4, mfrom)
        e_f, se_f = ef.mean(), ef.std()/np.sqrt(len(ef))
        e_c, se_c = ec.mean(), ec.std()/np.sqrt(len(ec))
        rows.append(dict(U=U, exact=ex, free=float(e_f), free_se=float(se_f),
                         sgn=float(sf.mean()), cp=float(e_c), cp_se=float(se_c),
                         bias=float(e_c-ex), killed=int(killed)))
        print(f"{U:5.1f} {ex:+11.5f} {e_f:+12.5f}+-{se_f:.5f} {sf.mean():11.4f} "
              f"{e_c:+11.5f}+-{se_c:.5f} {e_c-ex:+9.5f} {abs(e_c-ex)/abs(ex):7.4f} "
              f"{time.time()-t0:5.0f}", flush=True)
        json.dump(rows, open("resultsU.json", "w"), indent=1)
    print("\nCPMC has no sign to average -- every weight is positive by construction.")
    print("What it pays instead is the bias column, and that is the number that decides")
    print("whether trading the exponential for it is a good deal.")
