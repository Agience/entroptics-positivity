"""Experiment MM -- a self-consistent multi-determinant trial, built entirely without the answer.

The route, and why it is this one.  `expKK` showed the prize: four to six non-orthogonal
determinants cut the constrained-path bias by 3 to 9x.  Every determinant that achieved it was
fitted to the exact ground state.  Two honest generators were then measured by their variational
bound, which costs nothing and needs no exact answer:

    free determinant                        0.000 of the correlation energy
    free + ALL single excitations   k=31    0.000   <- Brillouin: at this filling the uniform
                                                       Hartree-Fock potential is a constant shift,
                                                       so the free determinant IS the HF solution
                                                       and singles do not couple to it
    symmetry-projected mean fields  k=9     0.273   and SATURATED -- 7 and 9 determinants gave
                                                       0.272 and 0.273
    CPMC WALKERS                    k=8     0.342   and still climbing: 0.508 at 16, 0.660 at 32

So the walkers win, and they should: imaginary-time projection has already shaped them toward the
ground state, and they are non-orthogonal, which is the character `expAA` showed a fixed
orthonormal basis cannot buy cheaply.

The loop.  Run CPMC with whatever trial is current, take walkers from the equilibrated ensemble,
solve H c = E S c in their span, use that as the next trial.  Repeat.

NOTHING IS FITTED, TUNED, OR THRESHOLDED.

  * The coefficients come from the Hamiltonian's own generalised eigenproblem.
  * The walkers are taken AFTER population control, which makes them weight-equal, so taking the
    first k is unbiased sampling and not a selection criterion.
  * The only threshold is the numerical rank of the overlap matrix, taken as the arithmetic's
    resolution (eps * k * lambda_max) rather than a chosen tolerance.
  * The variational bound is the gate and it needs no exact answer: it must never fall below the
    true ground energy, and if the loop is doing anything it should fall toward it.

The circularity is real and is what the bound is for: walkers are shaped by the current trial, so
a loop could in principle reinforce its own bias.  A variational energy cannot be fooled that way
-- it is an upper bound on the truth whatever produced the basis -- so it is reported at every
iteration alongside the bias.
"""
from __future__ import annotations

import numpy as np

from model2d import Model2D
from sector_ed import ground_energy
from trial import free_trial
from noci import noci
from cpmc_multi import MultiCPMC, run_multi
from expLL_honest_trial import distinct


def harvest(m, n_up, n_dn, dets, coeffs, k, *, n_walkers=64, beta=6.0, seed=0):
    """Equilibrate a walk with the given trial and return k walkers from it.

    Taken after population control, where the weights are equal, so the k returned are an
    unbiased sample of the walker distribution rather than a chosen subset.
    """
    ch = MultiCPMC(m, n_up, n_dn, dets, coeffs, n_walkers=n_walkers, seed=seed)
    for t in range(int(round(beta / m.dtau))):
        ch.step()
        if (t + 1) % 5 == 0:
            ch.orthonormalise()
        if (t + 1) % 10 == 0:
            ch.population_control(n_walkers)
    ch.orthonormalise()
    W = ch.phi[+1].shape[0]
    idx = np.linspace(0, W - 1, min(k, W)).astype(int)
    return [{s: ch.phi[s][i].copy() for s in (+1, -1)} for i in idx]


if __name__ == "__main__":
    Lx, Ly, nu, nd = 2, 4, 3, 3
    beta, dtau = 8.0, 0.05
    K_BASIS = 8
    N_ITER = 4
    print("=" * 112)
    print(f"SELF-CONSISTENT TRIAL, NO EXACT ANSWER USED  {Lx}x{Ly} at ({nu},{nd}), beta = {beta}")
    print(f"basis size {K_BASIS}, {N_ITER} iterations.  The variational bound is the gate: it can")
    print("never sit below the exact energy, whatever produced the basis.")
    for U in (4.0, 8.0, 12.0):
        m = Model2D(Lx=Lx, Ly=Ly, t=1.0, mu=0.0, U=U, dtau=dtau, L=1, theta=0.0)
        ex = ground_energy(m.K, U, nu, nd)
        free = free_trial(m.K, nu, nd)
        dets, c = [free], np.array([1.0])
        e0 = None
        print()
        print(f"U = {U}   exact {ex:+.5f}")
        print(f"{'iter':>5} {'k':>3} {'NOCI bound':>11} {'bound gap':>10} {'valid':>6} "
              f"{'CPMC':>19} {'bias':>10} {'vs free det':>12}")
        for it in range(N_ITER + 1):
            rs = [run_multi(m, nu, nd, dets, c, beta, n_walkers=400, seed=s, n_meas=150)
                  for s in (1, 2, 3)]
            e = np.array([r["e"] for r in rs])
            b = float(e.mean()) - ex
            if e0 is None:
                e0 = abs(b)
            eb, _ = noci(dets, m.K, U)
            valid = "--" if not np.isfinite(eb) else ("ok" if eb >= ex - 1e-9 else "BELOW")
            gap = f"{eb-ex:10.5f}" if np.isfinite(eb) else f"{'--':>10}"
            eb_s = f"{eb:+11.5f}" if np.isfinite(eb) else f"{'--':>11}"
            print(f"{it:5d} {len(dets):3d} {eb_s} {gap} {valid:>6} "
                  f"{e.mean():+12.5f}+-{e.std(ddof=1)/np.sqrt(3):.5f} {b:+10.5f} "
                  f"{abs(b)/e0:12.3f}", flush=True)
            if it == N_ITER:
                break
            w = harvest(m, nu, nd, dets, c, K_BASIS, seed=100 + it)
            dets = distinct(dets + w)[:K_BASIS]
            _, c = noci(dets, m.K, U)
    print()
    print("'vs free det' below 1 is the honest construction beating CPMC's default trial.")
    print("A bound marked BELOW would mean the matrix elements are wrong and the row means nothing.")
