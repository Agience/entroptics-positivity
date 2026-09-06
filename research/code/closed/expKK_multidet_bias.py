"""Experiment KK -- does the overlap gain actually convert into a smaller constrained-path bias?

`expGG` established the headroom: four non-orthogonal determinants beat sixteen fixed-basis ones
at every U, with the advantage growing with coupling.  But overlap is not the quantity anyone
cares about -- the BIAS is, and the two are related only through the node.  A trial with a better
overlap has a different node, not necessarily a better one, and Part 5 already produced one case
where those came apart badly: the variational energy prefers a symmetry-broken determinant whose
overlap and bias are both worse.

So this runs the actual walk.  `cpmc_multi.py` carries the multi-determinant trial (gated
bit-identical to the single-determinant walker at k = 1, worst 7.1e-15, negative control 32.95),
and the bias is scored against exact diagonalisation at each k.

THIS IS A CEILING, NOT A METHOD, and the distinction is the whole reason to measure it first.
The determinants are fitted to the exact ground state, which nobody has.  What the sweep answers
is whether solving the real problem -- producing good determinants WITHOUT the answer -- is worth
attempting, by showing how far the bias would fall if it were solved perfectly.  A bias that
barely moves as the overlap climbs would close the route for good; one that falls with the
overlap makes trial-wavefunction construction the thing to work on.
"""
from __future__ import annotations

import numpy as np

from model2d import Model2D
from sector_ed import ground_energy
from cpmc_multi import run_multi
from expGG_nonorthogonal import best_k_dets


if __name__ == "__main__":
    Lx, Ly, nu, nd = 2, 4, 3, 3
    beta, dtau = 8.0, 0.05
    ks = [1, 2, 4, 6]
    print("=" * 104)
    print(f"DOES A BETTER OVERLAP BUY A SMALLER BIAS?  {Lx}x{Ly} at ({nu},{nd}), "
          f"beta = {beta}, dtau = {dtau}")
    print("The determinants are fitted to the exact ground state: this is the CEILING of what a")
    print("multi-determinant trial could buy, not a method that could be run without the answer.")
    print()
    print(f"{'U':>5} {'k':>3} {'overlap':>9} {'CPMC':>19} {'exact':>11} {'bias':>10} "
          f"{'rel':>9} {'vs k=1':>9} {'killed':>7}")
    for U in (4.0, 8.0, 12.0):
        m = Model2D(Lx=Lx, Ly=Ly, t=1.0, mu=0.0, U=U, dtau=dtau, L=1, theta=0.0)
        ex, g = ground_energy(m.K, U, nu, nd, want_vec=True)
        base = None
        for k in ks:
            ov, dets, c = best_k_dets(m.K, U, nu, nd, g, k, n_restarts=4, seed=1)
            rs = [run_multi(m, nu, nd, dets, c, beta, n_walkers=400, seed=s, n_meas=150)
                  for s in (1, 2, 3)]
            e = np.array([r["e"] for r in rs])
            se = float(e.std(ddof=1) / np.sqrt(len(rs)))
            b = float(e.mean()) - ex
            if base is None:
                base = abs(b)
            ratio = f"{abs(b)/max(base,1e-12):9.3f}" if base else f"{'--':>9}"
            print(f"{U:5.1f} {k:3d} {ov:9.5f} {e.mean():+12.5f}+-{se:.5f} {ex:+11.5f} "
                  f"{b:+10.5f} {abs(b)/abs(ex):9.5f} {ratio} "
                  f"{sum(r['killed'] for r in rs):7d}", flush=True)
        print()
    print("'vs k=1' is the bias relative to the single-determinant walk at the same U -- the")
    print("number that says whether the overlap gain converted.  If it falls with k, the")
    print("constrained-path bias is a solvable problem and the work is in building the trial.")
