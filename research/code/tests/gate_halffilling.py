"""Gate: at half filling the constraint must do NOTHING, and CPMC must be exact.

On a bipartite lattice at n_up = n_dn = N/2 the spin-z Hirsch decoupling gives overlap ratios
that are positive for both branches of every field, so max(r, 0) = r identically: the clip never
fires, no walker is ever killed, and the constrained walk IS the free projection.  Free
projection is exact, so CPMC must reproduce exact diagonalisation to Trotter error.

This is the strongest available statement that the constraint is implemented as derived rather
than merely producing plausible numbers: a constraint that fired here would be wrong, and one
that silently changed the answer here would be wrong in a way no doped run can show, because
doped runs have no exact reference above 2x4.

BOUNDARY CONDITIONS ARE NOT A DETAIL HERE.  Under fully periodic boundaries half filling is an
OPEN shell on every bipartite lattice this rig reaches -- 2x2, 2x4, 2x6, 4x4 and 4x6 all
measured -- so Psi_T is ambiguous exactly where the gate needs it pinned, and the first version
of this file ran there and produced numbers that looked like a failing method (2x2: -8.37 and
-8.62 against an exact -9.65, with <sgn> reading 0.52 to 0.98 where the derivation says 1).
The two sizes whose half filling does close under PBC, 2x3 and 3x4, close it by having an odd
ring, which makes them non-bipartite and not sign-free either.  Anti-periodic boundaries in y
close the shell at 2x4, 4x4 and 2x6 and leave all three bipartite (verified: zero hopping
within a sublattice).
"""
from __future__ import annotations

import numpy as np

from model2d import Model2D
from sector_ed import ground_energy
from cpmc_fast import run

if __name__ == "__main__":
    print("HALF FILLING  the clip must never fire and CPMC must equal free projection")
    print(f"{'lattice':>8} {'n':>3} {'U':>5} {'dtau':>6} {'free':>12} {'CPMC':>12} "
          f"{'exact':>12} {'CPMC-free':>10} {'killed':>7} {'<sgn>':>8}")
    for (Lx, Ly) in ((2, 4),):
        for U in (4.0, 8.0, 12.0):
            for dtau in (0.05, 0.025):
                m = Model2D(Lx=Lx, Ly=Ly, t=1.0, mu=0.0, U=U, dtau=dtau, L=1, theta=0.0,
                            apbc=True)
                n = m.N // 2
                ex = ground_energy(m.K, U, n, n)
                f = run(m, n, n, 8.0, constrained=False, n_walkers=400, seed=1)
                c = run(m, n, n, 8.0, constrained=True, n_walkers=400, seed=1)
                print(f"{f'{Lx}x{Ly}a':>8} {n:3d} {U:5.1f} {dtau:6.3f} {f['e']:+12.5f} "
                      f"{c['e']:+12.5f} {ex:+12.5f} {c['e']-f['e']:+10.5f} "
                      f"{c['killed']:7d} {f['sgn']:8.5f}", flush=True)
    print()
    print("killed must be 0 and <sgn> must be 1 on every row: at half filling there is no sign")
    print("problem to remove, so any difference between the two columns is the implementation.")
