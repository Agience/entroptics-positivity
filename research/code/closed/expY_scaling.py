"""Experiment Y -- price the trade on the axis that drives the cost: lattice size.

Everything else measured here lives on 2x4, where exact diagonalisation reaches and therefore
where the sign problem is mild.  That is the right place to measure the BIAS and the wrong place
to measure what the bias buys.  Here the lattice grows past ED and the two branches run side by
side on the same fillings, the same beta, the same budget:

  free projection   exact, carries the sign, branches on |w|.  Its cost is what the sign does.
  constrained path  no sign at all.  Its cost is the walk.

No exact energy exists above 2x4, so nothing here is scored against truth.  What IS quoted is the
COST of the exact method against the cost of the biased one, which needs no ED.  Nothing here
bounds the bias: free projection's own error bar reaches +-3.9 where the bias measured against ED
at 2x4 is 0.004, so |free - CPMC| is dominated by free projection's noise.  A two-seed pass
suggested otherwise and was a fluke of two seeds.

AND THIS IS NOT A CLEAN SIZE SCAN.  The closed-shell requirement fixes which fillings exist at
each size and they differ -- 0.500, 0.750, 0.833, 0.889 -- while filling matters more than size,
because the sign problem vanishes exactly at half filling on a bipartite lattice and is worst some
way below it.  4x4 at 0.750 is the severe point here and 6x6 at 0.889 is milder despite being
larger.  Reading the table as "severity grows with N" reads the filling column as if it were
constant.

WHY THERE IS NO <sgn> COLUMN.  Two ways of measuring it were tried and both are wrong, in
opposite directions.  WITH branching, resampling resets every weight to +-1, so Sum(w)/Sum(|w|)
afterwards is the sign accumulated since the last resample -- ten steps -- and reads as
NON-DECAYING (4x6 fitted <sgn> ~ exp(+0.054 beta), growing with beta, which is not a thing).
WITHOUT branching, a handful of walkers acquire enormous |w| and dominate both sums, so the
estimator saturates -- measured as 0.75, 0.62, 0.45, 0.55, 0.55 across beta = 2 to 12, flat after
beta = 6.  That is not a fixable estimator: getting the average sign right in ground-state free
projection is itself as hard as the sign problem, which is Part 1's result arriving in a second
form.

So severity is reported by two quantities that need no sign estimator:

  the ERROR BAR on the free-projection energy at fixed budget, and
  `dead`, the fraction of measurements where Sum(w) is non-positive or non-finite -- total
  cancellation, at which point the signed estimator has no denominator and the method has simply
  stopped returning a number.

Fillings are CLOSED shells, doped off half filling, since an open shell makes Psi_T ambiguous
(measured on 2x2 as a factor of 300 in the error).  Anti-periodic boundaries in y, because under
fully periodic ones the closed shells below half filling sit far from it -- 4x4 gives n = 5, a
filling of 0.625, where there is barely a sign problem -- while anti-periodic gives n = 6 and
n = 10 at 4x4 and 4x6, fillings of 0.75 and 0.83, which is the severe region.

U MATTERS AS MUCH AS SIZE.  A first version ran at U = 4 and found 4x4 essentially sign-free out
to beta = 8.  A scaling study at a coupling where nothing decays measures nothing.
"""
from __future__ import annotations

import time
import numpy as np

from model2d import Model2D
from sector_ed import closed_shells
from cpmc_fast import run


def pick_filling(K):
    """The largest closed shell strictly below half filling -- doped, and unambiguous."""
    N = K.shape[0]
    cs = [n for n in closed_shells(K) if 0 < n < N // 2]
    return max(cs) if cs else None


def both(m, n, beta, n_walkers=400, seeds=(1, 2, 3)):
    out = {}
    for name, con in (("free", False), ("cpmc", True)):
        t0 = time.time()
        rs = [run(m, n, n, beta, constrained=con, n_walkers=n_walkers, seed=s, n_meas=150)
              for s in seeds]
        e = np.array([r["e"] for r in rs])
        ok = np.isfinite(e)
        out[name] = dict(
            e=float(e[ok].mean()) if ok.any() else float("nan"),
            se=float(e[ok].std(ddof=1) / np.sqrt(ok.sum())) if ok.sum() > 1 else float("nan"),
            dead=float(np.mean([r["dead_frac"] for r in rs])),
            secs=(time.time() - t0) / len(seeds))
    return out


def fmt(d):
    if not np.isfinite(d["e"]):
        return f"{'no number':>19}"
    if not np.isfinite(d["se"]):
        return f"{d['e']:+12.5f}+-{'--':>6}"
    return f"{d['e']:+12.5f}+-{d['se']:.5f}"


if __name__ == "__main__":
    betas = [2.0, 4.0, 6.0, 8.0, 12.0]
    lattices = [(2, 4), (4, 4), (4, 6), (6, 6)]
    U, dtau = 8.0, 0.05

    print("=" * 106)
    print(f"SCALING  U = {U}, dtau = {dtau}, 400 walkers, 3 seeds, closed shell below half filling")
    print(f"{'lattice':>8} {'N':>3} {'n':>3} {'fill':>5} {'beta':>5} "
          f"{'free E':>19} {'dead':>6} {'CPMC E':>19} {'dead':>6} {'|gap|':>8} {'s/run':>7}")
    for Lx, Ly in lattices:
        m = Model2D(Lx=Lx, Ly=Ly, t=1.0, mu=0.0, U=U, dtau=dtau, L=1, theta=0.0, apbc=True)
        n = pick_filling(m.K)
        if n is None:
            print(f"{f'{Lx}x{Ly}':>8}  no closed shell below half filling")
            continue
        for b in betas:
            r = both(m, n, b)
            gap = abs(r["free"]["e"] - r["cpmc"]["e"])
            gs = f"{gap:8.4f}" if np.isfinite(gap) else f"{'--':>8}"
            print(f"{f'{Lx}x{Ly}':>8} {m.N:3d} {n:3d} {2*n/m.N:5.3f} {b:5.1f} "
                  f"{fmt(r['free'])} {r['free']['dead']:6.2f} {fmt(r['cpmc'])} "
                  f"{r['cpmc']['dead']:6.2f} {gs} {r['cpmc']['secs']:7.1f}", flush=True)
    print()
    print("'dead' is the fraction of measurements with Sum(w) non-positive or non-finite: total")
    print("cancellation, where the signed estimator has no denominator left.  It is 0 for CPMC by")
    print("construction -- no weight there is ever negative.")
    print("'|gap|' is |free - CPMC|.  It does NOT bound the constrained-path bias: free")
    print("projection's own error bar reaches +-3.9 where the bias measured against ED at 2x4 is")
    print("0.004, so the gap is dominated by free projection's noise and says nothing about the")
    print("bias.  An earlier two-seed pass suggested it did; that was a fluke of two seeds.")
