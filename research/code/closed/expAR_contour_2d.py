"""Experiment AR -- a constant imaginary shift buys nothing in two dimensions.

Contour deformation is the one route that can change the EXPONENT rather than the prefactor: shift
the integration contour into the complex plane and the phase of the weight may become milder. In
one dimension a constant shift does this, and by a lot -- a mean phase of `0.0143` undeformed.

The question is whether it transfers. This file runs the same deformation on a two-dimensional
lattice at a point where the spin channel HAS a phase problem, so there is something for a shift to
fix, and scans the shift amplitude.

Two shift shapes are tried, both deterministic and neither fitted:

    uniform     the same `c` on every (slice, site) -- the 1-D deformation, carried over
    staggered   `+c` and `-c` by sublattice -- the 2-D Hubbard model's own structure, which a
                uniform shift cannot see

The undeformed value is measured in the same run and is the only thing the gain is scored against;
nothing here is compared to a figure from another rig.

WHY THE UNDEFORMED POINT MATTERS, AND HOW THE CASE WAS CHOSEN.  A shift can only help where there
is a phase problem to help with.  Run at a point whose mean phase is already ~1, any shift appears
to "do nothing" for a trivial reason and the scan says nothing at all.  That is not hypothetical:
`4x4` at `mu = 0` is half filling on a bipartite lattice, which section 5 shows is sign-free, and a
scan there returns a perfect undeformed value that every shift can only spoil.

So the point is chosen by MEASURING the undeformed phase first.  Of the cases probed, `4x4` at
`mu = 1.0, U = 6, beta = 6` sits at `0.50`, and it is the one used below.  The undeformed value is
re-measured in the same run and printed above the scan, so a reader can see there was something to
fix before reading whether anything fixed it.

The BASELINE is measured at twice the statistics of the scan points, deliberately.  The claim being
tested is that no shift improves on it, so the baseline is the number a gain would have to beat and
it is the one that must be tight; a scan point only has to be resolved well enough to show it is
not above.
"""
from __future__ import annotations

import numpy as np

from contour2d import run_contour
from model2d import Model2D


def block_se(x, min_blocks=24):
    """Blocked standard error: the largest over block sizes, so autocorrelation cannot hide."""
    best = 0.0
    for b in (1, 2, 4, 8, 16, 32):
        k = len(x) // b
        if k < min_blocks:
            break
        mm = x[:k * b].reshape(k, b).mean(1)
        best = max(best, float(np.abs(mm).std(ddof=1) / np.sqrt(k)))
    return best


def shift_field(m, c, kind):
    """The deformation, as a (L, N) field. `staggered` alternates by sublattice."""
    if kind == "uniform":
        return np.full((m.L, m.N), c)
    sgn = np.array([(-1.0) ** ((i // m.Ly) + (i % m.Ly)) for i in range(m.N)])
    return np.tile(c * sgn, (m.L, 1))


def phase_at(m, c, kind, n_meas=300, warm=100, seed=9):
    """(|<phase>|, blocked se, acceptance) at one shift amplitude."""
    p, acc = run_contour(m, shift_field(m, c, kind), n_meas=n_meas, warm=warm, seed=seed)
    return float(abs(p.mean())), block_se(p), float(acc)


if __name__ == "__main__":
    m = Model2D(Lx=4, Ly=4, t=1.0, mu=1.0, U=6.0, dtau=0.2, L=30, theta=0.0)
    print("=" * 92)
    print(f"CONTOUR DEFORMATION IN TWO DIMENSIONS.  4x4, mu = 1.0, U = 6, "
          f"beta = {0.2 * 30}, dtau = 0.2")
    print("The gain is measured against this run's OWN undeformed value, not a quoted one.")
    print()
    base, bse, bacc = phase_at(m, 0.0, "uniform", n_meas=600, warm=150)
    print(f"undeformed  |<phase>| = {base:.4f} +- {bse:.4f}   acceptance {bacc:.3f}",
          flush=True)
    print()
    print(f"{'shape':>10} {'c':>6} | {'|<phase>|':>18} {'gain':>7} {'acc':>7}")
    for kind in ("uniform", "staggered"):
        for c in (0.1, 0.35, 1.0, 4.0):
            v, se, acc = phase_at(m, c, kind, n_meas=300, warm=150)
            print(f"{kind:>10} {c:6.2f} | {v:10.4f} +- {se:.4f} {v / base:7.2f} {acc:7.3f}",
                  flush=True)
    print()
    print("A gain above 1 would mean the deformation mitigated the phase problem. The scan runs")
    print("to c = 4, far past any plausible optimum, because a thimble need not lie near the real")
    print("axis: a second regime at large shift is what would falsify the reading, and its absence")
    print("is what makes the closure a measurement rather than a failure to look.")
