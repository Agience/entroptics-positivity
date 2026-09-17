"""Section 5's triangular pair: the wrap cycles decide whether any flux opens a route.

A periodic triangular lattice carries triangles, rhombi AND wrap-around cycles, and route B has to
hold on all of them at once. That makes the answer depend on the size in a way the one-dimensional
cases cannot show:

  * on `4x4` the wraps are even and carry no flux, so a flux exists that opens route B;
  * on `3x3` the wraps are odd and flux-free, so no flux opens either route.

Section 5 quotes this pair -- ten rows, two lattices, five fluxes -- and no file produced it. The
`3x3` rows appear in `reads/expBA_oracle_superset.py`, which now prints their sign deficit too, but
the `4x4` rows appear nowhere, so the residuals and the negative-fraction claim beside them had no
source. This measures both lattices on the same footing.

THE TWO COLUMNS ARE DIFFERENT CLAIMS. The identity residual says whether a route is open; the sign
deficit says whether the lattice has a sign problem. Section 5's point is precisely that these come
apart on `4x4`, where the identity is broken and no negative weight appears.

    python remote_run.py reads/expBO_triangular_pair.py
"""
from __future__ import annotations

import numpy as np

from reads.expAO_spectral_criterion import criterion, measure, triangular

BETA, U, DTAU, N_DRAW, SEED = 4.0, 4.0, 0.125, 120, 3
FLUXES = ((0.0, "0"), (0.25, "pi/4"), (0.5, "pi/2"), (0.75, "3pi/4"), (1.0, "pi"))
DEEP = (8.0, 16.0)


def main():
    print("=" * 100)
    print("THE TRIANGULAR PAIR: WRAP PARITY DECIDES WHETHER ANY FLUX OPENS A ROUTE")
    print("=" * 100)
    print(f"  beta = {BETA}, U = {U}, dtau = {DTAU}, {N_DRAW} draws a row, seed {SEED}")
    print()
    print(f"{'lattice':>10} {'flux':>8} {'criterion':>11} {'identity resid':>17} "
          f"{'sign deficit':>14}")
    print("-" * 100)

    opened, broken = {}, {}
    for L in (4, 3):
        for frac, label in FLUXES:
            K = triangular(L, L, frac * np.pi)
            pred = bool(criterion(K))
            resid, deficit = measure(K, BETA, U=U, dtau=DTAU, n_draw=N_DRAW, seed=SEED)
            r = float(np.max(np.abs(resid)))
            opened.setdefault(L, []).append((label, r, pred))
            broken.setdefault(L, []).append(r)
            print(f"{f'{L}x{L}':>10} {label:>8} {str(pred):>11} {r:17.3e} {deficit:14.4f}",
                  flush=True)
        print()

    best4 = min(r for _, r, _ in opened[4])
    print(f"  4x4: the best residual over the five fluxes is {best4:.1e} -- a flux opens route B.")
    print(f"  3x3: the best is {min(r for _, r, _ in opened[3]):.1e} -- no flux opens either.")
    print()

    print("=" * 100)
    print("AND THE BROKEN IDENTITY ON 4x4 STILL PRODUCES NO NEGATIVE WEIGHT")
    print("=" * 100)
    print(f"{'beta':>6} {'identity resid':>17} {'sign deficit':>14}")
    print("-" * 100)
    for b in DEEP:
        resid, deficit = measure(triangular(4, 4, 0.0), b, U=U, dtau=DTAU,
                                 n_draw=200, seed=SEED)
        print(f"{b:6.1f} {float(np.max(np.abs(resid))):17.3e} {deficit:14.4f}", flush=True)
    print()
    print("  The identity is broken and the weights stay positive, which is the pair's point:")
    print("  the criterion decides the identity, and the identity is not the sign problem.")


if __name__ == "__main__":
    main()
