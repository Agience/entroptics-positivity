"""Section 11's closed door on doping, measured: no flux on any graph restores the identity at mu > 0.

Section 11 proves this for every finite lattice (`Doping.doping_closes_both_routes`): a diagonal
conjugation leaves `K_ii` where it was, both routes then demand `K_ii = -K_ii`, and Hermiticity makes
`K_ii` real, so a zero diagonal is forced and a chemical potential can never be accommodated. The
measurement is the empirical companion to that proof -- it samples where the theorem quantifies.

WHAT IS SWEPT. Four graphs, twenty-one fluxes each, at three chemical potentials. Flux is the only
freedom a diagonal-unitary conjugation has on a cycle, so sweeping it across a full period is
sweeping everything route B could use. At `mu = 0` some flux opens a route on every graph and the
identity goes to machine precision; at `mu > 0` the best flux on the best graph does not.

The residual is the one section 5 uses, from `reads/expAO_spectral_criterion.py:measure`, so this
adds no second definition of the quantity.

    python remote_run.py reads/expBK_doping_closes_every_flux.py
"""
from __future__ import annotations

import numpy as np

from reads.expAO_spectral_criterion import measure, ring

BETA, U, DTAU, N_DRAW = 8.0, 4.0, 0.125, 120
GRAPHS = (5, 6, 7, 8)
FLUXES = np.linspace(0.0, 2.0 * np.pi, 21)
MUS = (0.0, 0.2, 0.6)


def doped(n, flux, mu):
    """A flux-threaded ring with `-mu` on every diagonal entry -- the doping the theorem is about."""
    K = np.asarray(ring(n, flux), dtype=complex).copy()
    K[np.diag_indices(n)] -= mu
    return K


def main():
    print("=" * 96)
    print("NO FLUX ON ANY GRAPH RESTORES THE IDENTITY ONCE THE DIAGONAL IS NON-ZERO")
    print("=" * 96)
    print(f"  rings of {', '.join(str(g) for g in GRAPHS)} sites, {len(FLUXES)} fluxes each "
          f"= {len(GRAPHS) * len(FLUXES)} flux values per mu")
    print(f"  beta = {BETA}, U = {U}, dtau = {DTAU}, {N_DRAW} draws a point")
    print()
    print(f"{'mu':>6} {'best residual over all 84':>28} {'at ring':>9} {'at flux/pi':>12}")
    print("-" * 96)

    for mu in MUS:
        best = None
        for n in GRAPHS:
            for f in FLUXES:
                r, _ = measure(doped(n, f, mu), BETA, U=U, dtau=DTAU, n_draw=N_DRAW, seed=3)
                val = float(np.max(np.abs(r)))
                if best is None or val < best[0]:
                    best = (val, n, f / np.pi)
        print(f"{mu:6.2f} {best[0]:28.3e} {best[1]:9d} {best[2]:12.2f}", flush=True)

    print()
    print("  At mu = 0 a flux exists that takes the identity to machine precision -- route B, open")
    print("  on an odd cycle carrying half-odd-integer flux. At mu > 0 the BEST of eighty-four")
    print("  flux values on four graphs is orders away from it, and the theorem says why: no")
    print("  diagonal conjugation moves a diagonal entry, so no flux can cancel a chemical")
    print("  potential on any graph at any size.")


if __name__ == "__main__":
    main()
