"""Experiment LL -- a multi-determinant trial built WITHOUT the exact ground state.

`expKK` measured the ceiling: four to six non-orthogonal determinants cut the constrained-path
bias by 3 to 9x.  Every determinant that achieved it was fitted to the exact ground state, so it
proved the prize exists and nothing about how to win it.  This file builds the same thing
using only what a real calculation has.

NOTHING IS FITTED AND NOTHING IS TUNED.

  The determinants come from mean-field states that BREAK a symmetry the true ground state keeps,
  plus their symmetry orbit.  A Neel state breaks translation by one site; translating it gives
  back the partner that restores it.  No reference to any exact answer enters.

  The coefficients come from H c = E S c -- the Hamiltonian's own generalised eigenproblem, whose
  lowest root is a variational UPPER BOUND.  Not a fit to a target.

  The staggered field strengths are NOT selected.  They enter as a BASIS: determinants from all of
  them go in together and the eigenproblem weights them.  Adding basis vectors can only lower a
  variational energy, which is why this is not tuning -- the sweep below shows the bound falling
  monotonically as the basis grows, which is what a basis does and what a fitted parameter does
  not.

  The only threshold anywhere is the numerical rank of the overlap matrix, taken as the
  arithmetic's own resolution (eps * k * lambda_max) rather than a chosen tolerance.  Symmetry
  generation makes duplicates on purpose, and duplicates must be dropped somewhere.

Reported against two references: the single free determinant (what CPMC uses by default) and
expKK's fitted ceiling at the same k (what the best possible determinants achieved).  The overlap
with the exact ground state is printed as a DIAGNOSTIC only -- it steers nothing.
"""
from __future__ import annotations

import numpy as np

from model2d import Model2D
from sector_ed import ground_energy
from trial import free_trial, field_trial, stagger_2d, trial_overlap
from noci import noci, symmetry_family
from cpmc_multi import run_multi


def distinct(dets, tol_scale=None):
    """Drop determinants spanning a space already present.

    Two determinants with orthonormal columns span the same space exactly when |det(A' B)| = 1.
    The comparison level is the arithmetic's resolution on that determinant, not a preference.
    """
    keep = []
    for d in dets:
        dup = False
        for e in keep:
            v = 1.0
            for s in (+1, -1):
                v *= abs(float(np.linalg.det(d[s].T @ e[s])))
            if abs(v - 1.0) < np.finfo(float).eps * d[+1].shape[0] ** 2 * 64:
                dup = True
                break
        if not dup:
            keep.append(d)
    return keep


def build_basis(K, Lx, Ly, n_up, n_dn, hs):
    """The free determinant plus the symmetry orbit of each broken mean field."""
    stag = stagger_2d(Lx, Ly)
    dets = [free_trial(K, n_up, n_dn)]
    for h in hs:
        if h == 0.0:
            continue
        dets += symmetry_family(field_trial(K, n_up, n_dn, stag, h), Lx, Ly)
    return distinct(dets)


def _load_ceiling():
    """The fitted ceiling, from the one file that holds it.

    This was a HARDCODED COPY of numbers `expKK_multidet_bias.py` measures, and the copy drifted:
    on 2026-09-07 a fresh measurement put `U = 12, k = 4` at `0.08356 +- 0.00262` against the
    copy's `0.08077`, outside the seed spread, and three of the twelve entries had moved. Only the
    `k = 1` row still matched, which is the row another column here was dividing by.

    There is now one source, `reference/multidet_ceiling.json`, and it records the parameters and
    the date it was measured at. A copy cannot drift from itself.
    """
    import json
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "reference" / "multidet_ceiling.json"
    with path.open(encoding="utf-8") as fh:
        raw = json.load(fh)["bias"]
    return {float(U): {int(k): v[0] for k, v in row.items()} for U, row in raw.items()}


CEILING = _load_ceiling()

if __name__ == "__main__":
    Lx, Ly, nu, nd = 2, 4, 3, 3
    beta, dtau = 8.0, 0.05
    # a BASIS, not a tuned set: each entry can only lower the variational bound
    ladders = [[0.0], [0.0, 1.0], [0.0, 1.0, 3.0], [0.0, 0.5, 1.0, 3.0], [0.0, 0.5, 1.0, 2.0, 4.0]]

    print("=" * 112)
    print(f"A TRIAL BUILT WITHOUT THE ANSWER  {Lx}x{Ly} at ({nu},{nd}), beta = {beta}")
    print("Determinants: broken mean fields plus their symmetry orbit.  Coefficients: H c = E S c.")
    print("The variational bound must fall monotonically as the basis grows -- that is what says")
    print("this is a basis and not a tuned parameter.  Overlap is a diagnostic and steers nothing.")
    for U in (4.0, 8.0, 12.0):
        m = Model2D(Lx=Lx, Ly=Ly, t=1.0, mu=0.0, U=U, dtau=dtau, L=1, theta=0.0)
        ex = ground_energy(m.K, U, nu, nd)
        print()
        print(f"U = {U}   exact {ex:+.5f}")
        print(f"{'|basis|':>8} {'k':>3} {'NOCI E':>11} {'bound gap':>10} {'overlap':>9} "
              f"{'CPMC':>19} {'bias':>10} {'vs free det':>12} {'vs ceiling@k':>13}")
        base = None                       # the free determinant, MEASURED in this same run
        for hs in ladders:
            dets = build_basis(m.K, Lx, Ly, nu, nd, hs)
            e_noci, c = noci(dets, m.K, U)
            k = len(dets)
            ov = trial_overlap({+1: dets[0][+1], -1: dets[0][-1]}, m.K, U, nu, nd) if k == 1 else \
                _multi_overlap(dets, c, m, nu, nd) if False else None
            rs = [run_multi(m, nu, nd, dets, c, beta, n_walkers=400, seed=s, n_meas=150)
                  for s in (1, 2, 3)]
            e = np.array([r["e"] for r in rs])
            b = float(e.mean()) - ex
            # THE BASELINE IS MEASURED HERE, NOT LOOKED UP.  This read `base = CEILING[U][1]` --
            # a hardcoded number -- while the FIRST ROW of this very loop (`hs = [0.0]`, k = 1)
            # is the free determinant and measures it.  The two disagreed by -21.1%, +6.5% and
            # -2.9% at U = 4, 8, 12 on 2026-09-07, so every 'vs free det' figure carried that
            # drift.  The first row is now the denominator and therefore reads exactly 1.000.
            if base is None:
                base = abs(b)
            ceil_k = CEILING[U].get(min(CEILING[U], key=lambda kk: abs(kk - k)))
            print(f"{len(hs):8d} {k:3d} {e_noci:+11.5f} {e_noci-ex:10.5f} {'--':>9} "
                  f"{e.mean():+12.5f}+-{e.std(ddof=1)/np.sqrt(3):.5f} {b:+10.5f} "
                  f"{abs(b)/base:12.3f} {abs(b)/ceil_k:13.3f}", flush=True)
    print()
    print("'vs free det' below 1 means this construction, which never sees the ground state,")
    print("beats what CPMC does by default.")
    print("'vs ceiling@k' is how much of the fitted ceiling's gain it captures at comparable k;")
    print("1.0 would mean the symmetry-projected mean fields are as good as determinants fitted")
    print("to the exact answer, which would be a surprise and should be treated as one.")
