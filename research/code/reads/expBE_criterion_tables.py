"""Experiment BE -- section 5's criterion tables, printed.

WHY THIS FILE EXISTS.  Section 5 quotes four tables and attributes all of them to
`reads/expAO_spectral_criterion.py`.  That file supplies the machinery -- `signed_diagonal_conjugation`,
`route_A`, `route_B`, `chain`, `star`, `triangular_ladder`, `ring`, `tri_ladder`, `measure` -- and its
own `__main__` prints exactly ONE of the four: the spectral-asymmetry comparison that shows the
obvious criterion is the wrong one.  The other three were produced with the same functions and never
printed by anything.

That is the same defect `expBD` was written to fix for section 7's calibration table, found again in
section 5 and in three places instead of one.  The numbers were reproducible in the sense that the
functions are here; they were not reproducible in the sense the paper's own measurement note
requires, which is that running a named file prints them.

WHAT IT PRINTS, in the order section 5 quotes them:

  1. THE SIGNED-DIAGONAL CRITERION, on the full row set section 5 lists -- including the
     bond-disordered lattices, which are the rows that make the criterion predictive rather than
     descriptive.  No symmetry is designed into them: every bond strength is drawn at random.
  2. THE TOPOLOGY TABLE -- chains open and periodic, a star (a tree), a triangular ladder, even and
     odd rings.  The point of it is that the criterion is a statement about the hopping GRAPH and
     not about lattices, so the shapes here are ones the rest of the paper never uses.
  3. ROUTE A AGAINST ROUTE B, with the sign deficit beside the identity residual.  This is the one
     that carries section 5's sharpest claim: the two routes are not equivalent, and only route A
     leaves the weights real.  On the frustrated ladder threading the flux takes the identity from
     broken to exact WHILE the sign quality gets worse, which is what "the criterion decides the
     identity, not positivity" means when it is measured rather than asserted.

Every number the paper quotes in those three tables comes from here.  The fourth table -- the
spectral one -- stays where it is, in `expAO`'s own `__main__`, because that is the file whose
subject it is.

RUNTIME.  About four minutes: the row counts are small but each row samples 200 or 250 fields.
"""
from __future__ import annotations

import pathlib as _pathlib
import sys as _sys

# `expAO` is a sibling in `reads/`.  A bare `python reads/<this file>.py` puts `reads/` on the path
# and not `research/code`, so the shared rig would not import; the wrapper and the README both run
# from `research/code` with it on PYTHONPATH, and this makes the direct invocation work too.
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))

import numpy as np                                                    # noqa: E402

from reads.expAO_spectral_criterion import (                          # noqa: E402
    build, chain, measure, ring, route_A, route_B, signed_diagonal_conjugation, star,
    tri_ladder, triangular_ladder,
)

BETA, U = 8.0, 4.0


def _bond_disordered(Lx, Ly, spread, seed):
    """A bipartite lattice with every bond strength drawn at random.

    The criterion says the identity holds here, and nothing about these matrices was chosen to make
    that true -- which is what separates a predictive criterion from a descriptive one.
    """
    K = build(Lx, Ly, tp=0.0, mu=0.0, h=0.0)
    rng = np.random.default_rng(seed)
    N = K.shape[0]
    for i in range(N):
        for j in range(i + 1, N):
            if K[i, j] != 0:
                f = 1.0 + spread * (rng.random() - 0.5)
                K[i, j] *= f
                K[j, i] *= f
    return K


def table_one():
    """The signed-diagonal criterion against the measured identity."""
    rows = [
        ("bipartite, mu = 0", build(2, 4, 0.0, 0.0, 0.0)),
        ("bipartite, staggered h = 0.2", build(2, 4, 0.0, 0.0, 0.2)),
        ("bipartite, staggered h = 1.2", build(2, 4, 0.0, 0.0, 1.2)),
        ("tp = 0.3", build(2, 4, 0.3, 0.0, 0.0)),
        ("mu = 0.4", build(2, 4, 0.0, 0.4, 0.0)),
        ("bond-disordered (seed 1)", _bond_disordered(2, 4, 1.5, 1)),
        ("bond-disordered (seed 2)", _bond_disordered(2, 4, 1.5, 2)),
        ("bond-disordered (seed 3)", _bond_disordered(2, 4, 1.5, 3)),
        ("4x4 bond-disordered", _bond_disordered(4, 4, 1.5, 4)),
        ("2x6 bond-disordered", _bond_disordered(2, 6, 1.5, 5)),
        ("odd ring, 5 sites", ring(5)),
        ("odd ring, 7 sites", ring(7)),
    ]
    print("=" * 96)
    print(f"1. THE SIGNED-DIAGONAL CRITERION   S K S = -K for S = diag(+-1), beta = {BETA}, U = {U}")
    print("   The criterion costs O(N^2): no eigenvalues, no determinants, no field.")
    print()
    print(f"{'K':>32} | {'signed-diagonal S':>18} | {'identity resid':>15} | {'agrees':>7}")
    print("-" * 96)
    for name, K in rows:
        has_S = signed_diagonal_conjugation(K) is not None
        res, _ = measure(K, BETA, U=U)
        agrees = (res < 1e-8) == has_S
        print(f"{name:>32} | {('yes' if has_S else 'NO'):>18} | {res:15.3e} | "
              f"{('yes' if agrees else 'NO'):>7}", flush=True)


def table_two():
    """The same criterion on topologies the rest of the paper never uses."""
    rows = [
        ("chain, 8 sites, periodic", chain(8, periodic=True)),
        ("chain, 7 sites, periodic", chain(7, periodic=True)),
        ("chain, 8 sites, open", chain(8, periodic=False)),
        ("star graph (a tree), 8 sites", star(8)),
        ("triangular ladder, 8 sites", triangular_ladder(8)),
        ("even ring, 12 sites", ring(12)),
        ("odd ring, 9 sites", ring(9)),
    ]
    print()
    print("=" * 96)
    print("2. IT IS A STATEMENT ABOUT THE HOPPING GRAPH, NOT ABOUT LATTICES")
    print("   A tree satisfies it however it is drawn, having no cycles at all; any graph carrying")
    print("   an odd cycle never does.")
    print()
    print(f"{'topology':>32} | {'criterion':>10} | {'identity resid':>15}")
    print("-" * 96)
    for name, K in rows:
        has_S = signed_diagonal_conjugation(K) is not None
        res, _ = measure(K, BETA, U=U)
        print(f"{name:>32} | {('yes' if has_S else 'no'):>10} | {res:15.3e}", flush=True)


def table_three():
    """Route A against route B, with the sign deficit beside the identity residual."""
    rows = [
        ("2x4 real, half filled", build(2, 4, 0.0, 0.0, 0.0)),
        ("ring 6, flux pi/4", ring(6, flux=np.pi / 4)),
        ("ring 5, flux pi/2", ring(5, flux=np.pi / 2)),
        ("triangular ladder, flux 0", tri_ladder(6, theta=0.0)),
        ("triangular ladder, flux pi/2", tri_ladder(6, theta=np.pi / 2)),
    ]
    print()
    print("=" * 96)
    print("3. THE TWO ROUTES ARE NOT EQUIVALENT, AND ONLY ONE CARRIES POSITIVITY")
    print("   `sign deficit` is 1 - |<w/|w|>|: the ordinary negative-weight measure where the")
    print("   weight is real, and the mean-phase deficit where it is not.")
    print()
    print(f"{'K':>32} | {'route':>10} | {'identity resid':>15} | {'sign deficit':>13}")
    print("-" * 96)
    for name, K in rows:
        a, b = route_A(K), route_B(K)
        route = ("A and B" if a and b else "A only" if a else "B only" if b else "neither")
        res, deficit = measure(K, BETA, U=U)
        print(f"{name:>32} | {route:>10} | {res:15.3e} | {deficit:13.5f}", flush=True)
    print()
    print("  The last two rows are the point.  Threading the frustrated ladder takes the identity")
    print("  from broken to exact AND makes the sign quality worse: the flux converts a mild sign")
    print("  problem into a substantial phase problem.  So the criterion decides the IDENTITY, and")
    print("  the identity delivers positivity only where the weights stay real, which is route A.")


def table_four():
    """The criterion has no tolerance: the residual is linear in the violation from the first.

    `tests/test_spectral_criterion.test_the_identity_residual_is_linear_in_the_violation` asserts
    this over three decades, as a RATIO rather than a value -- correctly, since 85.4 is a property
    of this beta, N and U and not of the claim. What it does not do is print the table, and section
    4 quotes six rows of one. The sixth column is the one that makes the point: the negative
    fraction stays at zero while the identity degrades, so a broken identity and a sign problem are
    different quantities and the criterion predicts the first, not the second.
    """
    print()
    print("=" * 96)
    print("4. NO TOLERANCE AND NO ONSET: THE RESIDUAL IS LINEAR IN THE VIOLATION")
    print("   A staggered term of ANY amplitude makes S K S = -K unsatisfiable.  The ratio is what")
    print("   is claimed; its value is a property of this beta, N and U.")
    print()
    print(f"{'eps':>10} | {'criterion':>10} | {'identity resid':>15} | {'resid / eps':>12} | "
          f"{'neg fraction':>13}")
    print("-" * 96)
    for eps in (0.0, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1):
        K = build(2, 4, 0.0, 0.0, eps)
        has_S = signed_diagonal_conjugation(K) is not None
        res, neg = measure(K, BETA, U=U, n_draw=100, seed=11)
        ratio = "--" if eps == 0 else f"{res / eps:12.3f}"
        print(f"{eps:>10.0e} | {('yes' if has_S else 'no'):>10} | {res:15.3e} | {ratio:>12} | "
              f"{neg:13.4f}", flush=True)
    print()
    print("  A chemical potential behaves the same way; the row below is the same sweep on mu.")
    print()
    print(f"{'mu':>10} | {'criterion':>10} | {'identity resid':>15} | {'resid / mu':>12}")
    print("-" * 96)
    for mu in (1e-4, 1e-3, 1e-2):
        K = build(2, 4, 0.0, mu, 0.0)
        has_S = signed_diagonal_conjugation(K) is not None
        res, _neg = measure(K, BETA, U=U, n_draw=100, seed=11)
        print(f"{mu:>10.0e} | {('yes' if has_S else 'no'):>10} | {res:15.3e} | "
              f"{res / mu:12.3f}", flush=True)


if __name__ == "__main__":
    table_one()
    table_two()
    table_three()
    table_four()
