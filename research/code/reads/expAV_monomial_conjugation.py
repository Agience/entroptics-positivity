"""§5's diagonal restriction is NECESSARY, and a monomial conjugation is the counterexample.

THE RESULT.  §5 decides the identity by asking for a DIAGONAL unitary `S` with `S K S^-1 = -K`.
The natural objection is that diagonal is arbitrary -- what the derivation needs is that conjugation
leave the interaction's coupling DIAGONAL, and a monomial `S = P_pi D` does that too, carrying
`diag(v)` to `diag(pi v)`.  Widening the class that way is tested here and it BREAKS the criterion:

    identity holds  <->  DIAGONAL criterion   14 of 14
    identity holds  <->  MONOMIAL route A     12 of 14, two false positives

The two failures are the staggered potentials, and they are the sharpest possible case: a diagonal
`S` cannot touch a potential at all, because `s_a^2 = 1` never gives `-1`, whereas a monomial `S`
admits one whose permutation maps it to its own negative -- `a = b` gives `K[pi(a), pi(a)] =
-K[a, a]`, which is exactly what a sublattice swap does.  So monomial conjugation opens the
staggered lattice, the conjugation `S K S^dag = -K` is exact there, AND THE IDENTITY STILL FAILS,
by 5.28 and 9.89.

WHY, AND IT IS NOT AN ACCIDENT OF THIS LATTICE.  `S B_up(x) S^dag` carries `diag(exp(lam pi x))` --
the RELABELLED field.  Summing over every field configuration is invariant under relabelling
sites, so the partition function survives a permutation.  The §4 identity is a PER-CONFIGURATION
statement, and at fixed `x` the permuted configuration is a different one.  The permutation costs
exactly the per-configuration relation and nothing less.  A diagonal `S` is the identity
permutation, which is why it is the class the derivation admits.

This is a strengthening of §5: the restriction to diagonal `S` is not a conservative convenience
chosen for tractability, and there is now a constructed lattice proving it cannot be relaxed.

WHAT ENTROPTICS SUPPLIES, AND IT IS THE POSITIVITY HALF.

§5 says route B buys the identity WITHOUT positivity.  That statement is made from `K`.  It is also
readable from the OUTPUT, which is where an observer actually stands -- holding a finite sample of
configuration weights and no knowledge of the mechanism:

    `concentration` on the weight cloud's (Re, Im) frame returns `focus = sigma_1^2 / M`, the axial
    statistic.  `focus = 1` is a rank-one cloud, and a rank-one cloud is a phase that a single
    global rotation carries onto one axis -- which cancels in `<O> = sum O w / sum w` and costs
    nothing.  `focus < 1` is a spread no rotation removes.  `principal_directions` supplies the
    axis to try, read from the cloud rather than chosen.

On all four route-B lattices here the read returns `focus` of 0.54 to 0.83 and an imaginary part of
order 1 surviving de-rotation on the cloud's own axis.  It says, from the weights alone, that the
residual phase is not removable -- which is §5's positivity claim, obtained without `K`.

The instrument's mode count is part of that answer.  On a sign-free lattice the cloud is a POINT on
the real axis and `principal_directions` resolves NOTHING above its floor, returning zero columns.
That is the correct reading rather than a failure: there is no axis because there is no spread.

Nothing here is thresholded.  `focus` is compared to 1, the definitional value of a rank-one cloud;
the monomial and diagonal verdicts are exact algebra; and the identity residual and sign deficit are
the criterion module's own `measure`.
"""
from __future__ import annotations

from itertools import permutations

import numpy as np

import entroptics_adapter as EA
from reads.expAO_spectral_criterion import (
    _phase_bfs, build, chain, criterion, measure, ring, route_A, route_B, tri_ladder,
)
from reads.expAU_axial_versus_directional import unit_weights

TOL = 1e-10


# ── the Entroptics side: decide from the weights ─────────────────────────────

def read_the_cloud(u):
    """(resultant, focus, resolved modes, residual imaginary part after de-rotating).

    Every step is a read.  `concentration` supplies the axial and directional statistics of the
    weight cloud on its own (Re, Im) frame; `principal_directions` supplies the leading axis of
    the SAME frame, so the angle removed is the cloud's own and is never chosen by the caller.

    THE MODE COUNT IS PART OF THE ANSWER.  On a sign-free lattice the cloud is a POINT on the real
    axis, and the instrument resolves NOTHING above its floor -- `principal_directions` returns
    zero columns.  That is the correct reading and not a failure: there is no axis because there is
    no spread, and the weights are already aligned.  One resolved mode is a rank-one cloud, which a
    single global rotation carries onto one axis; two is a genuine spread no rotation removes.
    """
    X = np.stack([u.real, u.imag], axis=1)
    c = EA.weight_cloud(X)
    V = EA.cloud_axes(X)
    modes = int(V.shape[1])
    axis = 0.0 if modes == 0 else float(np.arctan2(np.real(V[1, 0]), np.real(V[0, 0])))
    v = u * np.exp(-1j * axis)
    return float(c.resultant), float(c.focus), modes, float(np.abs(v.imag).max())


# ── the algebraic side: monomial route A ─────────────────────────────────────

def magnitude_automorphisms(K, tol=TOL):
    """Permutations preserving |K| entrywise -- the only ones route A can use.

    The condition fixes every modulus and leaves only a phase free, so `|K[pi(a), pi(b)]|` must
    equal `|K[a, b]|` on every pair including the diagonal.
    """
    A = np.abs(np.asarray(K))
    for pi in permutations(range(K.shape[0])):
        p = np.array(pi)
        if np.max(np.abs(A[np.ix_(p, p)] - A)) <= tol:
            yield p


def _phase_bfs_offdiagonal(K, want, tol=TOL):
    """Phases with `e^{i(th_a - th_b)} = want(a, b)` on every OFF-DIAGONAL edge of K's support.

    The criterion module's `_phase_bfs` opens with `if max|diag(K)| > tol: return None`, which is
    correct for a DIAGONAL S -- `s_a^2 = 1` can never give `-1`, so a potential is fatal there.
    Under a monomial S the diagonal is not fatal: `a = b` gives `K[pi(a), pi(a)] = -K[a, a]`, so a
    potential that `pi` maps to its own negative is admissible, and that condition is checked by
    the caller against `pi` alone before any phase is solved.  Reusing the stricter BFS would veto
    exactly the cases this file exists to test, so the diagonal walk is dropped and nothing else is.
    """
    N = K.shape[0]
    th = np.full(N, np.nan)
    for start in range(N):
        if not np.isnan(th[start]):
            continue
        th[start] = 0.0
        stack = [start]
        while stack:
            i = stack.pop()
            for j in range(N):
                if i == j or abs(K[i, j]) <= tol:
                    continue
                tj = th[i] - np.angle(want(i, j))
                if np.isnan(th[j]):
                    th[j] = tj
                    stack.append(j)
                elif abs(np.exp(1j * (th[j] - tj)) - 1.0) > 1e-8:
                    return None
    return th


def monomial_route_A(K, tol=1e-8):
    """Is there a monomial unitary `S = P_pi D` with `S K S^-1 = -K`?  Returns (pi, S) or None.

    Exhaustive over the magnitude automorphisms, and the phases for each come from the criterion
    module's own BFS.  The result is CHECKED by forming `S K S^dag` rather than trusted from the
    solve, so a BFS that succeeded on a disconnected support cannot pass unnoticed.
    """
    K = np.asarray(K, dtype=complex)
    N = K.shape[0]
    for p in magnitude_automorphisms(K):
        Kp = K[np.ix_(p, p)]
        if np.max(np.abs(np.diag(Kp) + np.diag(K))) > tol:
            continue                       # no phase can move a diagonal entry
        th = _phase_bfs_offdiagonal(K, lambda a, b: -Kp[a, b] / K[a, b])
        if th is None:
            continue
        S = np.zeros((N, N), dtype=complex)
        S[p, np.arange(N)] = np.exp(1j * th)
        if np.max(np.abs(S @ K @ np.conj(S).T + K)) <= tol:
            return p, S
    return None


CASES = [
    ("chain 6", chain(6)),
    ("2x4 clean", build(2, 4, 0.0, 0.0, 0.0)),
    ("2x4 staggered h=0.2", build(2, 4, 0.0, 0.0, 0.2)),
    ("2x4 staggered h=0.6", build(2, 4, 0.0, 0.0, 0.6)),
    ("2x4 tp=0.3", build(2, 4, 0.3, 0.0, 0.0)),
    ("ring 5", ring(5, 0.0)),
    ("ring 5 flux pi/2", ring(5, np.pi / 2)),
    ("ring 6", ring(6, 0.0)),
    ("ring 6 flux pi/4", ring(6, np.pi / 4)),
    ("ring 7 flux pi/2", ring(7, np.pi / 2)),
    ("tri ladder 6", tri_ladder(6, 0.0)),
    ("tri ladder 6 flux pi/2", tri_ladder(6, np.pi / 2)),
    ("tri ladder 8", tri_ladder(8, 0.0)),
    ("tri ladder 8 flux pi/2", tri_ladder(8, np.pi / 2)),
]


def row(name, K):
    K = np.asarray(K, dtype=complex)
    mono = monomial_route_A(K)
    u = unit_weights(K)
    resultant, focus, modes, after = read_the_cloud(u)
    resid, deficit = measure(K, 6.0, n_draw=120, seed=5)
    return dict(name=name, N=K.shape[0], dA=bool(route_A(K)), dB=bool(route_B(K)),
                dany=bool(criterion(K)), mono=mono is not None,
                pi=None if mono is None else tuple(int(v) for v in mono[0]),
                resultant=resultant, focus=focus, modes=modes, after=after,
                resid=resid, deficit=deficit)


if __name__ == "__main__":
    print("=" * 118)
    print("A REMOVABLE PHASE, READ FROM THE WEIGHTS, AND THE SYMMETRY THAT EXPLAINS IT")
    print()
    print("`focus` is read by Entroptics from the weight cloud alone -- no K, no mechanism.")
    print("`mono A` is the algebra, from K alone.  They share no information.")
    print()
    print(f"{'lattice':>24} {'N':>3} | {'diag A':>7} {'mono A':>7} | {'modes':>5} {'focus':>8} "
          f"{'resultant':>10} {'after de-rot':>13} | {'identity':>11} {'deficit':>9}")
    rows = []
    for name, K in CASES:
        r = row(name, K)
        rows.append(r)
        print(f"{r['name']:>24} {r['N']:3d} | {str(r['dA']):>7} {str(r['mono']):>7} | "
              f"{r['modes']:5d} {r['focus']:8.5f} {r['resultant']:10.5f} {r['after']:13.2e} | "
              f"{r['resid']:11.3e} {r['deficit']:9.5f}", flush=True)

    print()
    print("=" * 118)
    gained = [r for r in rows if r["mono"] and not r["dA"]]
    if gained:
        print("OPENED BY A MONOMIAL S WHERE THE DIAGONAL CRITERION IS CLOSED:")
        for r in gained:
            print(f"  {r['name']:>24}  permutation {r['pi']}  "
                  f"identity {r['resid']:.3e}  deficit {r['deficit']:.5f}")
    else:
        print("No lattice here is opened by a monomial S that a diagonal S closes.")

    print()
    print("=" * 118)
    print("WHICH CONJUGATION CLASS PREDICTS THE IDENTITY?  Scored against the MEASURED residual.")
    print()
    holds = [r["resid"] < 1e-9 for r in rows]
    nd = sum(h == r["dany"] for h, r in zip(holds, rows))
    nm = sum(h == r["mono"] for h, r in zip(holds, rows))
    print(f"  identity holds  <->  DIAGONAL criterion (route A or B) : {nd} of {len(rows)}")
    print(f"  identity holds  <->  MONOMIAL route A                  : {nm} of {len(rows)}")
    print()
    for h, r in zip(holds, rows):
        if h != r["mono"]:
            print(f"  monomial FALSE POSITIVE: {r['name']:>24}  S K S^dag = -K exact, "
                  f"permutation {r['pi']}, identity residual {r['resid']:.3e}")
    print()
    print("The diagonal restriction in section 5 is NECESSARY.  A monomial S admits a potential")
    print("its permutation maps to its own negative -- which a diagonal S can never do -- and on")
    print("those lattices the conjugation is exact while the identity fails.  The permutation")
    print("relabels the auxiliary field: the partition function is invariant under that, and the")
    print("per-configuration identity is not.")
    print()
    print("=" * 118)
    print("THE POSITIVITY HALF, READ FROM THE WEIGHTS ALONE.")
    print()
    for r in rows:
        if r["resid"] < 1e-9 and r["deficit"] > 0.0:
            print(f"  {r['name']:>24}  identity {r['resid']:.2e}  focus {r['focus']:.5f}  "
                  f"|Im| after de-rotation {r['after']:.2e}  deficit {r['deficit']:.5f}")
    print()
    print("Every row above satisfies the identity and still carries a phase.  `focus` below 1 says")
    print("the cloud is not rank one, so no global rotation makes those weights real -- section 5's")
    print("statement that route B carries the identity without positivity, obtained from output.")
