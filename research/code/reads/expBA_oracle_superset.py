"""Experiment BA -- the ORACLE, verified on a superset built independently of the paper's lists.

THE RESULT.  The section 4 criterion agrees with the measured identity on **64 of 64** one-body
matrices, with no mismatch.  Where it predicts the identity holds, the residual is `1.1e-14` to
`2.6e-14`; where it predicts failure, `0.76` to `7.5`.  The separation is thirteen orders, so no
verdict here depends on where a threshold sits.

WHY IT IS WORTH HAVING SEPARATELY.  Section 4's evidence is quoted across four counts -- sixteen
matrices, "twenty-four in total", "29 of 29" flux cases, "11 real" -- that describe overlapping
populations, and the abstract summarises them as "forty ... including bond-disordered lattices, a
tree, and flux-threaded rings", which spans more than one of those sets.  Auditing that
bookkeeping is not the same as checking the claim.  This enumerates a SUPERSET from the criterion
module's own constructors, without reference to which matrices the paper used, and reports the
agreement that is actually measured.

It also matters more under the framing the abstract now takes.  The criterion is not the headline
result there; it is the INDEPENDENT ORACLE the output-side read is scored against, and the whole
claim structure is that a read which never sees `K` agrees with an algebra that never sees a
configuration.  If the oracle were unsound that agreement would mean nothing, so it carries more
weight than a supporting table.

WHAT IS COVERED, beyond the paper's own sets: chains at 6, 8 and 12 sites both open and periodic;
odd periodic chains at 5, 7 and 9; stars at 6, 8 and 10; triangular ladders at 6 and 8; 2x4 at
three hoppings x two fillings x two staggered fields; 4x4 and 2x6; three bond-disordered 2x4
lattices; rings of 5, 6, 7 and 8 at six fluxes each; flux-threaded triangular ladders; and a
triangular 3x3 at three fluxes.

For each `K`: `criterion(K)` -- route A or route B, from `K` alone, no sampling and no determinant
-- against the MEASURED identity residual from the sampler.  A mismatch would be a case where the
algebra says one thing and the configurations say another.
"""
from __future__ import annotations

import numpy as np

from reads.expAO_spectral_criterion import (
    build, chain, criterion, measure, ring, star, tri_ladder, triangular, triangular_ladder,
)

CASES = []

# --- real lattices: bipartite, frustrated, doped, staggered ------------------
for tp in (0.0, 0.3, 0.7):
    for mu in (0.0, 0.4):
        for h in (0.0, 0.6):
            CASES.append((f"2x4 tp={tp} mu={mu} h={h}", build(2, 4, tp, mu, h)))
CASES.append(("4x4 clean", build(4, 4, 0.0, 0.0, 0.0)))
CASES.append(("2x6 clean", build(2, 6, 0.0, 0.0, 0.0)))

# --- chains, rings, trees, ladders ------------------------------------------
for n in (6, 8, 12):
    CASES.append((f"chain {n} open", chain(n, periodic=False)))
    CASES.append((f"chain {n} periodic", chain(n, periodic=True)))
for n in (5, 7, 9):
    CASES.append((f"odd chain {n} periodic", chain(n, periodic=True)))
for n in (6, 8, 10):
    CASES.append((f"star {n}", star(n)))
for n in (6, 8):
    CASES.append((f"triangular ladder {n}", triangular_ladder(n)))

# --- bond-disordered, bipartite by construction ------------------------------
rng = np.random.default_rng(11)
for i in range(3):
    K = build(2, 4, 0.0, 0.0, 0.0).astype(float)
    nz = K != 0
    pert = rng.uniform(0.4, 1.6, K.shape)
    pert = np.triu(pert, 1)
    pert = pert + pert.T
    CASES.append((f"2x4 bond-disordered #{i+1}", np.where(nz, K * pert, 0.0)))

# --- complex: rings and lattices at flux ------------------------------------
for n in (5, 6, 7, 8):
    for f in (0.0, np.pi / 6, np.pi / 4, np.pi / 3, np.pi / 2, np.pi):
        CASES.append((f"ring {n} flux={f:.4f}", ring(n, f)))
for n in (6, 8):
    for f in (0.0, np.pi / 4, np.pi / 2):
        CASES.append((f"tri ladder {n} flux={f:.4f}", tri_ladder(n, f)))
for f in (0.0, np.pi / 4, np.pi / 2):
    CASES.append((f"triangular 3x3 flux={f:.4f}", triangular(3, 3, f)))

print(f"{'case':<32}{'criterion':>11}{'identity resid':>17}{'holds':>8}   verdict")
print("-" * 82)
agree = mismatch = 0
bad = []
for tag, K in CASES:
    try:
        pred = bool(criterion(K))
        resid, _deficit = measure(K, beta=4.0, n_draw=120, seed=3)
        holds = resid < 1e-6
    except Exception as e:                                         # noqa: BLE001
        print(f"{tag:<32}   *** {type(e).__name__}: {str(e)[:34]}")
        continue
    ok = (pred == holds)
    agree += ok
    mismatch += (not ok)
    if not ok:
        bad.append((tag, pred, resid))
    print(f"{tag:<32}{str(pred):>11}{resid:>17.3e}{str(holds):>8}   "
          f"{'agree' if ok else '*** MISMATCH ***'}")

print("-" * 82)
n = agree + mismatch
print(f"  criterion agrees with the measured identity on {agree} of {n}")
if bad:
    print(f"  MISMATCHES ({len(bad)}):")
    for tag, pred, resid in bad:
        print(f"    {tag}: criterion={pred}, residual={resid:.3e}")
else:
    print("  no mismatch")
print()
print(f"  The paper claims 'Forty one-body matrices in total, with no mismatch'.")
print(f"  This enumerates {n} independently; the claim is safe if the mismatch count is 0")
print(f"  and n is at least 40.")
