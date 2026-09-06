"""Experiment X -- the constrained-path bias is a dial, and this asks whether it can be TURNED.

Experiment W established the first half: at fixed U the bias tracks the overlap |<Psi_T|Psi_0>|,
and changing Psi_T alone at U = 8 moved the bias by a factor of 20.  So the bias is set by the
node, exactly as the derivation says.  That is useless on its own, because computing the overlap
needs the exact ground state, which is the thing nobody has.

The question that decides whether this scales is therefore NOT "does a better Psi_T help" -- it
does, by construction -- but:

    does a criterion computable WITHOUT the exact ground state pick the same Psi_T
    that the overlap would pick?

Three criteria are put on the same sweep of a one-parameter family Psi_T(h) -- a staggered field
of strength h, h = 0 being the free determinant:

  overlap       |<Psi_T|Psi_0>|            needs ED.  The truth, available only here.
  variational   <Psi_T|H|Psi_T>            free.  One determinant, Wick, no Monte Carlo.
  CP energy     the constrained-path answer  cheap.  But NOT variational, so a lower value is
                                             not evidence of a better answer on its own.

If argmax(overlap), argmin(variational) and argmin(|bias|) land on the same h, the dial has a
scalable handle and this is a research programme.  If they land apart, the dial exists and cannot
be turned without the answer, which is a different -- and still reportable -- result.

The whole curve is printed.  No h is selected, because selecting one here would be fitting to
exact-diagonalisation data that does not exist at the sizes this is meant to reach.
"""
from __future__ import annotations

import numpy as np

from model2d import Model2D
from sector_ed import ground_energy
from cpmc_fast import run
from trial import field_trial, stagger_2d, trial_overlap, trial_energy


def sweep(Lx, Ly, nu, nd, U, hs, beta=8.0, dtau=0.05, n_walkers=400, seeds=(1, 2, 3)):
    m = Model2D(Lx=Lx, Ly=Ly, t=1.0, mu=0.0, U=U, dtau=dtau, L=1, theta=0.0)
    stag = stagger_2d(Lx, Ly)
    ex = ground_energy(m.K, U, nu, nd)
    rows = []
    for h in hs:
        pt = field_trial(m.K, nu, nd, stag, h)
        ov = trial_overlap(pt, m.K, U, nu, nd)
        ev = trial_energy(pt, m.K, U, nu, nd)
        rs = [run(m, nu, nd, beta, n_walkers=n_walkers, seed=s, psi_t=pt) for s in seeds]
        if any(r["extinct_at"] is not None for r in rs):
            rows.append(dict(h=h, ov=ov, ev=ev, e=float("nan"), se=float("nan"),
                             ex=ex, extinct=True))
            continue
        e = np.array([r["e"] for r in rs])
        rows.append(dict(h=h, ov=ov, ev=ev, e=float(e.mean()),
                         se=float(e.std(ddof=1) / np.sqrt(len(seeds))), ex=ex, extinct=False))
    return rows, ex


def report(U, rows, ex):
    print(f"\nU = {U}   exact ground state {ex:+.5f}")
    print(f"{'h':>5} {'overlap':>9} {'<T|H|T>':>11} {'CP energy':>19} {'bias':>10} {'|bias|/|E|':>11}")
    for r in rows:
        if r["extinct"]:
            print(f"{r['h']:5.2f} {r['ov']:9.5f} {r['ev']:+11.5f} "
                  f"{'population extinct':>19} {'--':>10} {'--':>11}")
            continue
        b = r["e"] - ex
        print(f"{r['h']:5.2f} {r['ov']:9.5f} {r['ev']:+11.5f} "
              f"{r['e']:+12.5f}+-{r['se']:.5f} {b:+10.5f} {abs(b)/abs(ex):11.5f}")
    live = [r for r in rows if not r["extinct"]]
    if not live:
        return
    h_ov = max(live, key=lambda r: r["ov"])["h"]
    h_var = min(live, key=lambda r: r["ev"])["h"]
    h_bias = min(live, key=lambda r: abs(r["e"] - ex))["h"]
    h_cp = min(live, key=lambda r: r["e"])["h"]
    agree = "AGREE" if h_var == h_bias else "DISAGREE"
    print(f"  argmax overlap h={h_ov:.2f}   argmin variational h={h_var:.2f}   "
          f"argmin |bias| h={h_bias:.2f}   argmin CP energy h={h_cp:.2f}   -> {agree}")


if __name__ == "__main__":
    hs = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
    print("=" * 96)
    print("THE DIAL  2x4 at (3,3), beta = 8, dtau = 0.05, staggered-field trial family")
    for U in (4.0, 8.0, 12.0):
        rows, ex = sweep(2, 4, 3, 3, U, hs)
        report(U, rows, ex)
    print()
    print("A lower CP energy is NOT by itself evidence of a better answer: the constrained-path")
    print("estimator is not variational, and it can and does sit below the true ground state.")
    print("That is why the bias column is signed and the overlap column is printed beside it.")
