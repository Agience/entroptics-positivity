"""Experiment NN -- which COMPUTABLE criterion picks the trial wavefunction that actually works?

Two independent measurements now say the obvious criterion is the wrong one:

  Part 5   the variational energy of a single determinant prefers h = 3 and h = 4 where overlap
           and bias both want h = 0 -- anti-correlated with the truth.
  expMM    a NOCI state built from CPMC walkers has a BETTER variational energy than the plain
           free determinant (-16.23 against -15.00) and a WORSE overlap (0.809 against 0.844),
           and a 10x worse bias.  Its non-ground-state weight simply sits on lower excited
           states: 71% ground-state weight with the rest near -6.0, against 66% with the rest
           near -11.7.

So the quantity that decides the constrained path is the NODE, which tracks OVERLAP, and overlap
is exactly what cannot be computed without the answer.  That is the real obstacle -- not building
determinants, which the walkers do perfectly well.

This scores every computable criterion against the truth on a pool of ANSWER-FREE candidates:

  NOCI bound       variational, computable, and now twice shown to point the wrong way
  CPMC energy      computable.  The only criterion that has ever tracked the bias here (Part 5's
                   dial sweep, where argmin CP energy agreed with argmin |bias| at every U).
                   NOTE it is close to tautological WHEN the bias is positive, since then
                   argmin E_CPMC = argmin (E_CPMC - E_exact); the test is whether that holds
                   across a pool where it is not guaranteed, because CPMC is not variational and
                   was measured going BELOW exact at U = 2 in expV.
  overlap          the truth, printed for scoring only.  It steers nothing.

And the practical question the whole route turns on, asked plainly: does ANY answer-free
candidate beat the plain free determinant?  If none does, the route fails regardless of which
criterion would have picked the winner.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

from model2d import Model2D
from sector_ed import ground_energy
from trial import free_trial, _det_amplitudes
from noci import noci
from cpmc_multi import run_multi
from expMM_self_consistent import harvest
from expLL_unfitted_trial import build_basis, distinct
from expGG_nonorthogonal import best_k_dets


def overlap_of(dets, c, gn, N, n_up, n_dn):
    v = np.zeros(len(gn))
    for cj, d in zip(c, dets):
        v = v + cj * np.kron(_det_amplitudes(d[+1], N, n_up)[0],
                             _det_amplitudes(d[-1], N, n_dn)[0])
    n = np.linalg.norm(v)
    return float(abs(v @ gn) / n) if n > 0 else 0.0


def candidates(m, Lx, Ly, nu, nd, U):
    """Every entry here is ANSWER-FREE: no exact ground state is consulted to build it."""
    free = free_trial(m.K, nu, nd)
    out = [("free determinant", [free], np.array([1.0]))]
    for hs in ([0.0, 1.0], [0.0, 1.0, 3.0], [0.0, 0.5, 1.0, 2.0, 4.0]):
        d = build_basis(m.K, Lx, Ly, nu, nd, hs)
        _, c = noci(d, m.K, U)
        out.append((f"mean fields, {len(hs)} h", d, c))
    for beta_h, k, seed in ((4.0, 4, 100), (6.0, 4, 101), (6.0, 8, 102), (12.0, 8, 103)):
        w = harvest(m, nu, nd, [free], np.array([1.0]), k, n_walkers=64,
                    beta=beta_h, seed=seed)
        d = distinct([free] + w)[:k + 1]
        _, c = noci(d, m.K, U)
        out.append((f"walkers b={beta_h:g} k={len(d)}", d, c))
    # mean fields and walkers together
    w = harvest(m, nu, nd, [free], np.array([1.0]), 4, n_walkers=64, beta=6.0, seed=104)
    d = distinct(build_basis(m.K, Lx, Ly, nu, nd, [0.0, 1.0]) + w)
    _, c = noci(d, m.K, U)
    out.append((f"mean fields + walkers k={len(d)}", d, c))
    return out


if __name__ == "__main__":
    Lx, Ly, nu, nd = 2, 4, 3, 3
    U, beta, dtau = 8.0, 8.0, 0.05
    m = Model2D(Lx=Lx, Ly=Ly, t=1.0, mu=0.0, U=U, dtau=dtau, L=1, theta=0.0)
    ex, g = ground_energy(m.K, U, nu, nd, want_vec=True)
    gn = g / np.linalg.norm(g)

    print("=" * 112)
    print(f"WHICH COMPUTABLE CRITERION PICKS THE RIGHT TRIAL?  {Lx}x{Ly} at ({nu},{nd}), "
          f"U = {U}, beta = {beta}")
    print("Every candidate is answer-free.  'overlap' is the truth and steers nothing.")
    print()
    print(f"{'candidate':>28} {'k':>3} {'NOCI bound':>11} {'CPMC E':>19} {'bias':>10} "
          f"{'overlap':>9} {'vs free':>8}")
    rows = []
    base = None
    for name, dets, c in candidates(m, Lx, Ly, nu, nd, U):
        eb, _ = noci(dets, m.K, U)
        rs = [run_multi(m, nu, nd, dets, c, beta, n_walkers=400, seed=s, n_meas=150)
              for s in (1, 2, 3)]
        e = np.array([r["e"] for r in rs])
        b = float(e.mean()) - ex
        ov = overlap_of(dets, c, gn, m.N, nu, nd)
        if base is None:
            base = abs(b)
        rows.append(dict(name=name, k=len(dets), noci=eb, cpmc=float(e.mean()),
                         bias=b, ov=ov))
        print(f"{name:>28} {len(dets):3d} {eb:+11.5f} {e.mean():+12.5f}+-"
              f"{e.std(ddof=1)/np.sqrt(3):.5f} {b:+10.5f} {ov:9.5f} {abs(b)/base:8.3f}",
              flush=True)

    ovf, df, cf = best_k_dets(m.K, U, nu, nd, g, 4, n_restarts=2, seed=1)
    rs = [run_multi(m, nu, nd, df, cf, beta, n_walkers=400, seed=s, n_meas=150)
          for s in (1, 2, 3)]
    e = np.array([r["e"] for r in rs])
    print(f"{'[FITTED k=4, not a method]':>28} {4:3d} {noci(df, m.K, U)[0]:+11.5f} "
          f"{e.mean():+12.5f}+-{e.std(ddof=1)/np.sqrt(3):.5f} {e.mean()-ex:+10.5f} "
          f"{ovf:9.5f} {abs(e.mean()-ex)/base:8.3f}")

    print()
    print("=" * 112)
    b = np.array([abs(r["bias"]) for r in rows])
    # READ THE CPMC-ENERGY ROW AS A TAUTOLOGY WHERE IT IS ONE.  `bias = E_cpmc - E_exact`, so when
    # every candidate's bias carries the SAME SIGN -- as all nine do here, all positive -- ranking
    # by `E_cpmc` and ranking by `|bias|` are the same ranking, and Spearman must return exactly
    # 1.000 with `picks the best? YES`.  That is arithmetic, not a working selection criterion, and
    # it says nothing about whether minimising the CPMC energy would choose well in general (the
    # constrained path is not variational, so a negative bias breaks it).  The row is flagged in
    # the output so it cannot be quoted as a criterion that works.  The claim S8.6 makes is about
    # the NOCI bound, which is answer-free and anti-correlated.
    print(f"{'criterion':>14} {'Spearman vs |bias|':>19} {'p':>8} {'picks the best?':>16}")
    best = int(np.argmin(b))
    for label, key, sign in (("NOCI bound", "noci", +1), ("CPMC energy", "cpmc", +1),
                             ("overlap", "ov", -1)):
        v = np.array([r[key] for r in rows]) * sign
        rho, p = spearmanr(v, b)
        pick = int(np.argmin(v))
        note = ""
        if key == "cpmc" and (np.sign([r["bias"] for r in rows]) == np.sign(rows[0]["bias"])).all():
            note = "  <- TAUTOLOGY, not a criterion"
        print(f"{label:>14} {rho:19.3f} {p:8.4f} "
              f"{('YES' if pick == best else rows[pick]['name'][:16]):>16}{note}")
    print()
    print(f"best answer-free candidate: {rows[best]['name']} at bias {rows[best]['bias']:+.5f}")
    print(f"free determinant baseline:  {rows[0]['bias']:+.5f}")
    print()
    print("If no answer-free candidate beats the free determinant, the route fails whatever")
    print("criterion would have chosen among them.  A criterion is only worth having if")
    print("something in the pool is worth choosing.")
