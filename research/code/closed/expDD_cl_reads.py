"""Experiment DD -- can a read of the TRAJECTORY tell a correct complex-Langevin run from a wrong
one, without being told the answer?

This is the last live version of the original question.  Part 1 closed the reads against the sign
problem with a mechanism: every cheap read is the average sign under a different, cheaper measure,
and the measure that decides the answer is |D(x)|.  That argument cannot apply here, because
complex Langevin has NO average sign.  Its failure is a trajectory going where the derivation's
boundary term stops vanishing -- a property of a dynamical system, which is what a spectral read
is for.

Part 6 established that there IS a failure to predict: at mu = 1.0, U = 4 the charge channel's
density is wrong by z = 6.0 against a spin control that is exact, and one chain in 512 carries
99.5% of the double occupancy's spread.  It also found one statistic that separates the two --
max|K| / median|K|, 7.95 against 180-336 -- while q99 and q999 barely move.  Two classes on one
model is not a result; this is the grid.

THE GRID IS BUILT SO THE CHARGE CHANNEL CONTAINS BOTH CLASSES.  A grid where spin is always right
and charge is always wrong would be scored perfectly by any statistic that merely tells the two
channels apart, which is worth nothing.  At HALF FILLING the imaginary drift cancels by
particle-hole symmetry, the field stays real (measured |Im y| = 0.0000), and the charge channel is
correct.  Doped, it is not.  So the charge rows span correct and failing, and the question the
scoring actually asks is the hard one: separate a good charge run from a bad one.

LABELLED ON THE DENSITY, NOT THE DOUBLE OCCUPANCY.  Part 6 measured why: the double occupancy's
per-chain distribution has sd/mad = 225 and its mean is driven to a NEGATIVE probability by a
single runaway, so a z computed from it is meaningless.  The density's is well behaved
(sd/mad = 3.37) and its standard error scales as C^-0.48, so z on the density means what it says.

Reads compared:

  max|K| / med     the extreme tail.  The one that worked at a single point.
  q99, q999 / med  the same tail as a robust quantile -- what anyone would reach for first, and
                   what Part 6 measured as nearly blind (3.8 -> 4.1 across the two classes).
  tail slope       the tail as a power-law exponent.
  |Im| of field    how far the trajectory left the real axis at all.
  DecayRates       `entroptics.dynamics(W).rates()` -- the relaxation spectrum of the trajectory,
                   on TWO feature sets, because the choice of features is part of the claim:
                     field  one chain's own field components (2 L N of them), the literal
                            trajectory of the dynamical system;
                     ens    per-step ensemble summaries (mean and max of |X|, |Im X|, |K|), which
                            can see a rare runaway that chain 0 never has.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

from dqmc import Model
from clangevin import CLangevin
from gate_clangevin import exact_observables
from entroptics.dynamics import dynamics


def run_and_read(m, channel, eps=1e-3, chains=96, t_therm=8.0, t_meas=40.0,
                 n_rec=1200, seed=1):
    ch = CLangevin(m, eps=eps, seed=seed, adaptive=True, channel=channel, chains=chains)
    for _ in range(int(round(t_therm / eps))):
        ch.step()
    n_steps = int(round(t_meas / eps))
    every = max(1, n_steps // n_rec)
    Wf, We, nvals, dvals, drifts = [], [], [], [], []
    for k in range(n_steps):
        ch.step()
        if k % every:
            continue
        G = ch.greens()
        o = ch.observe(G)
        nvals.append(o["n"])
        dvals.append(o["docc"])
        K = ch.drift(G)
        drifts.append(np.abs(K).ravel())
        x = ch.X
        Wf.append(np.concatenate([x[0].real.ravel(), x[0].imag.ravel()]))
        aX, aI, aK = np.abs(x), np.abs(x.imag), np.abs(K)
        We.append(np.array([aX.mean(), aX.max(), aI.mean(), aI.max(), aK.mean(), aK.max()]))
    n = np.array(nvals).mean(axis=0)
    d = np.array(dvals).mean(axis=0)
    dr = np.concatenate(drifts)
    med = max(float(np.median(dr)), 1e-30)
    hi = np.sort(dr)[int(0.9 * len(dr)):]
    slope = float("nan")
    if len(hi) > 50:
        slope = float(np.polyfit(np.log(np.maximum(hi, 1e-30)),
                                 np.log(np.linspace(0.1, 1e-4, len(hi))), 1)[0])
    out = dict(n=complex(n.mean()), n_se=float(np.real(n).std(ddof=1) / np.sqrt(len(n))),
               n_med=float(np.median(np.real(n))),
               docc=complex(d.mean()), docc_med=float(np.median(np.real(d))),
               maxK=float(dr.max()) / med, q99=float(np.quantile(dr, 0.99)) / med,
               q999=float(np.quantile(dr, 0.999)) / med, slope=slope,
               im=float(np.abs(ch.X.imag).mean()))
    for tag, W in (("f", np.array(Wf)), ("e", np.array(We))):
        r = dynamics(W).rates()
        out[f"slow_{tag}"] = float(r.long_range)
        out[f"fast_{tag}"] = float(r.short_range)
        out[f"spread_{tag}"] = float(r.short_range) / max(float(r.long_range), 1e-30)
    return out


READS = [("max|K|/med", "maxK"), ("q99/med", "q99"), ("q999/med", "q999"),
         ("tail slope", "slope"), ("|Im|", "im"),
         ("slow (field)", "slow_f"), ("spread (field)", "spread_f"),
         ("slow (ens)", "slow_e"), ("spread (ens)", "spread_e")]


def score(rows, title):
    z = np.array([r["z"] for r in rows])
    wrong = np.array([r["label"] == "WRONG" for r in rows])
    print()
    print(f"{title}   ({wrong.sum()} wrong, {(~wrong).sum()} correct)")
    if not (wrong.any() and (~wrong).any()):
        print("  not both classes present -- nothing to score")
        return
    print(f"{'read':>15} {'Spearman vs z':>14} {'p':>8} {'separation':>12} "
          f"{'caught at 0 false alarms':>26}")
    for name, key in READS:
        v = np.array([r[key] for r in rows], float)
        ok = np.isfinite(v)
        if ok.sum() < 4:
            continue
        rho, p = spearmanr(v[ok], z[ok])
        g, b = v[(~wrong) & ok], v[wrong & ok]
        sep = (b.min() / g.max()) if b.min() > g.max() else (
            (g.min() / b.max()) if g.min() > b.max() else 0.0)
        caught = "0"
        for sign in (+1, -1):
            thr = g.max() if sign > 0 else g.min()
            hit = ((v > thr) if sign > 0 else (v < thr)) & wrong & ok
            if hit.sum():
                caught = f"{hit.sum()}/{(wrong & ok).sum()} ({'above' if sign>0 else 'below'})"
                break
        s = f"{sep:12.1f}" if sep else f"{'overlap':>12}"
        print(f"{name:>15} {rho:14.3f} {p:8.4f} {s} {caught:>26}")


if __name__ == "__main__":
    grid = [(0.0, 4.0), (0.3, 4.0), (0.6, 4.0), (1.0, 4.0), (0.0, 8.0), (0.6, 8.0)]
    rows = []
    print("=" * 118)
    print("GRID  N = 4, L = 10 (beta = 1), 96 chains, eps = 1e-3.  Labelled on the DENSITY.")
    print("NOTE: at half filling the density is pinned to exactly 1 by particle-hole symmetry")
    print("for both channels, so it carries NO label information there.  d_med is printed beside")
    print("it because the double occupancy is not symmetry-pinned and, where no runaway occurs,")
    print("its median is well behaved -- it is what establishes the mu = 0 labels.")
    print(f"{'mu':>4} {'U':>4} {'chan':>7} {'n exact':>8} {'n':>9} {'+-':>7} {'z':>6} "
          f"{'label':>6} {'d exact':>8} {'d med':>8} {'dd':>8} "
          f"{'max|K|':>8} {'q999':>7} {'|Im|':>7} {'slowF':>7} {'sprF':>7} "
          f"{'slowE':>7} {'sprE':>7}")
    for mu, U in grid:
        m = Model(N=4, t=1.0, t2=0.0, mu=mu, U=U, dtau=0.1, L=10)
        ne, de = exact_observables(m)
        for chan in ("spin", "charge"):
            r = run_and_read(m, chan)
            r["z"] = abs(r["n"].real - ne) / max(r["n_se"], 1e-12)
            r["label"] = "WRONG" if r["z"] > 3 else "ok"
            r.update(mu=mu, U=U, chan=chan, exact=ne)
            rows.append(r)
            print(f"{mu:4.1f} {U:4.1f} {chan:>7} {ne:8.5f} {r['n'].real:9.5f} {r['n_se']:7.5f} "
                  f"{r['z']:6.1f} {r['label']:>6} {de:8.5f} {r['docc_med']:8.5f} "
                  f"{r['docc_med']-de:+8.5f} {r['maxK']:8.1f} {r['q999']:7.2f} "
                  f"{r['im']:7.4f} {r['slow_f']:7.4f} {r['spread_f']:7.2f} "
                  f"{r['slow_e']:7.4f} {r['spread_e']:7.2f}", flush=True)

    print()
    print("=" * 118)
    score(rows, "ALL RUNS -- the easy question, and any channel-detector scores it")
    score([r for r in rows if r["chan"] == "charge"],
          "CHARGE ONLY -- the real question: a good charge run against a bad one")
    print()
    print("'separation' is the gap between the two classes as a ratio, or 'overlap' if they mix.")
    print("A read that separates only in the ALL table has learned the channel, not the failure.")
