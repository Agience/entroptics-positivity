"""Experiment EE -- does the complex-Langevin error bar shrink like 1/sqrt(C) at all?

Experiment CC was set up to ask whether complex Langevin converges to the WRONG answer here, and
answered a different question instead.  Going from 128 chains to 512 -- four times the sampling --
the quoted error on the double occupancy did not halve, it ROSE, from +-0.035 to +-0.096.  A
well-behaved estimator cannot do that.  So the earlier "2.2 sigma discrepancy" was not a
discrepancy, it was an under-estimated error bar, and the honest reading of the 512-chain run is
z = 0.13: consistent with exact, and consistent with almost anything.

That is a worse failure than a wrong answer, and a subtler one.  A wrong answer with a valid error
bar announces itself the moment anyone checks.  An estimator whose standard error does not
converge gives a tight-looking number at small sample size and a different tight-looking number at
the next, with nothing in either run to say which to believe.

So the question here is not "is the mean wrong" but "is the ERROR BAR real":

    take the per-chain values from one run and resample them at C = 8, 16, ... 512.
    A finite-variance estimator gives se(C) proportional to C^-0.5.
    Fit the exponent.  The SPIN channel is the control and must return -0.5.

and alongside it the shape of the per-chain distribution itself -- median against mean, MAD
against standard deviation, and the largest chain's share of the total spread.  A heavy tail shows
in all three, and none of them needs a theory of why.
"""
from __future__ import annotations

import numpy as np

from dqmc import Model
from clangevin import run_cl
from gate_clangevin import exact_observables


def se_scaling(v, sizes, n_draw=2000, rng=None):
    """Bootstrap the per-chain values at each size; fit log se against log C.

    WITH REPLACEMENT, and that is not a detail.  Drawing without replacement from a
    512-element pool makes the C = 512 draw the whole pool every time, so its spread is
    identically zero and the log fit is dominated by a -infinity point -- measured, that
    returned an exponent near -5 for every column INCLUDING the spin control, which must
    return -0.5.  The control catching it is the only reason it was caught.
    """
    rng = rng or np.random.default_rng(0)
    v = np.asarray(np.real(v), float)
    xs, ys = [], []
    for c in sizes:
        means = v[rng.integers(0, len(v), size=(n_draw, c))].mean(axis=1)
        xs.append(c)
        ys.append(float(means.std(ddof=1)))
    a = np.polyfit(np.log(xs), np.log(ys), 1)
    return float(a[0]), list(zip(xs, ys))


def shape(v):
    v = np.asarray(np.real(v), float)
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med))) * 1.4826
    sd = float(v.std(ddof=1))
    dev = np.abs(v - med)
    return dict(mean=float(v.mean()), med=med, sd=sd, mad=mad,
                ratio=sd / max(mad, 1e-30),
                top=float(dev.max() ** 2 / max((dev ** 2).sum(), 1e-30)))


if __name__ == "__main__":
    m = Model(N=4, t=1.0, t2=0.0, mu=1.0, U=4.0, dtau=0.1, L=10)
    ne, de = exact_observables(m)
    sizes = [8, 16, 32, 64, 128, 256, 512]
    print("=" * 100)
    print(f"IS THE ERROR BAR REAL?  mu = 1.0, U = 4, N = 4, L = 10.  exact d = {de:.6f}")
    print()
    import pickle, os
    cache = "expEE_chains.pkl"
    if os.path.exists(cache):
        runs = pickle.load(open(cache, "rb"))
        print("(per-chain values loaded from cache)")
    else:
        runs = {}
        for chan in ("spin", "charge"):
            runs[chan] = run_cl(m, t_therm=8.0, t_meas=40.0, n_meas=200, eps=2e-3, seed=5,
                                adaptive=True, channel=chan, chains=512)
        pickle.dump({k: {kk: vv for kk, vv in v.items() if kk != "drifts"}
                     for k, v in runs.items()}, open(cache, "wb"))
    for chan in ("spin", "charge"):
        r = runs[chan]
        print(f"{chan:>7}: n = {r['n'].real:.6f} +- {r['n_se'].real:.5f} "
              f"(exact {ne:.6f}, z = {abs(r['n'].real-ne)/max(r['n_se'].real,1e-12):6.1f})   "
              f"d = {r['docc'].real:.6f} +- {r['docc_se'].real:.5f} "
              f"(exact {de:.6f}, z = {abs(r['docc'].real-de)/max(r['docc_se'].real,1e-12):5.2f})",
              flush=True)

    print()
    print("SE SCALING  se(C) should go as C^-0.5 if the chain distribution has finite variance")
    print(f"{'channel':>8} {'obs':>5} {'exponent':>9} " + " ".join(f"{c:>9}" for c in sizes))
    for key in ("per_chain_n", "per_chain_docc"):
        lab = "n" if key.endswith("_n") else "docc"
        for chan, r in runs.items():
            a, rows = se_scaling(r[key], sizes)
            cells = {c: s for c, s in rows}
            print(f"{chan:>8} {lab:>5} {a:9.3f} "
                  + " ".join(f"{cells.get(c, float('nan')):9.5f}" for c in sizes))

    print()
    print("PER-CHAIN DISTRIBUTION  a heavy tail shows as sd >> mad and one chain owning the spread")
    print(f"{'channel':>8} {'obs':>5} {'mean':>10} {'median':>10} {'sd':>10} {'mad':>10} "
          f"{'sd/mad':>8} {'top chain share':>16}")
    for key in ("per_chain_n", "per_chain_docc"):
        lab = "n" if key.endswith("_n") else "docc"
        for chan, r in runs.items():
            s = shape(r[key])
            print(f"{chan:>8} {lab:>5} {s['mean']:10.5f} {s['med']:10.5f} {s['sd']:10.5f} "
                  f"{s['mad']:10.5f} {s['ratio']:8.2f} {s['top']:16.4f}")
    print()
    print()
    print("ROBUST CENTRE  the median across chains is immune to a single runaway")
    print(f"{'channel':>8} {'obs':>5} {'mean':>10} {'median':>10} {'exact':>10} "
          f"{'mean-exact':>11} {'median-exact':>13}")
    for key, ex in (("per_chain_n", ne), ("per_chain_docc", de)):
        lab = "n" if key.endswith("_n") else "docc"
        for chan, r in runs.items():
            v = np.real(r[key])
            print(f"{chan:>8} {lab:>5} {v.mean():10.5f} {np.median(v):10.5f} {ex:10.5f} "
                  f"{v.mean()-ex:+11.5f} {np.median(v)-ex:+13.5f}")
    print()
    print("An exponent near -0.5 with sd ~ mad is an estimator whose error bar means what it says.")
    print("An exponent well above -0.5, or sd many times mad, is one whose error bar does not --")
    print("and the spin channel is the control that says which of those the code itself produces.")
