"""Experiment FF -- break the confound: hold complexification ON and vary only the FAILURE.

Experiment DD's grid has a defect that no amount of scoring can fix.  In it, the only correct
charge runs are at half filling, and half filling is exactly where particle-hole symmetry cancels
the imaginary drift so the field never complexifies.  So

    correct  <=>  mu = 0  <=>  |Im y| = 0

are one partition wearing three names, and any read that merely notices the field went complex
separates it perfectly.  DD cannot tell "this read predicts failure" from "this read detects
complexification", and `spread (field)` -- which separated 17x there -- is under exactly that
cloud.

This fixes it by holding mu FIXED and nonzero, so EVERY row complexifies, and sweeping U so the
failure turns on across the rows.  lam = sqrt(dt U), so small U means a weak imaginary coupling:
the field still leaves the real axis, and the failure should be small enough to fall under the
labelling threshold.  If that happens, the sweep contains what DD structurally could not --

    a charge run that COMPLEXIFIES and is CORRECT

-- and the question becomes clean: with |Im y| > 0 on every row, does any read track z?

The prediction that would kill `spread (field)`: if it sits near 2.3 (its failing value) on the
correct-but-complexified rows, it is detecting complexification and closes like every other read
in Part 1.  If it stays near 40 (its correct value) there and only drops when z rises, it is
tracking the failure and is worth something.

The spin channel runs alongside as the control it has been throughout: same model, same code, no
complexification, and its reads were measured constant to three significant figures across DD.
"""
from __future__ import annotations

import pathlib as _pathlib
import sys as _sys

# `gate_clangevin` is a sibling of this file's parent, under `tests/`.  A bare import finds it only
# when that directory is already on the path, so a plain `python closed/<this file>` fails.  The
# package-style form is not available either: an unrelated installed package is also named `tests`.
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent / "tests"))


import numpy as np
from scipy.stats import spearmanr

from dqmc import Model
from gate_clangevin import exact_observables
from expDD_cl_reads import run_and_read, READS


if __name__ == "__main__":
    mu = 0.6
    Us = [0.25, 0.5, 1.0, 2.0, 4.0]
    print("=" * 122)
    print(f"BREAKING THE CONFOUND  mu = {mu} fixed and nonzero, so every charge row complexifies.")
    print("Only U varies, so only the FAILURE varies.  A read that tracks z here is tracking the")
    print("failure; one that is flat here was detecting complexification in expDD.")
    print(f"{'U':>5} {'chan':>7} {'n exact':>8} {'n':>9} {'+-':>7} {'z':>7} {'label':>6} "
          f"{'d exact':>8} {'d med':>8} {'dd':>8} {'max|K|':>8} {'q999':>7} {'|Im|':>7} "
          f"{'slowF':>7} {'sprF':>8} {'sprE':>11}")
    rows = []
    for U in Us:
        m = Model(N=4, t=1.0, t2=0.0, mu=mu, U=U, dtau=0.1, L=10)
        ne, de = exact_observables(m)
        for chan in ("charge", "spin"):
            r = run_and_read(m, chan)
            r["z"] = abs(r["n"].real - ne) / max(r["n_se"], 1e-12)
            r["label"] = "WRONG" if r["z"] > 3 else "ok"
            r.update(U=U, chan=chan, exact=ne, d_exact=de)
            rows.append(r)
            print(f"{U:5.2f} {chan:>7} {ne:8.5f} {r['n'].real:9.5f} {r['n_se']:7.5f} "
                  f"{r['z']:7.1f} {r['label']:>6} {de:8.5f} {r['docc_med']:8.5f} "
                  f"{r['docc_med']-de:+8.5f} {r['maxK']:8.1f} {r['q999']:7.2f} "
                  f"{r['im']:7.4f} {r['slow_f']:7.4f} {r['spread_f']:8.2f} "
                  f"{r['spread_e']:11.1f}", flush=True)

    ch = [r for r in rows if r["chan"] == "charge"]
    print()
    print("=" * 122)
    print("CHARGE ROWS ONLY, all complexified.  Does any read track z when only the failure moves?")
    good = [r for r in ch if r["label"] == "ok"]
    print(f"  complexified: {sum(1 for r in ch if r['im'] > 1e-6)}/{len(ch)}    "
          f"correct: {len(good)}/{len(ch)}")
    if not good:
        print("  NO correct-and-complexified row was produced -- U was not taken low enough, and")
        print("  the confound is NOT broken.  Nothing here may be credited to failure prediction.")
    z = np.array([r["z"] for r in ch])
    print(f"{'read':>15} {'Spearman vs z':>14} {'p':>9} {'value on correct':>18} "
          f"{'value on wrong':>16}")
    for name, key in READS:
        v = np.array([r[key] for r in ch], float)
        ok = np.isfinite(v)
        if ok.sum() < 3:
            continue
        rho, p = spearmanr(v[ok], z[ok])
        g = [r[key] for r in ch if r["label"] == "ok" and np.isfinite(r[key])]
        b = [r[key] for r in ch if r["label"] == "WRONG" and np.isfinite(r[key])]
        gs = f"{np.mean(g):18.4g}" if g else f"{'--':>18}"
        bs = f"{np.mean(b):16.4g}" if b else f"{'--':>16}"
        print(f"{name:>15} {rho:14.3f} {p:9.4f} {gs} {bs}")
    print()
    print("Spearman here is over rows that ALL complexify, so it cannot be earned by detecting")
    print("complexification.  A read that scored in expDD and is flat here was reading the wrong")
    print("thing, and expDD's separation numbers do not survive.")
