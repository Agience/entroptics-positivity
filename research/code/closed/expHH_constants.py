"""Experiment HH -- is the complex-Langevin failure a result, or is it my adaptive-step constant?

The rule this rig integrates with is

    dt = eps / max(1, |K|_max / ref),      ref = ref_scale * max(1, |x|_mean + 2 lam)

and the `1` and the `2` in `ref` are numbers I invented.  Adaptive stepping is standard practice
for complex Langevin, but the reference scale is a choice, and here it is NOT symmetric between
the two channels: the spin channel's drift peaks near 7 so the step is cut about fourfold, while
the charge channel's reaches 1e3-1e4 so it is cut about a thousandfold.  The two channels are
therefore integrated with effective steps differing by ~250x -- and the failure being reported
(density wrong by z = 6 to 133) is the charge channel's.

So the failure has to be shown to survive the removal of the constant.  Two ways, and they are
different questions:

  ADAPTIVE OFF   no constant at all, only the discretisation parameter eps, which is then
                 extrapolated.  This is the principled version: if the trajectory blows up at
                 fixed step that is a RESULT about the dynamics, not something to patch with a
                 threshold.
  REF SWEPT      keep adaptation but scale the invented reference over two orders of magnitude.
                 A z that is flat across the sweep is a z that does not know about the constant.

The spin channel runs alongside at every setting, because it is the control for whether the
INTEGRATOR is sound, independently of what the charge channel does.

A z that collapses toward zero at any setting means the failure was an artifact and Parts 6 and 7
come down with it.
"""
from __future__ import annotations

import pathlib as _pathlib
import sys as _sys

# `gate_clangevin` is a sibling of this file's parent, under `tests/`.  A bare import finds it only
# when that directory is already on the path, so a plain `python closed/<this file>` fails.  The
# package-style form is not available either: an unrelated installed package is also named `tests`.
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent / "tests"))


import numpy as np

from dqmc import Model
from clangevin import run_cl
from gate_clangevin import exact_observables


def one(m, ne, chan, eps, adaptive, ref_scale, chains=64):
    r = run_cl(m, t_therm=8.0, t_meas=40.0, n_meas=200, eps=eps, seed=3,
               adaptive=adaptive, channel=chan, chains=chains, ref_scale=ref_scale)
    # n_se comes back COMPLEX from run_cl; take the real part explicitly.  Writing
    # max(complex, float) raises a TypeError, and the first version of this file caught that
    # exception and printed "BLEW UP" -- reporting a harness bug as a physical divergence, which
    # is the failure mode this whole rig keeps producing.  Only non-finite output is a blow-up.
    z = abs(r["n"].real - ne) / max(float(np.real(r["n_se"])), 1e-12)
    return r, z


if __name__ == "__main__":
    mu, U = 0.6, 4.0
    m = Model(N=4, t=1.0, t2=0.0, mu=mu, U=U, dtau=0.1, L=10)
    ne, de = exact_observables(m)
    print("=" * 104)
    print(f"DOES THE FAILURE SURVIVE REMOVING MY CONSTANT?   mu = {mu}, U = {U}, "
          f"exact n = {ne:.6f}")
    print("expDD measured z = 85.4 here with adaptive stepping at ref_scale = 1.")
    print()
    print("ADAPTIVE OFF -- no constant at all, only eps, extrapolated")
    print(f"{'chan':>7} {'eps':>9} {'n':>10} {'+-':>8} {'z':>8} {'|Im|':>8} {'max|K|':>10} "
          f"{'finite':>7}")
    for chan in ("charge", "spin"):
        for eps in (2e-3, 1e-3, 5e-4):
            r, z = one(m, ne, chan, eps, False, 1.0)
            fin = bool(np.isfinite(r["n"].real) and np.isfinite(r["max_drift"]))
            print(f"{chan:>7} {eps:9.1e} {r['n'].real:10.5f} {float(np.real(r['n_se'])):8.5f} "
                  f"{z:8.1f} {r['im_frac']:8.4f} {r['max_drift']:10.2e} {str(fin):>7}", flush=True)

    print()
    print("REF SWEPT -- adaptation kept, the invented reference scaled over 100x")
    print(f"{'chan':>7} {'ref_scale':>10} {'n':>10} {'+-':>8} {'z':>8} {'|Im|':>8} "
          f"{'max|K|':>10}")
    for chan in ("charge", "spin"):
        for rs in (0.1, 0.3, 1.0, 3.0, 10.0):
            r, z = one(m, ne, chan, 1e-3, True, rs)
            print(f"{chan:>7} {rs:10.2f} {r['n'].real:10.5f} {float(np.real(r['n_se'])):8.5f} "
                  f"{z:8.1f} {r['im_frac']:8.4f} {r['max_drift']:10.2e}", flush=True)
    print()
    print("A charge z that stays large everywhere is a failure that does not know about the")
    print("constant.  A charge z that collapses at any setting means the constant was driving it,")
    print("and Parts 6 and 7 come down with it.")
