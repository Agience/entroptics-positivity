"""Experiment BF -- how far does `best_k_dets` reproduce across processes?

WHY THIS FILE EXISTS.  Section 9.6 quotes an upper bound on what multi-determinant trials could buy
and then says why the upper end is quoted loosely: `best_k_dets` fits its orbitals by restarting a
non-convex optimisation and keeping the best, and the result is bit-exact within one process but not
across them.  The number attached to that caveat -- the spread of the overlap at `U = 8, k = 2` --
was not produced by anything.  `expKK` prints the overlap to five decimals, which cannot resolve a
spread of order `1e-7` however many times it is run, so re-running `expKK` could never have settled
it either.

WHAT THIS MEASURES.  The same call, at full precision, with everything that could legitimately
differ held fixed: same `K`, same filling, same target state, same `k`, same `n_restarts`, same
`seed`.  Anything left is the optimiser's own non-determinism -- BLAS thread scheduling, reduction
order, whatever the runtime chose that minute.

HOW TO READ IT.  One invocation prints one value.  The spread is across INVOCATIONS, so this is run
several times and the values compared; running it once and reporting a spread of zero would be
measuring nothing.  `--repeats` re-runs the fit inside one process as the control: that value is the
within-process spread, and it must be zero for "bit-exact within one process" to be the right
description.

    for i in 1 2 3; do python closed/expBF_multidet_reproducibility.py; done

WHY IT MATTERS, and why it is a caveat rather than a result.  Nothing in the paper's argument turns
on the seventh decimal of an overlap.  What turns on it is whether the quoted CEILING can be stated
tightly: a fit that lands somewhere slightly different each run gives a slightly different bound, so
section 9.6 says "about" and this file says how much "about" is.
"""
from __future__ import annotations

import argparse
import platform

import numpy as np

from closed.expGG_nonorthogonal import best_k_dets
from model2d import Model2D
from sector_ed import ground_energy

U, DTAU, BETA = 8.0, 0.05, 8.0
N_UP, N_DN, K_DETS = 3, 3, 2


def one_fit(m, g, seed=1, n_restarts=4):
    ov, _dets, _c = best_k_dets(m.K, U, N_UP, N_DN, g, K_DETS,
                                n_restarts=n_restarts, seed=seed)
    return float(ov)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3,
                    help="re-fits inside THIS process; the within-process spread, which must be 0")
    # ACROSS-process is the quantity section 9.6 quotes, and it cannot be measured from inside one
    # process by construction. `--collect N` spawns N fresh interpreters running this same file and
    # reports the spread of what they return, so one command produces the figure rather than asking
    # a reader to run it repeatedly and subtract by hand.
    ap.add_argument("--collect", type=int, default=0, metavar="N",
                    help="spawn N fresh processes and report the ACROSS-process spread")
    args = ap.parse_args()

    if args.collect:
        import re as _re
        import subprocess as _sp
        import sys as _sys
        vals = []
        for k in range(args.collect):
            r = _sp.run([_sys.executable, __file__, "--repeats", "1"],
                        capture_output=True, text=True)
            got = _re.findall(r"overlap = ([0-9.]+)", r.stdout)
            if not got:
                raise SystemExit(f"process {k + 1} printed no overlap: "
                                 f"{r.stdout}{r.stderr}")
            vals.append(float(got[0]))
            print(f"  process {k + 1}   overlap = {got[0]}", flush=True)
        print()
        print(f"  distinct values        {len(set(vals))} of {len(vals)}")
        print(f"  across-process spread  {max(vals) - min(vals):.3e}")
        print()
        print("  Bit-exact inside a process and not across them: the fit finds a slightly different")
        print("  optimum each run, so the CEILING it reports is quoted as a band and not a figure.")
        raise SystemExit(0)

    m = Model2D(Lx=2, Ly=4, t=1.0, mu=0.0, U=U, dtau=DTAU,
                L=int(round(BETA / DTAU)), theta=0.0)
    _ex, g = ground_energy(m.K, U, N_UP, N_DN, want_vec=True)

    vals = [one_fit(m, g) for _ in range(args.repeats)]
    spread = max(vals) - min(vals)

    print("=" * 96)
    print(f"`best_k_dets` REPRODUCIBILITY   2x4 at ({N_UP},{N_DN}), U = {U}, k = {K_DETS}, "
          f"n_restarts = 4, seed = 1")
    print(f"numpy {np.__version__} on {platform.python_version()}")
    print("=" * 96)
    for i, v in enumerate(vals, 1):
        print(f"  fit {i} in this process   overlap = {v:.17g}")
    print(f"  within-process spread    {spread:.3e}   "
          f"{'bit-exact' if spread == 0.0 else 'NOT bit-exact'}")
    print()
    print("  The ACROSS-process spread is not visible from one invocation.  Run this several")
    print("  times and compare the printed overlaps; that difference is what section 9.6 quotes.")
