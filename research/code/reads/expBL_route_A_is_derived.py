"""Section 7.2's route-A table: `strength = -1 + 2r`, printed for every row the paper quotes.

`tests/test_spectral_criterion.py` asserts this relation exactly, which is the right thing for a
gate to do, but a gate prints nothing and it covers four rows where the paper's table has six. So
the table had no file that produces it, and a figure with no producer is how four separate stale
numbers survived in this paper already.

WHAT THE RELATION SAYS. Under route A the two channels satisfy `G_dn = 1 - conj(G_up)`, so the real
parts are exact negatives about `1/2` and the imaginary parts are exact positives. A single signed
alignment averages the two halves, and the result is fixed by how much of the centred variance is
imaginary:

    r = |Im A~|^2 / (|Re A~|^2 + |Im A~|^2),        strength = -1 + 2r

so the departure from `-1` is the frame's imaginary weight and carries nothing about the sign
problem, which is absent on every row here. The `neg` column is what says so.

The construction is `reads/expBD_route_calibration.channels`, which is the gate's arithmetic
verbatim, so this file and that gate cannot disagree for reasons unrelated to the physics.

    python remote_run.py reads/expBL_route_A_is_derived.py
"""
from __future__ import annotations

import numpy as np

import entroptics_adapter as EA
from reads.expAO_spectral_criterion import measure, ring, route_A, route_B
from reads.expBD_route_calibration import BETA, channels

ROWS = ((6, "pi/8", 1 / 8), (6, "pi/4", 1 / 4), (6, "pi/3", 1 / 3),
        (8, "pi/4", 1 / 4), (8, "pi/3", 1 / 3), (10, "pi/4", 1 / 4))


def main():
    print("=" * 96)
    print("ROUTE A: THE DEPARTURE FROM -1 IS THE FRAME'S IMAGINARY WEIGHT, EXACTLY")
    print("=" * 96)
    print(f"  beta = {BETA}; every row is route A only, and sign-free")
    print()
    print(f"{'K':>22} {'r':>10} {'-1 + 2r':>12} {'measured':>12} {'|diff|':>11} {'neg':>9}")
    print("-" * 96)

    worst = 0.0
    for n, label, frac in ROWS:
        K = np.asarray(ring(n, frac * np.pi), dtype=complex)
        assert route_A(K) and not route_B(K), f"ring {n} flux {label} is not route-A only"
        _, neg = measure(K, BETA, n_draw=100, seed=5)
        A, B = channels(K)
        Ac = A - A.mean(axis=0, keepdims=True)
        vre = float((Ac.real ** 2).sum())
        vim = float((Ac.imag ** 2).sum())
        r = vim / (vre + vim)
        got = float(EA.channel_alignment(A, B).strength)
        diff = abs(got - (-1.0 + 2.0 * r))
        worst = max(worst, diff)
        print(f"{f'ring {n}, flux {label}':>22} {r:10.5f} {-1.0 + 2.0 * r:12.5f} "
              f"{got:12.5f} {diff:11.2e} {neg:9.5f}", flush=True)

    print()
    print(f"  worst |measured - (-1 + 2r)| over the six rows: {worst:.2e}")
    print("  The relation is arithmetic, not a fit: `r` is computed from the frame and the read is")
    print("  computed by the library, and they agree to the precision above with nothing tuned.")


if __name__ == "__main__":
    main()
