"""Section 7.2: the deficit is the perpendicular part of the particle-hole violation, measured on
the capability axis rather than on synthetic frames.

Section 7.2 decomposes the violation `E = B~ + A~` into the part along `A~` and the part
perpendicular to it, and makes two claims: the parallel part is invisible to the read, and the
perpendicular part sets the deficit, to leading order as

    deficit ~= |E_perp|^2 / (2 |A~| |B~|).

Both are properties of the read and both hold on any frame; what needs measuring on this axis is
where the small-angle form stops describing the DQMC deficit, because the paper quotes that
boundary as a beta. The chains here are section 7's own -- the same model, betas and seeds as
`reads/expAN_capability_table.py` -- so the deficits in the last column are the capability table's
deficits and can be read against it directly.

    python remote_run.py reads/expBH_perpendicular_violation.py
"""
from __future__ import annotations

import numpy as np

import entroptics_adapter as EA
from dqmc import Model
from reads.expAN_capability_table import BETAS, DTAU, MU, N, SEEDS, T2, U
from reads.expQQ_coupling_vs_sign import run

R, WARM, N_MEAS = 64, 25, 40


def decompose(A, B):
    """(measured deficit, small-angle prediction) on one pair of channel frames.

    `E = B~ + A~` is the violation of `B = 1 - A`: it vanishes exactly when the two centred
    channels are exact negatives, which is the particle-hole point. `E_perp` removes the component
    along `A~`, which is the component a rescaling of one channel can absorb.
    """
    Ac = A - A.mean(axis=0, keepdims=True)
    Bc = B - B.mean(axis=0, keepdims=True)
    Ev = Bc + Ac
    Ep = Ev - ((Ev * Ac).sum() / (Ac ** 2).sum()) * Ac
    pred = float(np.linalg.norm(Ep) ** 2
                 / (2 * np.linalg.norm(Ac) * np.linalg.norm(Bc)))
    deficit = 1.0 + float(EA.channel_alignment(A, B).strength)
    return deficit, pred


def main():
    print("=" * 104)
    print("THE PERPENDICULAR VIOLATION AGAINST THE MEASURED DEFICIT, ON SECTION 5's OWN CHAINS")
    print("=" * 104)
    print(f"  N = {N}, U = {U}, dtau = {DTAU}, t2 = {T2}, mu = {MU}, "
          f"R = {R}, warm = {WARM}, n_meas = {N_MEAS}, seeds {SEEDS}")
    print("  Each row is the mean over seeds; the relative miss is |pred - deficit| / deficit.")
    print()
    print(f"{'beta':>6} {'deficit':>12} {'|E_perp|^2/(2|A||B|)':>22} {'relative miss':>15}")
    print("-" * 104)

    for beta in BETAS:
        ds, ps = [], []
        for seed in SEEDS:
            m = Model(N=N, t=1.0, t2=T2, mu=MU, U=U, dtau=DTAU,
                      L=int(round(beta / DTAU)))
            A, B, _, _ = run(m, R=R, warm=WARM, n_meas=N_MEAS, seed=seed)
            d, p = decompose(np.asarray(A), np.asarray(B))
            ds.append(d)
            ps.append(p)
        d, p = float(np.mean(ds)), float(np.mean(ps))
        print(f"{beta:6.1f} {d:12.5f} {p:22.5f} {abs(p - d) / d:14.2%}", flush=True)

    print()
    print("  The quadratic form is the leading term of `1 - cos`, so it is exact as the angle goes")
    print("  to zero and low once the angle is not small. The beta at which the miss becomes")
    print("  visible is what section 7.2 quotes; past it the exact cosine is what holds.")
    print()

    print("=" * 104)
    print("AND THE PARALLEL PART IS INVISIBLE: RESCALING ONE CHANNEL MOVES E AND NOT THE DEFICIT")
    print("=" * 104)
    m = Model(N=N, t=1.0, t2=T2, mu=MU, U=U, dtau=DTAU, L=int(round(1.0 / DTAU)))
    A, B, _, _ = run(m, R=R, warm=WARM, n_meas=N_MEAS, seed=SEEDS[0])
    A = np.asarray(A)
    base = 1.0 - A
    print(f"{'B':>18} {'|E|/|A~|':>12} {'deficit':>14}")
    print("-" * 104)
    for label, Bx in (("1 - A", base), ("1 - 2A", 1.0 - 2.0 * A), ("3 - 5A", 3.0 - 5.0 * A)):
        Ac = A - A.mean(axis=0, keepdims=True)
        Bc = Bx - Bx.mean(axis=0, keepdims=True)
        ratio = float(np.linalg.norm(Bc + Ac) / np.linalg.norm(Ac))
        deficit = 1.0 + float(EA.channel_alignment(A, Bx).strength)
        print(f"{label:>18} {ratio:12.5f} {deficit:14.8f}", flush=True)
    print()
    print("  `E` grows with the rescaling and the deficit does not move: the read has no notion of")
    print("  the two channels' relative size, only of their alignment.")


if __name__ == "__main__":
    main()
