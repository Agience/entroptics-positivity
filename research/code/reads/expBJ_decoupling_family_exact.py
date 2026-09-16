"""Section 9.3's sweep over the decoupling family, summed exactly rather than sampled.

Section 9.3 concludes that the spin channel wins with no interior optimum. It argued that from the
negative fraction of `Re w` over draws from the PRIOR, and section 8 says plainly that prior draws
support no expectation value: the effective sample size there is 1.5 to 13 of 400. A statement about
which decoupling has the worse sign problem is a statement about `<sgn>`, and `<sgn>` is an
expectation.

So it is summed here instead of sampled. Every auxiliary field is enumerated -- both of them, the
real spin field `X` and the imaginary charge field `Y`, which is `4^(N L)` configurations -- and the
average sign comes out exactly:

    <sgn> = |sum w| / sum |w|

with no sampler, no seed and no effective sample size to worry about. The lattice is small because
exactness costs exponentially; what the sweep establishes is the ORDERING across theta, and the
ordering is what section 9.3 claims.

The same enumeration reproduces the mean-phase deficit and the negative fraction of `Re w`, so the
quantities section 9.3 used are printed beside the one it needed.

    python remote_run.py reads/expBJ_decoupling_family_exact.py
"""
from __future__ import annotations

import itertools

import numpy as np

from model2d import Model2D
from stable import slogdet_one_plus_block, udt_product

LX, LY, L, DTAU, U = 2, 1, 4, 0.25, 4.0
THETAS = (0.0, 0.25, 0.5, 0.75, 1.0)
BATCH = 4096


def exact_sums(m):
    """(<sgn>, mean-phase deficit, negative fraction of Re w) by enumerating every field pair."""
    N = m.N
    tot_w = 0.0 + 0.0j
    tot_abs = 0.0
    tot_phase = 0.0 + 0.0j
    n_neg = 0
    n_all = 0
    fields = itertools.product([-1.0, 1.0], repeat=2 * N * L)
    while True:
        chunk = list(itertools.islice(fields, BATCH))
        if not chunk:
            break
        arr = np.array(chunk).reshape(len(chunk), 2, L, N)
        X, Y = arr[:, 0], arr[:, 1]
        dets = {}
        for sigma in (+1, -1):
            Bl = m.B_slices(X, Y, sigma)
            Uu, D, T = udt_product(Bl, 4)
            s, la = slogdet_one_plus_block(Uu, D, T)
            dets[sigma] = np.asarray(s) * np.exp(np.asarray(la))
        w = dets[+1] * dets[-1] * m.charge_phase(Y)
        aw = np.abs(w)
        keep = aw > 0
        tot_w += complex(w.sum())
        tot_abs += float(aw.sum())
        tot_phase += complex((w[keep] / aw[keep]).sum())
        n_neg += int((w.real < 0).sum())
        n_all += len(w)
    return (abs(tot_w) / tot_abs, 1.0 - abs(tot_phase) / n_all, n_neg / n_all, n_all)


def main():
    print("=" * 100)
    print("THE DECOUPLING FAMILY, SUMMED EXACTLY OVER EVERY AUXILIARY FIELD")
    print("=" * 100)
    print(f"  {LX}x{LY} lattice, L = {L}, dtau = {DTAU}, beta = {DTAU * L}, U = {U}")
    print(f"  {4 ** (LX * LY * L)} field pairs per row -- no sampler, no seed, no effective")
    print("  sample size. `<sgn>` here is |sum w| / sum |w|, the definition.")
    print()

    for mu in (0.0, 0.8):
        print(f"  mu = {mu}")
        print(f"{'theta':>8} {'<sgn> (exact)':>16} {'mean-phase deficit':>20} "
              f"{'neg fraction of Re w':>22}")
        print("-" * 100)
        rows = []
        for theta in THETAS:
            m = Model2D(Lx=LX, Ly=LY, t=1.0, mu=mu, U=U, dtau=DTAU, L=L, theta=theta)
            sgn, deficit, neg, n = exact_sums(m)
            rows.append((theta, sgn, deficit, neg))
            print(f"{theta:8.2f} {sgn:16.8f} {deficit:20.8f} {neg:22.8f}", flush=True)
        # A tie is a tie: at half filling every theta is sign-free and the values differ only in
        # floating-point dust, so `>` on raw floats would report an ordering that is not there.
        TIE = 1e-12
        interior = [r for r in rows[1:-1] if r[1] > rows[0][1] + TIE]
        best = max(r[1] for r in rows)
        at_best = [f"{r[0]:.2f}" for r in rows if r[1] > best - TIE]
        print()
        print(f"    best <sgn> = {best:.8f}, attained at theta = {', '.join(at_best)}")
        print(f"    interior theta beating theta = 0 by more than {TIE:g}: {len(interior)}"
              f"   ({'none' if not interior else 'SOME'})")
        print()


if __name__ == "__main__":
    main()
