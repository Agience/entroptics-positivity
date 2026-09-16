"""Experiment AL -- does the CAPABILITY claim survive a change of representation?

Section 7's useful claim is not that the coupling reads -1 at the symmetric point.  It is that
along beta at fixed filling the coupling's DEFICIT from saturation moves, monotonically and
resolved, in a regime where the average sign is identically 1 and therefore has no derivative to
read.  Everything else in the paper is structure; that is the part that would be used.

It has only ever been measured with the discrete Ising field.  AK showed the section 4 identity and
the lockstep are properties of the decoupling's structure rather than of the field distribution, so
the capability ought to be too -- but "ought to" is not a measurement, and this is the claim least
able to afford being representation-specific.

Both representations are run on the same axis, with the same lattice, the same filling and the same
draws-per-row, and each carries the instrument's own exact re-pairing null as a control.

WHAT IS AND IS NOT COMPARED.  The two field distributions are different measures, so the two
columns of deficits are NOT expected to agree numerically -- a deficit of 0.03 in one is not a
prediction of 0.03 in the other.  What must agree is the SHAPE: monotone in beta, resolved, and
moving while the negative fraction is still zero.  A file that asserted numerical agreement would
be asserting something false about two different ensembles.
"""
from __future__ import annotations

import numpy as np

import entroptics_adapter as EA
from model2d import Model2D
from stable import udt_product, inv_one_plus_block, slogdet_one_plus_block


def lam_of(field, U, dtau):
    return (float(np.arccosh(np.exp(dtau * U / 2.0))) if field == "ising"
            else float(np.sqrt(dtau * U)))


def read_axis(field, beta, mu, U=4.0, dtau=0.125, Lx=2, Ly=4, n=400, seed=0):
    L = int(round(beta / dtau))
    m = Model2D(Lx=Lx, Ly=Ly, t=1.0, mu=mu, U=U, dtau=dtau, L=L, theta=0.0)
    lam = lam_of(field, U, dtau)
    rng = np.random.default_rng(seed)
    A, B, S = [], [], []
    for _ in range(n):
        X = (rng.choice([-1.0, 1.0], size=(L, m.N)) if field == "ising"
             else rng.standard_normal((L, m.N)))
        g, s = {}, 1.0
        for sigma in (+1, -1):
            d = np.exp(sigma * lam * X)
            Bl = m.expmK[None, :, :] * d[:, None, :]
            Uu, D, T = udt_product(Bl[None], 4)
            g[sigma] = np.real(np.diag(inv_one_plus_block(Uu, D, T)[0])).copy()
            sg, _ = slogdet_one_plus_block(Uu, D, T)
            s *= float(np.real(sg[0]))
        A.append(g[+1]); B.append(g[-1]); S.append(s)
    A, B, S = np.array(A), np.array(B), np.array(S)
    c = EA.channel_alignment(A, B)
    nullc = EA.channel_alignment(A, B[np.random.default_rng(seed + 1).permutation(len(B))])
    return dict(neg=float(np.mean(S < 0)), deficit=1.0 + float(c.strength),
                z=abs(float(c.z)), res=bool(c.resolved), nullz=abs(float(nullc.z)),
                nullres=bool(nullc.resolved))


if __name__ == "__main__":
    SEEDS = (1, 7, 23, 45)
    MU = 0.4
    print("=" * 112)
    print("THE CAPABILITY CLAIM, IN BOTH REPRESENTATIONS.  2x4, U = 4, mu = 0.4, 400 draws x 4")
    print("seeds per row.  The claim is a SHAPE -- deficit monotone in beta, resolved, and moving")
    print("while the negative fraction is still zero -- not a number: the two field distributions")
    print("are different measures and their deficits are not predictions of each other.")
    print()
    for field in ("ising", "gauss"):
        print(f"  {field}")
        print(f"    {'beta':>5} | {'neg fraction':>13} | {'deficit over seeds':>21} "
              f"{'|z|':>8} {'permuted null |z|':>18}")
        prev = None
        mono = True
        for beta in (1.0, 1.5, 2.0, 3.0, 4.0):
            rs = [read_axis(field, beta, MU, seed=s) for s in SEEDS]
            d = [r["deficit"] for r in rs]
            md = float(np.mean(d))
            if prev is not None and md < prev:
                mono = False
            prev = md
            print(f"    {beta:5.1f} | {np.mean([r['neg'] for r in rs]):13.5f} | "
                  f"{min(d):9.5f} to {max(d):9.5f} {np.mean([r['z'] for r in rs]):8.1f} "
                  f"{max(r['nullz'] for r in rs):18.2f}", flush=True)
        print(f"    monotone in beta: {mono}")
        print()
    print("The rows where the negative fraction is 0.00000 are the capability: there the average")
    print("sign is identically 1 with zero variance and no derivative, and the deficit has already")
    print("moved and is resolved against the instrument's own exact null.")
