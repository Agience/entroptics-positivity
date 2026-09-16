"""Generality: the claims hold across lattice geometries, and fail on the control geometries.

Every table in sections 2 to 5 was measured on a single 2x4 lattice while the claims are stated
generally, so all three are re-asserted here across sizes and shapes -- N = 8, 12, 16, 24 -- with
the odd-side lattices as built-in controls.

A periodic lattice is bipartite when both sides are even; an odd side makes an odd ring and
destroys it.  Half filling is `mu = 0` only where the particle-hole symmetry that pins it is
present, which is exactly the bipartite case, so the odd geometries are controls rather than
matched-filling comparisons.

The control is the load-bearing part.  A suite that only checked the geometries where the claims
hold would pass equally on a build that had stopped being sensitive to the bipartite structure at
all.
"""
from __future__ import annotations

import numpy as np
import pytest

import entroptics_adapter as EA
from model2d import Model2D
from stable import udt_product, inv_one_plus_block, slogdet_one_plus_block

BIPARTITE = [(2, 4), (4, 4), (2, 6)]
ODD_RING = [(2, 3), (3, 4)]


def read_lattice(Lx, Ly, beta, U=4.0, dtau=0.125, n_draw=200, seed=0):
    L = int(round(beta / dtau))
    m = Model2D(Lx=Lx, Ly=Ly, t=1.0, mu=0.0, U=U, dtau=dtau, L=L, theta=0.0)
    lam = float(np.arccosh(np.exp(dtau * U / 2.0)))
    rng = np.random.default_rng(seed)
    res, su, sd, A, B = [], [], [], [], []
    for _ in range(n_draw):
        X = rng.choice([-1.0, 1.0], size=(m.L, m.N))
        row, lg, sg = {}, {}, {}
        for sigma in (+1, -1):
            d = np.exp(sigma * lam * X)
            Bl = m.expmK[None, :, :] * d[:, None, :]
            Uu, D, T = udt_product(Bl[None], 4)
            row[sigma] = np.real(np.diag(inv_one_plus_block(Uu, D, T)[0])).copy()
            s, la = slogdet_one_plus_block(Uu, D, T)
            sg[sigma] = float(np.real(s[0])); lg[sigma] = float(la[0])
        res.append(abs((lg[+1] - lg[-1]) -
                       (-dtau * L * float(np.trace(m.K)) + lam * float(X.sum()))))
        su.append(sg[+1]); sd.append(sg[-1]); A.append(row[+1]); B.append(row[-1])
    su, sd = np.array(su), np.array(sd)
    return dict(res=np.array(res), flip=float(np.mean(su < 0)),
                agree=float(np.mean(su == sd)), neg=float(np.mean(su * sd < 0)),
                coupling=EA.channel_alignment(np.array(A), np.array(B)))


@pytest.mark.parametrize("Lx,Ly", BIPARTITE)
def test_identity_and_lockstep_hold_on_every_bipartite_geometry(Lx, Ly):
    r = read_lattice(Lx, Ly, beta=10.0, seed=Lx * 100 + Ly)
    assert r["res"].max() < 1e-9, f"identity failed at {r['res'].max():.2e}"
    assert r["flip"] > 0.0, "no channel flipped -- the lockstep claim would be vacuous here"
    assert r["agree"] == 1.0
    assert r["neg"] == 0.0


@pytest.mark.parametrize("Lx,Ly", BIPARTITE)
def test_the_calibration_is_geometry_independent(Lx, Ly):
    r = read_lattice(Lx, Ly, beta=10.0, seed=Lx * 100 + Ly + 7)
    c = r["coupling"]
    assert c.resolved
    assert c.strength == pytest.approx(-1.0, abs=1e-4)


@pytest.mark.parametrize("Lx,Ly", ODD_RING)
def test_the_odd_ring_control_fails_all_three(Lx, Ly):
    """THE CONTROL. An odd side is not bipartite, and all three claims must break together."""
    r = read_lattice(Lx, Ly, beta=10.0, seed=Lx * 100 + Ly + 3)
    assert r["res"].max() > 1.0, "the identity held on a non-bipartite lattice"
    assert r["agree"] < 1.0, "the channels stayed locked without the bipartite structure"
    assert r["neg"] > 0.0, "no sign problem appeared, so nothing is being controlled for"
    c = r["coupling"]
    # Against a bipartite lattice, where the symmetry does force -1 -- measured here rather than
    # compared to `0.05`, which was a distance from -1 chosen by hand.
    bip = read_lattice(2, 4, beta=10.0, seed=7)["coupling"]
    assert abs(bip.strength + 1.0) < 1e-4, "the bipartite reference no longer reads -1"
    assert (not c.resolved) or abs(c.strength + 1.0) > abs(bip.strength + 1.0), \
        f"the coupling still read -1 without the symmetry that forces it: {c.strength:.5f}"
