"""THE MEASURED BOUNDARY. Where the desynchronisation reading stops, pinned as a gate.

Sections 2 to 5 read two sign-free mechanisms and both relate the two channels affinely.  A third
mechanism does not, and it bounds the claim:

  WU-ZHANG / KRAMERS.  A spin-dependent twist makes the two spins time-reversal partners,
      `K_dn = conj(K_up)`.  With attractive U in the charge channel `B_dn = conj(B_up)`, so the
      weight is `|det|^2 >= 0`.  Positivity is CONJUGATE PAIRING, not an affine relation.

These tests assert the FAILURE of "saturated if and only if sign-free", in both directions, because
that equivalence is the overstatement the paper is at risk of and only a test keeps it out:

  NOT NECESSARY   the time-reversal rows are sign-free and the combined read does not saturate.
  NOT SUFFICIENT  the control rows carry a sign problem and the combined read reads exactly 1.

Read BLOCKWISE the mechanism is fully visible -- +1 on the real part, -1 on the imaginary part --
which is why the failure is about a single signed scalar averaging two subspaces, and not about the
instrument.

A GUARD TRAVELS WITH THESE TESTS.  The total phase round the x-cycle is `Lx * phi`, so a twist that
is a multiple of pi is gauge-equivalent to a real boundary condition and does nothing; on a 2-wide
lattice the phase cancels outright.  `test_an_inert_twist_is_detected_as_inert` pins the guard,
because the first version of this experiment ran on a 2-wide lattice and printed a full table while
testing nothing.
"""
from __future__ import annotations

import numpy as np
import pytest

from reads.expAF_third_mechanism import run

SEEDS = (1, 7, 23, 45, 93)
LIVE = [(4, 3, 0.0625), (4, 3, 0.1875), (3, 4, 0.125), (5, 2, 0.1)]


@pytest.mark.parametrize("Lx,Ly,pp", LIVE)
def test_the_kramers_mechanism_is_sign_free_and_exact(Lx, Ly, pp):
    """The mechanism itself: conjugation exact, weight real, no negative weight."""
    for seed in SEEDS:
        r = run(Lx, Ly, 8.0, -4.0, 0.0, pp * np.pi, True, seed=seed)
        assert r["imag"] > 1e-8, "the twist is inert, so this row tests nothing"
        assert r["conj"] == 0.0, f"G_dn != conj(G_up): {r['conj']:.3e}"
        assert r["phase"] == 0.0, f"the weight was not real: {r['phase']:.3e}"
        assert r["neg"] == 0.0, "a negative weight appeared in a sign-free mechanism"


@pytest.mark.parametrize("Lx,Ly,pp", LIVE)
def test_saturation_is_not_necessary_for_sign_freedom(Lx, Ly, pp):
    """SIGN-FREE AND UNSATURATED. The read averages +1 and -1 over two orthogonal subspaces."""
    for seed in SEEDS:
        r = run(Lx, Ly, 8.0, -4.0, 0.0, pp * np.pi, True, seed=seed)
        assert r["neg"] == 0.0
        assert r["sre"] == pytest.approx(+1.0, abs=1e-4)
        assert r["sim"] == pytest.approx(-1.0, abs=1e-4)
        # The two subspace reads saturate; the combined one must depart from saturation by more
        # than they do. Both references are measured on the two lines above, so no level is picked.
        sub = max(abs(abs(r["sre"]) - 1.0), abs(abs(r["sim"]) - 1.0))
        assert abs(abs(r["s"]) - 1.0) > sub, \
            f"the combined read saturated on the Kramers mechanism: {r['s']:.4f} " \
            f"against subspace reads {r['sre']:.4f} / {r['sim']:.4f}"


@pytest.mark.parametrize("Lx,Ly,pp", LIVE)
def test_saturation_is_not_sufficient_for_sign_freedom(Lx, Ly, pp):
    """A SIGN PROBLEM UNDER A SATURATED READ.

    Same lattice, same interaction, same field; only time reversal removed. The channels go
    bit-identical, so the read saturates on a degenerate input, while the model's sign problem
    moves into a phase COMMON to both channels where no between-channel comparison reaches it.
    """
    for seed in SEEDS:
        r = run(Lx, Ly, 8.0, -4.0, 0.0, pp * np.pi, False, seed=seed)
        # Against the same lattice WITH time reversal, whose weight is exactly real -- the one
        # thing this test changes. `> 1.0` was a level; this is the comparison it stood in for.
        kept = run(Lx, Ly, 8.0, -4.0, 0.0, pp * np.pi, True, seed=seed)
        assert kept["phase"] == 0.0, "the time-reversal-preserved control is no longer real"
        assert r["neg"] > 0.0, "no sign problem in the control, so nothing is being shown"
        assert r["phase"] > kept["phase"], "the control's weight stayed real"
        assert r["s"] == pytest.approx(1.0, abs=1e-4), \
            f"the control did not saturate: {r['s']:.4f}"


@pytest.mark.parametrize("Lx,Ly,pp", [(4, 3, 0.25), (4, 3, 0.5)])
def test_an_inert_twist_is_detected_as_inert(Lx, Ly, pp):
    """THE GUARD. `Lx * phi` a multiple of pi is gauge-equivalent to a real boundary condition."""
    r = run(Lx, Ly, 8.0, -4.0, 0.0, pp * np.pi, True, seed=3)
    assert r["imag"] < 1e-8, \
        f"this twist was expected to be inert but Im G reached {r['imag']:.3e}"


def test_a_two_wide_lattice_is_refused():
    """The defect that produced a full table of nothing. It must now raise."""
    with pytest.raises(AssertionError):
        run(2, 4, 8.0, -4.0, 0.0, np.pi / 8, True, seed=1)


# ── the reading is derived, not merely unsaturated ──────────────────────────

@pytest.mark.parametrize("Lx,Ly,pp", LIVE)
@pytest.mark.parametrize("seed", [1, 45])
def test_the_kramers_reading_is_exactly_one_minus_twice_the_imaginary_share(Lx, Ly, pp, seed):
    """G_dn = conj(G_up) makes the real parts exact positives and the imaginary parts exact
    negatives, so a single signed alignment returns +1 - 2r with r the imaginary variance share.

    This is the same formula as §4's route A with one sign changed, and it means the departure
    from saturation is the frame's imaginary weight rather than a failure of the read. The
    seed-to-seed spread in the boundary table is r's spread.
    """
    r_ = run(Lx, Ly, 8.0, -4.0, 0.0, pp * np.pi, True, seed=seed)
    assert r_["neg"] == 0.0, "this row must be sign-free for the point to hold"

    import entroptics as E
    from reads.expAF_third_mechanism import hop_flux
    from scipy.linalg import expm
    from stable import udt_product, inv_one_plus_block

    L = int(round(8.0 / 0.125)); N = Lx * Ly
    Kup = hop_flux(Lx, Ly, phi=pp * np.pi)
    eup, edn = expm(-0.125 * Kup), expm(-0.125 * np.conj(Kup))
    lam = float(np.arccosh(np.exp(0.125 * 4.0 / 2.0)))
    rng = np.random.default_rng(seed)
    A, B = [], []
    for _ in range(200):
        X = rng.choice([-1.0, 1.0], size=(L, N))
        d = np.exp(lam * X)
        g = []
        for ek in (eup, edn):
            Bl = ek[None, :, :] * d[:, None, :]
            Uu, D, T = udt_product(Bl[None], 4)
            g.append(np.diag(inv_one_plus_block(Uu, D, T)[0]).copy())
        A.append(g[0]); B.append(g[1])
    A, B = np.array(A), np.array(B)
    assert float(np.abs(B - np.conj(A)).max()) < 1e-9, "G_dn = conj(G_up) does not hold here"

    Ac = A - A.mean(axis=0, keepdims=True)
    vre = float((Ac.real ** 2).sum()); vim = float((Ac.imag ** 2).sum())
    share = vim / (vre + vim)
    assert E.reads.coupling(A, B).strength == pytest.approx(1.0 - 2.0 * share, abs=1e-10)
