"""The boundary of §5 is intrinsic on one side and mechanism-dependent on the other.

Both halves are pinned because both are places a later draft could overstate:

  NOT AN ARTIFACT OF ONE SCALAR.  The +1 / -1 blockwise reading in `test_boundary.py` came from
      splitting the frame into real and imaginary parts, which uses knowledge of the mechanism.
      Asked for directions with no such hint, the instrument returns the directions of ONE
      CHANNEL'S OWN VARIANCE, and the per-direction couplings are not +-1.

  AN IMPOSSIBILITY, NOT A LIMITATION.  Three systems with `numpy.array_equal(A, B)` True, the same
      saturated read, and negative fractions of 0.0000, 0.0300 and 0.3333.  A pair with `B = A`
      carries no between-channel content, so no comparison between channels separates them.
      This says nothing about whether a read of a SINGLE channel could, and the tests do not
      assert anything in that direction.
"""
from __future__ import annotations

import numpy as np
import pytest

import entroptics as E
from reads.expAH_boundary_is_intrinsic import channels, embed, per_direction


def test_the_complex_frame_and_its_real_embedding_agree():
    """The library reduces a complex frame through iota(x) = (Re x, Im x). Checked, not assumed."""
    A, B, _ = channels(4, 3, 8.0, -4.0, np.pi / 16, True, seed=7)
    direct = E.reads.coupling(A, B)
    embedded = E.reads.coupling(embed(A), embed(B))
    assert direct.strength == pytest.approx(embedded.strength, abs=1e-9)


@pytest.mark.parametrize("phi_over_pi", [1 / 16, 3 / 16])
def test_the_instrument_does_not_recover_the_split_unaided(phi_over_pi):
    """The per-direction couplings are NOT +-1, so the split needed the mechanism to be known."""
    A, B, W = channels(4, 3, 8.0, -4.0, phi_over_pi * np.pi, True, seed=7)
    assert float(np.mean(np.real(W) < 0)) == 0.0, "this row must be sign-free"
    per = [v for v in per_direction(embed(A), embed(B)) if v is not None]
    assert len(per) >= 2, "nothing resolved, so there is no split to fail to recover"
    # The AIDED split -- real and imaginary parts, which is what knowing the mechanism buys --
    # saturates. At least one unaided direction must depart from saturation by more than that
    # split does. Both sides measured here; `0.05` was a window drawn round +-1 by hand.
    aided = [float(E.reads.coupling(np.real(A), np.real(B)).strength),
             float(E.reads.coupling(np.imag(A), np.imag(B)).strength)]
    aided_departure = max(abs(abs(v) - 1.0) for v in aided)
    assert max(abs(abs(v) - 1.0) for v in per) > aided_departure, \
        f"the instrument recovered the split unaided: {per} against aided {aided}"


def test_the_unsaturated_reading_is_not_a_hidden_magnitude():
    """A non-zero `phase` could mean the signed real part is averaging a magnitude away.

    It does not: the largest phase measured inflates the magnitude by 1/cos(phase) = 1.023, which
    moves 0.4794 to 0.490 and nowhere near saturation.
    """
    A, B, _ = channels(4, 3, 8.0, -4.0, 3 * np.pi / 16, True, seed=7)
    c = E.reads.coupling(A, B)
    magnitude = abs(c.strength) / np.cos(c.phase)
    # The inflated magnitude must still sit nearer the signed reading it came from than to
    # saturation -- both measured here. `> 0.2` was a distance chosen from saturation.
    assert abs(magnitude - 1.0) > abs(magnitude - abs(c.strength)), \
        f"the magnitude saturated where the signed part did not: {magnitude:.4f} " \
        f"from a signed read of {abs(c.strength):.4f}"


def test_identical_channels_collide_across_different_sign_behaviour():
    """THE IMPOSSIBILITY. Same read input shape, same output, sign behaviour spanning 33 points."""
    rows = []
    for phi, tr in ((0.0, True), (np.pi / 16, False), (3 * np.pi / 16, False)):
        A, B, W = channels(4, 3, 8.0, -4.0, phi, tr, seed=7)
        Ae, Be = embed(A), embed(B)
        assert np.array_equal(Ae, Be), "the two channels were not bit-identical"
        c = E.reads.coupling(Ae, Be)
        rows.append((float(np.mean(np.real(W) < 0)), float(c.strength), bool(c.resolved)))

    assert all(r for _, _, r in rows)
    strengths = [s for _, s, _ in rows]
    assert all(s == pytest.approx(1.0, abs=1e-4) for s in strengths), \
        f"the read did not collide: {strengths}"
    negs = [n for n, _, _ in rows]
    # THE COLLISION, stated as one: the sign behaviour must span MORE than the read does across
    # the same three rows. That is the whole claim, and it needs no level -- `> 0.3` was one.
    assert max(negs) - min(negs) > max(strengths) - min(strengths), \
        f"the sign behaviour spans no more than the read does: sign {negs}, read {strengths}"


# ── a second collision: exactly related channels, different sign behaviour ───

def test_an_exact_channel_relation_does_not_determine_the_sign():
    """§4 route B: the channels satisfy G_dn = 1 - G_up exactly AND the weight carries a phase.

    This is a different statement from the collision above. There the two channels are identical
    as data, so no comparison remains. Here they are distinct and exactly related -- the
    comparison is well posed -- and it is still blind, because the relation holds whether the
    weight is positive or spread across the circle.
    """
    import entroptics as E
    from scipy.linalg import expm
    from stable import udt_product, inv_one_plus_block, slogdet_one_plus_block
    from reads.expAO_spectral_criterion import ring, tri_ladder, build, route_B

    def probe(Kmat, beta=8.0, U=4.0, dtau=0.125, n=200, seed=5):
        L = int(round(beta / dtau)); N = Kmat.shape[0]
        eK = expm(-dtau * np.asarray(Kmat, dtype=complex))
        lam = float(np.arccosh(np.exp(dtau * U / 2.0)))
        rng = np.random.default_rng(seed)
        A, B, W = [], [], []
        for _ in range(n):
            X = rng.choice([-1.0, 1.0], size=(L, N)); g = {}; d_ = {}
            for sg in (+1, -1):
                d = np.exp(sg * lam * X)
                Bl = eK[None, :, :] * d[:, None, :]
                Uu, D, T = udt_product(Bl[None], 4)
                g[sg] = np.diag(inv_one_plus_block(Uu, D, T)[0]).copy()
                s, _ = slogdet_one_plus_block(Uu, D, T); d_[sg] = complex(s[0])
            A.append(g[+1]); B.append(g[-1]); W.append(d_[+1] * d_[-1])
        A, B = np.array(A), np.array(B)
        emb = lambda M: np.concatenate([M.real, M.imag], axis=1)
        return (float(np.abs(A + B - 1.0).max()),
                1.0 - float(abs(np.asarray(W).mean())),
                float(E.reads.coupling(emb(A), emb(B)).strength))

    rows, reads = [], []
    for K in (build(2, 4, 0, 0, 0), ring(5, np.pi / 2), tri_ladder(8, np.pi / 2)):
        Kc = np.asarray(K, dtype=complex)
        rel, deficit, strength = probe(Kc)
        assert rel < 1e-8, f"the channel relation is not exact here: {rel:.2e}"
        assert strength == pytest.approx(-1.0, abs=1e-3), f"the read did not saturate: {strength}"
        rows.append(deficit); reads.append(strength)

    # Same form as the collision above: the sign behaviour must span more than the read does.
    assert max(rows) - min(rows) > max(reads) - min(reads), \
        f"the sign behaviour spans no more than the read does: sign {rows}, read {reads}"
    assert min(rows) < 1e-6, "no row is actually sign-free, so nothing is being collided"


def test_the_phase_problem_is_a_common_mode_and_is_readable_one_sidedly():
    """Why the relation reads are blind, and that the quantity is not unmeasurable.

    Two assertions, and they belong together:
      * the two determinants' PHASES are aligned (coupling +1), so the weight's phase is a common
        mode -- invisible to any comparison between the two sides;
      * read one-sidedly, `concentration.resultant` on the unit-modulus weight equals the mean
        phase exactly, so nothing here is beyond measurement.
    """
    import entroptics as E
    from scipy.linalg import expm
    from stable import udt_product, slogdet_one_plus_block
    from reads.expAO_spectral_criterion import ring, tri_ladder

    def dets(Kmat, beta=8.0, U=4.0, dtau=0.125, n=300, seed=5):
        L = int(round(beta / dtau)); N = Kmat.shape[0]
        eK = expm(-dtau * np.asarray(Kmat, dtype=complex))
        lam = float(np.arccosh(np.exp(dtau * U / 2.0)))
        rng = np.random.default_rng(seed)
        D1, D2 = [], []
        for _ in range(n):
            X = rng.choice([-1.0, 1.0], size=(L, N)); d_ = {}
            for sg in (+1, -1):
                d = np.exp(sg * lam * X)
                Bl = eK[None, :, :] * d[:, None, :]
                Uu, D, T = udt_product(Bl[None], 4)
                s, _ = slogdet_one_plus_block(Uu, D, T)
                d_[sg] = complex(s[0])
            D1.append(d_[+1]); D2.append(d_[-1])
        return np.array(D1), np.array(D2)

    from reads.expAO_spectral_criterion import build as _build
    C1, C2 = dets(np.asarray(_build(2, 4, 0, 0, 0), dtype=complex))
    cw = C1 * C2
    clean = float(abs((cw / np.abs(cw)).reshape(-1, 1).mean()))

    for K in (ring(5, np.pi / 2), tri_ladder(8, np.pi / 2)):
        D1, D2 = dets(np.asarray(K, dtype=complex))
        u1 = (D1 / np.abs(D1)).reshape(-1, 1)
        u2 = (D2 / np.abs(D2)).reshape(-1, 1)
        c = E.reads.coupling(u1, u2)
        assert c.strength == pytest.approx(1.0, abs=1e-3), \
            f"the determinants' phases are not a common mode: {c.strength:.4f}"

        w = (D1 * D2); u = (w / np.abs(w)).reshape(-1, 1)
        truth = float(abs(u.mean()))
        # Against a lattice with no phase problem, run through the SAME code just above, rather
        # than against `0.95`. A row that carries a phase must read below the clean one.
        assert truth < clean, \
            f"this row carries no phase problem: {truth:.5f} against clean {clean:.5f}"
        assert float(E.reads.concentration(u).resultant) == pytest.approx(truth, abs=1e-9)
