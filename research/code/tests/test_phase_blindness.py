"""§9.2, gated: no read of magnitudes alone can see a sign, at any exponent.

This is the reason §7 reads a coupling between the two channels rather than a magnitude of either,
so it is worth pinning rather than asserting. `B = diag(sigma) A` leaves every magnitude untouched,
so every power marginal agrees EXACTLY -- not to a tolerance -- and the singular spectrum agrees
too because `diag(sigma)` is orthogonal.

The positive control is what makes it a statement about magnitude reads: a read that compares the
two sides does see the flip. Without it the test would be consistent with the frames simply being
indistinguishable.
"""
from __future__ import annotations

import numpy as np
import pytest

import entroptics_adapter as EA
from closed.expAP_phase_blindness import flipped, magnitude_moments

QS = (0.5, 1.0, 1.5, 2.0, 3.0)


def _frames(seed=0, T=400, N=8):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((T, N))
    half = np.where(rng.random(T) < 0.5, -1.0, 1.0)
    return A, flipped(A, half), flipped(A, -np.ones(T))


@pytest.mark.parametrize("seed", [0, 5, 11])
def test_every_power_marginal_is_exactly_blind_to_row_sign_flips(seed):
    A, B, C = _frames(seed)
    base = magnitude_moments(A, QS)
    for X in (B, C):
        assert float(np.abs(np.abs(X) - np.abs(A)).max()) == 0.0
        for q, v in magnitude_moments(X, QS).items():
            assert v == base[q], f"q = {q} differed by {abs(v - base[q]):.3e}"


@pytest.mark.parametrize("seed", [0, 5])
def test_the_singular_spectrum_is_blind_too(seed):
    A, B, C = _frames(seed)
    sa = np.linalg.svd(A, compute_uv=False)
    for X in (B, C):
        assert float(np.abs(np.linalg.svd(X, compute_uv=False) - sa).max()) < 1e-12


def test_the_positive_control_a_coupling_does_see_the_flip():
    """Without this, blindness would be consistent with the frames being indistinguishable."""
    A, B, C = _frames(0)
    assert EA.channel_alignment(A, A).strength == pytest.approx(1.0, abs=1e-9)
    assert EA.channel_alignment(A, C).strength == pytest.approx(-1.0, abs=1e-9)
    half = EA.channel_alignment(A, B)
    # Against the two saturated reads measured on the line above, and against the instrument's own
    # decision -- not against `0.1` and `4.0`, which were levels guessed at for both.
    assert not half.resolved, \
        f"a half-flipped frame resolved: {half.strength:.4f}, |z| = {abs(half.z):.2f}"
    assert abs(half.strength) < abs(EA.channel_alignment(A, C).strength), \
        f"a half-flipped frame read as strongly as a fully flipped one: {half.strength:.4f}"


# ── the classification's third row: one channel alone carries no indicator ───

def test_a_single_channel_read_does_not_track_the_sign_deficit():
    """Section 9.2b's third row: a single-channel read carries no indicator of the sign.

    Along a flux sweep the sign deficit rises and falls, and no single-channel spectral read
    follows it as well as the deficit follows itself across seeds. Asserted as a rank correlation
    below the deficit's own reproducibility, which is the claim and is the comparison the paper
    makes; a test that checked a couple of points could pass on a monotone stretch.

    `reads/expBG_single_channel_has_no_indicator.py` is the same question over a full flux period
    and across six lattices, and is where section 9.2b's figures come from.
    """
    from scipy.linalg import expm
    from scipy.stats import spearmanr

    from stable import udt_product, inv_one_plus_block, slogdet_one_plus_block
    from reads.expAO_spectral_criterion import tri_ladder

    def probe(K, beta=8.0, U=4.0, dtau=0.125, n=200, seed=5):
        L = int(round(beta / dtau)); N = K.shape[0]
        eK = expm(-dtau * np.asarray(K, dtype=complex))
        lam = float(np.arccosh(np.exp(dtau * U / 2.0)))
        rng = np.random.default_rng(seed)
        A, W = [], []
        for _ in range(n):
            X = rng.choice([-1.0, 1.0], size=(L, N)); d_ = {}; g = None
            for sg in (+1, -1):
                d = np.exp(sg * lam * X)
                Bl = eK[None, :, :] * d[:, None, :]
                Uu, D, T = udt_product(Bl[None], 4)
                s, _ = slogdet_one_plus_block(Uu, D, T); d_[sg] = complex(s[0])
                if sg == +1:
                    g = np.diag(inv_one_plus_block(Uu, D, T)[0]).copy()
            A.append(g); W.append(d_[+1] * d_[-1])
        so = EA.single_channel_optics(np.array(A))
        # circular distance from pi -- a plain |phase - pi| wraps and reads 6.2 for 0.04
        gap = abs(np.angle(np.exp(1j * (float(so.phase) - np.pi))))
        return 1.0 - float(abs(np.asarray(W).mean())), gap, float(so.attenuation)

    FLUX = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)
    rows = [probe(tri_ladder(8, t * np.pi)) for t in FLUX]
    # The SAME sweep on a second seed. It supplies both references this test needs, so neither is
    # a number: how far the deficit moves is compared to how much it wanders between seeds, and
    # 'the reads do not track it' is compared to how well the deficit tracks ITSELF across seeds.
    again = [probe(tri_ladder(8, t * np.pi), seed=17) for t in FLUX]

    deficit = [r[0] for r in rows]
    other = [r[0] for r in again]
    wander = max(abs(a - b) for a, b in zip(deficit, other))
    assert max(deficit) - min(deficit) > wander, \
        f"the sweep moved the sign by no more than it wanders between seeds " \
        f"({wander:.4f}): {deficit}"

    rho_self, _ = spearmanr(deficit, other)
    for idx, name in ((1, "phase"), (2, "attenuation")):
        rho, _ = spearmanr(deficit, [r[idx] for r in rows])
        assert abs(rho) < abs(rho_self), \
            f"{name} tracked the deficit after all: rho = {rho:+.3f} against the deficit's own " \
            f"reproducibility {rho_self:+.3f}"
