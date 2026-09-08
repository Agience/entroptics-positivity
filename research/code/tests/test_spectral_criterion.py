"""The criterion behind §4, gated: a signed-diagonal conjugation of the one-body matrix.

The §3 identity holds exactly when `S K S = -K` for some diagonal S of +-1 -- elementwise
`s_i s_j K_ij = -K_ij`, which is a 2-colouring of K's support graph plus a zero diagonal.  It is
decided from K alone: no eigenvalues, no determinants, no sampling.

Three things are gated, and the second and third are what stop the first being a re-description of
the cases it was built from:

  * the criterion agrees with the measured identity on every matrix tried;
  * it PREDICTS correctly on bond-disordered lattices, where no symmetry was designed in;
  * the SPECTRAL criterion -- symmetry of K's spectrum about zero, which for Hermitian K is exactly
    similarity to -K -- is asserted to be WRONG, because a staggered potential satisfies it to
    1e-15 and breaks the identity anyway.  Without that test a later draft could quietly replace
    the signed-diagonal criterion with the spectral one, which is the natural thing to reach for.
"""
from __future__ import annotations

import numpy as np
import pytest

import entroptics as E
from scipy.linalg import expm

from stable import udt_product, inv_one_plus_block

from reads.expAO_spectral_criterion import (
    build, chain, criterion, measure, ring, route_A, route_B, signed_diagonal_conjugation,
    spectral_asymmetry, star, tri_ladder, triangular, triangular_ladder)

BETA, DRAWS = 8.0, 120


def holds(K, seed=3):
    r, _ = measure(K, BETA, n_draw=DRAWS, seed=seed)
    return r < 1e-9, r


def disordered(Lx, Ly, strength, seed):
    """A bipartite lattice with RANDOM bond strengths: 2-colourable, zero diagonal, no design."""
    K = build(Lx, Ly, 0.0, 0.0, 0.0)
    rng = np.random.default_rng(seed)
    N = Lx * Ly
    for i in range(N):
        for j in range(i + 1, N):
            if abs(K[i, j]) > 1e-12:
                f = 1.0 + strength * rng.standard_normal()
                K[i, j] *= f; K[j, i] *= f
    return K


@pytest.mark.parametrize("tp,mu,h", [
    (0.0, 0.0, 0.0), (0.0, 0.0, 0.2), (0.0, 0.0, 1.2),
    (0.3, 0.0, 0.0), (0.0, 0.4, 0.0),
])
def test_the_criterion_agrees_with_the_measured_identity(tp, mu, h):
    K = build(2, 4, tp, mu, h)
    predicted = signed_diagonal_conjugation(K) is not None
    actual, r = holds(K)
    assert predicted == actual, \
        f"criterion said {predicted}, identity residual {r:.3e} at tp={tp} mu={mu} h={h}"


@pytest.mark.parametrize("strength", [0.3, 0.8, 1.5])
def test_the_criterion_predicts_on_a_lattice_with_no_designed_symmetry(strength):
    """PREDICTION, not description: random bonds, and the identity must still hold exactly."""
    K = disordered(2, 4, strength, seed=int(strength * 10))
    assert signed_diagonal_conjugation(K) is not None
    ok, r = holds(K)
    assert ok, f"the identity failed on a 2-colourable disordered lattice at {r:.3e}"


def test_a_staggered_potential_breaks_it_while_the_lattice_stays_bipartite():
    """The case §4's stated conditions do not cover: a diagonal term cannot be negated by signs."""
    K = build(2, 4, 0.0, 0.0, 0.6)
    assert signed_diagonal_conjugation(K) is None
    ok, r = holds(K)
    assert not ok and r > 1.0, f"the staggered potential did not break the identity: {r:.3e}"


def test_the_spectral_criterion_is_not_the_criterion():
    """THE NEGATIVE CONTROL ON THE CRITERION ITSELF.

    Symmetry of the spectrum about zero is, for Hermitian K, exactly similarity to -K. A staggered
    potential satisfies it to machine precision and breaks the identity, so the similarity that
    matters is not any unitary but one that commutes with the interaction's diagonal factor.
    """
    for h in (0.2, 0.6, 1.2):
        K = build(2, 4, 0.0, 0.0, h)
        assert spectral_asymmetry(K) < 1e-12, "the staggered spectrum is not symmetric here"
        ok, r = holds(K)
        assert not ok, f"the identity survived at h = {h}, so this control is vacuous"


@pytest.mark.parametrize("n", [5, 7])
def test_an_odd_ring_is_refused_by_the_colouring(n):
    K = np.zeros((n, n))
    for i in range(n):
        K[i, (i + 1) % n] = -1.0
        K[(i + 1) % n, i] = -1.0
    assert signed_diagonal_conjugation(K) is None
    ok, r = holds(K)
    assert not ok and r > 1.0, f"the identity held on an odd ring: {r:.3e}"


# ── the criterion is graph-theoretic, so it must hold off the square lattice ──

@pytest.mark.parametrize("name,K,expected", [
    ("chain 8 periodic", chain(8), True),
    ("chain 7 periodic", chain(7), False),
    ("chain 8 open", chain(8, False), True),
    ("star (a tree)", star(8), True),
    ("triangular ladder", triangular_ladder(8), False),
    ("even ring 12", ring(12), True),
    ("odd ring 9", ring(9), False),
])
def test_the_criterion_holds_on_topologies_that_are_not_lattices(name, K, expected):
    """The prediction is made from the graph and then measured, on shapes the paper never uses.

    The star graph is the one that matters: a tree, no cycles, nothing lattice-like, and the
    criterion says the identity holds there -- which it does.
    """
    predicted = signed_diagonal_conjugation(K) is not None
    assert predicted == expected, f"{name}: criterion said {predicted}"
    r, _ = measure(K, BETA, n_draw=100, seed=7)
    assert (r < 1e-9) == expected, f"{name}: identity residual {r:.3e}"


# ── sharpness: the criterion has no tolerance, and the sign problem does ─────

def test_the_identity_residual_is_linear_in_the_violation():
    """No tolerance and no onset: the residual is proportional to the perturbation from the first.

    Asserted as a RATIO holding across decades rather than as a value, so the test says "linear"
    and not "equal to 85.4", which is a property of this beta, N and U.
    """
    ratios, resid = [], []
    for eps in (1e-6, 1e-4, 1e-3):
        K = build(2, 4, 0.0, 0.0, eps)
        assert signed_diagonal_conjugation(K) is None, f"criterion accepted eps = {eps}"
        r, _ = measure(K, BETA, n_draw=100, seed=11)
        ratios.append(r / eps); resid.append(r)
    # Linearity as a comparison between two spreads measured in this same loop: the RATIO must
    # vary far less across the decades than the residual it is built from. `< 0.05` was a
    # tolerance chosen for what counts as linear.
    rel = lambda v: (max(v) - min(v)) / min(v)
    assert rel(ratios) < rel(resid), \
        f"the residual is not linear in the violation: ratios span {rel(ratios):.3e} against " \
        f"residuals spanning {rel(resid):.3e}: {ratios}"


def test_a_broken_identity_can_still_carry_no_sign_problem():
    """Structurally fragile, numerically robust -- and they are different quantities.

    This is what stops the criterion being read as a predictor of the negative fraction: at
    eps = 1e-3 the identity is broken by eight orders over its clean value and not one weight in
    200 draws is negative.
    """
    K = build(2, 4, 0.0, 0.0, 1e-3)
    assert signed_diagonal_conjugation(K) is None
    r, neg = measure(K, BETA, n_draw=200, seed=11)
    assert r > 1e-3, f"the identity was not measurably broken: {r:.3e}"
    assert neg == 0.0, f"a negative weight appeared, so the two are not separable here: {neg}"


# ── complex hoppings: two routes, and the identity holds when either is open ──

@pytest.mark.parametrize("N,flux_over_pi,expected", [
    (5, 0.0, False), (5, 0.25, False), (5, 0.5, True), (5, 1.0, False), (5, 1.5, True),
    (7, 0.0, False), (7, 0.5, True), (7, 1.5, True),
    (6, 0.0, True), (6, 0.25, True), (6, 0.5, True), (6, 1.5, True),
    (8, 0.25, True), (8, 1.0, True),
])
def test_the_two_route_criterion_on_flux_threaded_rings(N, flux_over_pi, expected):
    """An ODD cycle carrying half-odd-integer flux restores what the odd cycle destroys.

    Route A (`S K S^-1 = -K`) needs every cycle even and ignores flux; route B
    (`S K S^-1 = -conj(K)`) needs `(-1)^l e^{-2i Phi} = 1`, which odd cycles meet at flux
    `pi/2 mod pi`. The identity holds when either is open.
    """
    K = ring(N, flux_over_pi * np.pi)
    assert criterion(K) == expected,         f"criterion said {criterion(K)} for ring {N} at flux {flux_over_pi}pi"
    r, _ = measure(K, 6.0, n_draw=60, seed=5)
    assert (r < 1e-9) == expected, f"identity residual {r:.3e}"


def test_each_route_is_load_bearing_on_its_own():
    """Neither route alone explains the data, which is why the criterion is a disjunction.

    A 5-ring at flux pi/2 is open only by route B; a 6-ring at flux pi/4 only by route A. A
    criterion using one route would call one of them wrong.
    """
    odd = ring(5, np.pi / 2)
    assert route_B(odd) and not route_A(odd), "route B alone should open the odd ring"
    even = ring(6, np.pi / 4)
    assert route_A(even) and not route_B(even), "route A alone should open the even ring"
    for K in (odd, even):
        r, _ = measure(K, 6.0, n_draw=60, seed=5)
        assert r < 1e-9, f"the identity failed where a route is open: {r:.3e}"


@pytest.mark.parametrize("tp,mu,h", [(0.0, 0.0, 0.0), (0.0, 0.0, 0.6), (0.3, 0.0, 0.0)])
def test_the_general_criterion_reduces_to_the_colouring_on_real_matrices(tp, mu, h):
    """For real K the two routes coincide, so the signed-diagonal colouring is the whole story."""
    K = build(2, 4, tp, mu, h)
    assert criterion(K) == (signed_diagonal_conjugation(K) is not None)


# ── the criterion is constructive: flux restores a frustrated lattice ────────

@pytest.mark.parametrize("theta_over_pi,expected", [
    (0.0, False), (0.25, False), (0.5, True), (0.75, False), (1.0, False), (1.5, True),
])
def test_flux_restores_the_identity_on_a_frustrated_ladder(theta_over_pi, expected):
    """A graph where NO flux-free choice works, and route B names the flux that does."""
    K = tri_ladder(8, theta_over_pi * np.pi)
    assert criterion(K) == expected, f"criterion said {criterion(K)} at {theta_over_pi}pi"
    r, _ = measure(K, 6.0, n_draw=60, seed=5)
    assert (r < 1e-9) == expected, f"identity residual {r:.3e}"


def test_flux_restores_the_identity_and_makes_the_sign_worse():
    """THE CORRECTION. Route B restores the identity on an odd cycle and leaves a PHASE behind.

    The second return of `measure` is `1 - |<w/|w|>|`: the ordinary negative-weight measure when
    the weight is real, and the mean-phase deficit when it is not. An earlier version tested
    `real(s_up) * real(s_dn) < 0`, which is meaningless for a complex determinant -- `slogdet`
    returns a unit-modulus phase there -- and reported 0.0000 on a lattice whose weights are 93%
    phase-carrying. This test exists so that cannot come back.
    """
    bad, dbad = measure(tri_ladder(8, 0.0), 8.0, n_draw=200, seed=9)
    good, dgood = measure(tri_ladder(8, np.pi / 2), 8.0, n_draw=200, seed=9)
    assert bad > 1.0, "the unfluxed ladder's identity should be broken"
    assert good < 1e-9, "flux should restore the identity"
    # The claim is a direction, and it is asserted as one. A factor -- `3x` stood here -- would be
    # a number fitted to the pair it is measured on (0.060 -> 0.522 at this beta and seed).
    assert dgood > dbad, \
        f"the flux did not make the sign quality worse: {dbad:.4f} -> {dgood:.4f}"


def test_only_route_A_leaves_the_weights_real():
    """Route A gives positivity; route B gives the identity without it."""
    a = np.asarray(ring(6, np.pi / 4), dtype=complex)
    assert route_A(a) and not route_B(a)
    ra, da = measure(a, 8.0, n_draw=200, seed=9)
    assert ra < 1e-9 and da == pytest.approx(0.0, abs=1e-6),         f"route A did not deliver positivity: residual {ra:.2e}, deficit {da:.5f}"

    b = np.asarray(ring(5, np.pi / 2), dtype=complex)
    assert route_B(b) and not route_A(b)
    rb, db = measure(b, 8.0, n_draw=200, seed=9)
    assert rb < 1e-9, "route B should still satisfy the identity"
    # Against route A's deficit, measured six lines up and asserted to be zero. That contrast is
    # the whole content of this test; `> 0.05` was a level standing in for it.
    assert db > da, f"route B unexpectedly delivered positivity too: {db:.5f} against {da:.5f}"


# ── two dimensions: the periodic wrap cycles decide it ──────────────────────

@pytest.mark.parametrize("Lx,Ly,theta_over_pi,expected", [
    (4, 4, 0.0, False), (4, 4, 0.25, False), (4, 4, 0.5, True), (4, 4, 1.5, True),
    (3, 3, 0.0, False), (3, 3, 0.5, False), (3, 3, 0.75, False),
])
def test_the_criterion_on_a_periodic_triangular_lattice(Lx, Ly, theta_over_pi, expected):
    """Triangles, rhombi AND wrap cycles must all satisfy route B at once.

    On 4x4 the wraps are even and flux-free so pi/2 opens route B; on 3x3 they are odd and
    flux-free and nothing opens it. The criterion is size-dependent here and correctly so.
    """
    K = triangular(Lx, Ly, theta_over_pi * np.pi)
    assert criterion(K) == expected, f"criterion said {criterion(K)} for {Lx}x{Ly}"
    r, _ = measure(K, 6.0, n_draw=60, seed=5)
    assert (r < 1e-9) == expected, f"identity residual {r:.3e}"


def test_the_triangular_rows_are_about_the_identity_not_about_restoration():
    """Pinned so the 4x4 row is never quoted as positivity restored.

    Its unfluxed version carries a broken identity and no negative weight whatever, so there is
    nothing there to restore. The ladder is where restoration is demonstrated.
    """
    r, neg = measure(triangular(4, 4, 0.0), 12.0, n_draw=150, seed=13)
    assert r > 1.0, "the unfluxed 4x4 identity is no longer broken"
    assert neg == 0.0,         f"the unfluxed 4x4 now shows a sign problem ({neg}); this row could now show restoration"


# ── the corollary: a diagonal term is fatal, and no flux repairs it ──────────

def _dope(K, mu):
    K = np.asarray(K, dtype=complex).copy()
    np.fill_diagonal(K, np.diag(K) - mu)
    return K


@pytest.mark.parametrize("name,fn", [
    ("ring 6", lambda th: ring(6, th)),
    ("ring 5", lambda th: ring(5, th)),
    ("tri ladder", lambda th: tri_ladder(8, th)),
])
def test_no_flux_rescues_a_doped_graph(name, fn):
    """Every diagonal conjugation leaves diag(K) alone, so a chemical potential is fatal.

    Proved in a line -- both routes demand K_ii = -K_ii, and K_ii is real for Hermitian K -- and
    scanned anyway, because a proof about the criterion is not a measurement of the identity.
    """
    # 21 points, so the grid lands exactly on 0.5pi -- the flux route B needs on an odd
    # cycle. A coarser grid misses it and makes the undoped precondition fail for no reason.
    fluxes = np.linspace(0.0, 2.0, 21) * np.pi
    assert any(criterion(fn(f)) for f in fluxes),         f"{name}: no flux opens a route even undoped, so this scan shows nothing"
    for mu in (0.2, 0.6):
        assert not any(criterion(_dope(fn(f), mu)) for f in fluxes),             f"{name}: a flux opened a route at mu = {mu}"
    best = min(measure(_dope(fn(f), 0.2), 6.0, n_draw=40, seed=5)[0] for f in fluxes)
    assert best > 1.0, f"{name}: the identity nearly held at mu = 0.2 ({best:.3e})"


# ── the two routes are not interchangeable for the §5 calibration ───────────

def _channels(K, beta=8.0, U=4.0, dtau=0.125, n=200, seed=5):
    """The two channels' Green's diagonals, complex where K is."""
    L = int(round(beta / dtau)); N = K.shape[0]
    eK = expm(-dtau * np.asarray(K, dtype=complex))
    lam = float(np.arccosh(np.exp(dtau * U / 2.0)))
    rng = np.random.default_rng(seed)
    A, B = [], []
    for _ in range(n):
        X = rng.choice([-1.0, 1.0], size=(L, N))
        g = {}
        for sg in (+1, -1):
            d = np.exp(sg * lam * X)
            Bl = eK[None, :, :] * d[:, None, :]
            Uu, D, T = udt_product(Bl[None], 4)
            g[sg] = np.diag(inv_one_plus_block(Uu, D, T)[0]).copy()
        A.append(g[+1]); B.append(g[-1])
    return np.array(A), np.array(B)


@pytest.mark.parametrize("name,K", [
    ("ring 5 flux pi/2", ring(5, np.pi / 2)),
    ("ring 7 flux pi/2", ring(7, np.pi / 2)),
    ("tri ladder flux pi/2", tri_ladder(8, np.pi / 2)),
])
def test_route_B_gives_the_exact_particle_hole_relation_and_the_exact_minus_one(name, K):
    """Route B is the anti-similarity, and it forces G_dn = 1 - G_up configuration by config."""
    Kc = np.asarray(K, dtype=complex)
    assert route_B(Kc) and not route_A(Kc), f"{name} is not route-B-only"
    A, B = _channels(Kc)
    assert float(np.abs(A + B - 1.0).max()) < 1e-9, f"{name}: particle-hole relation broken"
    c = E.reads.coupling(A, B)
    assert c.resolved and c.strength == pytest.approx(-1.0, abs=1e-4)


@pytest.mark.parametrize("name,K", [
    ("ring 6 flux pi/4", ring(6, np.pi / 4)),
    ("ring 8 flux pi/4", ring(8, np.pi / 4)),
])
def test_route_A_alone_is_sign_free_without_the_calibration(name, K):
    """THE SEPARATION. The identity holds and the read is NOT -1.

    Positivity, the §3 identity and the exact -1 coincide everywhere in §§2-4 and come apart
    here, which is why the calibration is attributed to route B and not to the identity.
    """
    Kc = np.asarray(K, dtype=complex)
    assert route_A(Kc) and not route_B(Kc), f"{name} is not route-A-only"
    r, neg = measure(Kc, 8.0, n_draw=100, seed=5)
    assert r < 1e-9, f"{name}: the identity should still hold, got {r:.3e}"
    assert neg == 0.0, f"{name}: should still be sign-free, got {neg}"
    A, B = _channels(Kc)
    assert float(np.abs(A + B - 1.0).max()) > 1e-3,         f"{name}: the particle-hole relation held, so nothing separates the routes here"
    c = E.reads.coupling(A, B)
    assert abs(c.strength + 1.0) > 1e-3, f"{name}: the read saturated at {c.strength:.4f}"


def test_each_route_supplies_its_own_exact_channel_relation():
    """Route B gives G_dn = 1 - G_up; route A gives 1 - conj(G_up). Both configuration-wise."""
    Kb = np.asarray(ring(5, np.pi / 2), dtype=complex)
    assert route_B(Kb) and not route_A(Kb)
    A, B = _channels(Kb)
    assert float(np.abs(A + B - 1.0).max()) < 1e-9
    assert float(np.abs(A + np.conj(B) - 1.0).max()) > 1e-3, "both relations hold; nothing separates"

    Ka = np.asarray(ring(6, np.pi / 4), dtype=complex)
    assert route_A(Ka) and not route_B(Ka)
    A, B = _channels(Ka)
    assert float(np.abs(A + np.conj(B) - 1.0).max()) < 1e-9
    assert float(np.abs(A + B - 1.0).max()) > 1e-3, "both relations hold; nothing separates"


@pytest.mark.parametrize("N,flux_over_pi", [(6, 0.125), (6, 0.25), (8, 0.25), (10, 0.25)])
def test_the_route_A_reading_is_derived_not_merely_observed(N, flux_over_pi):
    """strength = -1 + 2r exactly, with r the imaginary share of the centred variance.

    The departure from -1 under route A is the frame's imaginary weight and nothing else. It is
    not a degradation of the read and carries no information about the sign problem, which is
    absent on every one of these rows.
    """
    K = np.asarray(ring(N, flux_over_pi * np.pi), dtype=complex)
    assert route_A(K) and not route_B(K)
    _, neg = measure(K, 8.0, n_draw=100, seed=5)
    assert neg == 0.0, "this row is meant to be sign-free"
    A, B = _channels(K)
    Ac = A - A.mean(axis=0, keepdims=True)
    vre = float((Ac.real ** 2).sum()); vim = float((Ac.imag ** 2).sum())
    r = vim / (vre + vim)
    assert E.reads.coupling(A, B).strength == pytest.approx(-1.0 + 2.0 * r, abs=1e-10)
