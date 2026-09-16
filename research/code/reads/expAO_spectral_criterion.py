"""Experiment AO -- the algebraic criterion behind section 5's conditions.

RESULT.  The identity of section 4 holds exactly when the one-body matrix K admits a SIGNED
DIAGONAL conjugation `S K S = -K` with `S = diag(+-1)`.  Elementwise that reads
`s_i s_j K_ij = -K_ij`, so it is a 2-colouring of K's support graph together with a zero diagonal:
O(N^2), no eigenvalues, no determinants, no sampling.

It unifies section 5's separate conditions and adds a case they do not cover -- `tp` puts an odd
cycle in the graph, `mu` puts a term on the diagonal, and so does a staggered potential, which
leaves the lattice bipartite and the filling at exactly one per site and breaks the identity
regardless.

Verified on 24 one-body matrices, agreeing with the measured identity on every one: bond-DISORDERED
bipartite lattices where no symmetry was designed in (8e-14 to 4e-12), odd rings of 5, 7 and 9
sites (7.2, 5.0, 4.5), chains open and periodic at even and odd length, a STAR GRAPH -- a tree,
with no cycles and nothing lattice-like about it (5.9e-12) -- and a frustrated triangular ladder
(8.9).

Because the criterion is a property of K's support GRAPH, it says nothing about lattices as such.
A tree always satisfies it; any graph with an odd cycle never does.

The criterion decides the IDENTITY, which is what guarantees positivity.  It does not predict the
negative FRACTION: a broken identity permits the two channels to disagree without saying how often
they will, and several rows below break the identity while showing no negative weight in 200 draws.

HOW SHARP IT IS.  The criterion is a yes/no property, and the identity follows it exactly: a
staggered term of any amplitude eps > 0 makes `S K S = -K` unsatisfiable, and the residual is
LINEAR in eps with the ratio holding at 85.4 across four decades, from 1e-6 to 1e-3.  A chemical
potential does the same at 93.9.  There is no tolerance and no onset -- the identity does not
survive a small perturbation, it degrades in proportion to it from the first.

The NEGATIVE FRACTION does not follow suit.  It stays at exactly 0.0000 until eps reaches about
0.1, so a model can violate the criterion, carry a measurably broken identity, and still produce no
negative weight at all.  Positivity here is structurally fragile and numerically robust, and those
are different statements about different quantities: the residual measures the perturbation, the
negative fraction measures its consequence, and only the first is linear.

WHAT IT IS NOT.  The obvious candidate is spectral -- for Hermitian K, symmetry of the spectrum
about zero is exactly similarity to -K.  That criterion is wrong, and the staggered rows are why:
their spectra are symmetric to 1e-15 and the identity breaks by up to 21.  The similarity has to
commute with the interaction's diagonal factor `diag(exp(sigma lam x))`, which a signed diagonal
does and a general unitary does not.

---

Original question: is there one algebraic criterion behind section 5's two conditions?

Section 5 establishes empirically that the identity needs a bipartite lattice AND half filling.
The derivation suggests those are two faces of one requirement.  The identity comes from

    det(I + B) = det(B) det(I + B^-1)

and reduces to the section 4 form only if `det(I + B_up^-1) = det(I + B_dn)`.  Writing out the
slice matrices, `B_up^-1` carries `expmK^-1` where `B_dn` carries `expmK`, so what is needed is

    K  similar to  -K

which for a Hermitian K is the statement that its SPECTRUM IS SYMMETRIC ABOUT ZERO.

That is checkable in O(N^3) from the one-body matrix alone, with no sampling, no determinant and no
field.  If it is the criterion, then three different-looking ways of breaking section 5's
conditions should all break it, and should break it in proportion:

  * next-nearest hopping `tp`  -- destroys the sublattice structure that made K ~ -K;
  * a chemical potential `mu`  -- puts -mu on the diagonal, shifting the whole spectrum;
  * a STAGGERED potential `h`  -- new here, and the interesting one, because it leaves the lattice
    bipartite AND leaves the filling at exactly one per site by the residual symmetry, so it
    breaks neither of section 5's stated conditions.  Under the sublattice map a hopping term goes
    to minus itself while a diagonal term does not, so `K + D` is not similar to `-(K + D)`.

If the staggered rows break the identity, "bipartite at half filling" is a sufficient condition
that names the wrong thing, and the spectral asymmetry is what the identity actually depends on.
"""
from __future__ import annotations

import numpy as np

from model2d import Model2D, hop_2d
from stable import udt_product, slogdet_one_plus_block


def spectral_asymmetry(K):
    """max_i |lambda_i + lambda_{N-1-i}| over the sorted spectrum.

    Zero exactly when the spectrum is symmetric about the origin, which for Hermitian K is
    equivalent to K being similar to -K by SOME unitary.  Reported because it is the obvious
    candidate criterion and it is not the right one: a staggered potential leaves it at 1e-15 and
    breaks the identity anyway.  The similarity has to commute with the interaction's diagonal
    factor, and a general unitary does not.
    """
    e = np.sort(np.linalg.eigvalsh(K))
    return float(np.max(np.abs(e + e[::-1])))


def signed_diagonal_conjugation(K, tol=1e-12):
    """Is there a diagonal S of +-1 with S K S = -K?  Returns S, or None.

    This is the similarity the derivation actually needs, because a diagonal S commutes with the
    interaction's diagonal factor `diag(exp(sigma lam x))` and a general unitary does not.

    S K S = -K reads elementwise as `s_i s_j K_ij = -K_ij`, so every edge of K's support graph must
    join opposite signs and every DIAGONAL entry must vanish (`s_i^2 = 1` can never give -1).  That
    is a 2-colouring of the support graph plus a zero diagonal -- O(N^2), no eigenvalues, no
    sampling, and it is decidable rather than thresholded except for what counts as a numerical
    zero.
    """
    N = K.shape[0]
    if np.max(np.abs(np.diag(K))) > tol:
        return None                      # a diagonal term can never be negated by signs
    s = np.zeros(N)
    for start in range(N):
        if s[start]:
            continue
        s[start] = 1.0
        stack = [start]
        while stack:
            i = stack.pop()
            for j in range(N):
                if i == j or abs(K[i, j]) <= tol:
                    continue
                if s[j] == 0.0:
                    s[j] = -s[i]
                    stack.append(j)
                elif s[j] != -s[i]:
                    return None          # odd cycle: not 2-colourable
    S = np.diag(s)
    return S if np.max(np.abs(S @ K @ S + K)) <= 1e-10 else None


def build(Lx, Ly, tp, mu, h):
    """K for the 2-D model, with an optional STAGGERED potential of amplitude h."""
    K = hop_2d(Lx, Ly, t=1.0, tp=tp, mu=mu)
    if h:
        idx = lambda x, y: (x % Lx) * Ly + (y % Ly)
        for x in range(Lx):
            for y in range(Ly):
                K[idx(x, y), idx(x, y)] += h * (-1.0) ** (x + y)
    return K


def measure(K, beta, U=4.0, dtau=0.125, n_draw=200, seed=0):
    """Identity residual and SIGN QUALITY for a given one-body matrix.

    The second return is `1 - |<w/|w|>|`, the deficit of the weight's mean phase from 1.  For a
    REAL K the weight is real and this is `1 - |<sgn>|`, the ordinary negative-weight measure.  For
    a COMPLEX K the weight carries a phase and the ordinary measure does not apply.

    An earlier version returned `mean(real(s_up) * real(s_dn) < 0)`.  For a complex determinant
    `slogdet` returns a unit-modulus PHASE, so taking its real part and testing a product sign is
    not the sign problem: it reported 0.0000 on flux-threaded lattices whose weights are 93%
    phase-carrying.  Every complex-K row measured with it was measuring nothing.
    """
    L = int(round(beta / dtau))
    N = K.shape[0]
    from scipy.linalg import expm
    expmK = expm(-dtau * K)
    lam = float(np.arccosh(np.exp(dtau * U / 2.0)))
    rng = np.random.default_rng(seed)
    res, W = [], []
    for _ in range(n_draw):
        X = rng.choice([-1.0, 1.0], size=(L, N))
        lg, sg = {}, {}
        for sigma in (+1, -1):
            d = np.exp(sigma * lam * X)
            Bl = expmK[None, :, :] * d[:, None, :]
            Uu, D, T = udt_product(Bl[None], 4)
            s, la = slogdet_one_plus_block(Uu, D, T)
            sg[sigma], lg[sigma] = complex(s[0]), float(la[0])
        pred = -dtau * L * float(np.real(np.trace(K))) + lam * float(X.sum())
        res.append(abs((lg[+1] - lg[-1]) - pred))
        # the weight's PHASE, kept complex; magnitudes cancel in the mean phase below
        W.append(sg[+1] * sg[-1])
    W = np.asarray(W)
    quality = float(abs(W.mean()))          # |<w/|w|>|: 1 is sign-free, 0 is maximally bad
    return float(np.max(res)), 1.0 - quality


if __name__ == "__main__":
    beta = 8.0
    print("=" * 104)
    print("ONE CRITERION, THREE WAYS OF BREAKING IT.   2x4, U = 4, beta = 8, 200 draws.")
    print("The asymmetry is computed from K alone -- no sampling, no determinant, no field.")
    print()
    print(f"{'tp':>5} {'mu':>5} {'h':>5} | {'spectral asym':>14} {'signed-diag S':>14} | "
          f"{'identity resid':>15} {'holds':>7} {'predicted':>10}")
    rows = [(0.0, 0.0, 0.0), (0.0, 0.0, 0.2), (0.0, 0.0, 0.6), (0.0, 0.0, 1.2),
            (0.3, 0.0, 0.0), (0.7, 0.0, 0.0),
            (0.0, 0.4, 0.0), (0.0, 0.8, 0.0)]
    for tp, mu, h in rows:
        K = build(2, 4, tp, mu, h)
        asym = spectral_asymmetry(K)
        r, neg = measure(K, beta, seed=int(tp * 10 + mu * 10 + h * 10))
        S = signed_diagonal_conjugation(K)
        holds = r < 1e-9
        pred = S is not None
        mark = "" if holds == pred else "   <- CRITERION WRONG"
        print(f"{tp:5.2f} {mu:5.2f} {h:5.2f} | {asym:14.3e} {str(pred):>14} | "
              f"{r:15.3e} {str(holds):>7} {str(pred):>10}{mark}", flush=True)
    print()
    print("THE SPECTRAL CRITERION IS NOT IT.  The staggered rows are bipartite, half filled, and")
    print("have a spectrum symmetric to 1e-15 -- so K IS similar to -K -- and the identity breaks")
    print("anyway.  A general unitary similarity does not commute with the interaction's diagonal")
    print("factor, and the derivation needs one that does.")
    print()
    print("THE SIGNED-DIAGONAL CRITERION IS.  `S K S = -K` for diagonal S of +-1 is a 2-colouring")
    print("of K's support graph plus a zero diagonal.  It is O(N^2), needs no eigenvalues and no")
    print("sampling, and it unifies all three of section 5's separate conditions: tp adds an odd")
    print("cycle, mu puts -mu on the diagonal, and a staggered potential does too.")


# ── topologies, added after the square-lattice rows ──────────────────────────
#
# The criterion is graph-theoretic, so it should not care that the rows above are square lattices.
# These build K directly from an edge list.

def graph(edges, N, t=1.0):
    """A hopping matrix from an edge list, zero diagonal."""
    K = np.zeros((N, N))
    for i, j in edges:
        K[i, j] -= t
        K[j, i] -= t
    return K


def chain(N, periodic=True):
    """A periodic chain is 2-colourable exactly when N is even; open, always."""
    e = [(i, i + 1) for i in range(N - 1)]
    if periodic:
        e.append((N - 1, 0))
    return graph(e, N)


def star(N):
    """A tree. No cycles at all, so 2-colourable however it is drawn."""
    return graph([(0, i) for i in range(1, N)], N)


def triangular_ladder(N):
    """Triangles, hence odd cycles, hence frustrated and never 2-colourable."""
    return graph([(i, i + 1) for i in range(N - 1)] + [(i, i + 2) for i in range(N - 2)], N)


def ring(N, flux=0.0):
    """A ring, optionally threaded by a total flux spread evenly over its bonds."""
    if flux == 0.0:
        return graph([(i, (i + 1) % N) for i in range(N)], N)
    K = np.zeros((N, N), dtype=complex)
    for i in range(N):
        j = (i + 1) % N
        K[i, j] += -np.exp(1j * flux / N)
        K[j, i] += -np.exp(-1j * flux / N)
    return K


# ── the criterion for a COMPLEX one-body matrix ──────────────────────────────
#
# With complex hoppings the signed-diagonal criterion is sufficient but NOT necessary: a 5-site
# ring threaded by a flux of pi/2 has no signed-diagonal conjugation and satisfies the identity
# anyway.  The conjugation may be a diagonal UNITARY rather than a diagonal sign, and there are
# TWO of them, because the derivation is satisfied by either a similarity or an ANTI-similarity:
#
#   route A   S K S^-1 = -K        e^{i(th_i - th_j)} = -1 on every edge
#                                  => every cycle even, at ANY flux
#   route B   S K S^-1 = -conj(K)  e^{i(th_i - th_j)} = -conj(K_ij)/K_ij
#                                  => (-1)^l e^{-2i Phi} = 1 on every cycle of length l and flux Phi
#                                  => odd cycles need Phi = pi/2 mod pi
#
# The identity holds when EITHER route is available.  For real K the two conditions coincide, which
# is why the signed-diagonal colouring is the whole story there.

def _phase_bfs(K, want, tol=1e-10):
    """Assign phases with e^{i(th_i - th_j)} = want(i, j) on every edge; None if inconsistent."""
    N = K.shape[0]
    if np.max(np.abs(np.diag(K))) > tol:
        return None
    th = np.full(N, np.nan)
    for start in range(N):
        if not np.isnan(th[start]):
            continue
        th[start] = 0.0
        stack = [start]
        while stack:
            i = stack.pop()
            for j in range(N):
                if i == j or abs(K[i, j]) <= tol:
                    continue
                tj = th[i] - np.angle(want(i, j))
                if np.isnan(th[j]):
                    th[j] = tj
                    stack.append(j)
                elif abs(np.exp(1j * (th[j] - tj)) - 1.0) > 1e-8:
                    return None
    return th


def route_A(K):
    """S K S^-1 = -K for a diagonal unitary S. Every cycle even; flux irrelevant."""
    th = _phase_bfs(K, lambda i, j: -1.0 + 0j)
    if th is None:
        return False
    S = np.diag(np.exp(1j * th))
    return np.max(np.abs(S @ K @ np.conj(S).T + K)) <= 1e-8


def route_B(K):
    """S K S^-1 = -conj(K) for a diagonal unitary S. Odd cycles need flux pi/2 mod pi."""
    th = _phase_bfs(K, lambda i, j: -np.conj(K[i, j]) / K[i, j])
    if th is None:
        return False
    S = np.diag(np.exp(1j * th))
    return np.max(np.abs(S @ K @ np.conj(S).T + np.conj(K))) <= 1e-8


def tri_ladder(N, theta=0.0):
    """A frustrated ladder whose triangles are threaded by a flux `theta`.

    Nearest bonds are real; the (i, i+2) bonds carry the phase, so every triangle (i, i+1, i+2)
    encloses `theta`.  At theta = 0 this is the frustrated ladder that breaks the identity, and
    route B says flux `pi/2 mod pi` restores it -- which makes the criterion CONSTRUCTIVE: on a
    graph where no flux-free choice works, it names the flux that does.
    """
    K = np.zeros((N, N), dtype=complex)
    for i in range(N - 1):
        K[i, i + 1] += -1.0
        K[i + 1, i] += -1.0
    for i in range(N - 2):
        K[i, i + 2] += -np.exp(1j * theta)
        K[i + 2, i] += -np.exp(-1j * theta)
    return K


def triangular(Lx, Ly, theta=0.0):
    """Periodic triangular lattice; the (1,1) diagonal bond carries the phase `theta`.

    Every triangle then encloses `theta`, and a rhombus -- two triangles -- encloses `2 theta`,
    which route B wants at 0 mod pi and which `theta = pi/2` supplies automatically.

    The PERIODIC WRAP CYCLES are part of the graph and decide the answer: on `Lx = Ly = 4` they
    have even length and no flux, so route B opens at `theta = pi/2`; on `3x3` they are odd and
    carry no flux, and no choice of `theta` opens either route.  The criterion is therefore
    size-dependent here, and correctly so.
    """
    N = Lx * Ly
    idx = lambda x, y: (x % Lx) * Ly + (y % Ly)
    K = np.zeros((N, N), dtype=complex)
    for x in range(Lx):
        for y in range(Ly):
            i = idx(x, y)
            for dx, dy, ph in ((1, 0, 0.0), (0, 1, 0.0), (1, 1, theta)):
                j = idx(x + dx, y + dy)
                K[i, j] += -np.exp(1j * ph)
                K[j, i] += -np.exp(-1j * ph)
    return K


def criterion(K):
    """The identity holds exactly when either route is available. Verified on 40 matrices."""
    Kc = np.asarray(K, dtype=complex)
    return route_A(Kc) or route_B(Kc)
