"""
Determinantal (auxiliary-field / BSS) QMC for a small Hubbard chain, WITH a fermion
sign problem, plus an exact-diagonalisation ground truth for the SAME Trotter
discretisation -- so any difference between the two is purely statistical.

Model
-----
    H = sum_sigma sum_ij K_ij c^dag_{i sigma} c_{j sigma}
        + U sum_i (n_{i up} - 1/2)(n_{i dn} - 1/2)

    K_ij = -t  for |i-j| = 1,   -t2 for |i-j| = 2   (open chain; t2 frustrates,
    K_ii = -mu                                       breaks particle-hole => sign problem)

Discretisation
--------------
    T = exp(-dtau H_K) exp(-dtau H_U),   Z = Tr T^L,   beta = L dtau

Hirsch HS transform (exact, discrete):
    exp(-dtau U (n_up - 1/2)(n_dn - 1/2))
        = (e^{-dtau U/4}/2) sum_{s=+-1} exp(lambda s (n_up - n_dn)),
      cosh lambda = exp(dtau U / 2)

so, with V_sigma(l) = diag(exp(sigma lambda s_{l,i})) and B_sigma(l) = e^{-dtau K} V_sigma(l),

    Z = C^{N L} sum_s det(I + B_up(s)) det(I + B_dn(s)),   C = e^{-dtau U/4}/2

The two determinants have independent signs once particle-hole symmetry is broken,
so w(s) changes sign: that IS the fermion sign problem, in its standard form.

Observable
----------
The imaginary-time single-particle propagator, per site:

    G_sigma(tau_l, 0)_ij = <c_{i sigma}(tau_l) c^dag_{j sigma}(0)>
                         = [B(l) ... B(1) (I + B)^{-1}]_ij

Its diagonal is an (L, N) array -- an ordered axis (imaginary time) and a feature
axis (site) -- exactly the shape Entroptics reads, and its decay rate is a gap.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import expm


# ---------------------------------------------------------------- the model

def hop_matrix(N: int, t: float, t2: float, mu: float) -> np.ndarray:
    """Open-chain hopping matrix with nn (t) and nnn (t2) terms and chemical potential."""
    K = np.zeros((N, N))
    for i in range(N - 1):
        K[i, i + 1] = K[i + 1, i] = -t
    for i in range(N - 2):
        K[i, i + 2] = K[i + 2, i] = -t2
    K[np.diag_indices(N)] = -mu
    return K


class Model:
    def __init__(self, N=4, t=1.0, t2=0.7, mu=0.6, U=4.0, dtau=0.2, L=20):
        self.N, self.t, self.t2, self.mu, self.U = N, t, t2, mu, U
        self.dtau, self.L = dtau, L
        self.beta = dtau * L
        self.K = hop_matrix(N, t, t2, mu)
        self.expmK = expm(-dtau * self.K)
        self.lam = float(np.arccosh(np.exp(dtau * U / 2.0)))

    # ---- the determinantal weight and the propagator, for one auxiliary field

    def _B_slices(self, s: np.ndarray, sigma: int) -> np.ndarray:
        """(L, N, N) stack of B_sigma(l) = e^{-dtau K} diag(exp(sigma lam s_l))."""
        d = np.exp(sigma * self.lam * s)                  # (L, N)
        return self.expmK[None, :, :] * d[:, None, :]     # column scaling == right-multiply by diag

    def weight_and_G(self, s: np.ndarray, want_G: bool = True):
        """Signed weight (up to the constant C^{NL}) and, if asked, the (L, N) diagonal
        of the imaginary-time propagator matrix, spin-averaged."""
        N, L = self.N, self.L
        logabs, sign, Gs = 0.0, 1.0, []
        for sigma in (+1, -1):
            Bl = self._B_slices(s, sigma)
            B = np.eye(N)
            for l in range(L):
                B = Bl[l] @ B
            sg, ld = np.linalg.slogdet(np.eye(N) + B)
            logabs += ld
            sign *= sg
            if want_G:
                G0 = np.linalg.inv(np.eye(N) + B)
                g = np.empty((L, N, N))
                g[0] = G0
                P = G0
                for l in range(1, L):
                    P = Bl[l - 1] @ P
                    g[l] = P
                Gs.append(g)
        G = (Gs[0] + Gs[1]) / 2.0 if want_G else None
        return sign, logabs, G


# ---------------------------------------------------------------- exact diagonalisation

def fock_operators(M: int):
    """M fermionic modes -> (2^M, 2^M) annihilation operators via Jordan-Wigner."""
    a = np.array([[0.0, 1.0], [0.0, 0.0]])            # a|1> = |0>
    Z = np.diag([1.0, -1.0])
    I = np.eye(2)
    ops = []
    for k in range(M):
        op = np.array([[1.0]])
        for j in range(M):
            op = np.kron(op, Z if j < k else (a if j == k else I))
        ops.append(op)
    return ops


class ExactTrotter:
    """Exact Tr over the full Fock space of the SAME Trotter product the DQMC samples."""

    def __init__(self, m: Model):
        N = m.N
        c = fock_operators(2 * N)                      # 0..N-1 up, N..2N-1 down
        self.c = c
        dim = 2 ** (2 * N)
        HK = np.zeros((dim, dim))
        for sp in (0, N):
            for i in range(N):
                for j in range(N):
                    if m.K[i, j] != 0.0:
                        HK += m.K[i, j] * (c[sp + i].T @ c[sp + j])
        n = [ci.T @ ci for ci in c]
        HU = np.zeros((dim, dim))
        for i in range(N):
            HU += m.U * (n[i] - 0.5 * np.eye(dim)) @ (n[N + i] - 0.5 * np.eye(dim))
        self.T = expm(-m.dtau * HK) @ expm(-m.dtau * HU)
        self.m = m
        self.Tp = [np.eye(dim)]
        for _ in range(m.L):
            self.Tp.append(self.Tp[-1] @ self.T)
        self.Z = float(np.trace(self.Tp[m.L]))

    def propagator(self) -> np.ndarray:
        """(L, N, N) exact <c_i(tau_l) c^dag_j(0)>, spin-averaged."""
        m, c = self.m, self.c
        out = np.zeros((m.L, m.N, m.N))
        for l in range(m.L):
            left, right = self.Tp[m.L - l], self.Tp[l]
            for sp in (0, m.N):
                for i in range(m.N):
                    A = left @ c[sp + i]
                    for j in range(m.N):
                        out[l, i, j] += float(np.trace(A @ right @ c[sp + j].T))
        return out / (2.0 * self.Z)
