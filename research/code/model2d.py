"""2-D Hubbard on a periodic Lx x Ly lattice, with a one-parameter DECOUPLING family.

The interaction can be traded for a random field in two different ways, because for
a, b in {0,1} both of these are exact:

    (n_up - 1/2)(n_dn - 1/2) =  1/4 - m^2 / 2         m    = n_up - n_dn      (SPIN)
                             =  rho^2 / 2 - 1/4       rho  = n_up + n_dn - 1  (CHARGE)

(they agree because rho^2 + m^2 = 1 identically on the four states).  Gaussian-integrating
each gives

    SPIN    exp(-dt U (n_up-1/2)(n_dn-1/2)) = e^{-dt U/4} Int N(x) exp( lam_s x m )
    CHARGE                                  = e^{+dt U/4} Int N(y) exp( i lam_c y rho )

with lam = sqrt(dt U).  The spin field couples REALLY, the charge field couples
IMAGINARILY -- for repulsive U that is forced, and it is why the two channels give
different signs for the same physics.

Splitting U = (1-theta) U + theta U and decoupling each piece in its own channel gives the
family this module exists to sweep:

    V_sigma(l)_ii = exp( sigma lam_s x[l,i] + i lam_c y[l,i] )
    weight        = prod_{l,i} exp(-i lam_c y[l,i])  *  det(I+B_up) det(I+B_dn)

theta = 0 is the standard spin decoupling (real weights); theta = 1 is pure charge.  Every
theta describes the SAME physics, so any observable must come out theta-independent -- which
is the gate this file is checked against, along with the single-site identity by quadrature.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import expm


def hop_2d(Lx, Ly, t=1.0, tp=0.0, mu=0.0, apbc=False):
    """Lx x Ly square lattice: nearest neighbour t, diagonal t', chemical potential.

    `apbc` makes the y-direction ANTI-periodic -- the wrap-around bonds carry -t instead of +t.
    That is not decoration.  Under fully periodic boundaries half filling is an OPEN shell on
    every bipartite lattice this rig can reach (2x2, 2x4, 2x6, 4x4, 4x6 all measured), so the
    trial wavefunction is ambiguous exactly where the half-filling sign-free property is meant
    to be checked; and the two sizes whose half filling does close, 2x3 and 3x4, close it by
    having an odd ring, which makes them non-bipartite and not sign-free either.  Anti-periodic
    boundaries shift the momenta off the degenerate points and close the shell without touching
    the sublattice structure.  Measured: 2x4 apbc has n = 4 closed, 4x4 apbc has n = 8 closed.
    """
    N = Lx * Ly
    idx = lambda x, y: (x % Lx) * Ly + (y % Ly)
    K = np.zeros((N, N))
    for x in range(Lx):
        for y in range(Ly):
            i = idx(x, y)
            for dx, dy in ((1, 0), (0, 1)):
                s = -t
                if apbc and dy == 1 and (y + 1) >= Ly:
                    s = +t
                K[i, idx(x + dx, y + dy)] += s
                K[idx(x + dx, y + dy), i] += s
            if tp:
                for dx, dy in ((1, 1), (1, -1)):
                    K[i, idx(x + dx, y + dy)] -= tp
                    K[idx(x + dx, y + dy), i] -= tp
    K[np.diag_indices(N)] = -mu
    return K


class Model2D:
    def __init__(self, Lx=2, Ly=2, t=1.0, tp=0.0, mu=0.0, U=4.0, dtau=0.2, L=20,
                 theta=0.0, apbc=False):
        self.Lx, self.Ly, self.N = Lx, Ly, Lx * Ly
        self.t, self.tp, self.mu, self.U = t, tp, mu, U
        self.dtau, self.L, self.beta = dtau, L, dtau * L
        self.theta = float(theta)
        self.apbc = bool(apbc)
        self.K = hop_2d(Lx, Ly, t, tp, mu, apbc=apbc)
        self.expmK = expm(-dtau * self.K)
        self.lam_s = float(np.sqrt(dtau * (1.0 - self.theta) * U))
        self.lam_c = float(np.sqrt(dtau * self.theta * U))

    # ---- the decoupled slice matrices and the weight

    def B_slices(self, X, Y, sigma):
        """(M, L, N, N) for a batch of field pairs.  X couples to spin (real), Y to charge
        (imaginary), so V is complex whenever theta > 0."""
        d = np.exp(sigma * self.lam_s * X + 1j * self.lam_c * Y)
        return self.expmK[None, None, :, :] * d[:, :, None, :]

    def charge_phase(self, Y):
        """prod exp(-i lam_c y) -- the c-number the `-1` in rho = n_up + n_dn - 1 leaves
        behind.  It is part of the weight and carries phase, so it is never dropped."""
        return np.exp(-1j * self.lam_c * Y.sum(axis=(1, 2)))


# ─────────────────────────────────────────── the algebra gate

def single_site_identity(U, dtau, theta, n_nodes=40):
    """Check the decoupling on ONE site by Gauss-Hermite quadrature, before any lattice.

    Returns (exact, quadrature) for each of the four states |n_up, n_dn>.  The identity is
    an algebraic claim; if it fails here nothing downstream can be right."""
    from numpy.polynomial.hermite_e import hermegauss
    xs, ws = hermegauss(n_nodes)
    ws = ws / ws.sum()
    lam_s = np.sqrt(dtau * (1.0 - theta) * U)
    lam_c = np.sqrt(dtau * theta * U)
    pre = np.exp(-dtau * (1.0 - theta) * U / 4.0) * np.exp(+dtau * theta * U / 4.0)
    out = []
    for nu in (0, 1):
        for nd in (0, 1):
            exact = np.exp(-dtau * U * (nu - 0.5) * (nd - 0.5))
            m, rho = nu - nd, nu + nd - 1
            gs = float(np.sum(ws * np.exp(lam_s * xs * m)))
            gc = complex(np.sum(ws * np.exp(1j * lam_c * xs * rho))) * np.exp(-1j * lam_c * 0)
            # the -1 c-number is per SITE and cancels against rho's own -1 inside the
            # exponent, so it is folded in by writing rho explicitly above
            out.append((nu, nd, exact, pre * gs * gc))
    return out


if __name__ == "__main__":
    print("GATE A -- the single-site decoupling identity, by 2-D Gauss-Hermite quadrature")
    print("=" * 78)
    ok = True
    for theta in (0.0, 0.25, 0.5, 0.75, 1.0):
        worst = 0.0
        for nu, nd, exact, quad in single_site_identity(4.0, 0.2, theta):
            worst = max(worst, abs(quad - exact) / abs(exact))
        ok &= worst < 1e-10
        print(f"  theta={theta:4.2f}   worst relative error over the four states = {worst:.2e}")
    print(f"\nGATE A: {'PASS' if ok else 'FAIL'}")
