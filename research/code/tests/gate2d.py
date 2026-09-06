"""GATE B -- the LATTICE weight, by exact quadrature over the auxiliary fields, against
exact diagonalisation.  Two claims at once:

  (1) Z from the decoupled field integral == Tr[(e^{-dt H_K} e^{-dt H_U})^L]
  (2) Z is INDEPENDENT of theta -- every decoupling describes the same physics, so if the
      family is right this is exact, not approximate.

Small enough to integrate exactly: 2 sites x L slices x 2 fields.
"""
import itertools
import numpy as np
from numpy.polynomial.hermite_e import hermegauss

from model2d import Model2D
from dqmc import ExactTrotter


def Z_quadrature(m, n_nodes=14, chunk=200_000):
    n = m.L * m.N
    xs, ws = hermegauss(n_nodes); ws = ws / ws.sum()
    grid = np.stack(np.meshgrid(*([xs]*(2*n)), indexing="ij"), -1).reshape(-1, 2*n)
    wt = np.prod(np.stack(np.meshgrid(*([ws]*(2*n)), indexing="ij"), -1)
                 .reshape(-1, 2*n), axis=1)
    pre = (np.exp(-m.dtau*(1-m.theta)*m.U/4.0) * np.exp(+m.dtau*m.theta*m.U/4.0))**(m.N*m.L)
    Z = 0j
    for i in range(0, len(grid), chunk):
        g, w = grid[i:i+chunk], wt[i:i+chunk]
        X = g[:, :n].reshape(-1, m.L, m.N); Y = g[:, n:].reshape(-1, m.L, m.N)
        eye = np.broadcast_to(np.eye(m.N, dtype=complex), (len(g), m.N, m.N))
        d = m.charge_phase(Y).astype(complex)
        for sigma in (+1, -1):
            Bl = m.B_slices(X, Y, sigma)
            B = eye.copy()
            for l in range(m.L): B = Bl[:, l] @ B
            d = d * np.linalg.det(eye + B)
        Z += np.sum(w * d)
    return pre * Z


if __name__ == "__main__":
    print("GATE B -- lattice Z by field quadrature vs exact diagonalisation")
    print("=" * 82)
    for Lx, Ly, L, nn in ((2, 1, 1, 14), (2, 1, 2, 8)):
        ref = None
        print(f"\n  {Lx}x{Ly} sites, L={L} slices, {nn} nodes "
              f"({nn**(2*Lx*Ly*L):,} quadrature points)")
        for theta in (0.0, 0.3, 0.7, 1.0):
            m = Model2D(Lx=Lx, Ly=Ly, t=1.0, mu=0.0, U=4.0, dtau=0.25, L=L, theta=theta)
            Zq = Z_quadrature(m, n_nodes=nn)
            if ref is None:
                ref = float(ExactTrotter(m).Z)
            rel = abs(Zq - ref)/abs(ref)
            print(f"    theta={theta:4.2f}  Z_quad = {Zq.real:.10f}{Zq.imag:+.2e}i   "
                  f"|Z_quad - Z_ED|/|Z_ED| = {rel:.2e}")
    print("\nA pass needs BOTH: agreement with ED, and the same Z at every theta.")
