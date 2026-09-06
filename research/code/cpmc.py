"""Constrained-path AFQMC (Zhang, Carlson & Gubernatis 1995) for the 2-D Hubbard model.

Everything else in this rig fights the sign problem and loses to it.  This one does not fight
it: it removes it by construction, and pays a BIAS instead.  That is the trade the practitioner
actually makes at 2-D scale, and it matches the stated criterion -- "decoupling, not necessarily
sign-free."

The method walks in Slater-determinant space rather than in the field path integral.  A walker
is a pair of orbital matrices (Phi_up: N x N_up, Phi_dn: N x N_dn) representing a determinant;
imaginary-time projection e^{-dt H} acting on it stays a determinant, so the walk is closed.
The ground state is reached as e^{-beta H}|Psi_T> for large beta.

The sign problem enters as walkers crossing the node <Psi_T|phi> = 0.  The CONSTRAINT is to
require <Psi_T|phi> > 0 and kill any walker that would cross.  That is exact if Psi_T is the
exact ground state and biased otherwise -- a bias that is measurable, which is the whole point:
this file exists to MEASURE it against exact diagonalisation, not to assume it is small.

Importance sampling, with the discrete Hirsch field: for each site the two branches have
overlap ratios r_+ and r_-, computed by the same rank-1 form the rest of this rig uses,

    r_sigma = 1 + (e^{sigma lam s} - 1) G_sigma[i,i],   G = phi (Psi_T^dag phi)^-1 Psi_T^dag

    NOTE the convention.  Here G[i,j] = <c^dag_j c_i>, the GROUND-STATE mixed density matrix.
    The finite-temperature rig in this same directory carries G = (I+B)^-1 = <c c^dag>, whose
    ratio is 1 + Delta (1 - G_ii) and whose rank-1 update has the opposite sign.  Writing the
    finite-T form here converged the walk to a state 0.45 above the exact ground state, with
    free projection agreeing with it -- an error that survived every check of the estimator and
    the propagator because BOTH were right.  What caught it was propagating one step by exact
    enumeration of all 2^N fields and comparing against the Trotter operator.

the branch is drawn with probability proportional to max(r, 0), and the weight is multiplied by
the mean of the clipped ratios.  Both branches non-positive kills the walker.  No weight is ever
negative, so nothing cancels and there is no average sign to collapse.
"""
from __future__ import annotations

import numpy as np


class CPMC:
    def __init__(self, m, n_up, n_dn, n_walkers=200, seed=0, ortho_every=5, pop_every=10,
                 psi_t=None):
        self.m, self.n_up, self.n_dn = m, int(n_up), int(n_dn)
        self.rng = np.random.default_rng(seed)
        self.ortho_every, self.pop_every = int(ortho_every), int(pop_every)
        self.lam = float(np.arccosh(np.exp(m.dtau * m.U / 2.0)))
        self.expK_half = _expm(-0.5 * m.dtau * m.K)
        # The trial wavefunction is the ONLY source of constrained-path bias: the constraint is
        # exact when psi_t is the exact ground state.  So it is an argument, not a constant.
        # Default is the free (U = 0) ground state, the standard first choice.
        if psi_t is None:
            w, v = np.linalg.eigh(m.K)
            psi_t = {+1: v[:, :self.n_up].copy(), -1: v[:, :self.n_dn].copy()}
        self.psiT = {s: np.asarray(psi_t[s], float).copy() for s in (+1, -1)}
        self.walk = [{+1: self.psiT[+1].copy(), -1: self.psiT[-1].copy()} for _ in range(n_walkers)]
        self.w = np.ones(n_walkers)
        self.killed = 0

    # ---- overlap and the mixed Green's function

    def _ov(self, phi, sigma):
        return float(np.linalg.det(self.psiT[sigma].T @ phi))

    def _green(self, phi, sigma):
        """G[i,j] = <Psi_T| c^dag_j c_i |phi> / <Psi_T|phi>."""
        P = self.psiT[sigma]
        return phi @ np.linalg.solve(P.T @ phi, P.T)

    # ---- one imaginary-time step

    def step(self):
        m = self.m
        alive = []
        for k, phi in enumerate(self.walk):
            if self.w[k] <= 0:
                continue
            ph = {s: self.expK_half @ phi[s] for s in (+1, -1)}
            G = {s: self._green(ph[s], s) for s in (+1, -1)}
            weight = self.w[k]
            dead = False
            for i in range(m.N):
                rs = {}
                for s in (+1, -1):
                    rs[s] = {}
                    for x in (+1.0, -1.0):
                        d = np.exp(s * self.lam * x) - 1.0
                        rs[s][x] = 1.0 + d * G[s][i, i]
                r_plus = rs[+1][+1.0] * rs[-1][+1.0]
                r_minus = rs[+1][-1.0] * rs[-1][-1.0]
                cp, cm = max(r_plus, 0.0), max(r_minus, 0.0)   # THE CONSTRAINT
                tot = cp + cm
                if tot <= 0.0:                                  # both branches cross the node
                    dead = True
                    break
                weight *= 0.5 * tot                             # the HS 1/2 sum, importance-sampled
                x = 1.0 if self.rng.random() < cp / tot else -1.0
                for s in (+1, -1):
                    d = np.exp(s * self.lam * x) - 1.0
                    R = rs[s][x]
                    ph[s][i, :] *= (1.0 + d)                    # V is diagonal: scale row i
                    col = -G[s][:, i].copy()
                    col[i] += 1.0                        # e_i - G[:, i]
                    G[s] = G[s] + (d / R) * np.outer(col, G[s][i, :])
            if dead:
                self.killed += 1
                continue
            ph = {s: self.expK_half @ ph[s] for s in (+1, -1)}
            if self._ov(ph[+1], +1) * self._ov(ph[-1], -1) <= 0.0:
                self.killed += 1
                continue
            alive.append((ph, weight))
        if not alive:
            raise RuntimeError("every walker was killed by the constraint")
        self.walk = [a[0] for a in alive]
        self.w = np.array([a[1] for a in alive])
        return len(alive)

    def orthonormalise(self):
        for phi in self.walk:
            for s in (+1, -1):
                q, r = np.linalg.qr(phi[s])
                phi[s] = q * np.sign(np.diag(r))[None, :]

    def population_control(self, target):
        """Comb resampling on the weights -- no weight is negative, so this is ordinary."""
        w = self.w / self.w.sum()
        u = (self.rng.random() + np.arange(target)) / target
        idx = np.searchsorted(np.cumsum(w), u)
        self.walk = [{s: self.walk[i][s].copy() for s in (+1, -1)} for i in idx]
        self.w = np.full(target, self.w.sum() / target)

    # ---- the mixed energy estimator

    def energy(self):
        m = self.m
        num = den = 0.0
        for k, phi in enumerate(self.walk):
            G = {s: self._green(phi[s], s) for s in (+1, -1)}
            kin = sum(float(np.sum(m.K * G[s].T)) for s in (+1, -1))
            nup, ndn = np.diag(G[+1]).real, np.diag(G[-1]).real
            pot = float(m.U * np.sum((nup - 0.5) * (ndn - 0.5)))
            num += self.w[k] * (kin + pot)
            den += self.w[k]
        return num / den


def _expm(A):
    from scipy.linalg import expm
    return expm(A)


def run_cpmc(m, n_up, n_dn, beta=8.0, n_walkers=200, seed=0, n_meas=200, equil=0.4):
    ch = CPMC(m, n_up, n_dn, n_walkers=n_walkers, seed=seed)
    steps = int(round(beta / m.dtau))
    n_eq = int(equil * steps)
    es = []
    for t in range(steps + n_meas):
        ch.step()
        if (t + 1) % ch.ortho_every == 0:
            ch.orthonormalise()
        if (t + 1) % ch.pop_every == 0:
            ch.population_control(n_walkers)
        if t >= n_eq + steps - int(equil * steps) and t >= n_eq:
            if t >= steps:
                es.append(ch.energy())
    return np.array(es), ch.killed
