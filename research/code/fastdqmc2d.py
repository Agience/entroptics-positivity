"""Rank-1 DQMC for the 2-D model with the mixed decoupling: CONTINUOUS fields, COMPLEX weights.

Three differences from `fastdqmc.py`, all of them local:

  * two fields per (site, slice) -- X couples to spin (real), Y to charge (imaginary) -- and
    both are continuous, so a move is a Gaussian shift rather than a sign flip;
  * for a shift (dx, dy) the slice matrix changes by the same diagonal rank-1 form,
        Delta_sigma = exp(sigma lam_s dx + i lam_c dy) - 1,
    so the Green's-function update is unchanged: G' = G - (Delta/R) (I-G)[:,i] (x) G[i,:];
  * the weight carries a phase, from the determinants AND from the c-number
    prod exp(-i lam_c y) that the `-1` in rho = n_up + n_dn - 1 leaves behind.  The Gaussian
    priors and |R_up R_dn| set the acceptance; the phase is carried, never used to accept.

The invariant is the one from `fastdqmc.py`, restated because getting it wrong cost four
attempts there:

    A_l = B(l-1) ... B(0) B(L-1) ... B(l),   G_l = (I + A_l)^{-1},  B(l) RIGHTMOST,
    A_{l+1} = B(l) A_l B(l)^{-1}             so the wrap uses the slice just finished.
"""
from __future__ import annotations

import numpy as np

from stable import udt_product, inv_one_plus_block, slogdet_one_plus_block


class Fast2D:
    def __init__(self, m, seed=0, n_stab=1, udt_block=4, step=1.0):
        self.m, self.n_stab, self.udt_block, self.step = m, int(n_stab), int(udt_block), step
        self.rng = np.random.default_rng(seed)
        self.X = self.rng.standard_normal((m.L, m.N))
        self.Y = self.rng.standard_normal((m.L, m.N)) if m.lam_c > 0 else np.zeros((m.L, m.N))
        self.l = 0
        self.G = {}
        self.acc = self.prop = 0
        self.max_drift = 0.0
        self.n_checks = self.n_bad = 0
        # RELATIVE, not absolute.  |G| runs from ~1 at N=4 to ~34 at N=16, so a fixed 1e-6
        # is a different demand at every size; the drift that matters is drift against the
        # scale of the thing that drifted.  The reference is the arithmetic's own resolution
        # times the matrix dimension -- the smallest difference float64 can carry here.
        self.drift_rel = float(np.finfo(float).eps) * (2 * m.N) ** 2
        self._refresh(check=False)

    def _slices(self, sigma, l):
        m = self.m
        order = [(l + k) % m.L for k in range(m.L)]
        d = np.exp(sigma * m.lam_s * self.X[order] + 1j * m.lam_c * self.Y[order])
        return m.expmK[None, :, :].astype(complex) * d[:, None, :]

    def _refresh(self, check=True):
        newG, ph, la = {}, 1.0 + 0j, 0.0
        for sigma in (+1, -1):
            U, D, T = udt_product(self._slices(sigma, self.l)[None], self.udt_block)
            newG[sigma] = inv_one_plus_block(U, D, T)[0]
            s, l_ = slogdet_one_plus_block(U, D, T)
            ph = ph * s[0]
            la += float(l_[0])
        if check and self.G:
            scale = max(float(np.abs(newG[s]).max()) for s in (+1, -1))
            d = (max(float(np.abs(newG[s] - self.G[s]).max()) for s in (+1, -1))
                 / max(scale, np.finfo(float).tiny))
            self.max_drift = max(self.max_drift, d)
            self.n_checks += 1
            self.n_bad += int(d > self.drift_rel)
        self.G = newG
        # the c-number from rho's -1, and the determinants' phase
        self.phase = complex(ph * np.exp(-1j * self.m.lam_c * self.Y.sum()))
        self.logabs = la

    def sweep(self):
        m = self.m
        for l in range(m.L):
            self.l = l
            if l % self.n_stab == 0:
                self._refresh()
            for i in range(m.N):
                dx = self.rng.normal(0.0, self.step)
                dy = self.rng.normal(0.0, self.step) if m.lam_c > 0 else 0.0
                x, y = self.X[l, i], self.Y[l, i]
                # Gaussian priors on the fields
                dS = -0.5 * ((x + dx) ** 2 - x ** 2) - 0.5 * ((y + dy) ** 2 - y ** 2)
                dU = np.exp(+m.lam_s * dx + 1j * m.lam_c * dy) - 1.0
                dD = np.exp(-m.lam_s * dx + 1j * m.lam_c * dy) - 1.0
                rU = 1.0 + dU * (1.0 - self.G[+1][i, i])
                rD = 1.0 + dD * (1.0 - self.G[-1][i, i])
                self.prop += 1
                if self.rng.random() < min(1.0, np.exp(min(dS, 700.0)) * abs(rU * rD)):
                    self.acc += 1
                    for sigma, dl, R in ((+1, dU, rU), (-1, dD, rD)):
                        G = self.G[sigma]
                        col = -G[:, i].copy()
                        col[i] += 1.0
                        self.G[sigma] = G - (dl / R) * np.outer(col, G[i, :])
                    self.X[l, i] = x + dx
                    self.Y[l, i] = y + dy
                    r = rU * rD
                    self.phase *= (r / abs(r)) * np.exp(-1j * m.lam_c * dy)
            self._wrap(l)
        self.l = 0
        return self.phase

    def _wrap(self, l):
        m = self.m
        d = np.exp(m.lam_s * self.X[l] + 1j * m.lam_c * self.Y[l])
        for sigma in (+1, -1):
            dd = np.exp(sigma * m.lam_s * self.X[l] + 1j * m.lam_c * self.Y[l])
            B = m.expmK.astype(complex) * dd[None, :]
            self.G[sigma] = B @ self.G[sigma] @ np.linalg.inv(B)

    def measure(self):
        self._refresh()
        g = (np.trace(self.G[+1]) + np.trace(self.G[-1])) / 2.0
        return self.phase, complex(g)


def run2d(m, n_meas=2000, warm=200, seed=0, n_stab=1, thin=2, step=1.0, udt_block=4):
    ch = Fast2D(m, seed=seed, n_stab=n_stab, udt_block=udt_block, step=step)
    for _ in range(warm):
        ch.sweep()
    ph, ob = [], []
    for _ in range(n_meas):
        for _ in range(thin):
            ch.sweep()
        p, g = ch.measure()
        ph.append(p)
        ob.append(g)
    return (np.array(ph), np.array(ob), ch.acc / max(ch.prop, 1),
            ch.max_drift, ch.n_bad / max(ch.n_checks, 1))
