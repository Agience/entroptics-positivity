"""Numerically stabilised sampler: every accept/reject weight and every measurement
goes through the UDT factorisation, so the chain samples |w| and not a float64
artefact.  The unstabilised sampler's <sign> collapse at large beta*U was breakdown,
not physics -- this is the version whose numbers mean anything.
"""
from __future__ import annotations

import numpy as np

from stable import udt_product, slogdet_one_plus, inv_one_plus


class StableChains:
    def __init__(self, m, R: int, seed: int = 0, n_stab: int = 5):
        self.m, self.R, self.n_stab = m, R, n_stab
        self.rng = np.random.default_rng(seed)
        self.S = self.rng.choice([-1.0, 1.0], size=(R, m.L, m.N))
        self.eye = np.broadcast_to(np.eye(m.N), (R, m.N, m.N)).copy()
        self.accepted = self.proposed = 0
        self.cur_sign, self.cur_log = self._weights(self.S)

    def _Bl(self, S, sigma):
        d = np.exp(sigma * self.m.lam * S)
        return self.m.expmK[None, None, :, :] * d[:, :, None, :]

    def _weights(self, S):
        sign = np.ones(self.R)
        logabs = np.zeros(self.R)
        for sigma in (+1, -1):
            U, D, T = udt_product(self._Bl(S, sigma), self.n_stab)
            sg, ld = slogdet_one_plus(U, D, T)
            sign *= sg
            logabs += ld
        return sign, logabs

    def sweep(self):
        m, R = self.m, self.R
        for l in range(m.L):
            for i in range(m.N):
                Sp = self.S.copy()
                Sp[:, l, i] *= -1.0
                new_sign, new_log = self._weights(Sp)
                acc = self.rng.random(R) < np.exp(np.clip(new_log - self.cur_log, -700, 700))
                self.proposed += R
                self.accepted += int(acc.sum())
                if acc.any():
                    self.S[acc] = Sp[acc]
                    self.cur_sign[acc] = new_sign[acc]
                    self.cur_log[acc] = new_log[acc]
        return self.cur_sign

    def measure(self):
        """(R,) sign and (R, L, N, N) spin-averaged imaginary-time correlator matrix."""
        m, R = self.m, self.R
        sign = np.ones(R)
        G = np.zeros((R, m.L, m.N, m.N))
        for sigma in (+1, -1):
            Bl = self._Bl(self.S, sigma)
            U, D, T = udt_product(Bl, self.n_stab)
            sg, _ = slogdet_one_plus(U, D, T)
            sign *= sg
            P = inv_one_plus(U, D, T)
            G[:, 0] += P
            for l in range(1, m.L):
                P = Bl[:, l - 1] @ P
                G[:, l] += P
        return sign, G / 2.0
