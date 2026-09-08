"""Contour deformation for the 2-D Hubbard spin channel.

The one thing in this whole investigation that measurably improved a phase problem: in the
1-D continuous-HS test a single constant imaginary shift lifted <e^{i theta}> from 0.0143 to
0.0896, a 6.3x gain, gated against exact Gauss-Hermite quadrature (Cauchy held to 4e-6).
This asks whether it carries to 2-D, where the spin channel is the best decoupling available
(measured: every charge admixture and the SU(2) vector channel are worse).

The integrand is entire in the auxiliary field, so for a real shift c

    Z = Int dx g(x) = Int dx g(x + i c)          Cauchy

with Z fixed while Int|g| moves.  Writing g(x + ic) = e^{-x^2/2} e^{-i c.x + c^2/2} D(x + ic),
the modulus measure is p_c(x) ~ e^{-x^2/2} |D(x + ic)| -- real, positive, ordinary Metropolis --
and

    <e^{i theta}>_c = | E_{p_c} [ exp(-i c.x) * phase(D(x + ic)) ] |

Sampled DIRECTLY on each contour.  Reweighting one contour's chain to another has a phase
problem of its own -- that was tried in expE2 and its Cauchy gate came back 0.35-16 instead of
0, so every number downstream was void.  One chain per candidate c, no exceptions.
"""
from __future__ import annotations

import numpy as np

from stable import udt_product, slogdet_one_plus_block


class ContourChain2D:
    """Metropolis on real x with target exp(-x^2/2) |D(x + i c)|; the phase is carried."""

    def __init__(self, m, c, seed=0, step=1.1, udt_block=4):
        self.m, self.c, self.step, self.udt_block = m, c, step, int(udt_block)
        self.rng = np.random.default_rng(seed)
        self.X = self.rng.standard_normal((m.L, m.N))
        self.ph, self.la = self._weight(self.X)
        self.acc = self.prop = 0

    def _weight(self, X):
        """phase and log|.| of D(X + i c) = det(I+B_up) det(I+B_dn)."""
        m = self.m
        Z = X + 1j * self.c
        ph, la = 1.0 + 0j, 0.0
        for sigma in (+1, -1):
            d = np.exp(sigma * m.lam_s * Z)
            Bl = (m.expmK[None, :, :].astype(complex) * d[:, None, :])[None]
            U, D, T = udt_product(Bl, self.udt_block)
            s, l_ = slogdet_one_plus_block(U, D, T)
            ph *= s[0]
            la += float(l_[0])
        return complex(ph), la

    def sweep(self):
        m = self.m
        for l in range(m.L):
            for i in range(m.N):
                old = self.X[l, i]
                new = old + self.rng.normal(0.0, self.step)
                self.X[l, i] = new
                ph, la = self._weight(self.X)
                dS = -0.5 * (new ** 2 - old ** 2)
                self.prop += 1
                if self.rng.random() < np.exp(min(dS + la - self.la, 700.0)):
                    self.acc += 1
                    self.ph, self.la = ph, la
                else:
                    self.X[l, i] = old
        return self.ph

    def observed_phase(self):
        """exp(-i c.x) * phase(D(x + ic)) -- the quantity whose mean is <e^{i theta}>_c."""
        return np.exp(-1j * float(np.sum(self.c * self.X))) * self.ph


def run_contour(m, c, n_meas=800, warm=200, seed=0, step=1.1, thin=1):
    ch = ContourChain2D(m, c, seed=seed, step=step)
    for _ in range(warm):
        ch.sweep()
    out = []
    for _ in range(n_meas):
        for _ in range(thin):
            ch.sweep()
        out.append(ch.observed_phase())
    return np.array(out), ch.acc / max(ch.prop, 1)
