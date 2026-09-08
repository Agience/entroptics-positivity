"""Fast determinantal QMC: rank-1 Green's-function updates with n_stab re-orthogonalisation.

The stabilised sampler in `sampler_stable.py` re-evaluates the whole UDT product for every
proposal -- O(L^2 N^3) per sweep -- which caps the study at N = 4 and <sgn> >= 0.38.  The
standard algorithm keeps the equal-time Green's function and updates it in place:

    G_l = (I + B(l) ... B(1) B(L) ... B(l+1))^{-1}

    flip s[l,i]:   Delta_sigma = exp(-2 sigma lam s[l,i]) - 1
                   R_sigma     = 1 + Delta_sigma (1 - G_sigma[i,i])
                   accept with min(1, |R_up R_dn|)
                   G' = G - (Delta/R) (I - G)[:, i] (X) G[i, :]        rank-1, O(N^2)

    wrap to the next slice:   G_{l+1} = B(l+1) G_l B(l+1)^{-1}

and recomputes G from the UDT factors every `n_stab` slices, because the wrap accumulates the
same conditioning error that made the naive product fake a sign problem earlier in this rig.

The SIGN is tracked the same way -- each accepted flip multiplies it by sgn(R_up R_dn) -- and
is recomputed exactly from the core matrix at every stabilisation, so a drift between the two
is a bug that the gate catches rather than a number that quietly ships.

O(L N^3) per sweep against O(L^2 N^3): ~100x at N=16, L=40.  Everything here is gated in
`test_fastdqmc.py` against `sampler_stable.StableChains` and, at N=4, against exact
diagonalisation.
"""
from __future__ import annotations

import numpy as np

from stable import udt_product, core_matrix, inv_one_plus


class FastDQMC:
    """One chain.  `m` is a dqmc.Model."""

    def __init__(self, m, seed=0, n_stab=8, udt_block=4):
        # TWO different numbers, conflated on the first attempt: `n_stab` is how often the
        # sweep rebuilds G, `udt_block` is the QR re-orthogonalisation stride INSIDE that
        # rebuild.  With udt_block = n_stab = 8 the "exact" reference was itself wrong by
        # 3e-5, so the gate was comparing an update against a bad baseline.
        self.m, self.n_stab, self.udt_block = m, int(n_stab), int(udt_block)
        self.rng = np.random.default_rng(seed)
        self.S = self.rng.choice([-1.0, 1.0], size=(m.L, m.N))
        self.l = 0
        self.G = {}
        self.acc = self.prop = 0
        self.max_drift = 0.0
        self.n_checks = 0
        self.n_bad = 0
        # RELATIVE, not absolute.  |G| runs from ~1 at N=4 to ~34 at N=16, so a fixed 1e-6
        # is a different demand at every size; the drift that matters is drift against the
        # scale of the thing that drifted.  The reference is the arithmetic's own resolution
        # times the matrix dimension -- the smallest difference float64 can carry here.
        self.drift_rel = float(np.finfo(float).eps) * (2 * m.N) ** 2
        self._refresh()

    # ---- exact recomputation from the UDT factors

    def _slices(self, sigma, l):
        """The factors of A_l = B(l-1) B(l-2) ... B(l+1) B(l), in udt_product's order
        (index 0 is the RIGHTMOST factor).  B(l) is rightmost because with B = expmK V a
        flip perturbs from the right, B' = B (I + Delta e e^T), and only then is the ratio
        R = 1 + Delta (1 - G_ii) with G = (I + A_l)^{-1}."""
        m = self.m
        order = [(l + k) % m.L for k in range(m.L)]
        d = np.exp(sigma * m.lam * self.S[order])              # (L, N)
        return m.expmK[None, :, :] * d[:, None, :]

    def _refresh(self, check=True):
        """Rebuild G_l exactly from the UDT factors.  When `check`, first record how far the
        carried G had drifted from it -- that comparison is only meaningful because the sweep
        keeps `self.G` at `self.l` at all times (see `sweep`)."""
        newG, sgn = {}, 1.0
        for sigma in (+1, -1):
            U, D, T = udt_product(self._slices(sigma, self.l)[None], self.udt_block)
            newG[sigma] = inv_one_plus(U, D, T)[0]
            sgn *= core_matrix(U, D, T)[1][0]
        if check and self.G:
            scale = max(float(np.abs(newG[s]).max()) for s in (+1, -1))
            d = (max(float(np.abs(newG[s] - self.G[s]).max()) for s in (+1, -1))
                 / max(scale, np.finfo(float).tiny))
            self.max_drift = max(self.max_drift, d)
            self.n_checks += 1
            self.n_bad += int(d > self.drift_rel)
            # The refresh REPLACES G, so a spike is corrected here and its damage is confined
            # to the slice that produced it.  `max_drift` over a long run therefore always
            # catches the rare tail and says nothing about whether the physics moved; the
            # statistic that means anything is the FRACTION of refreshes that exceeded
            # tolerance.
        self.G = newG
        self.sign = float(sgn)

    # ---- one sweep
    #
    # The invariant, stated once because getting it wrong is what three earlier versions did:
    #
    #     A_l = B(l-1) B(l-2) ... B(0) B(L-1) ... B(l+1) B(l),   G_l = (I + A_l)^{-1}
    #
    # B(l) is RIGHTMOST (a flip perturbs B = expmK V from the right), and
    # A_{l+1} = B(l) A_l B(l)^{-1}, so the wrap uses the slice JUST FINISHED.  `self.G` is
    # G_{self.l} on entry to every iteration and on exit from the sweep.

    def sweep(self):
        m = self.m
        for l in range(m.L):
            self.l = l                                  # G is G_l here, by the invariant
            if l % self.n_stab == 0 and l > 0:
                self._refresh()
            for i in range(m.N):
                s = self.S[l, i]
                dU = np.exp(-2.0 * m.lam * s) - 1.0
                dD = np.exp(+2.0 * m.lam * s) - 1.0
                rU = 1.0 + dU * (1.0 - self.G[+1][i, i])
                rD = 1.0 + dD * (1.0 - self.G[-1][i, i])
                self.prop += 1
                if self.rng.random() < min(1.0, abs(rU * rD)):
                    self.acc += 1
                    for sigma, dl, R in ((+1, dU, rU), (-1, dD, rD)):
                        G = self.G[sigma]
                        col = -G[:, i].copy()
                        col[i] += 1.0                   # (I - G)[:, i]
                        self.G[sigma] = G - (dl / R) * np.outer(col, G[i, :])
                    self.S[l, i] = -s
                    self.sign *= float(np.sign(rU * rD))
            self._wrap(l)                               # G_l -> G_{l+1}
        self.l = 0                                      # wrapped past L-1 is back to 0
        return self.sign

    def _wrap(self, l):
        """G_{l+1} = B(l) G_l B(l)^{-1}."""
        m = self.m
        for sigma in (+1, -1):
            B = m.expmK * np.exp(sigma * m.lam * self.S[l])[None, :]
            self.G[sigma] = B @ self.G[sigma] @ np.linalg.inv(B)

    # ---- measurement

    def measure(self):
        """sign, and tr G(0) as a cheap observable both samplers can be compared on."""
        self._refresh()
        return self.sign, float(np.trace(self.G[+1]) + np.trace(self.G[-1])) / 2.0


def run(m, n_meas=400, warm=100, seed=0, n_stab=8, thin=1, udt_block=4):
    ch = FastDQMC(m, seed=seed, n_stab=n_stab, udt_block=udt_block)
    for _ in range(warm):
        ch.sweep()
    sg, ob = [], []
    for _ in range(n_meas):
        for _ in range(thin):
            ch.sweep()
        s, o = ch.measure()
        sg.append(s)
        ob.append(o)
    return (np.array(sg), np.array(ob), ch.acc / max(ch.prop, 1), ch.max_drift,
            ch.n_bad / max(ch.n_checks, 1))
