"""Vectorised constrained-path AFQMC -- the same algorithm as `cpmc.py`, over all walkers at once.

`cpmc.py` is the readable reference and it is too slow to answer the question: its inner loop is
one Python iteration per (walker, site, spin) on 8x8 matrices, so interpreter overhead dominates
completely and a Trotter extrapolation does not finish.  Here the walker index is an ARRAY axis:
Phi has shape (W, N, n_sigma), the mixed Green's function (W, N, N), and the site loop runs once
for the whole population instead of once per walker.

The constrained and free-projection branches are the SAME code with two switches, because the
previous version kept them as separate classes and the bug that cost this rig a day -- the
finite-temperature Green's-function convention -- had to be found and fixed in both.  One
implementation cannot disagree with itself.

    constrained=True    clip:  c = max(r, 0), a walker with both branches non-positive dies.
                        No weight is ever negative.  Biased by the trial node.
    constrained=False   carry: c = |r|, and the sign multiplies the weight.  Exact, and the
                        average sign decays.

GATE.  At W = 1 this draws exactly one uniform per site in the same order `cpmc.py` does, so the
two must produce bit-identical trajectories.  That is checked in `gate_fast.py`, not asserted
here, and it is the only reason to trust a rewrite of a file whose last bug was invisible to
every component test.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import expm


class FastCPMC:
    def __init__(self, m, n_up, n_dn, n_walkers=200, seed=0, psi_t=None,
                 constrained=True, ortho_every=5, pop_every=10):
        self.m, self.n_up, self.n_dn = m, int(n_up), int(n_dn)
        self.rng = np.random.default_rng(seed)
        self.constrained = bool(constrained)
        self.ortho_every, self.pop_every = int(ortho_every), int(pop_every)
        self.lam = float(np.arccosh(np.exp(m.dtau * m.U / 2.0))) if m.U > 0 else 0.0
        self.expK_half = expm(-0.5 * m.dtau * m.K)
        if psi_t is None:
            v = np.linalg.eigh(m.K)[1]
            psi_t = {+1: v[:, :self.n_up], -1: v[:, :self.n_dn]}
        self.P = {s: np.array(psi_t[s], float) for s in (+1, -1)}
        W = int(n_walkers)
        self.phi = {s: np.repeat(self.P[s][None], W, axis=0) for s in (+1, -1)}
        self.w = np.ones(W)
        self.killed = 0

    # ---- (W, N, N) mixed Green's function.  G[w,i,j] = <c^dag_j c_i> for walker w.

    def _green(self, ph, s):
        P = self.P[s]
        O = np.einsum("ji,wjk->wik", P, ph)                       # P^T phi, (W, n, n)
        rhs = np.broadcast_to(P.T, (ph.shape[0],) + P.T.shape)
        return ph @ np.linalg.solve(O, rhs)

    def _overlap(self, ph, s):
        return np.linalg.det(np.einsum("ji,wjk->wik", self.P[s], ph))

    # ---- one imaginary-time step, whole population at once

    def step(self):
        m, lam = self.m, self.lam
        ph = {s: np.einsum("ij,wjk->wik", self.expK_half, self.phi[s]) for s in (+1, -1)}
        G = {s: self._green(ph[s], s) for s in (+1, -1)}
        W = ph[+1].shape[0]
        w = self.w.copy()
        live = np.ones(W, bool)

        for i in range(m.N):
            r = {}
            for s in (+1, -1):
                gii = G[s][:, i, i]
                for x in (+1.0, -1.0):
                    r[(s, x)] = 1.0 + (np.exp(s * lam * x) - 1.0) * gii
            rp = r[(+1, +1.0)] * r[(-1, +1.0)]
            rm = r[(+1, -1.0)] * r[(-1, -1.0)]
            if self.constrained:
                cp, cm = np.maximum(rp, 0.0), np.maximum(rm, 0.0)   # THE CONSTRAINT
            else:
                cp, cm = np.abs(rp), np.abs(rm)
            tot = cp + cm
            ok = tot > 0.0
            live &= ok
            safe = np.where(ok, tot, 1.0)
            x = np.where(self.rng.random(W) < cp / safe, 1.0, -1.0)
            w = w * (0.5 * tot)
            if not self.constrained:
                w = w * np.where(x > 0, np.sign(rp), np.sign(rm))
            for s in (+1, -1):
                d = np.exp(s * lam * x) - 1.0
                R = np.where(x > 0, r[(s, +1.0)], r[(s, -1.0)])
                R = np.where(R == 0.0, 1.0, R)                     # dead walkers only
                ph[s][:, i, :] *= (1.0 + d)[:, None]
                col = -G[s][:, :, i].copy()
                col[:, i] += 1.0                                    # e_i - G[:, i]
                G[s] = G[s] + (d / R)[:, None, None] * (col[:, :, None] * G[s][:, None, i, :])

        ph = {s: np.einsum("ij,wjk->wik", self.expK_half, ph[s]) for s in (+1, -1)}
        if self.constrained:
            live &= (self._overlap(ph[+1], +1) * self._overlap(ph[-1], -1)) > 0.0
        live &= np.isfinite(w)
        self.killed += int((~live).sum())
        if not live.any():
            raise RuntimeError("every walker was killed by the constraint")
        self.phi = {s: ph[s][live] for s in (+1, -1)}
        self.w = w[live]
        # Rescale to mean magnitude 1, on BOTH branches.  Every estimator here is a RATIO of
        # weight sums, so a factor common to all walkers cancels exactly and this changes no
        # answer.  Without it the arithmetic decides the experiment: the weight is a product of
        # one factor per (site, step), so at beta = 8 and dtau = 0.1 it carries ~3000 factors
        # slightly below 1 on the constrained branch and underflows to zero, after which the
        # comb in population_control divides by a zero total and the whole population goes
        # non-finite.  That was read as "the trial node kills every walker at U = 8" -- a
        # physical claim about the constraint -- when it was float64 range.  Free projection
        # overflows the other way for the same reason.
        a = np.abs(self.w).mean()
        if a > 0 and np.isfinite(a):
            self.w = self.w / a
        return int(live.sum())

    def orthonormalise(self):
        for s in (+1, -1):
            q, rr = np.linalg.qr(self.phi[s])
            d = np.sign(np.einsum("wii->wi", rr))
            self.phi[s] = q * d[:, None, :]

    def population_control(self, target):
        if not self.constrained:
            # Signed weights: comb on the MAGNITUDE and carry the sign, which is the standard
            # released-constraint baseline.  A plain comb on w is invalid (negative weights are
            # not a distribution), but doing nothing at all is not the fair comparison either:
            # without branching, free projection's variance grows from weight SPREAD as well as
            # from the sign, and quoting that against a branched CPMC overstates the sign's
            # share.  Sum(w)/Sum(|w|) and Sum(w E)/Sum(w) are both preserved in expectation.
            a = np.abs(self.w)
            tot = a.sum()
            if not (tot > 0 and np.isfinite(tot)):
                raise RuntimeError("population weight is no longer a positive finite number")
            u = (self.rng.random() + np.arange(target)) / target
            idx = np.searchsorted(np.cumsum(a / tot), u)
            self.phi = {s: self.phi[s][idx].copy() for s in (+1, -1)}
            self.w = np.sign(self.w[idx])
            return
        tot = self.w.sum()
        if not (tot > 0 and np.isfinite(tot)):
            raise RuntimeError("population weight is no longer a positive finite number")
        u = (self.rng.random() + np.arange(target)) / target
        idx = np.searchsorted(np.cumsum(self.w / tot), u)
        self.phi = {s: self.phi[s][idx].copy() for s in (+1, -1)}
        self.w = np.ones(target)              # all equal after the comb; the scale cancels

    # ---- observables

    def mean_sign(self):
        return float(self.w.sum() / np.abs(self.w).sum())

    def energy(self):
        m = self.m
        G = {s: self._green(self.phi[s], s) for s in (+1, -1)}
        kin = sum(np.einsum("ij,wji->w", m.K, G[s]) for s in (+1, -1))
        nu = np.einsum("wii->wi", G[+1])
        nd = np.einsum("wii->wi", G[-1])
        e = kin + m.U * np.sum((nu - 0.5) * (nd - 0.5), axis=1)
        return float((self.w @ e) / self.w.sum())


def run(m, n_up, n_dn, beta, *, constrained=True, n_walkers=400, seed=0, n_meas=300,
        psi_t=None, blocks=15, ortho_every=5, pop_every=10):
    ch = FastCPMC(m, n_up, n_dn, n_walkers=n_walkers, seed=seed, psi_t=psi_t,
                  constrained=constrained, ortho_every=ortho_every, pop_every=pop_every)
    steps = int(round(beta / m.dtau))
    es, sg, n_dead = [], [], 0
    for t in range(steps + n_meas):
        try:
            ch.step()
        except RuntimeError:
            # A trial node that kills the whole population is a RESULT about that Psi_T, not a
            # crash: it says the constraint surface has no region the walk can occupy.  Report
            # it rather than aborting the sweep that was measuring exactly this.
            return dict(e=float("nan"), se=float("nan"), sgn=float("nan"),
                        killed=ch.killed, extinct_at=t)
        if (t + 1) % ch.ortho_every == 0:
            ch.orthonormalise()
        if (t + 1) % ch.pop_every == 0:
            ch.population_control(n_walkers)
        if t >= steps:
            tot = ch.w.sum()
            if not (np.isfinite(tot) and abs(tot) > 0):
                n_dead += 1               # the signed estimator has no denominator left
                continue
            es.append(ch.energy())
            sg.append(ch.mean_sign())
    if len(es) < blocks:
        return dict(e=float("nan"), se=float("nan"), sgn=float("nan"), killed=ch.killed,
                    extinct_at=None, dead_frac=1.0)
    e = np.array(es)
    bl = np.array([b.mean() for b in np.array_split(e, blocks)])
    return dict(e=float(bl.mean()), se=float(bl.std(ddof=1) / np.sqrt(blocks)),
                sgn=float(np.mean(sg)), killed=ch.killed, extinct_at=None,
                dead_frac=n_dead / max(n_dead + len(es), 1))
