"""Constrained-path AFQMC with a MULTI-DETERMINANT, NON-ORTHOGONAL trial wavefunction.

Part 5 closed trial-wavefunction tuning three ways and one of the three was closed on a bad
construction.  `expAA` expanded the exact ground state in a fixed orthonormal determinant basis
and truncated, which is optimal in that basis and the least compact way to build a trial;
`expGG` then optimised over k NON-ORTHOGONAL determinants and their coefficients and found that
four of them beat sixteen fixed-basis ones at every U, with the advantage GROWING with coupling
(+0.023, +0.062, +0.092 at k = 4 for U = 4, 8, 12).  So the constrained-path bias is not floored,
and this is the machinery that tests whether the overlap gain actually converts into a smaller
bias.

    |Psi_T> = sum_j c_j |phi_j^up> (x) |phi_j^dn>

THE SPINS STOP FACTORISING, and that is the whole structural difference.  With one determinant
the overlap is det(P^up' phi^up) * det(P^dn' phi^dn) and the two species are independent one-body
problems.  With a sum they are not:

    <Psi_T|phi> = sum_j c_j D_j^up D_j^dn,      D_j^sigma = det(P_j^sigma' phi^sigma)

so every quantity carries a j index and the up channel's ratio depends on the down channel's
determinants.  The mixed Green's function becomes the overlap-weighted average of the
per-determinant ones,

    G^sigma = sum_j w_j G_j^sigma / sum_j w_j,   w_j = c_j D_j^up D_j^dn
    G_j^sigma = phi^sigma (P_j^sigma' phi^sigma)^-1 P_j^sigma'

and the ratio for scaling row i of spin sigma by (1 + d) is

    R = sum_j w_j (1 + d_up G_j^up[i,i]) (1 + d_dn G_j^dn[i,i]) / sum_j w_j

which reduces to the single-determinant `1 + d G[i,i]` at k = 1 -- checked, not assumed, by
`gate_multi.py` requiring bit-identical trajectories against `cpmc_fast` at k = 1.

Cost is k times the single-determinant walk: one Green's function and one rank-1 update per
determinant per site.  Everything is batched over walkers exactly as `cpmc_fast` is, with the
determinant index as a second array axis.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import expm


class MultiCPMC:
    def __init__(self, m, n_up, n_dn, dets, coeffs, n_walkers=200, seed=0,
                 constrained=True, ortho_every=5, pop_every=10):
        """`dets` is a list of k dicts {+1: (N,n_up), -1: (N,n_dn)}; `coeffs` the k weights."""
        self.m, self.n_up, self.n_dn = m, int(n_up), int(n_dn)
        self.rng = np.random.default_rng(seed)
        self.constrained = bool(constrained)
        self.ortho_every, self.pop_every = int(ortho_every), int(pop_every)
        self.lam = float(np.arccosh(np.exp(m.dtau * m.U / 2.0))) if m.U > 0 else 0.0
        self.expK_half = expm(-0.5 * m.dtau * m.K)
        self.k = len(dets)
        self.c = np.asarray(coeffs, float)
        self.P = {s: np.stack([np.asarray(d[s], float) for d in dets]) for s in (+1, -1)}
        W = int(n_walkers)
        # FIX THE GLOBAL SIGN, then start on the heaviest POSITIVE branch.
        #
        # The overall sign of Psi_T is physically arbitrary, and the constraint is not: it kills
        # any walker with <Psi_T|phi> <= 0.  The optimal coefficients c = S^-1 b come out with
        # whatever sign the fit produced -- measured, all-negative at k = 2 and mixed at k = 4 --
        # so starting every walker on argmax|c| gave an initial overlap of -0.845 and the entire
        # population died on the first step, reported as "every walker was killed by the
        # constraint".  That reads as a statement about the trial node and is a sign convention.
        # Pick the start by the quantity the CONSTRAINT actually tests, and fix the sign on that.
        #
        # An earlier version flipped the global sign so the largest COEFFICIENT was positive and
        # started on that determinant.  That is a proxy, not the thing: the constraint tests
        # <Psi_T|phi_j> = sum_k c_k <phi_k|phi_j>, which involves the overlaps with every OTHER
        # determinant and is routinely negative while c_j > 0.  At k = 3 it killed the entire
        # population on step one and reported it as "every walker was killed by the constraint" --
        # the same arbitrary-sign defect wearing the same physical-sounding message.
        ovl = np.array([sum(self.c[a] * np.linalg.det(self.P[+1][a].T @ dets[b][+1])
                                      * np.linalg.det(self.P[-1][a].T @ dets[b][-1])
                            for a in range(self.k)) for b in range(self.k)])
        j0 = int(np.argmax(np.abs(ovl)))
        if ovl[j0] < 0:
            self.c = -self.c                          # the global sign of Psi_T is arbitrary
        start = dets[j0]
        self.phi = {s: np.repeat(np.asarray(start[s], float)[None], W, axis=0) for s in (+1, -1)}
        self.w = np.ones(W)
        self.killed = 0

    # ---- per-determinant overlaps and Green's functions

    def _dets_and_greens(self, ph):
        """D[s]: (k, W).  G[s]: (k, W, N, N)."""
        D, G = {}, {}
        for s in (+1, -1):
            k, N, n = self.P[s].shape
            Wn = ph[s].shape[0]
            O = np.einsum("jai,wak->jwik", self.P[s], ph[s])          # (k, W, n, n) = P_j' phi
            D[s] = np.linalg.det(O)
            Pt = np.swapaxes(self.P[s], 1, 2)                          # (k, n, N)
            rhs = np.broadcast_to(Pt[:, None], (k, Wn, n, N))
            G[s] = np.einsum("wan,jwnb->jwab", ph[s], np.linalg.solve(O, rhs))
        return D, G

    def _mix(self, D, G):
        wj = self.c[:, None] * D[+1] * D[-1]                          # (k, W)
        tot = wj.sum(axis=0)
        Gm = {s: np.einsum("jw,jwab->wab", wj, G[s]) / tot[:, None, None] for s in (+1, -1)}
        return wj, tot, Gm

    # ---- one imaginary-time step

    def step(self):
        m, lam = self.m, self.lam
        ph = {s: np.einsum("ij,wjk->wik", self.expK_half, self.phi[s]) for s in (+1, -1)}
        D, G = self._dets_and_greens(ph)
        wj = self.c[:, None] * D[+1] * D[-1]
        tot = wj.sum(axis=0)
        W = ph[+1].shape[0]
        w = self.w.copy()
        live = np.isfinite(tot) & (np.abs(tot) > 0)

        for i in range(m.N):
            r = {}
            for x in (+1.0, -1.0):
                du = np.exp(+lam * x) - 1.0
                dd = np.exp(-lam * x) - 1.0
                fu = 1.0 + du * G[+1][:, :, i, i]                     # (k, W)
                fd = 1.0 + dd * G[-1][:, :, i, i]
                r[x] = (wj * fu * fd).sum(axis=0) / np.where(tot != 0, tot, 1.0)
            rp, rm = r[+1.0], r[-1.0]
            cp, cm = (np.maximum(rp, 0.0), np.maximum(rm, 0.0)) if self.constrained \
                else (np.abs(rp), np.abs(rm))
            s_tot = cp + cm
            ok = s_tot > 0.0
            live &= ok
            x = np.where(self.rng.random(W) < cp / np.where(ok, s_tot, 1.0), 1.0, -1.0)
            w = w * (0.5 * s_tot)
            if not self.constrained:
                w = w * np.where(x > 0, np.sign(rp), np.sign(rm))
            for s in (+1, -1):
                d = np.exp(s * lam * x) - 1.0                          # (W,)
                Rj = 1.0 + d[None, :] * G[s][:, :, i, i]               # (k, W)
                Rj = np.where(Rj == 0.0, 1.0, Rj)
                ph[s][:, i, :] *= (1.0 + d)[:, None]
                D[s] = D[s] * Rj
                col = -G[s][:, :, :, i].copy()
                col[:, :, i] += 1.0
                G[s] = G[s] + (d[None, :] / Rj)[:, :, None, None] * (
                    col[:, :, :, None] * G[s][:, :, None, i, :])
            wj = self.c[:, None] * D[+1] * D[-1]
            tot = wj.sum(axis=0)

        ph = {s: np.einsum("ij,wjk->wik", self.expK_half, ph[s]) for s in (+1, -1)}
        if self.constrained:
            Df, _ = self._dets_and_greens(ph)
            live &= (self.c[:, None] * Df[+1] * Df[-1]).sum(axis=0) > 0.0
        live &= np.isfinite(w)
        self.killed += int((~live).sum())
        if not live.any():
            raise RuntimeError("every walker was killed by the constraint")
        self.phi = {s: ph[s][live] for s in (+1, -1)}
        self.w = w[live]
        a = np.abs(self.w).mean()
        if a > 0 and np.isfinite(a):
            self.w = self.w / a
        return int(live.sum())

    def orthonormalise(self):
        for s in (+1, -1):
            q, rr = np.linalg.qr(self.phi[s])
            self.phi[s] = q * np.sign(np.einsum("wii->wi", rr))[:, None, :]

    def population_control(self, target):
        if not self.constrained:
            a = np.abs(self.w); t = a.sum()
            if not (t > 0 and np.isfinite(t)):
                raise RuntimeError("population weight is not positive and finite")
            idx = np.searchsorted(np.cumsum(a / t),
                                  (self.rng.random() + np.arange(target)) / target)
            self.phi = {s: self.phi[s][idx].copy() for s in (+1, -1)}
            self.w = np.sign(self.w[idx])
            return
        t = self.w.sum()
        if not (t > 0 and np.isfinite(t)):
            raise RuntimeError("population weight is not positive and finite")
        idx = np.searchsorted(np.cumsum(self.w / t),
                              (self.rng.random() + np.arange(target)) / target)
        self.phi = {s: self.phi[s][idx].copy() for s in (+1, -1)}
        self.w = np.ones(target)

    def energy(self):
        """The mixed estimator.  The ONE-body part uses the overlap-weighted Green's function;
        the TWO-body part must NOT.

        For a single determinant <n_up n_dn> factorises into G_up[i,i] G_dn[i,i], because given
        the determinant the two species are independent one-body problems.  For a SUM of
        determinants that is false: the two-body expectation is the weighted average of the
        per-determinant PRODUCTS, not the product of the weighted averages,

            <Psi_T| n_up n_dn |phi> / <Psi_T|phi> = sum_j w_j G_j^up[i,i] G_j^dn[i,i] / sum_j w_j

        Building it the other way is wrong by 6% to 31% at k = 2 to 4 while the overlap stays
        exact to 6e-16 -- and it is invisible at k = 1, where the two expressions coincide, so
        every k = 1 gate passed over it.  `gate_multi_k2.py` is what catches it.
        """
        m = self.m
        D, G = self._dets_and_greens(self.phi)
        wj, tot, Gm = self._mix(D, G)
        kin = sum(np.einsum("ij,wji->w", m.K, Gm[s]) for s in (+1, -1))
        nu = np.einsum("jwii->jwi", G[+1]) - 0.5                      # (k, W, N)
        nd = np.einsum("jwii->jwi", G[-1]) - 0.5
        pot = m.U * np.einsum("jw,jwi->w", wj, nu * nd) / tot
        e = kin + pot
        return float((self.w @ e) / self.w.sum())


def run_multi(m, n_up, n_dn, dets, coeffs, beta, *, n_walkers=400, seed=0, n_meas=150,
              constrained=True, blocks=15):
    ch = MultiCPMC(m, n_up, n_dn, dets, coeffs, n_walkers=n_walkers, seed=seed,
                   constrained=constrained)
    steps = int(round(beta / m.dtau))
    es = []
    for t in range(steps + n_meas):
        ch.step()
        if (t + 1) % ch.ortho_every == 0:
            ch.orthonormalise()
        if (t + 1) % ch.pop_every == 0:
            ch.population_control(n_walkers)
        if t >= steps:
            es.append(ch.energy())
    e = np.array(es)
    bl = np.array([b.mean() for b in np.array_split(e, blocks)])
    return dict(e=float(bl.mean()), se=float(bl.std(ddof=1) / np.sqrt(blocks)),
                killed=ch.killed)
