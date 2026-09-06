"""Complex Langevin for the Hubbard model -- the one remaining family with a DYNAMICAL failure.

Every other route in this rig fails statically: the average sign is exponentially small and the
error is set by `M <sgn>^2`, full stop.  Complex Langevin has no average sign at all.  It
complexifies the auxiliary field, evolves it by a stochastic differential equation whose drift is
the gradient of the (now complex) action, and averages observables along the trajectory with no
reweighting and nothing to cancel.  The cost is polynomial in every regime.

It also, sometimes, converges to the WRONG ANSWER, silently.  That is the whole problem with it,
and it is a different KIND of problem from the sign problem: not a variance that grows, but a
trajectory that wanders where the derivation's boundary term stops vanishing.  It is a question
about a dynamical system, and the answer is a decay rate -- the one thing in this investigation
that a spectral read measures natively and the average sign does not.

CHANNEL MATTERS, AND IT DECIDES WHETHER THIS IS COMPLEX LANGEVIN AT ALL.  Both continuous
Hubbard-Stratonovich transforms are exact, and only one gives a complex action:

  spin    exp(-dt U (n_up-1/2)(n_dn-1/2)) = e^{-dt U/4} Int N(x) exp(lam x (n_up - n_dn))
  charge                                  = e^{+dt U/4} Int N(y) exp(i lam y (n_up + n_dn - 1))

with lam = sqrt(dt U) in both.  In the SPIN channel the action is real for a real field, so the
drift is real, the noise is real, and the trajectory never leaves the real axis -- measured as
|Im x| = 0.0000 on every row of the first gate run.  That is ordinary Langevin on a signed
measure, which is not this method: the sign appears there as a determinant changing sign, not as
a phase, so there is nothing to complexify.  The CHARGE channel carries an explicit i, the action
is genuinely complex, and the field complexifies as the method requires.  Both are implemented,
the default is charge, and the spin channel is kept as the CONTROL: complex Langevin run there
must reduce to ordinary Langevin and reproduce exact diagonalisation.

    spin    S(x)    = sum x^2/2 - sum_sigma ln det(I + B_sigma(x))
            K_{l,i} = -x_{l,i} + sum_sigma sigma lam (1 - G_sigma(l)_{ii})

    charge  S(y)    = sum y^2/2 + i lam sum y - sum_sigma ln det(I + B_sigma(iy))
            K_{l,i} = -y_{l,i} - i lam + sum_sigma i lam (1 - G_sigma(l)_{ii})

with G_sigma(l) = (I + A_l)^{-1}, A_l = B(l-1)...B(0)B(L-1)...B(l), B(l) RIGHTMOST -- the same
invariant the rest of this rig uses, restated because getting it wrong cost four attempts in
`fastdqmc.py` and a day in `cpmc.py`.

MANY CHAINS AT ONCE, because one chain cannot buy the statistics.  The integrated
autocorrelation time was measured at 565 Langevin STEPS on a 4-site, 10-slice model, so a run of
60,000 steps carries about 53 effective samples and an error of +-0.005 on a quantity of 0.167 --
too coarse to separate a step-size bias from noise, which is the only question worth asking here.
Independent chains are the cheap axis: `stable.udt_product` is already batched, the field carries
a leading chain index, and the per-call numpy overhead that dominates a 4x4 problem is paid once
for the whole ensemble rather than once per chain.

Nothing here is trusted without `gate_clangevin.py`: the analytic drift against a finite
difference of the complexified action, and the sampler against exact diagonalisation of the same
Trotter product.
"""
from __future__ import annotations

import numpy as np

from stable import udt_product, inv_one_plus_block


class CLangevin:
    """Complex Langevin on the continuous field, batched over independent chains.

    The field is (C, L, N) complex.  `eps` is the Langevin step; `adaptive` rescales it by the
    largest drift seen, which is the standard guard against the runaway excursions this method is
    known for -- and it is NOT a fix for wrong convergence, only for blow-up.  A run that stays
    finite and converges to the wrong number is the failure this file exists to characterise.
    """

    def __init__(self, m, eps=5e-4, seed=0, adaptive=True, udt_block=4,
                 stable_every=4, channel="charge", chains=1, ref_scale=1.0):
        if channel not in ("spin", "charge"):
            raise ValueError("channel must be 'spin' or 'charge'")
        self.m = m
        self.channel = channel
        self.chains = int(chains)
        self.lam = float(np.sqrt(m.dtau * m.U))
        self.eps, self.adaptive = float(eps), bool(adaptive)
        self.ref_scale = float(ref_scale)
        self.udt_block = int(udt_block)
        self.stable_every = int(stable_every)
        self.rng = np.random.default_rng(seed)
        self.X = np.zeros((self.chains, m.L, m.N), dtype=complex)
        self.expmK = m.expmK.astype(complex)
        self.max_drift = 0.0
        self.n_steps = 0
        self.t_elapsed = 0.0

    # ---- the ordered slice stack, and the Green's function at every slice

    def _coupling(self, sigma):
        """The exponent multiplying the field: sigma*lam in the spin channel, i*lam in charge
        (the same for both spins there, because charge couples to n_up + n_dn)."""
        return sigma * self.lam if self.channel == "spin" else 1j * self.lam

    def _slices(self, sigma, l):
        """(C, L, N, N) -- one ordered slice stack per chain, as `udt_product` wants it."""
        m = self.m
        order = [(l + k) % m.L for k in range(m.L)]
        d = np.exp(self._coupling(sigma) * self.X[:, order, :])
        return self.expmK[None, None, :, :] * d[:, :, None, :]

    def greens(self, stable_every=None):
        """G_sigma(l) for every chain, slice and spin: dict[sigma] -> (C, L, N, N).

        G(l+1) = B(l) G(l) B(l)^-1, from the invariant A_{l+1} = B(l) A_l B(l)^-1.  A drift needs
        G at EVERY slice, so wrapping costs one matrix product per slice where a fresh stabilised
        product costs O(L) -- the difference is O(L) against O(L^2) on the innermost loop of the
        method.  Re-stabilised every `stable_every` slices; `gate_clangevin.py` checks the wrapped
        result against the fully stabilised one rather than trusting the interval (measured
        1.07e-15 at stable_every = 4).
        """
        m, C = self.m, self.chains
        k = int(stable_every if stable_every is not None else self.stable_every)
        out = {}
        for sigma in (+1, -1):
            g = np.empty((C, m.L, m.N, m.N), dtype=complex)
            d = np.exp(self._coupling(sigma) * self.X)
            for l in range(m.L):
                if l % k == 0:
                    U, D, T = udt_product(self._slices(sigma, l), self.udt_block)
                    g[:, l] = inv_one_plus_block(U, D, T)
                else:
                    B = self.expmK[None] * d[:, l - 1][:, None, :]
                    g[:, l] = B @ g[:, l - 1] @ np.linalg.inv(B)
            out[sigma] = g
        return out

    def drift(self, G=None):
        """-dS/dx.  See the module docstring for the two channels' forms."""
        G = self.greens() if G is None else G
        k = -self.X.copy()
        if self.channel == "charge":
            k = k - 1j * self.lam                 # from the -1 in rho = n_up + n_dn - 1
        for sigma in (+1, -1):
            k += self._coupling(sigma) * (1.0 - np.einsum("clii->cli", G[sigma]))
        return k

    # ---- one Langevin step

    def step(self):
        k = self.drift()
        a = float(np.abs(k).max())
        self.max_drift = max(self.max_drift, a)
        dt = self.eps
        if self.adaptive and a > 0:
            dt = self.eps / max(1.0, a / self._ref_drift())
        eta = self.rng.standard_normal(self.X.shape)
        self.X = self.X + dt * k + np.sqrt(2.0 * dt) * eta
        self.n_steps += 1
        self.t_elapsed += dt          # the ACTUAL Langevin time, not eps * n_steps
        return a

    def _ref_drift(self):
        """The reference scale the adaptive step divides by.

        THE TWO NUMBERS HERE ARE INVENTED AND THAT IS A PROBLEM, so `ref_scale` exists to sweep
        them.  Adaptive stepping is standard practice for complex Langevin, but the reference is
        a choice, and here it is NOT symmetric between the two channels: the spin channel's drift
        peaks near 7 so the step is cut about fourfold, while the charge channel's reaches 1e3 to
        1e4 so it is cut about a thousandfold.  The two channels are then integrated with
        effective steps differing by ~250x, and the failure being reported is the charge
        channel's.  A conclusion that moves when `ref_scale` moves is a conclusion about this
        constant, not about complex Langevin -- which is what `expHH_constants.py` measures.
        """
        return self.ref_scale * max(1.0, float(np.abs(self.X).mean()) + self.lam * 2.0)

    # ---- observables

    def observe(self, G=None):
        """Per-CHAIN observables, shape (C,) -- kept separate so the error bar can come from the
        spread across independent chains rather than from a correlated series."""
        G = self.greens() if G is None else G
        nu = 1.0 - np.einsum("clii->cli", G[+1])
        nd = 1.0 - np.einsum("clii->cli", G[-1])
        # <n_up n_dn> factorises across spin species at FIXED field, which is what the decoupling
        # buys: given the field, the two species are independent one-body problems.
        return dict(n=(nu + nd).mean(axis=(1, 2)), docc=(nu * nd).mean(axis=(1, 2)))

    def drift_magnitudes(self, G=None):
        """|K| per component -- the input to the Aarts correctness criterion, whose statement is
        that the distribution of the drift magnitude must fall off FASTER than any power."""
        return np.abs(self.drift(G)).ravel()


def run_cl(m, t_therm=8.0, t_meas=40.0, n_meas=200, eps=1e-3, seed=0, adaptive=True,
           collect_drift=False, stable_every=5, channel="charge", chains=32, ref_scale=1.0):
    """Run to a fixed Langevin TIME, not a fixed step count.

    This is easy to get backwards.  With the step count held fixed, halving eps halves the
    Langevin time covered, so a run still approaching equilibrium looks like a step-size bias
    that GROWS as the step shrinks -- measured as a double-occupancy error of -0.004, -0.0005,
    +0.014, +0.035 over eps = 2e-3 down to 2.5e-4, which reads as a diverging discretisation
    error and is nothing of the kind.  The field starts at zero, which is the U = 0 state, and
    what was being seen is the approach to equilibrium.  Holding t = eps * n_steps fixed
    separates them: what remains IS the step-size bias, and it is linear in eps.
    """
    ch = CLangevin(m, eps=eps, seed=seed, adaptive=adaptive, stable_every=stable_every,
                   channel=channel, chains=chains, ref_scale=ref_scale)
    n_therm = max(1, int(round(t_therm / eps)))
    thin = max(1, int(round(t_meas / eps / n_meas)))
    for _ in range(n_therm):
        ch.step()
    obs, drifts = [], []
    for _ in range(n_meas):
        for _ in range(thin):
            ch.step()
        G = ch.greens()
        obs.append(ch.observe(G))
        if collect_drift:
            drifts.append(ch.drift_magnitudes(G))
    # (n_meas, C): average over TIME within a chain, then take the error from the spread ACROSS
    # chains.  Chains are independent; successive measurements within one are not -- the
    # integrated autocorrelation time here is ~565 steps.
    n = np.array([o["n"] for o in obs]).mean(axis=0)
    d = np.array([o["docc"] for o in obs]).mean(axis=0)
    C = len(d)
    # The PER-CHAIN values are returned, not just their mean.  A quoted standard error is only
    # meaningful if it shrinks like 1/sqrt(C), and on the charge channel it does not -- measured,
    # it ROSE from +-0.035 at 128 chains to +-0.096 at 512.  That is a heavy-tailed chain
    # distribution, and the only way to see it is to keep the chains.
    return dict(n=complex(n.mean()), n_se=complex(n.std(ddof=1) / np.sqrt(C)),
                docc=complex(d.mean()), docc_se=complex(d.std(ddof=1) / np.sqrt(C)),
                per_chain_n=n, per_chain_docc=d,
                max_drift=ch.max_drift, n_therm=n_therm, thin=thin, n_steps=ch.n_steps,
                chains=C, t_actual=ch.t_elapsed, t_nominal=eps * ch.n_steps,
                drifts=np.concatenate(drifts) if collect_drift else None,
                im_frac=float(np.abs(ch.X.imag).mean()))
