# The Fermion Sign Problem, Read Before the Censoring

### The average sign is a censored measurement, and the quantity it censors is still readable

**Ikailo John Sessford**, Ikailo Inc., `john@ikailo.com`

*Draft. September 2026.*

> **A note on measurement.** Every empirical quantity in this paper is a deterministic read of a
> configuration produced by the determinantal quantum Monte Carlo sampler in `research/code/`,
> through the open-source *Entroptics* instrument,
> [github.com/Agience/entroptics](https://github.com/Agience/entroptics). The sampler is gated
> against a brute-force enumeration of every auxiliary field at 5e-15. Each read fixes its own
> resolution from the data's own entropy and standardises against its own exact null; no constant
> is supplied to any of them.
>
> **A note on the negatives.** Most of this paper is routes that closed. Each closed with a stated
> mechanism rather than an abandonment, and the surviving claim is narrow because the closures are
> what narrow it. A reader who wants only the positive result can read §2 and §7.

---

## Abstract

*(to write)* The fermion sign problem is normally quantified by the average sign
`<sgn> = Z / Z_||`, which decays exponentially in `beta N`. That estimator applies `sign()` to a
determinantal weight whose logarithm spans 8 to 16 orders of magnitude, and `sign()` **censors**:
it discards the magnitude and returns a binary. Where the weight's two factors are locked in sign
the estimator returns `1.00000 +- 0.00000` for every configuration and has no derivative at all,
because `sign()` has none. We show that the continuous quantity whose breakdown *causes* the sign
to flip — the particle-hole relation between the two spin channels — is measurable there, exactly
at the locked point and monotonically as it breaks, at 135 sigma against an exact re-pairing null.

---

## 1. The problem, and what is normally measured

*(to write — the standard formulation, `<sgn> = exp(-beta N df)`, Troyer-Wiese, and the cost
`O(1/<sgn>^2)`)*

## 2. The censoring argument

*(to write — the core of the paper)*

The weight of a configuration is

    w(x) = det(I + B_up(x)) det(I + B_dn(x))

a continuous quantity. The standard severity measure is `<sign(w)>`. The chain that produces a
sign problem runs

    particle-hole symmetry  =>  G_up[i,i](x) + G_dn[i,i](x) = 1   configuration by configuration
                            =>  the two determinants are locked
                            =>  sign(w) = +1 for every x

Break the first relation and the sign *may* flip; it need not yet. The first relation is
continuous and the last is discrete, so there is a regime in which the first has moved and the
last has not. **In that regime `<sgn>` is identically 1 with zero variance and no gradient, and
the relation that will eventually break it is already measurable.**

## 3. Reading the lock

*(to write — the coupling, its exact null, the calibration)*

Measured on the two channels' Green's-function diagonals across configurations, `reads.coupling`
returns **-1.0000 at every beta** on a bipartite lattice at half filling, recovering the identity
above to the last digit, with the permuted control at `|z| <= 1.7`.

| beta | mu | neg fraction | coupling strength | z | control z |
|---|---|---|---|---|---|
| 2 | 0.0 | 0.0000 | **-1.0000** | -49.7 | -0.31 |
| 4 | 0.0 | 0.0000 | **-1.0000** | -34.8 | -0.30 |
| 6 | 0.0 | 0.0000 | **-1.0000** | -48.6 | +1.31 |
| 8 | 0.0 | 0.0000 | **-1.0000** | -26.7 | -0.19 |

## 4. Reading it where the sign is blind

*(to write)*

On one importance-sampled chain per row, both columns from the same samples:

| beta | `<sgn>` | coupling deficit | z |
|---|---|---|---|
| 1.0 | **1.00000 +- 0.00000** | 0.03518 | -137.1 |
| 1.5 | **1.00000 +- 0.00000** | 0.05047 | -133.7 |
| 2.0 | 0.99922 +- 0.00078 | 0.06098 | -131.5 |
| 3.0 | 0.96172 +- 0.00542 | 0.09175 | -125.5 |
| 4.0 | 0.84688 +- 0.01051 | 0.16228 | -115.3 |
| 6.0 | 0.46094 +- 0.01754 | 0.25639 | -101.5 |

## 5. What the read predicts, and what it cannot

*(to write)*

Fitted on the sign-free rows alone (`beta <= 2`, `<sgn> = 1` exactly), the deficit's growth
predicts its own value to within **19%** out to `beta = 4`, where the sign has fallen to 0.847 —
a forecast made entirely from data in which no sign problem exists.

**It cannot give the severity.** `strength` lies in `[-1, 1]`, so the deficit lies in `[0, 2]`,
while `-ln<sgn> = beta N df` is unbounded. No monotone map connects a bounded quantity to an
unbounded one asymptotically, and the extrapolation duly overshoots by 2.2x by `beta = 6`. The
read is an onset detector, not a severity meter, and the boundary is provable rather than
empirical.

## 6. The routes that closed

*(to write — one subsection each, with the mechanism)*

6.1 Reads that score severity — each equals the average sign under a cheaper measure
6.2 Phase blindness — `|diag(sigma) A| = |A|` cell for cell, so no exponent helps
6.3 The scalar decoupling family — provably complete
6.4 Contour deformation — measured, including the falsification test it survived
6.5 Complex Langevin — wrong by 2% at `z ~ 28` against an exact control
6.6 The constrained path and its trial wavefunction — the variational criterion is
    anti-correlated with the bias; no answer-free construction beat the default
6.7 `extract()` on a mean — a category error, not a limitation of the read

## 7. What the instrument did that the estimator cannot

*(to write — the summary of §2 with the numbers of §4)*

## 8. Method

*(to write — the sampler, the stabilisation, the gates and their negative controls)*

## 9. Outlook

*(to write — the algebraic axis: what a positivity-restoring representation must reproduce, from
the measured fact that the lock constrains only the SIGN while the two magnitudes are independent
over 8 to 16 orders)*

## References

*(to write)*
