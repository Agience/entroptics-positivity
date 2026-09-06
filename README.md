# The Fermion Sign Problem, Read Before the Censoring

[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE.md)
[![PyPI](https://img.shields.io/pypi/v/entroptics?logo=pypi&logoColor=white&label=entroptics)](https://pypi.org/project/entroptics/)
[![Sponsor](https://img.shields.io/badge/Sponsor-ikailo-EA4AAA?logo=githubsponsors&logoColor=white)](https://github.com/sponsors/ikailo)

**The average sign is a censored measurement. Where it reads exactly 1 it has no gradient and
carries no information — and the continuous quantity whose breakdown causes the sign to flip is
still there to be read, at 135 sigma.**

This repository holds the paper, the reads, and the analysis behind the
[Entroptics](https://github.com/Agience/entroptics) reading of the fermion sign problem in
determinantal quantum Monte Carlo.

## What it establishes

A determinantal weight is `w(x) = det_up(x) det_dn(x)`, a continuous quantity whose logarithm
spans 8 to 16 orders of magnitude across configurations. Every classical estimator of the sign
problem's severity applies `sign()` to it and averages. That operation **censors**: it discards
the magnitude and returns a binary. In the regime where the weight's two factors are locked in
sign, the estimator returns `1.00000 +- 0.00000` for every configuration and has no derivative,
because `sign()` has none.

The Entroptics coupling read is taken one level up, on the quantity **before** the censoring. On
a bipartite lattice at half filling, particle-hole symmetry gives

    G_up[i,i](x) + G_dn[i,i](x) = 1        configuration by configuration

so the two channels are exact negatives once centred, and their coupling must read `-1`. It reads
**-1.0000 at every beta measured**, with the instrument's own exact re-pairing null placing the
permuted control at `|z| <= 1.7`. That is calibration against a known identity to the last digit.

Doped, the identity breaks configuration-wise and the coupling degrades smoothly — **while the
average sign is still exactly 1**. At `beta = 1` and `beta = 1.5` the sign reads
`1.00000 +- 0.00000` and the coupling reads deficits of `0.0352` and `0.0505` at `|z| ~ 135`,
monotone. The coupling reads O(1) Green's-function diagonals and has no dynamic-range problem;
the sign needs a weight spanning orders.

**Scope, stated tightly.** Onset detection where the classical estimator is structurally blind is
measured and mechanistic. Growth is predictable over a bounded range: an exponential read off the
sign-free rows alone (`beta <= 2`, where `<sgn> = 1` exactly) predicts the coupling deficit to
within **19%** out to `beta = 4`, where the sign has fallen to `0.847`. Severity is **not**
recoverable — the coupling deficit is bounded in `[0, 2]` while `-ln<sgn> = beta N df` is
unbounded, so no monotone map connects them asymptotically, and the prediction overshoots by
2.2x by `beta = 6`.

## What is closed, and why that matters

The reading above is what survived. The repository also carries the routes that were measured and
closed, because each closed with a mechanism rather than an abandonment, and the negative record
is what makes the surviving claim narrow enough to trust:

| route | how it closed |
|---|---|
| reads that score severity from the sampled data | every one equals the average sign under a different, cheaper measure — verified exactly on the anchor |
| `geometry()` and any exponent of it | phase-blind by construction: `\|diag(sigma) A\| = \|A\|` cell for cell, so `\|A\|^q == \|B\|^q` for every `q` |
| the scalar decoupling family | provably complete — two quadratic forms, coefficients pinned |
| the SU(2) vector decoupling, constant contour shifts | measured worse, including a falsification test the explanation could have failed |
| complex Langevin | wrong by 2% at `z ~ 28` against a spin-channel control exact to 1e-4 |
| trial-wavefunction tuning for the constrained path | the variational criterion is anti-correlated with the bias (`rho = -0.80`); no answer-free construction beat the default |
| `extract()` as an estimator of a mean | a category error — the mean lives in `info['centre']` and `clean` carries only deviations |

## Layout

| path | what |
|---|---|
| [`research/PAPER.md`](research/PAPER.md) | the paper |
| [`research/code/`](research/code) | the determinantal QMC machinery the reads run on |
| [`research/code/reads/`](research/code/reads) | the Entroptics reads — the result this repository is named for |
| [`research/code/closed/`](research/code/closed) | the routes measured and closed |
| [`research/code/tests/`](research/code/tests) | the gates |

## The gates

Nothing here is quoted that is not gated, because in this rig a defect does not look like a crash,
it looks like a result. Every gate carries a negative control that makes it able to fail:

| gate | holds | negative control |
|---|---|---|
| `validate.py` | the sampler's weight and propagator against brute-force enumeration of every auxiliary field | fails at 3e-1 |
| `gate_fast.py` | the vectorised walker IS the reference walker at one walker, bit for bit — 5.9e-14 | reinstating the finite-temperature Green's convention gives 3.2e+01 |
| `gate_multi_k2.py` | multi-determinant overlap AND energy against exact sector arithmetic — 5e-16 | the k = 1 gate alone passes over a two-body error of 6-31% |
| `gate_halffilling.py` | at half filling the constraint never fires: killed = 0, `<sgn>` = 1.00000 | under periodic boundaries half filling is an OPEN shell and the same gate reads a 13% error |
| `gate_clangevin.py` | the analytic drift against a finite difference of the complexified action — 8.4e-10 | the wrong convention gives 0.40 |

## Provenance

Every number in the paper is a deterministic read of a configuration produced by the sampler in
`research/code/`, through the open-source Entroptics instrument. The sampler is gated against
brute-force enumeration at 5e-15.
