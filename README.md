# The Fermion Sign Problem is a Desynchronisation

[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE.md)
[![PyPI](https://img.shields.io/pypi/v/entroptics?logo=pypi&logoColor=white&label=entroptics)](https://pypi.org/project/entroptics/)
[![Gates](https://img.shields.io/badge/gates-11%20with%20negative%20controls-0F9D58)](#the-gates)
[![Sponsor](https://img.shields.io/badge/Sponsor-Agience-EA4AAA?logo=githubsponsors&logoColor=white)](https://github.com/sponsors/Agience)

**A configuration's weight is a product of two determinants, one per spin channel. Each changes
sign as its spectrum crosses a boundary. At half filling the two channels cross that boundary
in lockstep — on around 12% of configurations at `beta = 12`, on exactly the same
configurations — so the product's sign never changes. Doping desynchronises
them, and the fraction on which they disagree IS the sign problem, exactly.**

Throughout, the object read is the standard **spin decoupling** of the Hubbard interaction. That
qualifier is measured rather than conventional, and where it binds is stated below: the sign
problem is a property of a Hamiltonian *and* a decoupling, and so is the reading of it.

This repository holds the paper, the reads, and the determinantal quantum Monte Carlo machinery
behind the [Entroptics](https://github.com/Agience/entroptics) reading of the fermion sign problem.

## The result

**The sign problem is the loss of a synchrony, and the synchrony is measurable.**

A negative weight is, by definition, a configuration on which the two channels' determinant signs
disagree. What is not a definition is that the disagreement has structure. Measured on 600
configurations per row, both channels read separately in the numerically conditioned frame:

| beta | mu | flip rate per channel | agree | disagree | = neg fraction |
|---|---|---|---|---|---|
| 4 | 0.0 | 0.003 ± 0.001 | **1.0000 ± 0.0000** | 0.0000 | exact |
| 8 | 0.0 | 0.040 ± 0.006 | **1.0000 ± 0.0000** | 0.0000 | exact |
| 12 | 0.0 | **0.116 ± 0.011** | **1.0000 ± 0.0000** | 0.0000 | exact |
| 6 | 0.4 | 0.010 ± 0.004 | 0.9837 ± 0.0030 | **0.0163** | exact |
| 8 | 0.4 | 0.020 ± 0.002 | 0.9620 ± 0.0068 | **0.0380** | exact |
| 12 | 0.4 | 0.048 ± 0.010 | 0.9183 ± 0.0129 | **0.0817** | exact |

The half-filled rows establish two things together. The individual channels **do** change sign —
on around 12% of configurations — so the lockstep is not vacuous. And they change sign on
**exactly the same configurations**: that column is `1.0000` with zero spread across seeds, not a
number that averages to it. The doped rows establish
the third: the disagreement fraction equals the negative-weight fraction exactly, everywhere.

**The lock is lockstep, not avoidance.** A symmetry could protect positivity by keeping
configurations away from the boundary where a determinant vanishes, or by making both channels
cross it together. At `beta = 10` the distance distribution has a bulk that does not move — median
**0.50 to 0.55** across every sample size and seed — and a tail that reaches the boundary, with the
1st percentile **25 to 77 times below** the median. The smallest distance observed has no stable
value, ranging over `0.0002` to `0.012` across seeds, which is itself the statement that no distance
is respected. The negative fraction stays exactly **0.0000** throughout. The symmetry synchronises
the crossing rather than preventing it.

## The order parameter

The channels' alignment is directly measurable. The Entroptics coupling read, standardised against
its own exact re-pairing null with no constant supplied, returns

* **exactly `-1.0000`** where the lockstep is perfect — recovering the particle-hole identity
  `G_up[i,i](x) + G_dn[i,i](x) = 1`, which holds configuration by configuration to a maximum
  residual of **6.7e-15**;
* a continuous departure from `-1` as the lockstep breaks, at **|z| = 137** against a permuted
  control at `|z| <= 1.7`;
* and it does so **while the average sign is still identically 1.00000 +- 0.00000**.

**What the deficit measures, exactly.** The read is a normalised alignment, so its deficit is an
angle: `1 - cos∠` between the two centred channel frames. Rescaling one channel — `B = 1 - 2A`
instead of `1 - A` — leaves the deficit at **exactly zero** while the raw violation grows to the
size of the channel itself. Only the component of the particle-hole violation *perpendicular* to
the channel registers, and to leading order the deficit is `|E_perp|² / (2|A||B|)`. So the order
parameter tracks an angle, which is bounded — and `-ln<sgn>` is not. That is why it detects onset
and not severity.

That last point is the capability. The average sign is an average of a binary: where the channels
are locked it is a constant, and a constant has no derivative. The coupling reads the alignment
that produces the lock, so it moves first.

Its scope is bounded and the boundary is provable: `strength` lies in `[-1, 1]`, so the departure
lies in `[0, 2]`, while `-ln<sgn>` is unbounded. **The read is an onset detector, not a severity
meter** — and no fixed map between them can exist, nor be calibrated, because on the rows where the
departure is useful `-ln<sgn>` is identically zero and offers nothing to calibrate against.

## The criterion

For a **real** one-body matrix `K`, the identity holds exactly when it admits a **signed-diagonal
conjugation**
`S K S = -K` with `S = diag(±1)`. Elementwise that is `s_i s_j K_ij = -K_ij`, so it is a
**2-colouring of `K`'s support graph together with a zero diagonal** — `O(N^2)`, no eigenvalues,
no determinants, no sampling.

It unifies the conditions that look separate. Next-nearest hopping puts an odd cycle in the graph;
a chemical potential puts a term on the diagonal; and so does a **staggered potential**, which
leaves the lattice bipartite and the filling at exactly one per site — satisfying both of the
conditions usually stated — and breaks the identity anyway, because a diagonal term cannot be
negated by signs.

Twenty-four one-body matrices, agreeing with the measured identity on every one, including cases
chosen so the criterion has to predict rather than describe: bond-disordered lattices with every
bond drawn at random and no symmetry designed in (residual `1e-13`), a **star graph** — a tree with
no cycles and nothing lattice-like about it — and frustrated topologies where it must fail.

**The two routes are not interchangeable, and only one supplies the calibration.** Both give the
identity and both give positivity. Only route B — the anti-similarity — forces `G_dn = 1 - G_up`
configuration by configuration, and with it the exactly calibrated `-1`. Where only route A is open
the lattice is still sign-free and the identity still exact, but the channels are no longer exact
negatives about `1/2` and the read comes back at **-0.945 to -0.993**: resolved, close, and not
calibrated. Each route supplies its own exact relation — route B gives `G_dn = 1 - G_up`, route A
the same composed with conjugation, `G_dn = 1 - conj(G_up)` — so under route A the real parts read
`-1` and the imaginary parts `+1`, and the single signed alignment averages them. That average is
exact: writing `r` for the imaginary share of the centred variance, `strength = -1 + 2r`, verified
to **1e-16**. The departure from `-1` is the frame's imaginary weight and carries no information
about the sign problem, which is absent throughout. So `-1.0000` is a consequence of the particle-hole structure route B supplies, not of
positivity and not of the identity — three things that coincide in every unfluxed measurement and
come apart under flux.

**With complex hoppings there are two routes.** The conjugation may be a diagonal *unitary* rather
than a diagonal sign, and the derivation accepts either a similarity or an anti-similarity:
`S K S⁻¹ = -K` needs every cycle even at any flux; `S K S⁻¹ = -conj(K)` needs `(-1)^ℓ e^{-2iΦ} = 1`,
so odd cycles need flux `π/2 mod π`. An odd cycle carrying half-odd-integer flux restores what the
odd cycle destroys — a 5-site ring at flux `π/2` satisfies the identity at `1.4e-14`. The identity
holds when either route is open, verified on **40 one-body matrices with no mismatch**, and for
real `K` the two conditions coincide, which is why the colouring is the whole story there.

**The obvious criterion is the spectral one, and it is wrong.** For Hermitian `K`, a spectrum
symmetric about zero *is* similarity to `-K`. The staggered cases have spectra symmetric to `1e-15`
and break the identity by up to 21. The similarity exists; it is not a signed diagonal, so it does
not commute with the interaction's diagonal factor and the derivation cannot use it.

**The criterion is sharp, and positivity is fragile in a way the sign problem is not.** A staggered
term of *any* amplitude makes `S K S = -K` unsatisfiable, and the identity follows exactly: its
residual is **linear in the perturbation across four decades**, from `1e-6` to `1e-3`, with no
tolerance and no onset. The negative fraction does not follow suit — it stays at exactly `0.0000`
until the perturbation reaches `~0.1`. A model can violate the criterion, carry a measurably broken
identity, and produce no negative weight at all.

**The two routes are not equivalent, and only one carries positivity.** Route A leaves the weights
real — at flux `π/4` on an even ring the identity holds to `2e-14` and the mean phase is exactly 1.
Route B restores the identity on an **odd** cycle and leaves a phase behind: threading a frustrated
ladder takes the identity from broken to `8e-14` and the sign quality from `0.060` to **`0.522`**,
nine times worse. So the criterion decides the **identity**, and the identity delivers positivity
only where the weights are real. An odd cycle has no flux-free route, so it can be made to satisfy
the identity and cannot be made positive this way.

The criterion decides the **identity**, and nothing further on its own: it carries positivity only
along route A, where the weights stay real, and it does not predict the negative *fraction* on
either route — a broken identity permits the two channels to disagree without saying how often they
will.

## Running the criterion on output instead of on the Hamiltonian

Everything above decides the identity from `K` — build the support graph, two-colour it, check the
flux on every odd cycle. A running simulation does not have a clean `K` to hand; it has
configurations and determinants. Read as a statement about **data**, §3's identity says that across
configurations

```
ln|det_up|(x) − ln|det_dn|(x)     is an exact affine function of     Σx
```

which is one channel against another on a shared index — what `coupling` reads. The read is
invariant to the offset and the scale, so neither `−dτ·L·tr(K)` nor `λ` has to be known. An exact
affine relation saturates a normalised alignment at 1; a broken identity cannot.

| | `1 − \|strength\|` |
|---|---|
| identity holds (7 lattices) | `0` to `2.2e-16` — machine epsilon |
| identity fails (8 lattices) | `1.0e-3` to `2.9e-2` |

**It agrees with the measured identity on 15 of 15 lattices, and with the algebraic criterion on 15
of 15, using no Hamiltonian.** The two sides share no information: `coupling` sees two columns and
never sees `K`; the criterion sees `K` and never sees a configuration. The departure is systematic
rather than sampling — across a factor of four in sample size and three seeds, the saturated
lattices stay at machine zero and the broken ones stay at `9.2e-4` to `8.5e-3` without shrinking.

**To use it**, log two scalars per configuration — both already computed by any DQMC code:
`ln|det_up| − ln|det_dn|` from the `slogdet` formed every sweep, and `Σx` over the auxiliary field.
Then call `coupling(A, B)` and read whether it saturates. It costs `O(n)` over two vectors.

Three uses: a **pilot** before committing cluster time; a **regression check** when a term is added
(`t'`, a staggered field, a twist, a chemical potential) and nobody re-derives the colouring; and a
**build check**, because the read follows what the code did rather than what the model was meant
to be.

**It fires before `<sgn>` does**, which is the point. On `2x4` with `t' = 0.3` at `β = 6`, 400
configurations: the negative-weight fraction is **0.0000** — not one negative weight — while the
identity is broken by **10.53** and the read flags it at **2.3e-3**. The average sign reports a
clean run on a model that has already left the sign-free class. `<sgn>` is a sampled quantity that
sits at `1.0000` until the problem is large enough to show in a finite sample, and the cost of it
showing is `O(1/<sgn>²)` with `<sgn> = exp(−βNΔf)` — exponential in `β` and in size. This read's
cost is flat in both.

## The scope, measured

The correspondence above is exact everywhere it was measured, and the measurements that establish
where it ends are results rather than caveats. Three of them:

**The reading is a synchronisation, and its sign is set by the mechanism.** This model is
sign-free in two structurally opposite places. Repulsive `U` in the spin channel at half filling
has `G_dn = 1 - G_up` — anti-synchronised, and the read gives `-1`. Attractive `U` in the charge
channel is sign-free at **any** filling because both spins see the identical field — synchronised,
and the read gives `+1`. Doping separates them: it takes the repulsive read from `-1.0000` to
unresolved and leaves the attractive one at `+1.0000`. So the read tracks whether the channels
move together, not the filling and not distance from a symmetric point.

**One formula covers all three mechanisms.** Each supplies an exact relation between the two
channels, and the read follows from it with no freedom left: particle-hole (`G_dn = 1 - G_up`)
gives `-1`; the same composed with conjugation gives `-1 + 2r`; the Kramers relation
(`G_dn = conj(G_up)`) gives `+1 - 2r` — where `r` is the imaginary share of the centred variance.
Verified to `1e-16`. A frame carrying imaginary weight cannot read `±1` on a single signed
alignment, and how far it falls short is that weight and nothing else.

**No read of the channel relation can see a phase problem.** Route B restores the identity on an
odd cycle by threading it with flux, and the two channels then satisfy `G_dn = 1 - G_up` to machine
precision — while the weight carries a phase. Three systems with an exact relation, an identical
saturated read, and sign deficits of **0.000, 0.156 and 0.478**. The relation holds whether the
weight is positive or spread across the circle, so the comparison is well posed and blind. The
screen's `match` behaves better than the coupling on other cases and is blind here in the same way.

**Outside real determinantal weights the correspondence between saturation and sign-freedom fails
in both directions, and we give the reason.** Put to a third positivity mechanism — conjugate pairing under time reversal, where the
weight is `|det|^2` — the read does *not* saturate although the model is sign-free; and on the
matched control, which carries up to **37.5%** negative weight, it reads exactly `1.0000`. The
second half is an impossibility rather than a limitation: there the two channels are bit-identical,
so every statistic of the pair is a statistic of one channel, and no comparison *between* channels
can reach a sign problem that lives in a phase common to both.

**The read is a property of a (Hamiltonian, decoupling) pair.** Along the decoupling family in
which every parameter value is the same physics and no representation has a sign problem, the read
still traverses its whole range from `-1.0000` to `+1.0000`. The lockstep is what carries
positivity at one end of that family and not in its interior, where the two determinants' signs
agree at chance while the model stays exactly sign-free.

**What does not depend on the representation:** the identity, the lockstep, and the capability all
reproduce in the *continuous Gaussian* spin decoupling, with its own closed-form constant
`sqrt(dtau U) = 0.70711` in place of the discrete `arccosh(exp(dtau U / 2)) = 0.73690`. Each
constant is exact to `1e-14` in its own representation and fails by fourteen orders in the other —
four percent apart, neither fitted, both fixed before any determinant is formed.

## What a read of this problem can carry

Three families, and the classification is closed rather than a list of things not yet tried. The
first two are proofs.

| family | status |
|---|---|
| aggregate signed configurations | **capped** at `n·<sgn>²` |
| read the relation between the two channels | **blind**, two ways |
| read one channel alone | **no indicator**, measured |
| identify an operator from a correlation sequence | not excluded by any of the above |

**The ceiling** is the sign problem restated. Correct importance sampling draws at `|w|`, so a
configuration enters any weighted average carrying only its sign, and Kish's effective sample size
of those weights is exactly `n·<sgn>²` — a property of the *weights*, so it bounds every
aggregation of them. That is why the order parameter above detects onset and cannot report
severity: not a limitation of the read, a consequence of its kind.

**The blindness** is two separate statements. Where the two channels coincide as data, nothing
remains to compare. Where they are distinct and *exactly related* — at `5e-15` — the comparison is
well posed and still carries nothing, because the relation holds whether the weight is positive or
spread across the circle.

The last row survives for a structural reason: identifying an operator from a correlation sequence
aggregates no signed configuration and compares no channels. That is a statement about where to
look, not a result — what it costs to identify such an operator from data a sign problem permits is
not settled here.

## Layout

| path | what |
|---|---|
| [`research/PAPER.md`](research/PAPER.md) | the paper |
| [`research/code/`](research/code) | the determinantal QMC machinery the reads run on |
| [`research/code/reads/`](research/code/reads) | the reads — the result this repository is named for |
| [`research/code/closed/`](research/code/closed) | the structural constraints of §8, each measured with its mechanism |
| [`research/code/tests/`](research/code/tests) | the gates |

## The gates

Nothing here is quoted that is not gated, because in this rig a defect does not look like a crash,
it looks like a result. Every gate carries a negative control that makes it able to fail.

| gate | holds | negative control |
|---|---|---|
| `validate.py` | the sampler's weight and propagator against brute-force enumeration of every auxiliary field — 5e-15 | fails at 3e-1 |
| `test_lockstep.py` | at half filling the channels agree on every configuration while individually flipping | doping breaks it, and the break equals the negative fraction |
| `test_conditioned_frame.py` | two independent routes to the sign agree; the answer does not move with the stabilisation block | the naive product reports a 50.75% negative fraction at half filling, where positivity is provable |
| `gate_fast.py` | the vectorised walker IS the reference walker at one walker, bit for bit — 5.9e-14 | the finite-temperature Green's convention gives 3.2e+01 |
| `gate_multi_k2.py` | multi-determinant overlap AND energy against exact sector arithmetic — 5e-16 | the k = 1 gate alone passes over a two-body error of 6-31% |
| `test_identity.py` | the identity, and that a *neighbouring* interaction's constant must fail it | borrowing a lambda 47% away fails by fourteen orders |
| `test_representation_independence.py` | the identity holds in both field distributions with each one's own constant | each representation's constant fails in the other, four percent away |
| `test_two_mechanisms.py` | the two sign-free mechanisms read `-1` and `+1`, asserted together | doping must desaturate the repulsive read and leave the attractive one |
| `test_boundary.py` | saturation is neither necessary nor sufficient outside real weights | an inert twist, and a 2-wide lattice, must be refused |
| `test_representation.py` | the read spans its full range where the physics is fixed | the effective sample size must stay too small to permit a reweighted quantity |
| `test_capability_across_representations.py` | the deficit is monotone, resolved and reproducible in both fields | it must stop being reproducible at the onset of the sign problem |

## No fit, no force, no constant

Every threshold here is derived from the arithmetic rather than chosen, and where a constant was
invented it was swept:

* the sign is a `slogdet`, never a count of eigenvalues below a chosen level — counting would need
  a tolerance to decide which eigenvalues are real, and the parity of real negative eigenvalues
  **is** the determinant's sign, exactly;
* the order parameter is a minimum modulus, which needs no tolerance;
* the numerical rank floor is `eps * k * lambda_max`, the resolution float64 carries on a Gram
  matrix;
* the stabilisation block is not chosen — every row is computed at three of them and the spread
  reported, measured at **0.0000**;
* the coupling's null is the instrument's own exact re-pairing null.

Security issues: email **connect@agience.ai** rather than opening a public issue.

Licensed under Apache-2.0 — see [`LICENSE.md`](LICENSE.md) and [`NOTICE`](NOTICE).

## Declaration of generative AI use

The author used Anthropic's Claude Opus (versions 4.8 and 5) in the preparation of this work. Its
contribution was to write code, and to generate and validate content. The ideas, the construction
and the claims are the author's. No other generative AI tool was used. The author reviewed and
edited all output and takes full responsibility for the content of this publication.
