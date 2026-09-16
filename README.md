# Reading Positivity from the Weights Alone

[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE.md)
[![PyPI](https://img.shields.io/pypi/v/entroptics?logo=pypi&logoColor=white&label=entroptics)](https://pypi.org/project/entroptics/)
[![Gates](https://img.shields.io/badge/gates-16%20with%20negative%20controls-0F9D58)](#the-gates)
[![Sponsor](https://img.shields.io/badge/Sponsor-Agience-EA4AAA?logo=githubsponsors&logoColor=white)](https://github.com/sponsors/Agience)

**Given only the weights a running simulation already holds — no Hamiltonian, no knowledge of the
decoupling, and no constant chosen by the caller — what can be decided about its positivity?**

That question is answered here with an instrument built for a different subject.
[Entroptics](https://github.com/Agience/entroptics) reads a 2-D ordered-by-feature field as a
finite optical aperture; a determinantal weight cloud is such a field, and the reads transfer to it
without modification. `concentration` on the cloud's `(Re, Im)` frame returns a *directional*
statistic and an *axial* one, and the pair sorts a run into three regimes — sign-free, a real sign
problem, or a phase problem no rotation reaches. **The sign problem is the rank-one case of a phase
problem**: rank one in the plane means the phase takes two values `pi` apart, and a phase confined
to `Z2` is what a sign *is*.

The reading is scored against an independent oracle computed from the Hamiltonian, and the two
sides share no information — the oracle sees `K` and never a configuration, the read sees two
logged columns and never `K`. They agree on **15 of 15 lattices**. The physics underneath is the
lockstep: a configuration's weight is a product of two determinants, one per spin channel, and at
half filling they cross the sign boundary together — on around 11% of configurations at
`beta = 12`, on exactly the same ones — so the product's sign never changes. Doping desynchronises
them, and the fraction on which they disagree *is* the sign problem, exactly.

Throughout, the object read is the standard **spin decoupling** of the Hubbard interaction. That
qualifier is measured rather than conventional, and where it binds is stated below: the sign
problem is a property of a Hamiltonian *and* a decoupling, and so is the reading of it.

This repository holds the paper, the reads, and the determinantal quantum Monte Carlo machinery
they run on.

## The result

**The sign problem is the loss of a synchrony, and the synchrony is measurable.**

A negative weight is a configuration on which the two channels' determinant signs disagree --
arithmetic, immediate from the product. What is not immediate is that the disagreement has
structure. Measured on 600
configurations per row, both channels read separately in the numerically conditioned frame:

| beta | mu | up flips | dn flips | agree | disagree | = neg fraction |
|---|---|---|---|---|---|---|
| 4 | 0.0 | 0.0020 ± 0.0018 | **0.0020** | **1.0000 ± 0.0000** | 0.0000 | exact |
| 8 | 0.0 | 0.0400 ± 0.0097 | **0.0400** | **1.0000 ± 0.0000** | 0.0000 | exact |
| 12 | 0.0 | **0.1120 ± 0.0103** | **0.1120** | **1.0000 ± 0.0000** | 0.0000 | exact |
| 6 | 0.4 | 0.0093 ± 0.0025 | 0.0093 | 0.9847 ± 0.0027 | **0.0153** | exact |
| 8 | 0.4 | 0.0220 ± 0.0040 | 0.0200 | 0.9647 ± 0.0079 | **0.0353** | exact |
| 12 | 0.4 | 0.0413 ± 0.0077 | 0.0417 | 0.9283 ± 0.0049 | **0.0717** | exact |

The half-filled rows establish two things together. The individual channels **do** change sign —
on around 11% of configurations — so the lockstep is not vacuous. And they change sign on
**exactly the same configurations**: that column is `1.0000` with zero spread across seeds, not a
number that averages to it. The doped rows add the third: the disagreement fraction equals the
negative-weight fraction exactly, everywhere.

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
  `G_up[i,i](x) + G_dn[i,i](x) = 1`, which holds configuration by configuration at a maximum
  residual of **4.9e-15** at `beta = 2`, rising to `1.2e-12` at `beta = 8`;
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

Sixty-nine one-body matrices, agreeing with the measured identity on every one, including cases
chosen so the criterion has to predict rather than describe: bond-disordered lattices with every
bond drawn at random and no symmetry designed in (residual `5e-14` to `1e-12`), a **star graph** — a tree with
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
ladder takes the identity from `1.2e+01` to `4.3e-14` and the sign deficit from `0.19000` to
**`0.50071`**. So the criterion decides the **identity**, and the identity delivers positivity
only where the weights are real. An odd cycle has no flux-free route, so it can be made to satisfy
the identity and cannot be made positive this way.

The criterion decides the **identity**, and nothing further on its own: it carries positivity only
along route A, where the weights stay real, and it does not predict the negative *fraction* on
either route — a broken identity permits the two channels to disagree without saying how often they
will.

## Running the criterion on output instead of on the Hamiltonian

Everything above decides the identity from `K` — build the support graph, two-colour it, check the
flux on every odd cycle. A running simulation does not have a clean `K` to hand; it has
configurations and determinants. Read as a statement about **data**, §4's identity says that across
configurations

```
ln|det_up|(x) − ln|det_dn|(x)     is an exact affine function of     Σx
```

which is one channel against another on a shared index — what `coupling` reads. The read is
invariant to the offset and the scale, so neither `−dτ·L·tr(K)` nor `λ` has to be known. An exact
affine relation saturates a normalised alignment at 1; a broken identity cannot.

| | `1 − \|strength\|` |
|---|---|
| identity holds (8 lattices) | `0` to `2.2e-16` — machine epsilon |
| identity fails (7 lattices) | `1.0e-3` to `2.9e-2` |

**It agrees with the measured identity on 15 of 15 lattices, and with the algebraic criterion on 15
of 15, using no Hamiltonian.** The two sides share no information: `coupling` sees two columns and
never sees `K`; the criterion sees `K` and never sees a configuration. The departure is systematic
rather than sampling — across a factor of four in sample size and three seeds, the saturated
lattices stay at machine zero and the broken ones stay at `9.8e-4` to `1.4e-2` without shrinking.

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
precision — while the weight carries a phase. Four systems with an exact relation, an identical
saturated read, and sign deficits of **0.00000, 0.09241, 0.15581 and 0.46110**. The relation holds whether
the weight is positive or spread across the circle, so the comparison is well posed and blind.

**Outside real determinantal weights the correspondence between saturation and sign-freedom fails
in both directions, and the reason is given.** Put to a third positivity mechanism — conjugate
pairing under time reversal, where the weight is `|det|^2` — the read does *not* saturate although
the model is sign-free; and on the matched control, which carries up to **37.5%** negative weight,
it reads exactly `1.0000`. The second half is an impossibility: there the two channels are
bit-identical,
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

Three families were put to the question. The first two are proofs.

| family | status |
|---|---|
| reweight by the sign | **capped** at `n·<sgn>²` |
| read the relation between the two channels | **blind**, two ways |
| read one channel alone | **measured: no ordering across lattices** |
| identify an operator from a correlation sequence | not excluded by any of the above |

**The ceiling** is the sign problem restated. Correct importance sampling draws at `|w|`, so a
configuration enters any weighted average carrying only its sign, and Kish's effective sample size
of those weights is exactly `n·<sgn>²` — a property of the *weights*, so it bounds every
estimator that reweights by them. The order parameter above does not reweight — it is an
unweighted statistic of the configurations the `|w|` chain visits — which is why it is readable
where `<sgn>` is not. What stops it reporting severity is its own bound: the deficit lies in
`[0, 2]` and `-ln<sgn>` does not.

**The blindness** is two separate statements. Where the two channels coincide as data, nothing
remains to compare. Where they are distinct and *exactly related* — at `5e-15` — the comparison is
well posed and still carries nothing, because the relation holds whether the weight is positive or
spread across the circle.

The last row survives for a structural reason: identifying an operator from a correlation sequence
aggregates no signed configuration and compares no channels. It is a place to look rather than a
result — what it costs to identify such an operator from data a sign problem permits is not settled
here.

## Layout

| path | what |
|---|---|
| [`research/PAPER.md`](research/PAPER.md) | the paper |
| [`research/code/`](research/code) | the determinantal QMC machinery the reads run on |
| [`research/code/reads/`](research/code/reads) | the reads — the result this repository is named for |
| [`research/code/closed/`](research/code/closed) | the structural constraints of §9, each measured with its mechanism |
| [`research/code/tests/`](research/code/tests) | the gates |
| [`research/code/entroptics_adapter.py`](research/code/entroptics_adapter.py) | the one way in to the instrument — each read under the paper's name for it, and the version pin |
| [`research/lean/`](research/lean) | the machine-checked algebraic core of §11 |

Of the 47 experiments the paper cites, 26 read through Entroptics and 21 do not, and all three
gates cited in the text are instrument-free. §§3–5 derive and verify the identity and its
criterion from the one-body matrix alone, §7 reads the same systems from their output, and the two
share no code. They agree on 15 of 15 lattices.

Every Entroptics read goes through the adapter, and a gate refuses any file that imports the
library directly. The adapter adds no arithmetic: `test_entroptics_adapter.py` asserts each read *is*
the library call it names, value for value, with a negative control that makes the comparison able
to fail. What it does add is the naming — `channel_alignment` rather than `reads.coupling` — the
version pin read from `requirements.txt`, and the two calls that look right and are not.

## What is proved rather than measured

The results here are measurements, and a measurement is not a theorem. But a few of the paper's
statements are algebra, and a few of *those* are universally quantified — "at any size", "on any
graph", "for every exponent" — where the text supports them with a sweep. A sweep says no case it
tried was a counterexample. It does not say none exists.

Those are machine-checked in Lean 4 / Mathlib under [`research/lean/`](research/lean): 80 theorems
across twelve modules, `sorry`-free, declaring no axiom of their own, every one elaborating against
the three foundational axioms. They cover the model's algebra and, in `Alignment.lean`, what the
read itself guarantees — that it is invariant to the offset and scale §4's identity carries, so the
criterion can run on output with no Hamiltonian; that saturation is *equivalent* to an exact affine
relation; and that the deficit cannot leave `[0, 2]`, which is why §7 detects onset and not severity.
The two that move from a sweep to a proof are

* **the closed door on doping** — a diagonal conjugation leaves `K_ii` where it was, so both routes
  demand `K_ii = -K_ii`; with `K` Hermitian that forces zero. The paper measures 84 flux values
  across four graphs; `doping_closes_both_routes` quantifies over every lattice, every `K`, every
  flux and every size.
* **magnitude blindness at every exponent** — measured at `q = 0.5, 1, 2, 3`;
  `sum_abs_pow_blind` covers every real `q`, so the cube is provably as blind as the square. Its
  companion `coupling_is_not_blind` carries the scope: the two-sided read is *not* blind, which is
  what makes this a statement about magnitude reads and not about the instrument.

§11 of the paper has the full correspondence table and says what the Lean development deliberately
does not cover — which is everything read from a running sampler.

## Running it

Everything runs from `research/code`, which has to be on the import path: a bare
`python reads/<file>.py` puts `reads/` there instead and fails on `import dqmc`. Nothing else is
needed -- the files that reach into `tests/` or `closed/` bootstrap those themselves.

```sh
cd research/code
PYTHONPATH=. python tests/gate_fast.py                    # a gate, ~40 s
PYTHONPATH=. python tests/validate.py                     # the sampler against enumeration, ~2 s
PYTHONPATH=. python reads/expBA_oracle_superset.py        # the criterion against the identity
PYTHONPATH=. python -m pytest tests -q                    # the suite: 495 tests
```

On Windows `cmd` use `set PYTHONPATH=.` on its own line first; in PowerShell, `$env:PYTHONPATH='.'`.

The suite is real computation rather than filesystem checks, so its runtime is the machine's: two
measured runs on a 22-core box took **5m42s and 12m00s**, and a laptop will be slower. Nothing in it
is skipped for want of data — every experiment here generates its own configurations from a seed.

Reproducing the reads needs [entroptics](https://github.com/Agience/entroptics) 0.2.3 or later;
the sampler, the criterion and the gates need only numpy and scipy. Seven experiments run past ten
minutes and are meant to be run when their section is being checked, not routinely.

Two wrappers hold the question of which machine runs the expensive work, so it is answered once
rather than in every caller. Both default to this one, so a fresh clone reproduces everything with
nothing configured:

```sh
python research/code/remote_run.py --describe                       # where would a run go?
python research/code/remote_run.py --module pytest tests -q         # the suite, wherever that is
python research/code/lean_build.py                                  # the proofs, wherever that is
```

Point them somewhere with a compute host by copying `research.local.env.example` to
`research.local.env` (git-ignored) and filling in `COMPUTE_*` and `LEAN_BUILD_*`. A remote run ships
the sources and runs them there, so what runs is what is in your tree; there is no data store to
move, because every experiment here generates its own configurations from a seed.

Checking the proofs needs [Lean 4 and `elan`](https://leanprover-community.github.io/get_started.html).
`lake exe cache get` in `research/lean` fetches the prebuilt Mathlib, which takes about a minute;
building from source instead takes hours.

## The gates

Nothing here is quoted that is not gated, because in this rig a defect does not look like a crash,
it looks like a result. Every gate carries a negative control that makes it able to fail.

| gate | holds | negative control |
|---|---|---|
| `validate.py` | the sampler's weight and propagator against brute-force enumeration of every auxiliary field — 5e-15 | fails at 3e-1 |
| `test_lockstep.py` | at half filling the channels agree on every configuration while individually flipping | doping breaks it, and the break equals the negative fraction |
| `test_conditioned_frame.py` | two independent routes to the sign agree; the answer does not move with the stabilisation block | the naive product reports a 50.75% negative fraction at half filling, where positivity is provable |
| `gate_fast.py` | the vectorised walker *is* the reference walker at one walker, bit for bit — 5.7e-14 | the finite-temperature Green's convention gives 3.2e+01 |
| `gate_multi_k2.py` | multi-determinant overlap *and* energy against exact sector arithmetic — 5e-16 | the k = 1 gate alone passes over a two-body error of 6-31% |
| `test_identity.py` | the identity, and that a *neighbouring* interaction's constant must fail it | borrowing a lambda 47% away fails by fourteen orders |
| `test_representation_independence.py` | the identity holds in both field distributions with each one's own constant | each representation's constant fails in the other, four percent away |
| `test_two_mechanisms.py` | the two sign-free mechanisms read `-1` and `+1`, asserted together | doping must desaturate the repulsive read and leave the attractive one |
| `test_boundary.py` | saturation is neither necessary nor sufficient outside real weights | an inert twist, and a 2-wide lattice, must be refused |
| `test_representation.py` | the read spans its full range where the physics is fixed | the effective sample size must stay too small to permit a reweighted quantity |
| `test_capability_across_representations.py` | the deficit is monotone, resolved and reproducible in both fields | it must stop being reproducible at the onset of the sign problem |
| `gate_main.py` (gate 1) | the complex-Langevin analytic drift against a central finite difference of the complexified action — 7.4e-10 spin, 1.4e-09 charge | the wrong convention (`G_ii` for `1 - G_ii`) fails at 0.40 and 0.36 |
| `test_nstab_is_safe.py` | the hand-picked stabilisation block against `n_stab = 1`, on the log weight as well as the sign | removing the stabilisation entirely must make the discrepancy appear |
| `test_entroptics_adapter.py` | every adapter read *is* the library call it names, value for value | comparing against a different argument must fail, or the equality proves nothing |
| `test_the_instrument_is_reached_through_the_adapter.py` | no file imports the instrument directly, over the AST rather than a regex | all five import spellings must be detected, including the dynamic one |
| `test_the_read_has_the_proved_properties.py` | the read obeys the three properties `Alignment.lean` proves of it — offset and scale invariance, saturation exactly on affine data, the `[0, 2]` bound — each to 1e-12 | an uncentred cosine must fail the invariance, and unrelated columns must not saturate |

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
