# The Fermion Sign Problem is a Desynchronisation

### Two spin channels cross the same boundary together, until they do not

**Ikailo John Sessford**, Ikailo Inc., `john@ikailo.com`

*Draft. September 2026.*

> **A note on measurement.** Every empirical quantity in this paper is a deterministic read of a
> configuration produced by the determinantal quantum Monte Carlo sampler in `research/code/`,
> through the open-source *Entroptics* instrument,
> [github.com/Agience/entroptics](https://github.com/Agience/entroptics). The sampler is gated
> against a brute-force enumeration of every auxiliary field at 5e-15. No constant, threshold or
> fitted parameter is supplied to any read; where a coefficient appears it is derived and the
> derivation is checked by requiring the wrong coefficient to fail.

---

## Abstract

The fermion sign problem in determinantal quantum Monte Carlo is normally diagnosed from the
average sign, a quantity that costs `O(1/<sgn>^2)` to resolve and is identically 1 until the
problem is already large. We ask a different question, and answer it with an instrument built for
a different subject: **given only the weights a running simulation already holds -- no Hamiltonian,
no knowledge of the decoupling, and no constant chosen by the caller -- what can be decided about
its positivity?**

The instrument is *Entroptics*, which reads a 2-D ordered-by-feature field as a finite optical
aperture. A determinantal weight cloud is such a field, and the reads transfer without
modification: `concentration` on the cloud's `(Re, Im)` frame returns a *directional* statistic
`resultant` and an *axial* one `focus`, and the pair classifies the run into three regimes.
`focus = 1` with `resultant = 1` is sign-free. `focus = 1` with `resultant < 1` is a real sign
problem: rank one in the plane means the phase takes two values `pi` apart, and a phase confined
to `Z2` is what a sign *is*. `focus < 1` is a genuine phase problem that no rotation reaches.
Measured, `focus` is `1.0000` on every real-weight row and `0.573` to `0.833` on every genuinely
complex one, and where it saturates the de-rotation angle is read off the cloud's own leading
direction -- no angle chosen -- returning a maximum imaginary part of `1e-16` and recovering
`|<sgn>|` exactly.

**That classification is actionable, and getting it wrong is expensive.** A global phase cancels
in `<O> = sum(O w)/sum(w)` and costs nothing, but the estimator carried over from real weights,
`mean(Re w)/mean(|w|)`, reads `cos(theta)` too small -- falling from `0.93500` to `0.25011` across
rotations that leave `|<w>|` unmoved -- and would overstate a cost that goes as `1/<sgn>^2` by
fourteen times.

**The reading is validated against an independent oracle computed from the Hamiltonian, and the
two sides share no information.** A configuration's weight is a product of two determinants, one
per spin channel; on a bipartite lattice at half filling they change sign **in lockstep** -- around
12% of configurations at `beta = 12`, on exactly the same ones -- so the product's sign never
changes. That lockstep follows from an identity between the two determinants, derived and verified
to `1e-14` in two field distributions whose closed-form constants differ by 4% and each fail in the
other by fourteen orders. The identity holds exactly when `K` admits a diagonal-unitary
conjugation, by either of two routes -- `S K S^-1 = -K`, which asks every cycle of `K`'s support
graph to be even, or `S K S^-1 = -conj(K)`, which admits odd cycles carrying half-odd-integer flux.
For real `K` the two coincide and reduce to a 2-colouring with a zero diagonal: `O(N^2)`, no
eigenvalues, no determinants, no field. The criterion agrees with the measured identity on
**64 of 64 one-body matrices with no mismatch** -- chains open and periodic, odd rings, stars,
triangular ladders, bond-disordered lattices, and flux-threaded complex cases -- with the residual
at `1e-14` where it predicts the identity holds and `0.76` to `7.5` where it predicts failure, a
separation of thirteen orders (§4; `reads/expBA_oracle_superset.py`).

The oracle sees `K` and never a configuration; the read sees two logged columns and never `K`. They
agree on **15 of 15 lattices**. Read as a statement about data, the identity says
`ln|det_up| - ln|det_dn|` is an exact affine function of `sum(x)`, which a normalised alignment
saturates at 1 and a broken identity cannot -- and because that read is invariant to both the
offset and the scale, it needs neither `lambda` nor `tr(K)`. That is what makes it usable as a
**build check**: on a lattice built with a periodic wrap where open was intended, the read fires at
`beta = 1` to `3` while not one negative weight has appeared and the criterion computed on the
*intended* lattice still reports sign-free.

**The scope is measured, and it separates a comparison from a one-sided read.** A comparison
between the two channels cannot reach a complex weight -- its phase is common to both, so the
comparison is blind by construction, and we exhibit systems whose negative fractions differ by 33
percentage points and whose reads are identical. The same weight read one-sidedly is exact. What
neither can do is bounded from below: correct importance sampling draws at `|w|`, so Kish's
effective sample size of those weights is exactly `n <sgn>^2`, a property of the weights and
therefore a ceiling on every aggregation of them. The read detects onset; it cannot report
severity, and no fixed map between the two can exist.

---

## 1. What is normally measured

Determinantal quantum Monte Carlo evaluates a fermionic partition function by decoupling the
interaction into an auxiliary field and integrating the fermions out, leaving a sum over field
configurations with weight `w(x)`. When `w` is non-negative it is a probability density and the
sum is a Monte Carlo average. When it is not, the standard workaround samples `|w|` and carries
the sign as an observable:

    <O> = <O sgn>_|w| / <sgn>_|w| ,        <sgn> = Z / Z_||

The denominator is the average sign, and it is exponentially small: `<sgn> = exp(-beta N df)` with
`df` the free-energy density difference between the physical ensemble and the one that samples
`|w|`. Both numerator and denominator are then small numbers computed as differences of large
ones, and the relative error of the ratio grows like `1 / sqrt(M <sgn>^2)`. The cost of a fixed
accuracy is therefore exponential in `beta N`. Deciding the general case is NP-hard, so no
representation-independent cure is available and the practical question is always which
representation a given model admits.

That is the quantity this paper is about, and the first thing to say about it is what it cannot
do. The average sign is an average of a binary. Where the two channels are locked it returns
`1.00000 +- 0.00000` on every configuration, and an average of a constant has no derivative: it
carries no information about how close the system is to losing the lock.

## 2. The sign is a disagreement between two channels

The weight of an auxiliary-field configuration is

    w(x) = det(I + B_up(x)) * det(I + B_dn(x))

so `sign w(x) < 0` **if and only if** the two channels' determinant signs disagree. That is a
definition. What is a result is that the disagreement has structure.

**Measured** (`reads/expVV_lockstep.py`, 600 configurations per row, 2x4, U = 4, both channels
read separately in the conditioned frame of §6; five seeds, errors from the seed spread):

| beta | mu | flip rate per channel | agree | disagree | disagree = neg fraction |
|---|---|---|---|---|---|
| 4 | 0.0 | 0.003 +- 0.001 | **1.0000 +- 0.0000** | 0.0000 | exact |
| 6 | 0.0 | 0.019 +- 0.006 | **1.0000 +- 0.0000** | 0.0000 | exact |
| 8 | 0.0 | 0.040 +- 0.006 | **1.0000 +- 0.0000** | 0.0000 | exact |
| 10 | 0.0 | 0.075 +- 0.013 | **1.0000 +- 0.0000** | 0.0000 | exact |
| 12 | 0.0 | **0.116 +- 0.011** | **1.0000 +- 0.0000** | 0.0000 | exact |
| 6 | 0.4 | 0.010 +- 0.004 | 0.9837 +- 0.0030 | 0.0163 | exact |
| 8 | 0.4 | 0.020 +- 0.002 | 0.9620 +- 0.0068 | 0.0380 | exact |
| 10 | 0.4 | 0.037 +- 0.008 | 0.9387 +- 0.0103 | 0.0613 | exact |
| 12 | 0.4 | 0.048 +- 0.010 | 0.9183 +- 0.0129 | 0.0817 | exact |

**Two columns are estimates and two are not, and the difference is the point.** The flip rate is a
proportion of 600 draws and is quoted with its spread across seeds. The agreement is not an
estimate: at half filling the two channels change sign on *exactly* the same configurations, so the
column reads `1.0000` with zero spread on every seed rather than averaging to it. The last column
is likewise an identity checked configuration by configuration, not a fitted correspondence -- the
disagreement fraction **is** the negative-weight fraction, on doped rows as well as half-filled
ones.

The half-filled rows establish two things together, and neither alone would do. The individual
channels **do** change sign, on around 12% of configurations by `beta = 12`, so the lockstep is not
a statement about a quantity that never moves. And they change sign on the same configurations
every time. Away from half filling the two channels' flip rates differ from each other as well --
`0.0133` against `0.0033` at `beta = 6` -- so the equality of the two rates is itself a half-filling
statement and not a generic one.

**On four geometries, not one.** Every table above is a 2x4 lattice, and the claim is general, so
it was re-measured across sizes and shapes (`reads/expAC_geometry.py`, `mu = 0`, `beta = 10`):

| lattice | N | identity residual | flip rate | agree | neg fraction | coupling |
|---|---|---|---|---|---|---|
| 2x4 | 8 | 3.0e-12 | 0.07 +- 0.014 | **1.0000** | 0.0000 | **-1.0000** |
| **4x4** | **16** | 1.9e-11 | **0.44 +- 0.028** | **1.0000** | 0.0000 | **-1.0000** |
| 2x6 | 12 | 2.2e-12 | 0.06 +- 0.014 | **1.0000** | 0.0000 | **-1.0000** |
| 4x6 | 24 | 6.2e-13 | 0.009 +- 0.006 | **1.0000** | 0.0000 | **-1.0000** |
| 2x3 | 6 | 1.1e+01 | 0.0467 | 0.9067 | 0.0933 | not resolved |
| 3x4 | 12 | 7.0e+00 | 0.0100 | 0.9733 | 0.0267 | -0.6387 |

The last two rows are controls: a periodic lattice with an odd side has an odd ring and is not
bipartite, and there all three claims fail together -- the identity breaks, the channels
disagree, and negative weights appear. A test that held everywhere would not be sensitive to the
property the argument depends on.

The `4x4` row is the strongest form of the statement available here. An individual channel changes
sign on **44.67%** of configurations and the two channels agree on **every one of them**.

One row is reported and not counted: `4x6` at `beta = 6` has a flip rate of **0.0000**, so its
perfect agreement is vacuous -- nothing crossed. Agreement is only evidence where the channels
actually flip, so `tests/test_lockstep.py` requires a non-zero flip rate alongside it.

## 3. The identity that produces the lockstep

Two facts about the two determinants look incompatible. Their signs agree on every configuration,
and their log-magnitudes differ by 8 to 16 orders. A relation `det_up = c det_dn` for constant `c`
would fix both, so the relation cannot be of that form: it must preserve the sign and leave the
magnitude free.

It does, and it is exact. The two channels' diagonal factors are inverses of one another, so
`det(B_up)/det(B_dn) = exp(2 lambda sum x)`; routing that through
`det(I + B) = det(B) det(I + B^-1)` carries one factor into the ratio of the full determinants:

    ln|det(I + B_up(x))| - ln|det(I + B_dn(x))| = -dtau * L * tr(K) + lambda * sum(x)

with `lambda = arccosh(exp(dtau U / 2))` the decoupling's own constant.

**Verified** (`reads/expWW_ratio.py`, `tests/test_identity.py`) at half filling on a bipartite
lattice:

| beta | 2 | 4 | 6 | 8 |
|---|---|---|---|---|
| residual, coefficient `lambda` | **1.07e-14** | **2.13e-14** | **4.16e-13** | **3.06e-13** |
| residual, coefficient `2 lambda` | 2.51e+01 | 3.83e+01 | 5.31e+01 | 3.98e+01 |

The second row is the point: a relation that held for any coefficient would be vacuous. The
derived coefficient is exact and the alternative is wrong by fourteen orders.

**Across the interaction, not at one value of it.** `lambda` is a function of both `U` and `dtau`,
so varying them tests the derivation rather than repeating a measurement: each row predicts a
different coefficient and has to be exact at it (`reads/expAD_coupling_strength.py`, residual max
over 200 configurations):

| dtau | U = 2 | U = 4 | U = 8 | U = 12 |
|---|---|---|---|---|
| 0.0625 | 2.5e-14 | 1.1e-13 | 2.5e-13 | 5.3e-12 |
| 0.125 | 2.7e-14 | 5.7e-14 | 4.7e-12 | 1.0e-11 |
| 0.25 | 6.8e-14 | 1.4e-12 | 2.2e-10 | 1.0e-08 |

`lambda` spans 0.357 to 2.180 across this table, a factor of 6.1. The residual grows gently with
it as accumulated arithmetic -- the last cell sits at 1e-8, which is ten orders below the wrong
coefficient's value on the same row and is float64 rather than a limit of the relation.

**And the coefficient is resolved to better than its own variation.** Borrowing a neighbouring
row's `lambda` -- a difference of as little as 0.51 against 0.74 -- fails by 15 to 44 while the
predicted one holds at 1e-14:

| U | own lambda | borrowed | residual, own | residual, borrowed |
|---|---|---|---|---|
| 2 | 0.51048 | 0.73690 | 3.0e-14 | **15.4** |
| 4 | 0.73690 | 1.08504 | 1.2e-12 | **18.8** |
| 8 | 1.08504 | 1.38202 | 2.6e-12 | **17.8** |
| 12 | 1.38202 | 0.51048 | 4.3e-11 | **43.6** |

So it is not that any coefficient of approximately the right size will do. The relation resolves a
44% difference in `lambda` by fourteen orders, at every interaction strength measured.

**One relation explains both observations.** The ratio is an exponential, hence positive, hence the
two determinants share a sign -- the lockstep, and so the absence of a sign problem. Its exponent
is a sum over `L * N` field components, hence the ratio's magnitude ranges over 8 to 16 orders --
so the two determinants' magnitudes look unrelated while their signs are pinned.

**The identity is a property of the structure, not of the Ising field.** Its derivation uses one
property of the decoupling -- that the two spins' diagonal factors are inverses, `V_up = V_dn^-1`.
The *continuous* Gaussian spin decoupling has that property too, with its own closed-form constant
`lambda_s = sqrt(dtau U)`, so the same derivation predicts the same identity there with a different
coefficient. At `U = 4, dtau = 0.125` the two constants are `0.73690` and `0.70711` -- **4.2%
apart** -- and neither is fitted (`reads/expAK_identity_across_representations.py`, 150 draws per
cell):

| field | beta | its own lambda | the OTHER representation's lambda | twice its own |
|---|---|---|---|---|
| Ising | 2 | **2.1e-14** | 9.5e-01 | 2.4e+01 |
| Ising | 4 | **3.3e-14** | 1.4e+00 | 3.4e+01 |
| Ising | 6 | **4.6e-12** | 1.5e+00 | 3.8e+01 |
| Gaussian | 2 | **1.4e-14** | 8.8e-01 | 2.1e+01 |
| Gaussian | 4 | **2.7e-14** | 1.2e+00 | 2.9e+01 |
| Gaussian | 6 | **7.3e-14** | 1.5e+00 | 3.5e+01 |

This is the sharpest control in the paper. Each representation's own constant is exact to `1e-14`
and the other's, four percent away, fails by fourteen orders. A relation that held for any
coefficient of roughly the right size could not do this, and neither constant can have been tuned
to its data: both are closed forms of their own decoupling, fixed before any determinant is formed.

What the identity carries travels with it. In both representations the two determinants' signs
agree on **1.0000** of configurations with no negative weight, and the coupling reads **-1.0000**:

| field | beta | flip rate | signs agree | negative weight | strength | agreement is |
|---|---|---|---|---|---|---|
| Ising | 4 | 0.0000 | 1.0000 | 0.0000 | -1.0000 | *vacuous, no flips* |
| Ising | 8 | 0.0350 | **1.0000** | 0.0000 | **-1.0000** | real |
| Ising | 12 | 0.1450 | **1.0000** | 0.0000 | **-1.0000** | real |
| Gaussian | 4 | 0.0000 | 1.0000 | 0.0000 | -1.0000 | *vacuous, no flips* |
| Gaussian | 8 | 0.0200 | **1.0000** | 0.0000 | **-1.0000** | real |
| Gaussian | 12 | 0.0900 | **1.0000** | 0.0000 | **-1.0000** | real |

The `beta = 4` rows are marked because neither representation flips a determinant sign in 200 draws
there, so perfect agreement is agreement about nothing; the rows with content are the deeper ones.
§7 shows the read is a property of a (Hamiltonian, decoupling) pair -- this is the other half of
that statement: within the spin decoupling it does not depend on which field distribution realises
it.

## 4. What the identity requires, controlled

Every row that verified the identity had a bipartite lattice **and** half filling. Those are
confounded, and separating them required care: next-nearest hopping `tp` destroys the bipartite
structure, but it also moves the band, so `mu = 0` is no longer half filling. Compared without
that control the data appears to show that breaking the bipartite structure REMOVES the sign
problem -- an artifact, because the non-bipartite cells sat at `<n> = 0.75`.

The filling is therefore fixed by the **non-interacting** density, a closed-form function of the
single-particle spectrum that carries no sampling noise and is computed without touching the sign.

**Measured** (`reads/expZZ_matched_filling.py`, U = 4, beta = 8, every row at `n = 1.00000`):

| tp | tuned mu | bipartite | identity residual | signs lock | neg fraction |
|---|---|---|---|---|---|
| 0.00 | **-0.0000** | yes | **1.3e-12** | **1.0000** | **0.0000** |
| 0.15 | 0.6000 | no | 4.25e+01 | 0.9500 | 0.0500 |
| 0.30 | 1.1992 | no | 8.32e+01 | 0.8150 | 0.1850 |
| 0.70 | 1.5548 | no | 1.05e+02 | 0.9200 | 0.0800 |

The tuner recovers `mu = -0.0000` for the bipartite lattice without being told that half filling
lies there, which is the control validating itself. At identical filling, the bipartite lattice has
no sign problem and the identity is exact; every non-bipartite one has a sign problem and the
identity is broken.

**For a real one-body matrix the two conditions are one criterion, checkable without sampling.**
The criterion below is a statement about the spin decoupling, as everything in §§2-5 is: it is
derived from that decoupling's diagonal factor and says nothing about the charge channel, whose
positivity comes from a different mechanism (§5). The derivation reduces to §3's form only when
`det(I + B_up^-1) = det(I + B_dn)`, which needs
`expmK^-1` similar to `expmK` -- and the similarity has to commute with the interaction's diagonal
factor. That restricts it to a **signed diagonal**: `S K S = -K` with `S = diag(+-1)`. Elementwise
`s_i s_j K_ij = -K_ij`, so the criterion is a 2-colouring of K's support graph together with a zero
diagonal. It costs `O(N^2)`, and uses no eigenvalues, no determinants and no field
(`reads/expAO_spectral_criterion.py`, 2x4, `U = 4`, `beta = 8`):

| K | signed-diagonal S exists | identity residual | agrees |
|---|---|---|---|
| bipartite, `mu = 0` | yes | 1.7e-12 | yes |
| bipartite, staggered `h = 0.2` | **no** | 3.8e+00 | yes |
| bipartite, staggered `h = 1.2` | **no** | 2.1e+01 | yes |
| `tp = 0.3` | no | 1.2e+01 | yes |
| `mu = 0.4` | no | 3.2e+01 | yes |
| bipartite, bond-disordered x3 | **yes** | 8e-14 to 3e-13 | yes |
| 4x4 and 2x6 bond-disordered | **yes** | 4e-12, 2e-13 | yes |
| odd rings of 5 and 7 sites | no | 7.2, 5.0 | yes |

Sixteen one-body matrices, agreeing with the measured identity on every one. The criterion unifies
§4's two conditions and covers a case neither names: a **staggered potential** leaves the lattice
bipartite and the filling at exactly one per site, and breaks the identity anyway, because a
diagonal term cannot be negated by signs.

The bond-disordered rows are the ones that make it predictive rather than descriptive. No symmetry
was designed into them -- every bond strength is drawn at random, up to a spread of 1.5 -- and the
criterion says the identity holds, which it does to `1e-13`.

**It is a statement about the hopping graph, not about lattices.** The same criterion was applied
to topologies the rest of the paper never uses, with the prediction made before the measurement:

| topology | criterion | identity residual |
|---|---|---|
| chain, 8 sites, periodic | yes | 7.5e-14 |
| chain, 7 sites, periodic | no | 4.3e+00 |
| chain, 8 sites, open | yes | 1.6e-13 |
| **star graph (a tree), 8 sites** | **yes** | **5.9e-12** |
| triangular ladder, 8 sites | no | 8.9e+00 |
| even ring, 12 sites | yes | 9.9e-14 |
| odd ring, 9 sites | no | 4.5e+00 |

A tree satisfies it however it is drawn, having no cycles at all; any graph carrying an odd cycle
never does. Twenty-four one-body matrices in total, agreeing with the measured identity on every
one.

**With complex hoppings there are two routes, and the identity holds when either is open.** The
signed-diagonal criterion is sufficient but not necessary once `K` is complex: a 5-site ring
threaded by a flux of `pi/2` has no signed-diagonal conjugation and satisfies the identity anyway,
at `1.4e-14`. The conjugation may be a diagonal **unitary** rather than a diagonal sign, and the
derivation is satisfied by either a similarity or an **anti**-similarity:

| route | condition | consequence |
|---|---|---|
| A | `S K S^-1 = -K` | every cycle **even**, at any flux |
| B | `S K S^-1 = -conj(K)` | `(-1)^l e^{-2i Phi} = 1` on every cycle: **odd** cycles need flux `pi/2 mod pi` |

An odd cycle carrying half-odd-integer flux restores what the odd cycle destroys. Measured on rings
of 5, 6, 7 and 8 sites at six fluxes each, plus a 5x2 lattice at five fluxes -- **29 of 29 agree**,
including every case where exactly one route is open.

For real `K` the two conditions coincide, which is why the signed-diagonal colouring is the whole
story there: checked on 11 real matrices, the general criterion and the colouring return the same
answer on every one. **Forty one-body matrices in total, with no mismatch.**

**The two routes are not equivalent, and only one carries positivity.** Route A leaves the weights
real: on `ring 6` and `ring 8` at flux `pi/4` the identity holds at `2e-14` and the weight's mean
phase is **exactly 1** -- no sign problem. Route B restores the identity on an odd cycle and leaves
a phase behind. Measured with `1 - |<w/|w|>|`, which is the ordinary negative-weight measure when
the weight is real and the mean-phase deficit when it is not
(`reads/expAO_spectral_criterion.py`, `beta = 8`, 200 draws):

| K | route | identity residual | sign deficit |
|---|---|---|---|
| 2x4 real, half filled | A and B | 6.3e-13 | **0.00000** |
| ring 6, flux `pi/4` | **A only** | 3.6e-14 | **0.00000** |
| ring 5, flux `pi/2` | **B only** | 2.4e-14 | 0.15893 |
| triangular ladder, flux 0 | neither | 7.8e+00 | 0.06000 |
| triangular ladder, flux `pi/2` | **B only** | 8.0e-14 | **0.52160** |

The last two rows are the point. Threading the frustrated ladder does exactly what the criterion
says -- the identity goes from broken to exact at `8e-14` -- and the sign quality gets **nine times
worse**, from `0.060` to `0.522`. The flux converts a mild sign problem into a substantial phase
problem.

So the criterion decides **the identity**, and the identity delivers **positivity only where the
weights are real**, which is route A. On an odd cycle no flux-free choice exists and route B is the
only one available, so an odd cycle cannot be made positive this way at all -- it can only be made
to satisfy the identity.

**On a two-dimensional frustrated lattice the answer depends on the size, and the criterion tracks
it.** A periodic triangular lattice has triangles, rhombi *and* wrap-around cycles, and route B has
to hold on all of them at once. On `4x4` the wraps are even and carry no flux, so `theta = pi/2`
opens route B and the identity holds to `3e-13`; on `3x3` the wraps are odd and flux-free, no
`theta` opens either route, and the identity stays broken at every flux tried. Ten rows across the
two lattices and five fluxes, all predicted correctly.

*What that pair does and does not show.* It is a statement about the **identity**. The unfluxed
`4x4` carries a broken identity -- residual `8.7` to `14.0` -- and still shows **no negative weight
at all** in 200 draws out to `beta = 16`, so nothing was restored there that was measurably absent.
The `3x3` lattice does have a real sign problem, `0.4150` unfluxed and `0.1300` at `pi/2`, and the
criterion correctly says no flux removes it. This pair extends the criterion to two dimensions and
shows the wrap cycles deciding the outcome; like the ladder, it is a statement about the identity
and not about positivity, since `theta = pi/2` opens route B.

**The obvious criterion is the spectral one, and it is wrong.** For Hermitian `K`, a spectrum
symmetric about zero is exactly similarity to `-K`. The staggered rows have spectra symmetric to
**1e-15** and break the identity by up to `21`. The similarity exists; it is not a signed diagonal,
so it does not commute with `diag(exp(sigma lam x))` and the derivation cannot use it.

**The criterion is sharp: the identity is fragile where the sign problem is robust.** A staggered
term of any amplitude `eps > 0` makes `S K S = -K` unsatisfiable, and the identity follows exactly:

| eps | criterion | identity residual | residual / eps | negative fraction |
|---|---|---|---|---|
| 0 | yes | 3.3e-13 | -- | **0.0000** |
| 1e-06 | no | 8.542e-05 | **85.425** | **0.0000** |
| 1e-04 | no | 8.543e-03 | **85.425** | **0.0000** |
| 1e-03 | no | 8.548e-02 | **85.482** | **0.0000** |
| 1e-02 | no | 9.185e-01 | 91.848 | **0.0000** |
| 1e-01 | no | 3.217e+00 | 32.165 | 0.0200 |

The residual is **linear in the perturbation across four decades**, with no tolerance and no onset:
the identity does not survive a small violation, it degrades in proportion to it from the first. A
chemical potential behaves the same way, at a ratio of `93.9`.

The negative fraction does not follow suit. It is exactly `0.0000` until `eps` reaches about `0.1`,
so a model can violate the criterion, carry a measurably broken identity, and still produce no
negative weight at all. **The identity is structurally fragile and the sign problem numerically
robust**, and those are statements about two different quantities -- the residual measures the
perturbation, the negative fraction measures its consequence, and only the first is linear.

So the criterion decides the **identity**, and nothing further on its own: it carries positivity
only along route A, where the weights stay real, and it does not predict the negative *fraction* on
either route. That is the mechanism behind the next point.

**The residual is not a severity measure** (`reads/expAB_coupling_controlled.py`). It grows
monotonically with `tp` while the negative fraction does not (0.050, 0.185, 0.080). The identity's failure permits the signs to differ; it
does not say by how much they will.

**What this does and does not establish about positivity in general.** The rows above are an
`if and only if` on this axis: at matched filling, bipartite means the identity holds and there is
no sign problem, and non-bipartite means both fail. They are **not** a claim that this is where
positivity comes from. Wei et al. [7] give a unified account of the known sign-free models and
identify sign-free models with *repulsive* interactions and *without* particle-hole symmetry, and
Wu and Zhang [6] derive positivity from a conjugate-pairing property tied to time-reversal rather
than to a lattice bipartition. §5 measures one of those other places directly -- the attractive
charge channel, sign-free at every filling -- and finds the read saturated there too, at the
opposite sign. So the symmetric point is one source of positivity that this rig can read, and the
scope of §4 is the axis it was measured on.

### 4.1 The criterion performed on output, with no Hamiltonian

Everything above decides the identity from `K`: build the support graph, look for a diagonal
unitary with `S K S^-1 = -K`, two-colour it, check the flux on every odd cycle. That is a decision
about a matrix. A running simulation does not have a clean matrix to hand -- it has configurations
and determinants -- and the terms most likely to have broken the identity, a doping term or a
staggered potential someone added, are exactly the ones nobody re-derives the criterion for.

Read as a statement about DATA rather than about `K`, §3's identity says something a read can act
on. Across configurations,

    ln|det_up|(x) - ln|det_dn|(x)   is an EXACT AFFINE FUNCTION of   sum(x)

with slope `lambda` and offset `-dtau L tr(K)`. That is one channel against another on a shared
index, which is what `coupling` reads, and the read is invariant to both the offset and the scale --
so neither constant has to be known. An exact affine relation must saturate a normalised alignment
at 1. A broken identity cannot (`reads/expAW_criterion_from_output.py`, 2x4 and rings, `U = 4`,
`beta = 6`, 400 configurations):

| | `1 - \|strength\|` |
|---|---|
| identity holds (7 lattices) | `0` to `2.2e-16` -- machine epsilon |
| identity fails (8 lattices) | `1.0e-3` to `2.9e-2` |

**The read agrees with the measured identity residual on 15 of 15 lattices, and with §4's algebraic
criterion on 15 of 15, using no Hamiltonian.** The two sides share no information: `coupling` sees
two columns and never sees `K`; the criterion sees `K` and never sees a configuration.

*The departure is systematic, not sampling noise.* This is the one way the read could be right for
the wrong reason, so it is measured rather than argued: across a factor of four in sample size and
three seeds, the saturated lattices stay at machine zero and the broken ones stay at `9.2e-4` to
`8.5e-3` without shrinking. A departure that fell as `n` grew would be the read reporting its own
variance. It is gated as an assertion, not left as a remark.

*What it costs and what it is for.* Two columns a simulation already has, and one read -- `O(n)`
after determinants that were computed anyway, against a criterion that otherwise needs the
Hamiltonian's structure. It turns "is this model sign-free" from a question asked once on paper
into a check a run can perform on itself.

#### How it is used

Log two scalars per configuration. Both are already computed by any determinant QMC code:

  * `ln|det_up| - ln|det_dn|`, which comes out of the stable propagation -- `slogdet` is formed
    every sweep to build the weight;
  * `sum(x)`, a reduction over the auxiliary field.

Then one call, `coupling(A, B)`, and read whether it saturates. Three places this is worth doing:

  1. **As a pilot, before committing to a production run.** A few hundred configurations at small
     `beta` decide whether the model *as assembled* sits in the sign-free class.
  2. **As a regression check.** A sign-free setup acquires a term -- `t'`, a staggered field, a
     boundary twist, a chemical potential. Nobody re-derives the two-colouring when a term is
     added, and this catches it without anyone having to.
  3. **As a build check.** The read sees what the code *did*, not what the model was meant to be.
     Where an assembly routine and its documentation disagree, the read follows the code.

#### The build check, measured

The third use is the one the other two cannot cover, because it is the case where the criterion of
§4 is *unavailable by construction*: if the code may not be running the model its author believes,
then the `K` and the `lambda` one would feed to a direct test of §3's identity are the suspect
quantities themselves. The read needs neither -- it is invariant to the offset and the scale -- so
it is the only instrument left standing (`reads/expAZ_build_check.py`).

Intended: an **open** 7-site chain, bipartite, and §4's criterion computed on it reports sign-free.
As built: a periodic wrap, making a 7-site **ring** -- an odd cycle, outside the class. The oracle
is silent for every row below, because it reads the model rather than the code.

| `beta` | negative fraction (4 seeds) | `1 - \|strength\|` | seed spread | correct-build control |
|---|---|---|---|---|
| 1 | **0.00000** | `1.904e-07` | `2.1e-09` | `1.1e-16` |
| 2 | **0.00000** | `6.319e-05` | `5.8e-06` | `-2.2e-16` |
| 3 | **0.00000** | `3.336e-04` | `6.1e-05` | `1.1e-16` |
| 4 | 0.00125 | `7.381e-04` | `1.0e-04` | `0.0e+00` |

At `beta = 1` to `3` not one negative weight appears across four seeds, so every standard health
check passes, while the read stands eleven orders above the correctly-built control at `beta = 2`
and five to thirty times its own seed spread. From `beta = 4` the sign fires too and the read is no
longer needed.

**The scope is two provable limits, not two gaps.** A wrong decoupling constant -- the continuous
Gaussian `sqrt(dtau U)` used in the discrete field -- is **not** detected: `1 - |strength|` is
`-2.2e-16`, exactly saturated, because rescaling an affine relation leaves it affine. Scale
invariance is why the read needs no `lambda` and why it cannot see a wrong one; the two are the same
fact, and the direct residual catches that case at `1.61`, so the two checks are complementary and
neither dominates. Applying the field with one sign to both channels leaves the two determinants
identical, so the logged difference is identically zero and the read returns **unresolved** rather
than a departure -- a degenerate input, visible before any read is taken.

#### Why this is not just watching the average sign

Every DQMC code already reports `<sgn>`, so the read is only worth having if it fires earlier. It
does. On `2x4` with `t' = 0.3`, at `beta = 6` over 400 configurations:

| quantity | value |
|---|---|
| negative-weight fraction | **0.0000** -- not one negative weight |
| §3 identity residual | **10.53** -- broken by an order of magnitude |
| `1 - \|strength\|` | **2.3e-3** -- flagged |

The average sign reports a clean run on a model that has already left the sign-free class. The same
holds on `ring 5` (deficit `0.0167`, identity broken by `6.26`) and on both staggered lattices.

`<sgn>` is a *sampled* quantity: it sits at `1.0000` until the sign problem grows large enough to
appear in a finite sample, and the cost of it appearing is `O(1/<sgn>^2)` with
`<sgn> = exp(-beta N df)`. So the expense of discovering the problem that way rises exponentially
in `beta` and in system size, while this read's cost is flat in both -- `O(n)` over two columns, at
whatever `beta` is cheapest to sample. What it buys is the difference between learning at small
`beta`, where a pilot is minutes, and learning at production `beta` from error bars that will not
shrink.

## 5. Reading the weights: which problem is it, how far from the symmetric point, and at what cost

An observer standing at the output holds a cloud of configuration weights and nothing else -- no
`K`, no knowledge of the decoupling, and no constant to supply. This section is what can be
decided from there, and it has three parts, in the order a practitioner meets them.

**First, which problem the run has.** `concentration` on the weight cloud's `(Re, Im)` frame
returns a *directional* statistic and an *axial* one, and only the pair classifies: `focus = 1`
with `resultant = 1` is sign-free; `focus = 1` with `resultant < 1` is a real sign problem, because
rank one in the plane means the phase takes two values `pi` apart and a phase confined to `Z2` is
what a sign *is*; `focus < 1` is a phase problem no rotation reaches. That triage is §5.3, and it
is the part that decides what the rest of the section is even measuring.

**Second, how far the run sits from the protecting symmetry**, which is the coupling read below --
exactly calibrated where the lockstep is perfect, and moving while the average sign is still
identically 1.

**Third, what any of it can cost**, which §8 bounds.

The order is deliberate: the coupling read that follows is a statement about a *real* determinantal
weight, so the classification has to come first or the reading is being taken on an object it does
not apply to.

### 5.1 The coupling read, and its exact calibration

The lockstep of §2 is produced by the identity of §3, and the identity holds exactly at the
symmetric point of §4. That suggests a continuous measure of how far a system sits from that
point, and there is one: the alignment between the two channels, read as a coupling.

**The calibration is exact.** At half filling on a bipartite lattice, particle-hole symmetry gives
`G_up[i,i](x) + G_dn[i,i](x) = 1` configuration by configuration -- verified to a maximum residual
of **6.7e-15** (`reads/expTT_order_parameter.py`) -- so the two centred channels are exact
negatives and their coupling must read `-1`. It reads **-1.0000**, at every beta, with the
instrument's own exact re-pairing null placing the permuted control at `|z| <= 1.7`. On the
controlled axis of §4 it reads `-1.0000` with `tightness = 1.000`: the entire coupling in a single
mode.

**What the deficit measures, exactly.** The read is a normalised alignment, so its deficit from
saturation is an angle:

    deficit = 1 - cos(angle between the two centred channel frames)

Write the violation of the particle-hole relation as `E = B~ + A~`, and split it into the part
along `A~` and the part perpendicular to it. Two consequences follow, and both are measured rather
than argued:

* **The parallel part is invisible.** Rescaling one channel -- `B = 1 - 2A` instead of `1 - A` --
  moves `|E|/|A~|` to `1.0` and leaves the deficit at **exactly** `0.000000`. The read has no
  notion of the two channels' relative size, only of their alignment.
* **The perpendicular part is what registers**, and to leading order
  `deficit = |E_perp|^2 / (2 |A~| |B~|)`. Along the capability axis of this section that
  approximation agrees with the measured deficit to `1e-5` at `beta = 1` and `4e-4` at `beta = 3`;
  by `beta = 4`, where the deficit reaches `0.093`, the small-angle form is 4.4% low and the exact
  cosine is what holds.

So the order parameter is the **angle between the two centred channel frames**, and what it tracks
as filling or temperature moves is the growth of the component of the particle-hole violation that
cannot be absorbed into a rescaling. That is why it is an onset detector: an angle is bounded, and
`-ln<sgn>` is not.

**The calibration comes from one of §4's two routes, not from the identity.** §4 shows the identity
holds when either `S K S^-1 = -K` (route A) or `S K S^-1 = -conj(K)` (route B) is available. Both
give positivity. Only route B gives the exact particle-hole relation, and with it the exact `-1`
(`reads/expAO_spectral_criterion.py`, 250 configurations per row):

| K | route A | route B | `max\|G_up + G_dn - 1\|` | strength |
|---|---|---|---|---|
| 2x4 real | yes | yes | 6.0e-10 | **-1.0000** |
| ring 8, real | yes | yes | 1.0e-12 | **-1.0000** |
| ring 6, flux `pi` | yes | yes | 7.4e-11 | **-1.0000** |
| ring 5, flux `pi/2` | no | **yes** | **5.3e-15** | **-1.0000** |
| ring 7, flux `pi/2` | no | **yes** | **4.3e-15** | **-1.0000** |
| triangular 4x4, flux `pi/2` | no | **yes** | **3.2e-12** | **-1.0000** |
| ring 6, flux `pi/4` | **yes** | no | **1.4e-01** | -0.9925 |
| ring 6, flux `pi/3` | **yes** | no | **1.9e-01** | -0.9863 |
| ring 8, flux `pi/4` | **yes** | no | **6.4e-01** | **-0.9453** |

Route B is the anti-similarity, and it is what forces `G_dn = 1 - G_up` configuration by
configuration -- the identity this section's calibration rests on. Where only route A is open the
lattice is still sign-free and the identity of §3 still holds exactly, but the two channels are no
longer exact negatives about `1/2`, and the read comes back at `-0.945` to `-0.993`: resolved,
close, and not calibrated.

**Each route supplies its own exact relation between the channels.** Route B gives
`G_dn = 1 - G_up`; route A gives the same thing composed with complex conjugation,
`G_dn = 1 - conj(G_up)`. Both hold configuration by configuration, to `5e-15` and `3e-15`
respectively. Under route A the real parts are therefore exact negatives about `1/2` and the
imaginary parts are exact *positives*, so the two halves of the frame read `-1` and `+1` and a
single signed alignment averages them.

That average is not approximate. Writing `r` for the imaginary part's share of the centred
variance, the read is

    strength = -1 + 2r

and it holds to **1e-16** across six flux-threaded rings:

| K | r | `-1 + 2r` | measured |
|---|---|---|---|
| ring 6, flux `pi/8` | 0.00091 | -0.99818 | **-0.99818** |
| ring 6, flux `pi/4` | 0.00370 | -0.99259 | **-0.99259** |
| ring 6, flux `pi/3` | 0.00674 | -0.98653 | **-0.98653** |
| ring 8, flux `pi/4` | 0.02713 | -0.94573 | **-0.94573** |
| ring 8, flux `pi/3` | 0.02213 | -0.95575 | **-0.95575** |
| ring 10, flux `pi/4` | 0.00225 | -0.99549 | **-0.99549** |

So the departure from `-1` under route A is not a degradation of the reading and carries no
information about the sign problem, which is absent throughout: it is the imaginary weight of the
frame, exactly.

So `-1.0000` is not a consequence of positivity, and it is not a consequence of the §3 identity
either. It is a consequence of the particle-hole structure that route B supplies, which the
half-filled bipartite case happens to have. That is worth separating, because the three coincide in
every measurement in §§2-4 and come apart under flux.

**It is readable where the average sign is not.** Both columns come from the same
importance-sampled chains at each beta (`reads/expAN_capability_table.py`, `StableChains` gated
against brute-force enumeration at 5e-15; `N = 8`, `U = 4`, `dtau = 0.125`, `t2 = 0.7`, `mu = 1.0`,
four seeds, errors from the seed spread):

| beta | `<sgn>` | coupling deficit | reproducibility | `\|z\|` |
|---|---|---|---|---|
| 1.0 | **1.00000 +- 0.00000** | 0.03574 +- 0.00022 | 0.6% | 136.9 |
| 1.5 | **1.00000 +- 0.00000** | 0.04893 +- 0.00035 | 0.7% | 134.0 |
| 2.0 | 0.99863 +- 0.00037 | 0.06141 +- 0.00039 | 0.6% | 131.3 |
| 3.0 | 0.94961 +- 0.00535 | 0.1079 +- 0.0033 | 3.1% | 123.2 |
| 4.0 | 0.81836 +- 0.01234 | 0.1251 +- 0.0019 | 1.5% | 118.7 |
| 6.0 | 0.48633 +- 0.00733 | 0.201 +- 0.021 | 10.4% | 106.5 |

The first two rows are the capability. `<sgn>` is an average of a binary: where the channels are
locked it is identically 1 with zero variance and no derivative. The coupling has already moved,
and is resolved at 137 sigma.

**The rows are quoted to the precision they reproduce to, which is not the same precision.** The
deficit is reproducible to under 1% where the claim lives and to 10% at `beta = 6`, so the deep
rows carry fewer digits. That difference is itself the reason §5 claims onset and not severity: the
deficit is a converged estimate on the sign-free rows, and on the deepest row its error does not
shrink with sampling in the way a converged estimate's must.

**The capability is not a property of the field distribution either.** The claim above is the one
that would be used, and it had only been measured with the discrete Ising field. Run on the same
axis with the *continuous Gaussian* spin decoupling -- a different measure realising the same
decoupling (`reads/expAL_capability_across_representations.py`, `mu = 0.4`, 400 draws x 4 seeds):

| beta | neg fraction | Ising deficit | Gaussian deficit | `\|z\|` (both) | permuted null |
|---|---|---|---|---|---|
| 1.0 | **0.00000** | 0.00563 - 0.00598 | 0.00627 - 0.00629 | 52.0 | <= 1.7 |
| 1.5 | **0.00000** | 0.00902 - 0.00958 | 0.00985 - 0.01058 | ~49 | <= 2.3 |
| 2.0 | **0.00000** | 0.01412 - 0.01480 | 0.01438 - 0.01530 | ~48 | <= 1.6 |
| 3.0 | **0.00000** | 0.02887 - 0.03184 | 0.02891 - 0.03164 | ~47 | <= 3.4 |
| 4.0 | 0.005 / 0.0006 | 0.09336 - **0.95636** | 0.05197 - 0.09512 | 32 / 46 | <= 1.6 |

Monotone in beta and resolved at `|z| ~ 50` in both, against a permuted null that stays under 3.4,
with the negative fraction identically zero on the first four rows. What is being compared is the
*shape* -- the two field distributions are different measures and neither deficit is a prediction
of the other -- though in the event they agree to about 10% wherever the negative fraction is zero,
which was not required and is recorded rather than relied on.

**The last row is a limitation, and it is the first row where the sign problem appears.** The Ising
deficit there spans `0.09336` to `0.95636` across four seeds -- a spread ten times its own lower
value. The deficit is reproducible while the negative fraction is zero and stops being so at the
onset, which bounds the claim to exactly the regime §5 makes it in. Reproducibility across seeds
is how that boundary is located, and it is why every row here is replicated.

**On the controlled axis of §4 it does neither.** Along `tp` at fixed filling the read gives the
exact calibration at `tp = 0` and then nothing usable: `strength` is `-1.0000`, `-0.0480`, not
resolved (`z = -0.55`), `-0.2794` -- not monotone, with the unresolved row being the one with the
worst sign problem (0.1825). The library returns zero there rather than a value, which is the
instrument declining to report rather than reporting nothing.

A monotone quantity does appear on that axis -- `tightness` falls 1.000, 0.848, 0.676, 0.564 --
and it is **not** a property of the coupling. At the symmetric point `G_dn = 1 - G_up` exactly, so
the centred channels satisfy `B~ = -A~` and the cross-covariance is `-A~' A~`, whose spectrum is
the single channel's own. Measured directly: the dominant share of channel A alone is 0.9986 where
the coupling's tightness is 1.0000, and away from the symmetric point the coupling's value sits
between the two channels' individual shares (0.770/0.963, 0.772/0.423, 0.470/0.613). `tightness`
is inherited from the channels' internal structure, which also varies with `tp`. It is a property
of each channel on its own, not of the coupling between them.

**So the scope is by axis, and it is narrow.** Along beta at fixed filling the deficit is monotone
and resolved at `|z| ~ 130` where the average sign is identically 1 -- that is the capability, and
it is what §5 claims. Along `tp` the read supplies an exact calibration at the symmetric point and
no usable signal away from it.

**A read of each channel against the system's own zero, which is not an aggregation and not a
relation** (`reads/expAT_balance_at_the_systems_own_zero.py`). `Screen.register` takes a lens's own
laws -- `entry`, `inverse`, `energy`, `zero` -- and its documentation is explicit that a side
withholding them inherits defaults that are *"the right laws for a system that has no others and the
wrong ones for a system that does"*. This system has its own zero: at half filling
`G_up[i,i] + G_dn[i,i] = 1` configuration by configuration, so each channel balances at **1/2**,
the particle-hole point. The library's default is the column mean, which is the sample's own centre
and says nothing about where the system balances.

`Screen.balance` scores what each side's zero leaves against exact no-drift moments -- a scaled
chi-square whose degrees of freedom are the covariance spectrum's participation ratio, with no
tolerance anywhere. Every row below sits at `<n> = 1.00000`, so filling is not the variable; 12
independent runs per row:

| K | sign problem | deficit | **own zero**, median p | **default zero**, median p |
|---|---|---|---|---|
| 2x4 clean | no | 0.0000 | **0.34104** | 1.00000 |
| ring 6, flux `pi/4` | no | 0.0000 | **0.56477** | 1.00000 |
| ring 8, flux `pi/4` | no | 0.0000 | **0.54925** | 1.00000 |
| staggered `h = 0.2` | **yes** | 0.0661 | **0.02489** | 1.00000 |
| staggered `h = 0.6` | **yes** | 0.0644 | **0.00000** | 1.00000 |
| ring 5, flux `pi/2` | phase | 0.1545 | 0.48993 | 1.00000 |
| tri ladder, flux `pi/2` | phase | 0.4792 | 0.37657 | 1.00000 |

**The last column is the control and it carries the result.** Withhold the system's law and the read
is blind on every row -- median `1.00000` throughout, sign-free and sign-problem alike. The
separation belongs to the zero, not to the read.

*What the read is, stated plainly.* `balance` scores the joint residual `||r||^2` against the
exact no-drift moments `E||r||^2 = tr(S)/T` and `Var||r||^2 = 2 tr(S^2)/T^2`, which fix a scaled
chi-square whose degrees of freedom are `(tr S)^2 / tr(S^2)` -- the **participation ratio of the
covariance spectrum**. It is therefore a correlation-aware joint test, and that is what separates
it from the obvious alternative: a per-column deviation-of-the-mean test broadly tracks it and
**orders two rows the opposite way**. On a staggered `2x4` and a flux-`pi/2` triangular ladder over
six seeds the naive test puts the staggered row higher (`z = 2.85` against `2.30`) while `balance`
puts the ladder higher (median pvalue `0.280` against `0.0089`). The two channels here are strongly
correlated, so the effective degrees of freedom sit far below the column count and a test assuming
full rank misreads them.

*How strong that separation is, measured.* The `31x` between those medians is a median effect and
it does **not** survive seed by seed: the ladder's per-seed pvalues run `0.0017` to `0.862` and the
staggered row's `2e-15` to `0.072`, so the two populations overlap and the ordering holds on five of
the six seeds rather than all six. What is asserted here is therefore the reversal of ordering,
which is the claim, and not a clean split, which the data does not support. The overlap is pinned in
the gate so a later draft cannot upgrade one into the other. The classical instrument of the same shape is Hotelling's `T^2`; what the read
supplies is the rank, derived from the spectrum rather than assumed.

*Read the pvalue, not the boolean.* `closed` is a per-run decision at the reader's level and
fluctuates -- a sign-free ring fires on 9 runs of 12. The pvalue behind it is the quantity and its
median across independent runs separates with no overlap, which is the same discipline every other
sampled figure here uses.

*Scope.* This separates **real-weight** sign problems at fixed filling. It does not see a phase
problem: the route B rows, whose channels are exactly related and whose weights carry a phase, read
`0.49` and `0.38`. That is consistent rather than contradictory -- §4's route B leaves the channels
exactly related, and this read asks each channel against its own zero, which that relation leaves
undisturbed.

**Why no weighted read can be a severity meter** (`closed/expAS_weighted_reads_are_capped.py`).
Correct importance sampling draws at `|w|`, so a configuration enters any weighted average carrying
only its **sign**. Kish's effective sample size of those weights is then

    ESS = (sum w)^2 / sum w^2 = (n <sgn>)^2 / n = n <sgn>^2

which is the `O(1/<sgn>^2)` cost of the sign problem written as a property of the weights.
`carriage` returns exactly that as `effective_n` -- verified at a ratio of `1.0000` across eight
`(n, <sgn>)` combinations -- and it is documented as *a property of the weights, not of the frames,
and the ceiling on what any aggregation of them can support*.

So the bound binds **every** read that aggregates signed configurations, this section's coupling
included. It is not a limitation of the instrument; it is the sign problem restated as a ceiling on
the ensemble. A route that could escape it must not aggregate signed configurations at all.

**The read is an onset detector and not a severity meter, and the two must not be conflated.**
Extrapolating the deficit outward is bounded by construction -- the deficit lies in `[0, 2]` while
`-ln<sgn>` is unbounded -- so no fixed map from one to the other can hold globally. Nor can such a
map be calibrated: on the rows where the deficit is useful, `<sgn>` is identically 1 and
`-ln<sgn>` is identically 0, so there is nothing there to calibrate against, and calibrating it
elsewhere would be the fitted constant this project does not permit.

**The reading is a synchronisation, and the sign of it is set by the mechanism.** A read that
returned `-1` wherever the model is sign-free would be indistinguishable from a read that had
learned one number, so it is put to the other place this model is provably sign-free. The two
places are structurally opposite. Repulsive `U` decoupled in the spin channel is sign-free at half
filling because `G_dn = 1 - G_up`: the channels are exact negatives, anti-synchronised. Attractive
`U` decoupled in the charge channel is sign-free at **any** filling because the field couples to
`n_up + n_dn - 1`, both spins see the identical diagonal factor, and the weight is `det^2`. Same
instrument, same call (`reads/expAE_both_mechanisms.py`, 2x4, `|U| = 4`, `beta = 10`, 300
configurations per seed, **six seeds per row reported as a range**):

| U | channel | mu | negative fraction | strength | `max|G_up - G_dn|` |
|---|---|---|---|---|---|
| +4 | spin | 0.0 | 0.0000 | **-1.0000 to -1.0000** | 3.1e+03 |
| +4 | spin | 0.4 | 0.0433 - 0.0800 | -0.3532 to 0.0000 | 1.9e+02 |
| +4 | spin | 0.8 | 0.0033 - 0.0133 | -0.8424 to -0.6143 | 8.4e+00 |
| -4 | charge | 0.0 | 0.0000 | **+1.0000 to +1.0000** | 0.0e+00 |
| -4 | charge | 0.4 | 0.0000 | **+1.0000 to +1.0000** | 0.0e+00 |
| -4 | charge | 0.8 | 0.0000 | **+1.0000 to +1.0000** | 0.0e+00 |

**The ranges carry the result.** Every saturated row is saturated on all six seeds to four
decimals, and the entire seed spread lies in the unsaturated rows -- the repulsive `mu = 0.4` row
runs from `-0.3532` to `0.0000`, a spread the size of the value. That asymmetry is what makes the
contrast readable. The replication also sets the scale for the controls: on the non-bipartite `3x4`
control one seed in six shows **no** negative weight in 300 draws, so a control row is read across
seeds and never from one.

**The last column is why the `+1` rows are not the evidence.** On the attractive rows the two
channels come out bit-identical, so the read is being handed the same array twice; `+1` is the
sign of the reading and says nothing about the instrument's discrimination. What the table
establishes is the **contrast**. Doping is exactly what takes the repulsive read from `-1.0000` to
unresolved, and it leaves the attractive read where it was -- because the attractive model stays
sign-free under doping and the repulsive one does not. So the read is not tracking filling, and it
is not tracking distance from a particle-hole symmetric point; it tracks whether the channels move
together, which is the property the sign actually depends on.

Saturation and a zero negative fraction are **not** equivalent. At `mu = 0.8, beta = 6` the
negative fraction runs `0.0000` to `0.0100` across seeds while the read runs `-0.8665` to
`-0.5358`: the departure from saturation exceeds the negative fraction on every seed, including
the seed where the negative fraction is exactly zero. That is §5's own result -- the read departs
before the average sign does -- appearing again rather than a contradiction of it.

No filling is estimated in that table. These are uniform draws over the field rather than
importance samples, so no expectation value taken from them carries information; the rows are
labelled by their inputs.

**The boundary of the reading, measured.** Both saturated cases above relate the channels
*affinely*, so "the read measures synchronisation" may be describing something narrower than it
sounds. The test is a mechanism where positivity comes from somewhere else. Wu and Zhang [6] and
Wei et al. [7] give one: with a spin-dependent twist the two spins become time-reversal partners,
`K_dn = conj(K_up)`, and with attractive `U` in the charge channel `B_dn = conj(B_up)`, so the
weight is `|det(I + B_up)|^2 >= 0`. Positivity here is *conjugate pairing*, not an affine relation.
The control applies the **same** twist to both spins: time reversal is gone, the weight becomes
`det^2` and acquires a phase, with everything else identical
(`reads/expAF_third_mechanism.py`, attractive `|U| = 4`, `beta = 8`, 200 draws x 5 seeds):

| Lx x Ly | phi/pi | T-rev | `G_dn = conj(G_up)` | `max\|Im w / Re w\|` | negative fraction | strength over seeds | Re only | Im only |
|---|---|---|---|---|---|---|---|---|
| 4x3 | 0.0625 | yes | **0.00e+00** | **0** | **0.0000** | 0.8575 to 0.9445 | +1.0000 | -1.0000 |
| 4x3 | 0.0625 | no | 4.49e+00 | 3.3e+02 | 0.0400 | **1.0000 to 1.0000** | +1.0000 | +1.0000 |
| 4x3 | 0.1875 | yes | **0.00e+00** | **0** | **0.0000** | 0.4889 to 0.7346 | +1.0000 | -1.0000 |
| 4x3 | 0.1875 | no | 6.84e+00 | 2.7e+02 | 0.3750 | **1.0000 to 1.0000** | +1.0000 | +1.0000 |
| 3x4 | 0.1250 | yes | **0.00e+00** | **0** | **0.0000** | 0.0649 to 0.7039 | +1.0000 | -1.0000 |
| 3x4 | 0.1250 | no | 1.58e+01 | 7.8e+02 | 0.2650 | **1.0000 to 1.0000** | +1.0000 | +1.0000 |
| 5x2 | 0.1000 | yes | **0.00e+00** | **0** | **0.0000** | 0.6201 to 0.8618 | +1.0000 | -1.0000 |
| 5x2 | 0.1000 | no | 6.35e+00 | 1.4e+03 | 0.1400 | **1.0000 to 1.0000** | +1.0000 | +1.0000 |

**This falsifies "saturated if and only if sign-free" in both directions.** The time-reversal rows
are sign-free -- the conjugation is exact, the weight's imaginary part is identically zero, the
negative fraction is 0.0000 -- and the read does **not** saturate: 0.06 to 0.94. The control rows
carry a real sign problem, up to 37.5% negative weight, and the read is **exactly 1.0000** on every
seed and every geometry.

Both follow from the same thing. Under time reversal the relation between the channels is
deterministic but carries *opposite signs on two orthogonal subspaces* -- `+1` on the real part,
`-1` on the imaginary part -- and one signed scalar averages them; read blockwise, the mechanism is
fully visible and exactly saturated. In the control the channels are bit-identical, so the read
saturates on a degenerate input, and that model's sign problem lives in a **phase common to both
channels**, which no comparison *between* the channels can reach.

### 5.2 Where the comparison stops: a phase common to both channels

**So the correspondence between saturation and sign-freedom is a statement about real
determinantal weights**, whose sign is a product of two determinant signs. That is the setting
every row of §§2-4 was measured in, and it is where that correspondence holds.

It is not where the reading stops, and the distinction matters. A complex weight is unreachable by
a **comparison** between the two channels -- for a reason given below, that its phase is common to
them -- and is read exactly by a **one-sided** read of the weight itself. What the two-mode
diagnosis further separates is whether that phase is global, in which case it cancels in
`<O> = sum(O w)/sum(w)` and costs nothing -- leaving a real weight whose own sign problem
`resultant` then measures -- or configuration-dependent, in which case it is the problem itself and
no rotation reaches it.

**The Kramers reading is the same arithmetic as §4's route A, with one sign changed.** There
`G_dn = conj(G_up)`, so the real parts are exact positives and the imaginary parts exact negatives,
and the read returns

    strength = +1 - 2r

with `r` again the imaginary share of the centred variance. Verified to **2e-16** across five
geometries and two seeds each, `r` running from `0.025` to `0.423` and the read from `0.949` to
`0.154`. The spread across seeds in the table above is `r`'s spread, not noise in the instrument.

So the three mechanisms this paper measures are one formula. Route B gives `G_dn = 1 - G_up` and
`r = 0`, hence `-1` exactly; route A gives `1 - conj(G_up)` and `-1 + 2r`; Kramers gives
`conj(G_up)` and `+1 - 2r`. A frame carrying imaginary weight cannot read `+-1` on a single signed
alignment, and how far it falls short is that weight and nothing else.

That is why saturation is not necessary for sign-freedom, and it is a sharper statement than the
tables alone: the Kramers rows are sign-free and read `0.59` **because** their frames are 20%
imaginary, not because the reading has failed on them.

**Is the boundary intrinsic, or an artifact of reading one scalar?** Two things were checked, and
they answer in opposite directions (`reads/expAH_boundary_is_intrinsic.py`).

*The `+1 / -1` split needed the mechanism to be known.* Handing the complex frame to the read
directly reproduces the concatenated values -- the library reduces a complex frame through the same
real embedding `iota(x) = (Re x, Im x)` -- and the experiment asserts that agreement rather than
assuming it. The read's `phase` field is `0.0101` and `0.2120` on the two Kramers rows, so the
alignment is not purely real and the signed real part is not the whole of it; but a phase of
`0.2120` rad inflates the magnitude by `1/cos(phase) = 1.023`, taking `0.4794` to `0.490`. The
unsaturated reading is not a magnitude hidden behind a sign. Asking the instrument for the
directions itself,
via `principal_directions` on channel A, does **not** recover the split: the per-direction couplings
come out `-0.462, +0.854` and `-0.509, --, +0.232`, not `+1` and `-1`. Those are the directions of
A's own variance, not the eigendirections of the A-to-B relation. So the blockwise saturation of the
previous table was recovered using knowledge of the mechanism, and a mechanism-agnostic application
of this read stays unsaturated.

**Why the relation reads are blind to a phase: it is a COMMON MODE.** Placing the two
determinants on the screen rather than the two Green's functions shows what is happening. On every
route B row the unit-modulus determinants couple at `+1.000` with `z ~ 21` -- their phases are
perfectly aligned. The weight is `det_up * det_dn`, so its phase is twice a phase the two channels
SHARE, and it varies configuration to configuration. Two aligned quantities moving together are
invisible to any comparison between them, which is what every two-sided read reported.

### 5.3 The triage: which problem is it?

**The triage recovers §4's positivity verdict from the weights alone**
(`reads/expAV_monomial_conjugation.py`). §4 establishes from `K` that route B buys the identity
*without* positivity. That same verdict is readable from the output, where an observer actually
stands -- holding a finite sample of weights and no knowledge of the mechanism. On four route-B
lattices the identity holds to machine precision, so every comparison between the channels is
saturated and blind:

| lattice | §3 identity residual | `focus` | `\|Im\|` after de-rotation | sign deficit |
|---|---|---|---|---|
| ring 5, flux `pi/2` | `1.42e-14` | 0.75254 | 1.000 | 0.10998 |
| ring 7, flux `pi/2` | `2.49e-14` | 0.83339 | 0.999 | 0.08482 |
| tri ladder 6, flux `pi/2` | `3.77e-14` | 0.54294 | 1.000 | 0.31001 |
| tri ladder 8, flux `pi/2` | `3.55e-14` | 0.57340 | 1.000 | 0.36442 |

Every row satisfies the identity and still carries a phase. `focus` below 1 says the cloud is not
rank one, so no global rotation makes those weights real -- which is §4's statement that route B
carries the identity without positivity, obtained with no `K` and no mechanism. Nothing here is
thresholded: `focus` is compared to 1, the definitional value of a rank-one cloud.

This is the case that separates the two halves of §5. The channel comparison of §5.1 is exactly
saturated on all four rows and reports nothing; the one-sided triage reads the answer off the
cloud's own geometry. A reader who takes only §5.1 from this section would conclude these lattices
are fine.

The same quantity read one-sidedly is available. On the weight's unit-modulus frame,
`concentration.resultant` -- the length of the mean row, the von Mises-Fisher sufficient statistic
-- **is** `|<w/|w|>|` by construction rather than by measurement, so its agreement is arithmetic
and not a finding. What it supplies is the value, on rows the comparison could not reach at all:

| K | `\|<phase>\|` | `resultant` |
|---|---|---|
| 2x4 clean | 1.00000 | **1.00000** |
| ring 6, flux `pi/4` | 1.00000 | **1.00000** |
| staggered `h = 0.6` | 0.93500 | **0.93500** |
| ring 5, flux `pi/2` | 0.84133 | **0.84133** |
| tri ladder, flux `pi/2` | 0.53586 | **0.53586** |

**And the companion read separates the two failure modes** (`reads/expAU_axial_versus_directional.py`).
`concentration` reports a DIRECTIONAL statistic and an AXIAL one, and they differ exactly where it
matters: an antipodal cloud reads `resultant ~ 0` and `focus ~ 1`. A real sign problem *is*
antipodal -- the weights sit at `+-1`, on one line through the origin -- while a phase problem is
spread around the circle. Measured, `focus` is **1.0000** on every real-weight row and `0.573` to
`0.833` on every genuinely complex one.

`focus = 1` is rank one in the `(Re, Im)` plane, which is the statement that **a single global
rotation makes every weight real** -- and that is actionable rather than descriptive. The rotation
is read off the cloud's own leading direction, no angle chosen, and applying it returns a real sign
problem carrying the same `|<sgn>|`: rows rotated by `0.7` and `1.9` present a maximum imaginary
part of `0.64` and `0.95`, de-rotate to `9e-16` and `4e-16`, and recover `0.93500` and `0.92500`
exactly. Where `focus < 1` no rotation helps and the residual stays at `1.0`.

**A global phase is harmless, and that is worth stating because it looks alarming.** A phase
common to every configuration multiplies numerator and denominator of `<O> = sum(O w)/sum(w)`
alike, so it cancels exactly and costs nothing. Measured: rotating the weights by `0.3` to `1.3`
leaves `|<w>|` at **0.93500** throughout, unmoved.

It is only visible to an estimator that takes the real part. `mean(Re w)/mean(|w|)` -- which is
correct for real weights and is the natural thing to carry over -- reads `cos(theta)` too small,
falling from `0.93500` to `0.25011` across the same rotations, and would overstate a cost that
goes as `1/<sgn>^2` by fourteen times. So `focus` is worth reading not because a global phase must
be removed but because it identifies when the real-part estimator has stopped being the right one.

*What that leaves, and it takes both numbers rather than one.* `focus` and `resultant` answer
different questions, and only the pair classifies. `focus = 1` says the cloud is rank one in the
`(Re, Im)` plane, so a single global rotation makes every weight real -- it does **not** say the
run is sign-free. What survives that rotation is a real problem whose severity is `resultant`:
`1.00000` on the clean lattice and on the inert twist, `0.93500` and `0.92500` on the staggered and
doped rows, which are sign problems and are exactly the ones §4's criterion speaks to. `focus < 1`
says no rotation reaches it and the phase varies configuration to configuration: it does not cancel
in the ratio, and it is the problem itself. So the three regimes are `(focus = 1, resultant = 1)`
sign-free, `(focus = 1, resultant < 1)` a real sign problem, and `focus < 1` a genuine phase
problem. The read separates a representation artefact from the physics, which is a diagnosis and
not a saving.

**The sign problem is the rank-one case of a phase problem.** Rank one in the plane is the
statement that the phase takes exactly two values `pi` apart -- a `Z2` subgroup of `U(1)` -- and a
phase confined to `Z2` is what a sign *is*. So `focus` reads the rank, the rank decides which of
the two a run has, and both are read from the weights alone with no constant supplied and no
Hamiltonian consulted.

So a caller with complex weights can ask which problem they have before deciding what to do about
it: `focus = 1` says a phase that a rotation removes, leaving the sign problem §4's criterion
speaks to; `focus < 1` says a phase that no rotation removes, which is what route B produces.

So the phase is readable, and the blindness above is a statement about *comparisons* rather than
about what can be measured. The two halves fit together: a common mode is invisible to a read of the
relation and visible to a read of one side. And what the one-sided read returns is `|<sgn>|` itself,
so it carries the ceiling of the paragraph below -- seeing the phase exactly and paying the same
`O(1/<sgn>^2)` to resolve it.

**A second collision, and this one has the channels exactly related rather than identical.** §4's
route B restores the identity on an odd cycle by threading it with flux, and the two channels then
satisfy `G_dn = 1 - G_up` to machine precision -- while the weight carries a phase. Three systems
whose channel relation is exact and whose sign behaviour is not
(`reads/expAO_spectral_criterion.py`, `beta = 8`, 250 draws; the deficit is `1 - |<w/|w|>|`):

| K | relation residual | sign deficit | coupling | Screen `match` |
|---|---|---|---|---|
| 2x4 real, half filled | 6.0e-10 | **0.00000** | -1.0000 | 1.0000 |
| ring 5, flux `pi/2` | **5.3e-15** | **0.15581** | -1.0000 | 1.0000 |
| ring 7, flux `pi/2` | **4.3e-15** | **0.09241** | -1.0000 | 1.0000 |
| triangular ladder, flux `pi/2` | **2.8e-14** | **0.47773** | -1.0000 | 1.0000 |

The relation is exact on every row, both reads return the same saturated value on every row, and
the sign deficit runs from `0` to `0.478`. **No read of the relation between the two channels can
see a phase problem**, because the relation does not carry it: `G_dn = 1 - G_up` holds identically
whether the weight is positive or spread across the circle.

That is a second impossibility of the same kind as the one below and it is not the same statement.
Below, the two channels are identical *as data*, so nothing remains to compare. Here they are
distinct and exactly related, and the comparison is well posed and still blind. The screen's
`match` was checked alongside the coupling for this reason -- it is better behaved on the route A
cases, returning `0.9994` where the coupling falls to `-0.9453` for reasons unconnected to the sign
-- and on these rows it is blind in exactly the same way.

*The other direction is an impossibility, not a limitation.* In the control the two channels are
bit-identical, and three systems make that concrete -- `numpy.array_equal(A, B)` is `True` in every
row:

| system | `B == A` | negative fraction | strength | z | resolved |
|---|---|---|---|---|---|
| attractive, no flux | **True** | **0.0000** | 1.0000 | 54.5 | yes |
| control, `phi = pi/16` | **True** | **0.0300** | 1.0000 | 57.1 | yes |
| control, `phi = 3pi/16` | **True** | **0.3333** | 1.0000 | 60.3 | yes |

The read returns the same saturated value on all three while the negative fraction spans 33
percentage points, so its output does not determine the sign behaviour. And when `B = A` the pair
carries no between-channel content whatsoever: every statistic of the pair collapses to a statistic
of one channel. That is a limit on **any** comparison between channels, not on this read. The
control's sign problem sits in the phase of `det(I + B_up)`, a global property of the determinant,
while the frame the read is given is the Green's function's diagonal. Whether some read of a single
channel could recover that phase is not measured here and is not claimed either way.

*Which twists are live.* The total phase round the x-cycle is `Lx * phi`, so a twist that is a
multiple of `pi` is gauge-equivalent to a real boundary condition and the determinants stay real;
on `Lx = 4` that makes `pi/4` and `pi/2` inert. A periodic lattice of width 2 is inert at every
`phi`, because its x-direction has one distinct bond per row reached from both ends and the two
Peierls phases add to `-2t cos(phi)`. Every row therefore reports `max |Im G|`, and the width is
asserted, so a row cannot be read unless its flux is doing something.

Three quantities have now been checked for whether they measure severity -- the identity residual
(§4), the coupling strength, and the coupling tightness -- and none does. The severity is the
negative fraction, and the negative fraction is a count of parity disagreements. §7 gives the
reason no continuous summary reaches it.

## 6. The conditioned frame is not optional

Every number above is a determinant sign, and a determinant sign has to be read in a frame that
stays conditioned. `I + B` does not. Its condition number grows like `exp(beta * bandwidth)`, and
past `beta ~ 4` its eigenvalues stop meaning anything.

This is not a caution, it is a measured failure. Forming `B` as the plain ordered product and
reading its spectrum reports, at **half filling on a bipartite lattice where positivity is
provable**:

| beta | 2 | 4 | 6 | 8 |
|---|---|---|---|---|
| negative fraction, naive product | 0.0000 | 0.0000 | 0.0225 | **0.5075** |
| negative fraction, conditioned frame | 0.0000 | 0.0000 | **0.0000** | **0.0000** |

with "distances to a sign flip" of 56, 1117 and 31683 for a quantity that is O(1). The naive frame
does not merely lose precision; it manufactures the phenomenon the paper is about.

The conditioned frame comes from the UDT factorisation. With `I + U D T = U Db M T` and
`prod(Db) > 0`, the sign is carried entirely by `det(U) det(T) det(M)`, and `M` is bounded by
construction because `Db >= 1` divides the large block and `Ds <= 1` is the small one.

Three properties are asserted as gates rather than assumed (`tests/test_conditioned_frame.py`):

* **two independent routes agree.** The block formulation and the core factorisation reach the
  sign by different arithmetic and agree on every configuration measured. The two routes must be
  arithmetically independent for this to mean anything: a parity count compared against
  `sign(prod(1 + lambda))` from the *same* eigenvalues agrees whatever those eigenvalues are.
* **the answer does not move with the stabilisation block.** Every configuration's sign is
  compared across three block sizes, not just the totals, so agreement cannot come from
  cancellation.
* **the naive frame FAILS.** Without this the suite would pass equally on a build that had
  silently reverted, by being a different and wrong calculation.

The third is the one that matters. This frame was lost and re-derived three times during the work,
in three different files.

## 7. What does not carry the sign

The sign is a per-configuration fact, and a per-configuration order parameter would be worth more
than an ensemble one. Three candidates were derived and tested. All three fail, and they fail for
the same reason.

| candidate | what it is | why it fails |
|---|---|---|
| `max_i \|G_up[i,i] + G_dn[i,i] - 1\|` | the residual of the particle-hole identity | a coarse summary of the object carrying the sign. Median 1.1x to 6.4x higher on negative-weight configurations, but positive-weight ones reach 190 while negative ones go down to 0.22 |
| `min_k \|lambda_k(M)\|` | the distance to a sign flip, in the conditioned frame | a **distance is sign-blind**. Negative-weight configurations sit 2 to 8x closer to a zero in median, and the distributions overlap completely: a configuration can approach a crossing without making it, or cross and travel far past |
| the §3 identity residual | the exact failure of an exact relation | a statement about **magnitudes**. Median ratio 0.74 to 1.10 -- no discrimination at all |

The reason is structural. `det(I + B) = prod_k (1 + lambda_k)`, so the sign is the **parity** of the
number of eigenvalues past the crossing. A parity is a global property of the whole spectrum, and
no scalar summary of a spectrum determines its parity. The §3 identity *holding* forces the signs
to lock; the identity *failing* only permits them to differ, and says nothing about which
configurations do.

It is the same wall §8 meets from the other side, and it is why the constrained path's node is
hard: what has to be predicted is a discrete global invariant, and the available cheap observables
are continuous local ones.

**The parity argument does not reach a complex weight, and the sweep that shows it says something
larger.** A parity is not continuous; a *phase* is, so the paragraph above is an argument about
real weights only. The model carries the instrument for testing that: both
`(n_up - 1/2)(n_dn - 1/2) = 1/4 - m^2/2` and `= rho^2/2 - 1/4` are exact identities on the four
states of a site, so splitting `U = (1-theta)U + theta U` and decoupling each piece in its own
channel gives a family in which **every theta is the same physics** -- the model's own gate --
while `theta = 0` is real and `theta = 1` is complex
(`reads/expAI_phase_is_not_a_parity.py`, 2x4, `U = 4`, `beta = 4`, 400 prior draws x 4 seeds):

| mu | theta | mean `\|Im w\| / \|w\|` | `1 - \|<w/\|w\|>\|` | neg fraction of `Re w` | strength over seeds |
|---|---|---|---|---|---|
| 0.0 | 0.00 | 0.0e+00 | **0.0000** | **0.0000** | **-1.0000 to -1.0000** |
| 0.0 | 0.50 | 3.1e-15 | **0.0000** | **0.0000** | -0.1725 to 0.0653 |
| 0.0 | 1.00 | 3.0e-14 | **0.0000** | **0.0000** | **+1.0000 to +1.0000** |
| 0.8 | 0.00 | 0.0e+00 | 0.0000 | 0.0006 | -0.8956 to -0.8895 |
| 0.8 | 0.50 | 6.4e-01 | 0.7159 | 0.4050 | -0.0790 to -0.0458 |
| 0.8 | 1.00 | 6.4e-01 | 0.6460 | 0.4956 | **+1.0000 to +1.0000** |

*The two complex-weight columns, and why neither is a count.* `|Im w| / |w|` is bounded by 1 and is
a share of the weight's magnitude; a ratio to `Re w` instead would be unbounded and would diverge
wherever `Re w` passes near zero, which is exactly what a weight crossing between positive and
negative does. `1 - |<w/|w|>|` is the mean-phase deficit, and it is zero when the weights are real
and positive whatever their spread in magnitude.

A *fraction of draws carrying a phase* stood in the second column until 2026-09-07 and has been
removed rather than re-cut, because that quantity has no cut-free form at finite precision. Counting
the draws whose argument is neither near `0` nor near `pi` requires an angle for "near" -- the value
used was `0.1` radians, chosen -- and it undercounted, reading `0.9350` where every draw in the row
carries a phase. Removing the window does not repair it: `Im w != 0` reads `0.9925` to `1.0000` on
the `mu = 0` rows, whose weights are real, because at that point it is counting floating-point dust
at `1e-15`. The two columns above need no window and separate the same two populations by fourteen
orders of magnitude, so the claim is made with them.

**The phase is not a function of the field either, and this is asked on the whole field.** §3 gives
the magnitude ratio exactly from the field,
`ln|det_up| - ln|det_dn| = -dtau L tr(K) + lambda sum(x)`, so the natural question for a complex
weight is whether `arg(w)` has an analogous relation. It matters more than most negatives here: if
it did, the phase could be computed without the determinant and the sign problem would not be a
sampling problem at all.

Three chosen functionals -- `exp(i sum x)`, the staggered field sum, the nearest-slice product --
return `z` between `-1.3` and `+1.8`, none resolved. But three functionals is a guess at a basis,
and a null there says only that those three do not carry it. So the question is put again on the
LARGEST linear basis available (`reads/expAX_is_the_phase_a_read_of_the_field.py`): the frame is
every one of the `L x N` Ising variables per configuration, nothing selected and nothing
summarised, read by `carriage` -- whose null is the exact re-pairing of weights with frames --
against the phase's two components. Any functional the paragraph above could have chosen is a
vector in that basis, so this subsumes them.

| | `carried` | against its own null |
|---|---|---|
| `ln\|det_up\| - ln\|det_dn\|` (control) | `1.95` to `2.68` | **`+12.1` to `+22.9` sigma** |
| `cos arg w` | `0.83` to `1.16` | `-1.8` to `+2.1` sigma |
| `sin arg w` | `0.84` to `1.13` | `-1.7` to `+1.4` sigma |

`carried` is `1` when the weights carry nothing. Every phase reading sits inside the null's own
observed range across six lattices, while the magnitude control -- read on the *same frames*, in
the same call -- stands twelve to twenty-three sigma clear of it. The per-coordinate `resolved`
counts are not the statistic and are not quoted: at `far = 0.05` over 384 coordinates about 19
clear the level by chance, which is the range those counts come out in.

The phase behaves as the sign does -- the magnitudes are given by a closed form and the argument
carries no linear information from the field that this instrument resolves on the full basis. What
this does not decide, and does not claim: a nonlinear function of the field is outside a linear
alignment's reach, and a relation below the instrument's resolution at 400 configurations would
not register.

**The `mu = 0` block is the result.** Same Hamiltonian, same physics, weights real and positive at
every theta -- there is no sign problem anywhere in that block, charge channel included -- and the
read runs the full range from `-1.0000` to `+1.0000`. A quantity that moves that far while both the
model and its sign structure hold still is not a reading of the model. **The read is a property of
a (Hamiltonian, decoupling) pair.** That is not a defect, because the sign problem is a property of
the same pair; it does mean the read cannot be quoted about a Hamiltonian without saying how it was
decoupled, and every claim in §§2-5 is therefore a claim about the *spin* decoupling named there.

The doped block supplies the complex weights the parity argument does not reach: the weights are
about 60% imaginary by magnitude -- `mean |Im w| / |w|` runs `0.568` to `0.647` there against
`3e-15` on the half-filled rows of the same family -- so there the sign problem is a *phase*
problem. No *fraction* of phase-carrying draws is quoted, because that quantity has no cut-free
form. A windowed count needs an angle to decide when an argument is "near" 0 or `pi`, and removing
the window does not fix it: `Im w != 0` reads `0.9925`-`1.0000` on the half-filled rows, whose
weights are real, because it is counting floating-point dust at `1e-15`. The share above and the
mean-phase deficit need no window and separate the same two populations by fourteen orders. The read does not reach it either -- at `theta = 1` it is saturated
at exactly `+1.0000` while 49.6% of draws have negative real part, the collision of §5 again, since
`theta = 1` sets `lambda_s = 0` and the two channels coincide.

**And in the interior of the family there is no lockstep at all.** Measuring what actually delivers
positivity at each theta (`reads/expAJ_how_positivity_arises.py`, half filling, 300 draws x 3
seeds): at `theta = 0` the determinants are real and their signs agree on **1.0000** of
configurations -- the mechanism of §§2-3. At `theta = 1` they agree on `1.0000` trivially, because
`lambda_s = 0` makes them the *same number*. In between, at `theta = 0.25, 0.50, 0.75`, the two
determinants' real parts agree on **0.5167, 0.5178, 0.5289** of configurations -- chance -- while
the weight is exactly real and positive on every draw. The lockstep is absent there and the model
is sign-free anyway. That is the mechanical reason the read decays across the family: it compares
the two channels, and in the interior the channels are not what carries positivity.

*One caution about that experiment.* The quantity `arg(charge_phase) + arg(det_up) + arg(det_dn)`
is `arg(w)`, so its vanishing is positivity restated rather than an independent invariant, and its
*maximum* over draws is not a summary of anything: a maximum of `pi` means one draw was
negative-real, not that all draws were real. Reading it as the latter briefly suggested the family
had no complex weights at all, which the distribution above refutes.

*No reweighted expectation is quoted from that table.* Fields are drawn from the prior and the
effective sample size is 1.5 to 13 of 400, which supports none. Every column is a property of the
prior ensemble: the negative fraction of `Re w` among the draws, the typical size of `Im w` against
`Re w`, and the coupling of the two channels over those draws. The §5 coupling along beta is a
different object, measured on importance-sampled chains, and the two are not compared here.

*Where the complex weights are.* Turning the decoupling toward the charge channel does not by
itself produce them: at half filling the whole family is sign-free, the charge channel included,
with the weight real to 1e-15 and no negative real part at any theta. Complex weights appear under
doping, which is why the table carries both blocks.

## 8. What any read of the sign must contend with

Each result below is a structural constraint on what can be read from sampled data, measured rather
than argued, and together they are why §5 reads a coupling between two channels and claims onset
rather than severity.

**8.1 Reads that score severity from the sampled data.** Every one reduces to the average sign
under a different, cheaper measure. The anchor is where the reduction can be written down and
checked (`closed/expAQ_anchor_reduces_to_the_sign.py`): with `B = diag(sigma) A`, the read returns
the average of `sigma` **reweighted by each row's squared length**, agreeing with it to `8e-05` to
`1.3e-03` across five draws. The residual is the centring the read performs and the plain average
does not; against its own definition on the centred frames the read is exact, which is arithmetic
rather than a finding. So a read of this kind does not fail to see the sign -- it sees a reweighted
one. They fail for one reason: the sign correlates
with `|D|`, since negative weights cluster near the zeros of the determinant, so any weighting that
is not `|D|`-weighted mis-weights exactly the configurations that decide the answer.

**8.2 A magnitude read cannot see a sign, for any exponent**
(`closed/expAP_phase_blindness.py`). With `B = diag(sigma) A` the magnitudes agree cell for cell,
so `sum |A|^q` and `sum |B|^q` agree for **every** `q` -- measured at `q = 0.5, 1, 2, 3` across
three frames differing by row sign flips, with a worst disagreement of **exactly 0.000e+00**, not a
tolerance. The singular spectrum is blind for the same reason, `diag(sigma)` being orthogonal:
maximum singular-value difference `0.000e+00`. The file carries its own positive control, and it is
what makes the blindness a property of magnitude reads rather than of the frames -- the coupling
*between* the two sides returns `+1.0000`, `0.0000` and `-1.0000` on the same three frames. That is
why §5 reads a coupling and not a magnitude: the sign of a determinantal weight lives in the
relation between the channels, not in the size of either.

**8.2b What can and cannot see the sign, classified.** Three families of read were put to the
question, and the classification is now complete rather than a list of attempts.

| family | status |
|---|---|
| aggregation of signed configurations | **capped** at `n <sgn>^2` -- §5, proved from Kish's effective sample size |
| a read of the relation between the two channels | **blind** -- §5, twice: identical channels, and exactly related channels |
| a read of one channel alone | **measured, no indicator** |

The third is the one neither impossibility rules out, so it was measured rather than argued. Along
a continuous flux sweep of the frustrated ladder the sign deficit runs `0.087` to `0.462` and back,
and the single-channel spectral reads do not follow it: `phase` against the deficit gives a rank
correlation of `+0.23` at `p = 0.49` over eleven points, and it is not even monotone -- a deficit of
`0.124` shows a larger phase departure than one of `0.222`. `attenuation`, `top_share` and
`dominance` fail to separate a sign-free row (`ring 6` at flux `pi/4`, deficit `0.00000`,
attenuation `0.44`) from a row with a real problem (staggered, deficit `0.060`, attenuation `0.56`).

What survives all three is identification of an operator from a correlation sequence, which
aggregates no signed configuration and reads no channel relation. That is the family §9 points at.

**8.3 The scalar decoupling family is complete** (`model2d.single_site_identity`,
`reads/expAI_phase_is_not_a_parity.py`). Using `n^2 = n`, the interaction has exactly two
quadratic forms, with the `m^2` coefficient pinned at `-U/2` and the `rho^2` coefficient at `+U/2`
for any rewriting; the "shift around alpha" freedom is one-body and is absorbed into the hopping
matrix. Both rewritings are checked by quadrature on a single site at every mixing, worst relative
error **3.6e-16**. So splitting `U` between the spin and charge channels is the whole family, and it
was swept: at `mu = 0.8` the negative fraction runs `0.0006`, `0.244`, `0.405`, `0.463`, `0.496`
across `theta = 0` to `1`, monotone over four seeds a row, so the spin channel wins with **no
interior optimum**. The SU(2) vector decoupling, which
lies outside the family, was derived and built and is 2.2x worse at 2x2 and unresolvable at 4x4.

**8.4 Contour deformation buys nothing in two dimensions** (`closed/expAR_contour_2d.py`).
Deforming the integration contour is the one route that can change the exponent rather than the
prefactor. Measured on `4x4`, `mu = 1.0`, `U = 6`, `beta = 6`, against an undeformed mean phase of
**0.4167 +- 0.0554** re-measured in the same run:

| shape | c | `\|<phase>\|` | gain |
|---|---|---|---|
| uniform | 0.10 | 0.0898 +- 0.0248 | 0.22 |
| uniform | 1.00 | 0.0626 +- 0.0249 | 0.15 |
| uniform | **4.00** | 0.0451 +- 0.0278 | **0.11** |
| staggered | 0.10 | 0.0334 +- 0.0307 | 0.08 |
| staggered | 1.00 | 0.0966 +- 0.0256 | 0.23 |
| staggered | **4.00** | 0.0647 +- 0.0260 | **0.16** |

Two shift shapes, neither fitted: a uniform `c`, which is the one-dimensional deformation carried
over, and one staggered by sublattice, which is the model's own structure and which a uniform shift
cannot see. Every gain is below 1 -- the deformation makes the phase problem worse at every
amplitude tried.

The scan runs to `c = 4`, far past any plausible optimum, and that is the part that makes this a
measurement rather than a failure to look: a thimble need not lie near the real axis, so a second
regime at large shift is what would have falsified the reading. There is none.

*The choice of point is load-bearing.* A shift can only help where there is something to help with,
and at `mu = 0` this lattice is at half filling on a bipartite graph -- the sign-free point of §4 --
where the undeformed phase is `1.0000` and every shift can only spoil it. The point above was
selected by measuring undeformed phases first, and the undeformed value is printed beside the scan
so a reader can see there was a problem before reading whether anything fixed it.

**8.5 Complex Langevin** (`closed/expCC_cl_fails.py`, `closed/expDD_cl_reads.py`,
`closed/expEE_no_error_bar.py`, `closed/expJJ_seed_spread.py`). Built and gated: the analytic drift against a finite difference of the
complexified action at 8.4e-10, with the wrong-convention control at 0.40. It converges to the
wrong answer here -- the density is off by 2% at `z ~ 28` against a spin-channel control exact to
1e-4. The uncertainty on that figure is estimated from reproducibility across seeds; a within-run
formula and a seed-to-seed spread differ by more than an order of magnitude on this data, and the
seed spread is the one that holds.

**8.6 The constrained path's trial wavefunction** (`closed/expW_trial.py`,
`closed/expX_dial.py`, `closed/expKK_multidet_bias.py`, `closed/expLL_unfitted_trial.py`,
`closed/expMM_self_consistent.py`, `closed/expNN_selection.py`). The bias is set by the node and by nothing else,
and it moves by a factor of 24 with the trial at fixed coupling. But the computable criterion
points the wrong way: `argmin <Psi_T|H|Psi_T>` is anti-correlated with the bias over a
nine-candidate pool (`rho = -0.80`, `p = 0.0096`). Fitted to the exact ground state, four to six
non-orthogonal determinants cut the bias by `1.7x` to `10.4x` depending on the coupling -- so the
prize is real -- and **no answer-free construction beat the plain free determinant**, every
candidate coming in 5 to 15x worse. What blocks it is not generating determinants but that the only
objective which selects them correctly is the one that needs the answer.

*The ceiling is not monotone in `k`, which is why the range is quoted rather than a single factor.*
Against the single-determinant walk at the same coupling (`closed/expKK_multidet_bias.py`, bias
relative to `k = 1`, three seeds):

| U | k = 2 | k = 4 | k = 6 |
|---|---|---|---|
| 4.0 | 0.810 | 0.327 | 0.368 |
| 8.0 | 1.119 | 0.530 | 0.096 |
| 12.0 | **1.188** | 0.574 | 0.212 |

**The second determinant moves the bias in opposite directions at different couplings, and only
two of those moves are resolved by the seed spread.** At `U = 4` two determinants are better than
one by `4.4` standard errors; at `U = 12` they are *worse* by `2.7`; at `U = 8` the difference is
`0.68` and is consistent with nothing. The `k = 6` against `k = 4` gap at `U = 4` is `0.64` and is
likewise unresolved.

So the shape of the ceiling is: four to six determinants beat one at every coupling, by a factor
between about `1.7x` and about `9-10x`, and the path there is not monotone -- at strong coupling
the second determinant is a step backwards. A claim of the form "`k` determinants cut the bias by
a factor" needs both the `k` and the `U` attached.

*Why the upper end is quoted loosely.* `best_k_dets` fits the determinants by restarting a
non-convex optimisation and keeping the best. It is bit-exact within one process and does **not**
reproduce across them: the same call at `U = 8, k = 2` returns overlaps spanning `1.7e-7` over
three runs, and pinning BLAS to a single thread only halves that, so it is not merely thread
scheduling. A `1e-7` shift in which optimum is found moves the bias by more than the walker seed
spread -- three independent runs of `U = 8, k = 4` give `0.02912`, `0.03001` and `0.03251` against
a quoted `+-0.0027`, and the largest reduction reads `10.43x` on one full run and `9.00x` on
another. **The error bar recorded beside each entry understates it**, because it varies walker
seeds with the fit held fixed and has no term for the fit itself. The low end is stable across runs
(`1.74x` and `1.73x`); the high end is not, and is quoted as a band rather than a figure.

**8.7 `extract()` as an estimator of a mean** (`reads/expRR_natural_noise.py`). Scored on the
sign-corrected correlator against the raw block mean over nine rows -- three `(t2, mu, beta)` points
by three block sizes, 24 blocks each -- the error ratio runs `0.725` to `1.637` with a **median of
`1.117`**, and `extract` *beats* the raw mean on two of the nine. So it is roughly a wash with a
tail the wrong way, not a uniform loss:

| `t2`, `mu`, `beta` | per block | raw err | `extract` err | ratio |
|---|---|---|---|---|
| 0.7, 0.6, 2.0 | 2 / 8 / 32 | 0.00603 / 0.00586 / 0.00115 | 0.00663 / 0.00581 / 0.00188 | 1.099 / **0.991** / 1.637 |
| 0.7, 1.0, 2.0 | 2 / 8 / 32 | 0.00523 / 0.00119 / 0.00111 | 0.00621 / 0.00148 / 0.00080 | 1.187 / 1.242 / **0.725** |
| 0.7, 0.6, 3.0 | 2 / 8 / 32 | 0.01003 / 0.00361 / 0.00213 | 0.01120 / 0.00395 / 0.00265 | 1.117 / 1.095 / 1.243 |

The reason is structural rather than a tuning failure, but not the one an earlier draft gave. What
`extract` returns is `(clean, info)` with `clean` a shrunk projection of the data onto its own
resolved modes, **in `W`'s own units and carrying a mean of its own** -- `clean.mean(axis=0)` is
neither `W.mean(axis=0)` nor `info['centre']` exactly, differing from them by `2e-3` to `1.3e-2` on
these rows. Optimal singular-value shrinkage is the right operation for recovering a low-rank signal
and the wrong one for an unbiased mean, because it shrinks the mean-carrying mode along with
everything else. The sign problem's difficulty is entirely in a mean; `extract` denoises frames.
They are different tasks, and the median ratio above is the size of that mismatch on this problem.

## 9. Outlook

The reframing changes what a positivity-restoring representation has to achieve, and the change is
not cosmetic.

**Not avoidance -- synchrony.** A natural reading of a sign-free model is that its configurations
stay away from the boundary where a determinant vanishes. That is measurably false here, and the
way it fails is the evidence. At `beta = 10` and half filling the distance distribution has a bulk
that does not move and a tail that reaches the boundary: the median holds at **0.50 to 0.55** across
every sample size and seed, while the 1st percentile sits **25 to 77 times below it**, at `0.007` to
`0.020`. The smallest distance observed has no stable value at all -- it ranges over `0.0002` to
`0.012` across seeds at fixed sample size, and falls further as draws are added -- which is itself
the statement that no distance is being respected. Throughout, the negative fraction is exactly
**0.0000**.

So configurations reach arbitrarily close to the boundary, and it is a tail doing it rather than
the bulk drifting inward. The individual channels cross on around 12% of configurations. What the
symmetry supplies is not distance but **synchrony** -- the two channels cross together.

**And the criterion says what a construction has to achieve, and what it cannot.** §4 turns the
requirement into a decidable property of the one-body matrix: a diagonal unitary `S` with either
`S K S^-1 = -K` or `S K S^-1 = -conj(K)`. It is constructive on the hopping graph -- it names the flux that
opens a route where frustration closes every flux-free one -- but only route A leaves the weights
real, so on a graph with an odd cycle the identity can be restored and the positivity cannot.

It is also a **closed door on doping**, in one line. Every diagonal matrix commutes with every
diagonal conjugation, so `S K S^-1` leaves `diag(K)` untouched; both routes then demand
`K_ii = -K_ii` -- and `K_ii` is real because `K` is Hermitian -- so `diag(K) = 0` is forced. A
chemical potential can therefore never be accommodated, by any flux, on any graph, at any size.
Measured over 84 flux values across four graphs, the best identity residual at `mu = 0.2` is `6.4`
and at `mu = 0.6` is `19.3`, while every one of those graphs reaches `1e-14` at `mu = 0`
(`reads/expAO_spectral_criterion.py`).

So a construction seeking positivity away from half filling cannot get there by choosing a lattice,
a flux or a size: the obstruction is a diagonal term, and no conjugation of this kind removes one.
It does not need to push the spectrum away from anything either. It needs to reproduce a **pairing**
between two channels whose magnitudes are unrelated over 8 to 16 orders and whose crossings coincide
exactly. That is what the algebraic
positivity results -- Majorana, Kramers, split-orthogonal -- achieve when they achieve it, by
making the weight a squared modulus so the question cannot arise; and it is why a perturbative or
reweighting fix has nothing to expand in, since there is no small residual, only a discrete
coincidence that either holds or does not.

**What a read of this problem can carry, classified.** Three families were put to the question and
the classification is closed, not a list of things not yet tried. Each entry is established in the
section named, and the first two are proofs rather than measurements.

| family | status | where |
|---|---|---|
| aggregate signed configurations | **capped** at `n <sgn>^2` | §5 |
| read the relation between the two channels | **blind**, two ways | §5 |
| read one channel alone | **no indicator**, measured | §8.2b |
| identify an operator from a correlation sequence | not excluded by any of the above | -- |

The ceiling is the sign problem restated: correct importance sampling draws at `|w|`, so a
configuration enters any weighted average carrying only its sign, and Kish's effective sample size
of those weights is exactly `n <sgn>^2`. Because that number is a property of the **weights**, it
bounds every aggregation of them, which is why the order parameter of §5 detects onset and cannot
report severity -- not as a limitation of that read, but as a consequence of its being a read of
that kind.

The blindness is two distinct statements. Where the two channels coincide as data, no comparison
between them remains. Where they are distinct and **exactly related** -- §4's route B, at `5e-15` --
the comparison is well posed and still carries nothing, because the relation holds whether the
weight is positive or spread across the circle.

What survives is the last row, and it survives for a structural reason rather than by not having
been tried: identifying an operator from a correlation sequence aggregates no signed configuration
and compares no channels, so neither the ceiling nor the blindness applies to it. Read this as a
statement about where to look and not as a result: what it costs to identify such an operator from
data a sign problem actually permits is not settled here.

**One attempt on that route is recorded, and it failed** (`reads/expAY_order_from_the_instrument.py`).
The prize is concrete: a correlation sequence `C(tau) = sum_i c_i lambda_i^tau` is measured at short
`tau`, where the sign is still mild and sampling is cheap, and the eigenvalues it identifies fix the
operator that gives the long-time behaviour one would otherwise pay `1/<sgn>^2` to sample. The
bottleneck is **model order** -- how many `lambda_i` to keep -- and every standard answer is a
chosen number: an information criterion picks a penalty, a singular-value cut picks a level, a
stability window picks a width. Under this paper's rules none is admissible.

`spectral_optics` reports `resolved_modes`, a count against a floor derived from the data rather
than supplied, which is model-order selection performed by the read. It does not work here, and the
reason is structural rather than a floor set wrong: on the Hankel embedding of a sum of decaying
exponentials it returns **1 for every true order above 1**, at every noise level including none,
while the same matrix has exact numerical rank 2 and 3 with singular values an order of magnitude
apart (5.42 and 0.46 at order 2). The leading mode carries a `top_share` of 0.994 to 0.996 there:
the modes are real and wildly unequal, and a floor that separates signal from a noise sea is asking
a different question from "how many modes are there".

So the row stays open, and one route into it is now closed by measurement rather than left for a
reader to re-attempt.

**What the measurement can and cannot supply.** The order parameter of §5 gives an exactly
calibrated, continuously varying measure of distance from the protecting symmetry, readable at
137 sigma in a regime where the average sign is identically 1 and has no derivative. It does not
forecast severity: it is bounded in `[0, 2]` while `-ln<sgn>` is unbounded, and §7 gives the
reason no continuous summary reaches the severity -- the severity is a count of parity
disagreements, and a parity is a global property of a spectrum that no scalar determines.

**The open question this leaves.** Whether the synchrony can be restored by construction, rather
than measured. Every route in §8 that attempted it worked on the field, the contour, or the trial
wavefunction, and the identity of §3 says the object to work on is neither: it is the relation
between the two channels' determinants. Nothing in this work has attempted to build a
representation in which that relation is imposed rather than inherited from a symmetry.

## References

Every entry below was checked against the publisher's or arXiv's own record rather than recalled,
and each is annotated with where this paper leans on it.

**The method.**

[1] R. Blankenbecler, D. J. Scalapino and R. L. Sugar, *Monte Carlo calculations of coupled
boson-fermion systems. I*, Phys. Rev. D **24**, 2278-2286 (1981).
doi:10.1103/PhysRevD.24.2278 — the determinantal formulation this paper measures in.

[2] J. E. Hirsch, *Discrete Hubbard-Stratonovich transformation for fermion lattice models*,
Phys. Rev. B **28**, 4059(R) (1983). doi:10.1103/PhysRevB.28.4059 — the decoupling whose own
constant `lambda = arccosh(exp(dtau U / 2))` is the coefficient in §3's identity, and whose two
channels (spin and charge) are the two mechanisms compared in §5.

[3] Z. Bai, C.-R. Lee, R.-C. Li and S. Xu, *Stable solutions of linear systems involving long
chain of matrix multiplications*, Linear Algebra Appl. **435**, 659-673 (2011) — the conditioning
problem of §6, and why `I + B` has to be inverted through a stratified factorisation rather than
formed.

**The sign problem.**

[4] E. Y. Loh Jr., J. E. Gubernatis, R. T. Scalettar, S. R. White, D. J. Scalapino and R. L.
Sugar, *Sign problem in the numerical simulation of many-electron systems*, Phys. Rev. B **41**,
9301 (1990). doi:10.1103/PhysRevB.41.9301 — the exponential decay of the average sign, and the
paper's own statement of the exception this one is about: the decay holds *"unless the measure is
forced to be positive by an explicit symmetry."*

[5] M. Troyer and U.-J. Wiese, *Computational complexity and fundamental limitations to fermionic
quantum Monte Carlo simulations*, Phys. Rev. Lett. **94**, 170201 (2005).
doi:10.1103/PhysRevLett.94.170201; arXiv:cond-mat/0408370 — NP-hardness of the general problem,
which is why §9 claims a reading and not a solution.

**Where positivity comes from.**

[6] C. Wu and S.-C. Zhang, *Sufficient condition for absence of the sign problem in the fermionic
quantum Monte Carlo algorithm*, Phys. Rev. B **71**, 155115 (2005).
doi:10.1103/PhysRevB.71.155115; arXiv:cond-mat/0407272 — positivity from the eigenvalues of the
fermion matrix pairing into complex conjugates, tied to time-reversal symmetry.

[7] Z. C. Wei, C. Wu, Y. Li, S. Zhang and T. Xiang, *Majorana positivity and the fermion sign
problem of quantum Monte Carlo simulations*, Phys. Rev. Lett. **116**, 250601 (2016).
doi:10.1103/PhysRevLett.116.250601; arXiv:1601.01994 — a unified account of the known sign-free
models, and the reason §4's scope is stated as a measured boundary rather than a general one:
sign-free models exist *with repulsive interactions and without particle-hole symmetry*, so the
symmetric point this paper reads is one place positivity comes from and not the only one.

[8] Z.-X. Li, Y.-F. Jiang and H. Yao, *Majorana-time-reversal symmetries: a fundamental principle
for sign-problem-free quantum Monte Carlo simulations*, Phys. Rev. Lett. **117**, 267002 (2016).
doi:10.1103/PhysRevLett.117.267002; arXiv:1601.05780 — the classification into Majorana and
Kramers classes.

**Routes measured and closed (§8).**

[9] S. Zhang, J. Carlson and J. E. Gubernatis, *Constrained path Monte Carlo method for fermion
ground states*, Phys. Rev. B **55**, 7464-7477 (1997). doi:10.1103/PhysRevB.55.7464 — the
constrained-path method benchmarked in §8.

[10] S. Zhang and H. Krakauer, *Quantum Monte Carlo method using phase-free random walks with
Slater determinants*, Phys. Rev. Lett. **90**, 136401 (2003).
doi:10.1103/PhysRevLett.90.136401; arXiv:cond-mat/0208340 — the phase-free extension.

[11] S. Chandrasekharan and U.-J. Wiese, *Meron-cluster solution of fermion sign problems*,
Phys. Rev. Lett. **83**, 3116-3119 (1999). doi:10.1103/PhysRevLett.83.3116 — a solution that works
by making the cancelling configurations cancel exactly, rather than by estimating their residue.

[12] G. Parisi, *On complex probabilities*, Phys. Lett. B **131**, 393-395 (1983) — stochastic
quantisation for complex actions.

[13] G. Aarts, *Can stochastic quantization evade the sign problem? The relativistic Bose gas at
finite chemical potential*, Phys. Rev. Lett. **102**, 131601 (2009).
doi:10.1103/PhysRevLett.102.131601 — the complex-Langevin route attempted in §8.

**The instrument.**

[E] Entroptics. `reads.coupling`, `Aperture.extract`, and the exact re-pairing null used as the
negative control throughout. The null is exact by construction rather than asymptotic; where a
Tracy-Widom comparison is made, the external reference is [14].

[14] M. Chiani, *Distribution of the largest eigenvalue for real Wishart and Gaussian random
matrices and a simple approximation for the Tracy-Widom distribution*, J. Multivariate Anal.
**129**, 69-81 (2014); arXiv:1209.3394.

The identity of §3 and the lockstep of §2 are derived and verified here, and no source is cited
for them. Every entry above was read before being cited.
