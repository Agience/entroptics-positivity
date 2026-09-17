# Reading Positivity from the Weights Alone

### A general-purpose instrument in determinantal quantum Monte Carlo, scored against a criterion computed from the Hamiltonian

**Ikailo John Sessford**, Ikailo Inc., `john@ikailo.com`

*Version 0.2.0. September 2026.*

> **A note on measurement.** Every empirical quantity in this paper is a deterministic read of a
> configuration produced by the determinantal quantum Monte Carlo sampler in `research/code/`. The
> sampler is gated against a brute-force enumeration of every auxiliary field at 5e-15. No
> fitted parameter is supplied to any read, and one constant is: the particle-hole zero `1/2` of
> §7's own-zero read, which follows from `G_up[i,i] + G_dn[i,i] = 1` rather than being tuned. The
> one threshold in play is the library's own significance level `far = 0.05`, left at its default;
> it gates `resolved`, which §6.2 reports on the degenerate input. Elsewhere a coefficient
> that appears is derived, and the derivation is checked by requiring the wrong coefficient to fail.
>
> **Which results use the instrument, and which do not.** Of the 54 experiments cited here, 28 read
> through the open-source *Entroptics* instrument
> ([github.com/Agience/entroptics](https://github.com/Agience/entroptics), version 0.2.3,
> [doi:10.5281/zenodo.22687899](https://doi.org/10.5281/zenodo.22687899)) and 26 do not. All three
> checks the argument rests on -- the lockstep, the identity, and the conditioned frame -- are
> instrument-free. §§3-5 derive and verify the identity and its criterion from the one-body matrix;
> §7 reads the output with the instrument. Neither side is given the other's input -- the read never
> receives `K`, the criterion never receives a configuration -- and they agree on 15 of 15
> lattices. Reads are reached through one domain adapter
> (`research/code/entroptics_adapter.py`) which names each library read after the question §7 asks
> of it and holds the version the figures were read through.
>
> **What is proved.** The paper's algebraic statements are machine-checked in Lean 4 / Mathlib
> (§11, `research/lean/`), `sorry`-free and on the three foundational axioms: the log-determinant
> identity, the positivity criterion and its two routes, the closed door on doping, the
> effective-sample ceiling, magnitude blindness at every exponent, and what the alignment read
> guarantees. Every machine-precision figure quoted here is printed by the file it is attributed to.

---

## Abstract

The fermion sign problem in determinantal quantum Monte Carlo is normally diagnosed from the
average sign, a quantity that costs `O(1/<sgn>^2)` to resolve and is identically 1 until the
problem is already large. This paper asks a different question: **given only the weights a running
simulation already holds -- no Hamiltonian, no knowledge of the decoupling, and no constant chosen
by the caller -- what can be decided about its positivity?**

**The answer is a build check a run can perform on itself.** §4 derives an identity between the two
spin channels' determinants: `ln|det_up| - ln|det_dn|` is an exact affine function of the field sum
`sum(x)`. A normalised alignment saturates at 1 on an affine relation and cannot on a non-affine
one, and it is invariant to both the offset and the slope, so the check needs neither the
decoupling constant nor `tr(K)` -- only two scalars every determinant QMC code already forms each
sweep. On a lattice built with a periodic wrap where open was intended, the read fires at
`beta = 1` to `3` while **not one negative weight has appeared**, every standard health check
passes, and the criterion computed on the *intended* lattice still reports sign-free. It stands
eleven orders above the correctly-built control and between `5.5` and `91` times its own seed
spread. That is a realistic, common and expensive lattice-assembly bug, caught at a beta where a
pilot run costs minutes.

For two scalar columns the read is the Pearson correlation of the centred columns, exactly, with
one addition: it is standardised against an exact re-pairing null and returns `0` rather than a
small correlation where the value does not clear it (§1.1). The cost is `O(n)` on columns already
computed, against a diagnostic that otherwise costs `O(1/<sgn>^2)`.

**A second reading classifies the run, from the weight cloud alone.** `concentration` on the
cloud's `(Re, Im)` frame returns a *directional* statistic `resultant` -- the mean resultant
length, which for a weight cloud is `|<w/|w|>|` -- and an *axial* one `focus`, the leading
eigenvalue of the cloud's orientation matrix. Because the rows are unit vectors in the plane,
`focus` lies in `[1/2, 1]`, with `1/2` the isotropic floor and `1` exactly a rank-one cloud. The
pair classifies: `focus = 1` with `resultant = 1` is sign-free; `focus = 1` with `resultant < 1` is
a real sign problem, because rank one in the plane means the phase takes two values `pi` apart and
a phase confined to `Z2` is what a sign *is*; `focus < 1` is a phase problem no rotation reaches.
Measured, `focus` is `1.0000` on every real-weight row and `0.543` to `0.833` on every genuinely
complex one.

**The classification decides which estimator is safe.** A global phase cancels in
`<O> = sum(O w)/sum(w)`. The estimator carried over from real weights,
`mean(Re w)/mean(|w|)`, reads `cos(theta)` -- falling from `0.93500` to `0.25011` across
rotations that leave `|<w>|` unmoved -- and would overstate a cost that goes as `1/<sgn>^2` by
fourteen times.

**The reading is validated against an independent oracle computed from the Hamiltonian, and neither
side is given the other's input.** A configuration's weight is a product of two determinants, one
per spin channel; on a bipartite lattice at half filling they change sign **in lockstep** -- around
11% of configurations at `beta = 12`, on exactly the same ones -- so the product's sign never
changes. That lockstep follows from an identity between the two determinants, derived and verified
to `1e-14` in two field distributions whose closed-form constants differ by 4% and each fail in the
other by fourteen orders. The identity holds exactly when `K` admits a diagonal-unitary
conjugation, by either of two routes -- `S K S^-1 = -K`, which asks every cycle of `K`'s support
graph to be even, or `S K S^-1 = -conj(K)`, which admits odd cycles carrying half-odd-integer flux.
For real `K` the two coincide and reduce to a 2-colouring with a zero diagonal: `O(N^2)`, no
eigenvalues, no determinants, no field.

That criterion is the standard one: for real `K` the two-colouring is bipartiteness and the zero
diagonal is the absence of a staggered potential, the long-known particle-hole route to positivity.
No new positivity criterion is claimed. It is the yardstick, and the agreement figures quoted in §5
-- 69 one-body matrices with no mismatch, and the read agreeing with the criterion on 15 of 15
lattices from the output alone -- are checks that this rig computes the known criterion correctly,
which is what licenses the build check above. The oracle sees `K` and never a configuration; the
read sees two logged columns and never `K`.

The read has one blind spot, and it is the same fact as its scale invariance: a wrong decoupling
constant changes the slope, leaves the relation affine, and does not desaturate the read. The
direct residual catches that case at `1.25`, so the two checks are complementary (§6).

**The scope is measured, and it separates a comparison from a one-sided read.** A phase common to
both channels is invisible to a comparison between them: systems whose negative fractions differ by
33 percentage points give identical reads. The ceiling on what any such read can deliver is exact:
correct importance sampling draws at `|w|`, so Kish's effective sample size of those weights is
exactly `n <sgn>^2`, a property of the weights and therefore a ceiling on every aggregation of
them. The read detects onset rather than severity: it moves on rows where `<sgn>` is still
identically 1, and severity is set by a crossing count the read has already summed away.

---

## 1. What is normally measured

Determinantal quantum Monte Carlo [1] evaluates a fermionic partition function by decoupling the
interaction into an auxiliary field and integrating the fermions out, leaving a sum over field
configurations with weight `w(x)`. When `w` is non-negative it is a probability density and the
sum is a Monte Carlo average. When it is not, the standard workaround samples `|w|` and carries
the sign as an observable:

    <O> = <O sgn>_|w| / <sgn>_|w| ,        <sgn> = Z / Z_||

The denominator is the average sign, and it is exponentially small [4]:
`<sgn> = exp(-beta N df)` with `df` the free-energy density difference between the physical
ensemble and the one that samples `|w|`. Both numerator and denominator are then small numbers
computed as differences of large ones, and the relative error of the ratio grows like
`1 / sqrt(M <sgn>^2)`. The cost of a fixed
accuracy is therefore exponential in `beta N`. Deciding the general case is NP-hard [5], so no
representation-independent cure is available and the practical question is always which
representation a given model admits.

That is the quantity this paper is about. The average sign is an average of a binary, so where the
two channels are locked it returns `1.00000 +- 0.00000` on every configuration, and an average of a
constant has no derivative: it carries no information about how close the system is to losing the
lock.

*That is a statement about one regime.* Where `<sgn>` does vary it carries a great deal: Mondaini,
Tarat and Scalettar [19] link it quantitatively to quantum critical behaviour, and read the low
average sign of the doped square-lattice Hubbard model as reflecting the onset of pseudogap physics
rather than only obstructing its simulation. This paper's reading is aimed at the complementary
regime, where the sign is identically 1 and has nothing to move, and the order parameter of §7 is
offered for the approach to that boundary.

### 1.1 The reads, as formulas

The instrument is general-purpose and its reads carry its own names. Every one used in this paper
is a formula on the frame it is handed, and each reduces to a standard statistic or is stated here
as one. A frame `W` is an `n x k` array -- `n` draws, `k` features -- and `W~` is `W` with each
column's mean removed. Reads are reached through `research/code/entroptics_adapter.py`, which names
each after the question asked of it.

**`coupling(A, B)`, the two-channel read** -- §§3-7. `A` and `B` are frames sharing their `n`. The
read is the normalised alignment of `A~` and `B~`, and it returns:

  * `strength`, in `[-1, 1]`. **For `k = 1` this is exactly the Pearson correlation coefficient of
    the two centred columns**, agreeing with `numpy.corrcoef` to `1e-16` on affine, noisy-affine
    and weakly-related pairs. It is invariant to an offset and a positive scale on either column
    (§11, `strength_affine_invariant`), which is why §6 needs neither `lambda` nor `tr(K)`; it is
    bounded by Cauchy-Schwarz (`abs_strength_le_one`); and it saturates at `+-1` exactly when the
    two columns are affinely related (`abs_strength_eq_one_iff`).
  * `z` and `resolved`, against the read's **exact re-pairing null**: the null re-pairs each frame
    row with every weight, giving the null distribution's mean and variance in closed form rather
    than asymptotically, so a sigma here is quoted with no constant supplied. Where the value does
    not clear its own null the read returns `strength = 0` and `resolved = False` rather than a
    small correlation -- on an independent pair of columns `corrcoef` reads `0.086` and the read
    reads `0.0` at `z = 1.71`.
  * `tightness`, the leading singular value's share of the squared cross-covariance spectrum,
    `s_1^2 / sum_i s_i^2`. It is `1` when the whole coupling sits in a single mode.

**`concentration(P)`, the one-sided cloud read** -- §7.1. `P` is the `(Re, Im)` frame of the
unit-modulus weights `u_i = w_i / |w_i|`, one row a draw. It returns a *directional* statistic and
an *axial* one, and the pair is what classifies:

  * `resultant = |mean(u)|`, the mean resultant length. On a weight cloud this **is** `|<w/|w|>|`,
    by construction rather than by measurement.
  * `focus`, the leading eigenvalue of the orientation matrix `T = (1/n) sum_i u_i u_i^T`. Because
    the `u_i` are unit vectors in the plane, `tr(T) = 1` and therefore **`focus` lies in
    `[1/2, 1]`**: `1/2` is the isotropic floor, and `1` is exactly a rank-one cloud, every phase on
    one line through the origin. The two differ where it matters -- an antipodal cloud reads
    `resultant ~ 0` and `focus = 1`.

    *The floor is why `focus` is read against 1 and not against 0.* §7.1's complex rows run `0.543`
    to `0.833`, which is a span from just above the isotropic floor to strongly axial; what every
    one of them shares is being below `1`, which is the definitional value the classification uses.

**`carriage(...)`, the evidence read** -- §7, §9. `effective_n` is Kish's effective sample size
`(sum w)^2 / sum w^2`. On a `+-1` sign sequence that is exactly `n <sgn>^2` (§11,
`kish_of_signs`). `carried` is the same read's statement of how much of a frame a set of weights
accounts for, standardised against the same exact re-pairing null; `1` is the value when the
weights carry nothing, which is how §8's table is read.

**`Screen().balance`, the own-zero read** -- §7. Defined where it is used: the joint residual
`||r||^2` against the exact no-drift moments `E||r||^2 = tr(S)/T` and `Var||r||^2 = 2 tr(S^2)/T^2`,
which fix a scaled chi-square whose degrees of freedom are the participation ratio
`(tr S)^2 / tr(S^2)`.

**`spectral_optics`, the single-channel read** -- §9.2b, §10. `top_share` is the leading
eigenvalue's share of the covariance spectrum, `lambda_1 / sum_i lambda_i`; `attenuation`,
`dominance` and `phase` are further summaries of the same spectrum, and `resolved_modes` is a count
against a floor derived from the data. §9.2b uses all of them only through a **rank** correlation,
which depends on ordering alone, so nothing there turns on how they are normalised.

## 2. The frame the determinants are read in

Every empirical number in this paper is a determinant sign or a function of one, and a
determinant sign has to be read in a frame that stays conditioned. `I + B` does not. Its condition
number grows like `exp(beta * bandwidth)`, and past `beta ~ 4` its eigenvalues stop meaning
anything. This section comes first because every table after it runs at a `beta` where that
matters.

This is a measured failure (`reads/expBM_naive_frame_manufactures_a_sign.py`, 400 draws a beta).
Forming `B` as the plain ordered product and reading its spectrum reports, at **half filling on a
bipartite lattice where positivity is provable** and the true negative fraction is therefore zero:

| beta | 2 | 4 | 6 | 8 |
|---|---|---|---|---|
| negative fraction, naive product | 0.0000 | 0.0000 | 0.0125 | **0.2400** |
| negative fraction, conditioned frame | 0.0000 | 0.0000 | **0.0000** | **0.0000** |
| largest `\|1 + lambda\|`, naive | 4.4e+06 | 1.4e+12 | 6.2e+17 | **5.8e+22** |
| largest `\|1 + lambda\|`, conditioned | 61.1 | 180.8 | 113.2 | **576.5** |

The last two rows are why. The conditioned frame's spectrum stays within three orders across the
whole range; the naive product's runs to `5.8e+22`, so its eigenvalues were never near a crossing
and its sign is not a reading of one. The naive frame does not merely lose precision; it
manufactures the phenomenon the paper is about.

The conditioned frame comes from the UDT factorisation [3]. With `I + U D T = U Db M T` and
`prod(Db) > 0`, the sign is carried entirely by `det(U) det(T) det(M)`, and `M` is bounded by
construction because `Db >= 1` divides the large block and `Ds <= 1` is the small one.

Three properties hold, and each is checked against a case that must break it
(`tests/test_conditioned_frame.py`):

* **two independent routes agree.** The block formulation and the core factorisation reach the
  sign by different arithmetic and agree on every configuration measured. The two routes must be
  arithmetically independent for this to mean anything: a parity count compared against
  `sign(prod(1 + lambda))` from the *same* eigenvalues agrees whatever those eigenvalues are.
* **the answer does not move with the stabilisation block.** Every configuration's sign is
  compared across three block sizes, not just the totals, so agreement cannot come from
  cancellation.
* **the naive frame fails.** A check that cannot fail establishes nothing, so the unstabilised
  product is run alongside and must report the sign problem that is not there.

Everything that follows is read in this frame.

## 3. The sign is a disagreement between two channels

The weight of an auxiliary-field configuration is

    w(x) = det(I + B_up(x)) * det(I + B_dn(x))

so `sign w(x) < 0` **if and only if** the two channels' determinant signs disagree. That is
arithmetic, immediate from the product. What is a result is that the disagreement has structure.

**Measured** (`reads/expVV_lockstep.py`, 600 configurations per row, 2x4, U = 4, both channels
read separately in the conditioned frame of §2; five seeds, errors from the seed spread):

| beta | mu | up flips | dn flips | agree | disagree | disagree = neg fraction |
|---|---|---|---|---|---|---|
| 4 | 0.0 | 0.0020 +- 0.0018 | **0.0020** | **1.0000 +- 0.0000** | 0.0000 | exact |
| 6 | 0.0 | 0.0133 +- 0.0046 | **0.0133** | **1.0000 +- 0.0000** | 0.0000 | exact |
| 8 | 0.0 | 0.0400 +- 0.0097 | **0.0400** | **1.0000 +- 0.0000** | 0.0000 | exact |
| 10 | 0.0 | 0.0753 +- 0.0138 | **0.0753** | **1.0000 +- 0.0000** | 0.0000 | exact |
| 12 | 0.0 | **0.1120 +- 0.0103** | **0.1120** | **1.0000 +- 0.0000** | 0.0000 | exact |
| 6 | 0.4 | 0.0093 +- 0.0025 | 0.0093 | 0.9847 +- 0.0027 | 0.0153 | exact |
| 8 | 0.4 | 0.0220 +- 0.0040 | 0.0200 | 0.9647 +- 0.0079 | 0.0353 | exact |
| 10 | 0.4 | 0.0323 +- 0.0035 | 0.0317 | 0.9433 +- 0.0070 | 0.0567 | exact |
| 12 | 0.4 | 0.0413 +- 0.0077 | 0.0417 | 0.9283 +- 0.0049 | 0.0717 | exact |

**Some columns are estimates and some are not.** The two flip
rates are proportions of 600 draws and the up column carries its spread across seeds.
The agreement is not an estimate: at half filling the two channels change sign on *exactly* the
same configurations, so the column reads `1.0000` with zero spread on every seed rather than
averaging to it. With the agreement exact, the two flip rates are equal by entailment and the down
column repeats the up one; it is printed because on the doped rows, where the agreement is broken,
the two come apart. The last column
is likewise an identity checked configuration by configuration, not a fitted correspondence -- the
disagreement fraction **is** the negative-weight fraction, on doped rows as well as half-filled
ones.

The half-filled rows establish two things together, and neither alone would do. The individual
channels **do** change sign, on around 11% of configurations by `beta = 12`, so the lockstep is not
a statement about a quantity that never moves. And they change sign on the same configurations
every time, which is what makes the two rates **exactly** equal at half filling -- `0.1120` and
`0.1120`, and so on every half-filled row to four decimals -- while the doped rows are merely close
(`0.0220` against `0.0200` at `beta = 8`). The agreement is a half-filling statement and not a
generic one: it is
`det_dn(x) = det_up(-x)` meeting a field distribution symmetric under `x -> -x`, and doping breaks
the second of those.

**On four geometries, not one.** Every table above is a 2x4 lattice, and the claim is general, so
it was re-measured across sizes and shapes (`reads/expAC_geometry.py`, `mu = 0`, `beta = 10`,
300 configurations x **four seeds on every row, controls included**):

| lattice | N | states at `E_F` | identity residual | flip rate | agree | neg fraction | coupling |
|---|---|---|---|---|---|---|---|
| 2x4 | 8 | 2 | 7.2e-12 | 0.0817 +- 0.0084 | **1.0000 +- 0.0000** | 0.0000 +- 0.0000 | **-1.0000** |
| **4x4** | 16 | **6** | 7.0e-11 | **0.4542 +- 0.0241** | **1.0000 +- 0.0000** | 0.0000 +- 0.0000 | **-1.0000** |
| 2x6 | 12 | 2 | 5.5e-12 | 0.0508 +- 0.0103 | **1.0000 +- 0.0000** | 0.0000 +- 0.0000 | **-1.0000** |
| 4x6 | 24 | 2 | 9.5e-12 | 0.0050 +- 0.0019 | **1.0000 +- 0.0000** | 0.0000 +- 0.0000 | **-1.0000** |
| 2x3 | 6 | 1 | 1.4e+01 | 0.0467 +- 0.0027 | 0.9042 +- 0.0242 | 0.0958 +- 0.0242 | not resolved |
| 3x4 | 12 | 1 | 7.5e+00 | 0.0075 +- 0.0074 | 0.9858 +- 0.0100 | 0.0142 +- 0.0100 | -0.7466 |

The last two rows are controls: a periodic lattice with an odd side has an odd ring and is not
bipartite, and there all three claims fail together -- the identity breaks, the channels
disagree, and negative weights appear. A test that held everywhere would not be sensitive to the
property the argument depends on. Every row here is read across four seeds, controls included,
because a control read from one seed cannot say whether its failure is the lattice or the draw.

*The flip rate is not monotone in `N`, and the `E_F` column is why.* `4x4` flips on `0.4542` of
configurations against `0.0050` on the larger `4x6`. For a square lattice with nearest-neighbour
hopping the dispersion is `E = -2t(cos kx + cos ky)`, so a state sits at the half-filling Fermi
level exactly when `cos kx + cos ky = 0`; `4x4` has **six** such states where every other lattice
here has two, and a determinant changes sign when an eigenvalue crosses zero. The count is exact
and needs no threshold. What the lockstep claim is about is the *agree* column, and that reads
`1.0000` with zero spread across seeds on every bipartite row whatever the flip rate does.

The `4x4` row is therefore the strongest form of the statement available here. An individual
channel changes sign on **45.4%** of configurations and the two channels agree on **every one of
them**, on all four seeds.

One row is reported and not counted: `4x6` at `beta = 6` has a flip rate of **0.0000 +- 0.0000**,
so its perfect agreement is vacuous -- nothing crossed. Agreement is only evidence where the
channels actually flip, so `tests/test_lockstep.py` requires a non-zero flip rate alongside it.

## 4. The identity that produces the lockstep

Two facts about the two determinants look incompatible. Their signs agree on every configuration,
and their log-magnitudes differ by 8 to 16 orders. A relation `det_up = c det_dn` for constant `c`
would hold that log-difference fixed, and it is not fixed -- it ranges across configurations, which
is the second fact. So the relation cannot be of that form: it must preserve the sign and leave the
magnitude free.

It does, and it is exact. The two channels' diagonal factors are inverses of one another, so
`det(B_up)/det(B_dn) = exp(2 lambda sum x)`; routing that through
`det(I + B) = det(B) det(I + B^-1)` carries one factor into the ratio of the full determinants:

    ln|det(I + B_up(x))| - ln|det(I + B_dn(x))| = -dtau * L * tr(K) + lambda * sum(x)

with `lambda = arccosh(exp(dtau U / 2))` the constant of Hirsch's discrete decoupling [2].

**Verified** (`reads/expWW_ratio.py`, `tests/test_identity.py`) at half filling on a bipartite
lattice:

| beta | 2 | 4 | 6 | 8 |
|---|---|---|---|---|
| residual, coefficient `lambda` | **1.51e-14** | **4.35e-14** | **1.08e-12** | **2.09e-12** |
| residual, coefficient `2 lambda` | 2.51e+01 | 3.83e+01 | 5.31e+01 | 4.86e+01 |

The second row is the control: a relation that held for any coefficient would be vacuous. The
derived coefficient is exact and the alternative is wrong by thirteen to fifteen orders, the
separation narrowing with beta as the derived residual accumulates arithmetic.

**Across the interaction, not at one value of it.** `lambda` is a function of both `U` and `dtau`,
so varying them tests the derivation rather than repeating a measurement: each row predicts a
different coefficient and has to be exact at it (`reads/expAD_coupling_strength.py`, residual max
over 200 configurations):

| dtau | U = 2 | U = 4 | U = 8 | U = 12 |
|---|---|---|---|---|
| 0.0625 | 2.7e-14 | 6.8e-13 | 7.3e-13 | 4.4e-12 |
| 0.125 | 2.3e-14 | 6.8e-14 | 3.7e-11 | 1.0e-11 |
| 0.25 | 6.0e-14 | 7.4e-12 | 1.5e-10 | 2.8e-08 |

`lambda` spans 0.357 to 2.180 across this table, a factor of 6.1. The residual grows gently with
it as accumulated arithmetic -- the last cell sits at `3e-8`, which is nine and a half orders below
the wrong coefficient's value on the same row and is float64 rather than a limit of the relation.

**And the coefficient is resolved to better than its own variation.** Borrowing a neighbouring
row's `lambda` -- a difference of as little as 0.51 against 0.74 -- fails by 15 to 44 while the
predicted one holds at 1e-14:

| U | own lambda | borrowed | residual, own | residual, borrowed |
|---|---|---|---|---|
| 2 | 0.51048 | 0.73690 | 2.3e-14 | **15.4** |
| 4 | 0.73690 | 1.08504 | 9.6e-13 | **18.8** |
| 8 | 1.08504 | 1.38202 | 1.3e-12 | **17.8** |
| 12 | 1.38202 | 0.51048 | 5.1e-11 | **43.6** |

The relation resolves a 44% difference in `lambda` by fourteen orders, at every interaction
strength measured.

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

| field | beta | its own lambda | the *other* representation's lambda | twice its own |
|---|---|---|---|---|
| Ising | 2 | **1.1e-14** | 9.5e-01 | 2.4e+01 |
| Ising | 4 | **3.1e-14** | 1.4e+00 | 3.4e+01 |
| Ising | 6 | **2.8e-12** | 1.5e+00 | 3.8e+01 |
| Gaussian | 2 | **3.7e-14** | 8.8e-01 | 2.1e+01 |
| Gaussian | 4 | **3.6e-14** | 1.2e+00 | 2.9e+01 |
| Gaussian | 6 | **4.3e-14** | 1.5e+00 | 3.5e+01 |

Each representation's own constant is exact to `1e-14` and the other's, four percent away, fails
by fourteen orders. A relation that held for any
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
§8 shows the read is a property of a (Hamiltonian, decoupling) pair -- this is the other half of
that statement: within the spin decoupling it does not depend on which field distribution realises
it.

## 5. What the identity requires, controlled

Every row that verified the identity had a bipartite lattice **and** half filling. Those are
confounded, and separating them required care: next-nearest hopping `tp` destroys the bipartite
structure, but it also moves the band, so `mu = 0` is no longer half filling. Compared without
that control the data appears to show that breaking the bipartite structure removes the sign
problem -- an artifact, because the non-bipartite cells sat at `<n> = 0.75`.

The filling is therefore fixed by the **non-interacting** density, a closed-form function of the
single-particle spectrum that carries no sampling noise and is computed without touching the sign.

**Measured** (`reads/expZZ_matched_filling.py`, U = 4, beta = 8, every row at `n = 1.00000`):

| tp | tuned mu | bipartite | identity residual | signs lock | neg fraction |
|---|---|---|---|---|---|
| 0.00 | **0.0000** | yes | **4.4e-13** | **1.0000** | **0.0000** |
| 0.15 | 0.6000 | no | 4.25e+01 | 0.9500 | 0.0500 |
| 0.30 | 1.1992 | no | 8.32e+01 | 0.8150 | 0.1850 |
| 0.70 | 1.5548 | no | 1.05e+02 | 0.9200 | 0.0800 |

The tuner recovers `mu = -0.0000` for the bipartite lattice without being told that half filling
lies there, which is the control validating itself. At identical filling, the bipartite lattice has
no sign problem and the identity is exact; every non-bipartite one has a sign problem and the
identity is broken.

**For a real one-body matrix the two conditions are one criterion, checkable without sampling.**
The criterion below is a statement about the spin decoupling, as everything in §§3-7 is: it is
derived from that decoupling's diagonal factor and says nothing about the charge channel, whose
positivity comes from a different mechanism (§7). The derivation reduces to §4's form only when
`det(I + B_up^-1) = det(I + B_dn)`, which needs
`expmK^-1` similar to `expmK` -- and the similarity has to commute with the interaction's diagonal
factor. That restricts it to a **signed diagonal**: `S K S = -K` with `S = diag(+-1)`. Elementwise
`s_i s_j K_ij = -K_ij`, so the criterion is a 2-colouring of K's support graph together with a zero
diagonal. It costs `O(N^2)`, and uses no eigenvalues, no determinants and no field
(`reads/expBE_criterion_tables.py`, 2x4, `U = 4`, `beta = 8`):

| K | signed-diagonal S exists | identity residual | agrees |
|---|---|---|---|
| bipartite, `mu = 0` | yes | 2.4e-12 | yes |
| bipartite, staggered `h = 0.2` | **no** | 5.7e+00 | yes |
| bipartite, staggered `h = 1.2` | **no** | 2.0e+01 | yes |
| `tp = 0.3` | no | 1.4e+01 | yes |
| `mu = 0.4` | no | 3.2e+01 | yes |
| bipartite, bond-disordered x3 | **yes** | 5.2e-14 to 1.2e-12 | yes |
| 4x4 and 2x6 bond-disordered | **yes** | 2.1e-12, 6.4e-13 | yes |
| odd rings of 5 and 7 sites | no | 7.3, 6.3 | yes |

Twelve one-body matrices, agreeing with the measured identity on every one. The criterion unifies
§5's two conditions and covers a case neither names: a **staggered potential** leaves the lattice
bipartite and the filling at exactly one per site, and breaks the identity anyway, because a
diagonal term cannot be negated by signs.

The bond-disordered rows are the ones that make it predictive rather than descriptive. No symmetry
was designed into them -- every bond strength is drawn at random, up to a spread of 1.5 -- and the
criterion says the identity holds, which it does to `1e-13`.

**It is a statement about the hopping graph, not about lattices.** The same criterion was applied
to topologies the rest of the paper never uses, with the prediction made before the measurement:

| topology | criterion | identity residual |
|---|---|---|
| chain, 8 sites, periodic | yes | 3.7e-13 |
| chain, 7 sites, periodic | no | 6.3e+00 |
| chain, 8 sites, open | yes | 2.0e-13 |
| **star graph (a tree), 8 sites** | **yes** | **6.8e-13** |
| triangular ladder, 8 sites | no | 1.0e+01 |
| even ring, 12 sites | yes | 6.5e-13 |
| odd ring, 9 sites | no | 5.9e+00 |

A tree satisfies it however it is drawn, having no cycles at all; any graph carrying an odd cycle
never does. Seven more topologies, agreeing with the measured identity on every one.

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
of 5, 6, 7 and 8 sites at six fluxes each, plus a 5x2 lattice at five fluxes
(`reads/expBA_oracle_superset.py`) -- **29 of 29 agree**,
including every case where exactly one route is open.

*A phase condition on the cycles of a graph built from the Hamiltonian is a known shape.* Hen [18]
gives a criterion of exactly that form for quantum Monte
Carlo generally: the simulation is sign-problem-free precisely when the total phases along the
chordless cycles of the weighted graph whose adjacency matrix is the Hamiltonian vanish. Route B is
the same shape one level down -- the graph here is the support graph of the one-body matrix `K`,
not of the Hamiltonian, and what the condition decides is the pairing identity of §4 rather than
positivity directly, which is why §5 separates the two and finds cases where the identity holds
and positivity does not.

For real `K` the two conditions coincide, which is why the signed-diagonal colouring is the whole
story there: checked on 11 real matrices, the general criterion and the colouring return the same
answer on every one.

The tables above overlap -- a bond-disordered `4x4` appears in one and a `4x4` flux row in another
-- so they are not added up here. `reads/expBA_oracle_superset.py` enumerates the population once,
independently of how the sections group it: **69 one-body matrices, no mismatch**, of which the
flux-route subset is **29 of 29**.

**The two routes are not equivalent, and only one carries positivity.** Route A leaves the weights
real: on `ring 6` and `ring 8` at flux `pi/4` the identity holds at `2e-14` and the weight's mean
phase is **exactly 1** -- no sign problem. Route B restores the identity on an odd cycle and leaves
a phase behind. Measured with `1 - |<w/|w|>|`, which is the ordinary negative-weight measure when
the weight is real and the mean-phase deficit when it is not
(`reads/expBE_criterion_tables.py`, `beta = 8`, 200 draws):

| K | route | identity residual | sign deficit |
|---|---|---|---|
| 2x4 real, half filled | A and B | 2.4e-12 | **0.00000** |
| ring 6, flux `pi/4` | **A only** | 2.8e-14 | **0.00000** |
| ring 5, flux `pi/2` | **B only** | 2.0e-14 | 0.14968 |
| triangular ladder, flux 0 | neither | 1.2e+01 | 0.19000 |
| triangular ladder, flux `pi/2` | **B only** | 4.3e-14 | **0.50071** |

*How many digits of a sign deficit are real.* The deficit is `1 - |<w/|w|>|`, a sampled mean over
draws, so it carries that mean's error and each table quotes its own run. Measured across six
seeds (`reads/expBG_single_channel_has_no_indicator.py`): at 200 draws the `ring 5` deficit spans
`0.14457` to `0.17446` with a standard deviation of `0.01059`, and the `tri ladder 8` deficit spans
`0.44611` to `0.54008` at `0.03284`; at 800 draws both spreads roughly halve, to `0.00707` and
`0.01918`. So a deficit quoted here is reproducible in its second decimal, and two sections quoting
the same lattice from different runs differ by that much and not by more.

The last two rows show the two coming apart. Threading the frustrated ladder does what the
criterion says -- the identity goes from `1.2e+01` to `4.3e-14` -- and the sign deficit rises from
`0.19000` to **`0.50071`**. The flux converts a mild sign problem into a substantial phase
problem.

So the criterion decides **the identity**, and the identity delivers **positivity only where the
weights are real**, which is route A. On an odd cycle no flux-free choice exists and route B is the
only one available, so an odd cycle cannot be made positive this way at all -- it can only be made
to satisfy the identity.

*That separation is not local to this model.* A pairing identity holding while positivity fails is
the same distinction reflection positivity turns on, where "the measure is invariant under the
reflection" is the easy half and the content is whether the weight can be written as a pairing at
all. They coincide on the well-behaved cases and come apart on route B over an odd cycle: the
identity exact at `4.3e-14`, the sign deficit `0.50071` against `0.19000` unfluxed.

**On a two-dimensional frustrated lattice the answer depends on the size, and the criterion tracks
it.** A periodic triangular lattice has triangles, rhombi *and* wrap-around cycles, and route B has
to hold on all of them at once. On `4x4` the wraps are even and carry no flux, so `theta = pi/2`
opens route B and the identity holds to `4.7e-13`; on `3x3` the wraps are odd and flux-free, no
`theta` opens either route, and the best residual over five fluxes is `3.2` -- the identity stays
broken at every one. Ten rows across the two lattices and five fluxes, all predicted correctly
(`reads/expBO_triangular_pair.py`).

*What the pair shows.* It is a statement about the **identity**. The unfluxed `4x4` carries a broken
identity -- residual `9.26` at `beta = 8`, rising to `14.59` at `beta = 16` -- and still shows **no
negative weight at all** in 200 draws at either, so nothing was restored there that was measurably
absent. The `3x3` lattice does have a real sign problem, `0.3500` unfluxed and `0.1351` at `pi/2`,
and the criterion correctly says no flux removes it. This pair extends the criterion to two
dimensions and
shows the wrap cycles deciding the outcome; like the ladder, it is a statement about the identity
and not about positivity, since `theta = pi/2` opens route B.

**The obvious criterion is the spectral one, and it is wrong.** For Hermitian `K`, a spectrum
symmetric about zero is exactly similarity to `-K`. The staggered rows have spectra symmetric to
**1e-15** and break the identity by up to `21`. The similarity exists; it is not a signed diagonal,
so it does not commute with `diag(exp(sigma lam x))` and the derivation cannot use it.

**The criterion is sharp: the identity is fragile where the sign problem is robust.** A staggered
term of any amplitude `eps > 0` makes `S K S = -K` unsatisfiable, and the identity follows exactly
(`reads/expBE_criterion_tables.py`, 100 draws a row):

| eps | criterion | identity residual | residual / eps | negative fraction |
|---|---|---|---|---|
| 0 | yes | 3.2e-13 | -- | **0.0000** |
| 1e-06 | no | 8.542e-05 | **85.425** | **0.0000** |
| 1e-04 | no | 8.543e-03 | **85.425** | **0.0000** |
| 1e-03 | no | 8.548e-02 | **85.482** | **0.0000** |
| 1e-02 | no | 9.185e-01 | 91.848 | **0.0000** |
| 1e-01 | no | 3.217e+00 | 32.165 | 0.0800 |

The residual is **linear in the perturbation**, with no tolerance and no onset: the identity does
not survive a small violation, it degrades in proportion to it from the first. The ratio holds to
`0.07%` across three decades of `eps`, from `1e-06` to `1e-03`, and is `7.5%` high by `1e-02`. A
chemical potential behaves the same way, at a ratio of `92.4` across three decades.

The negative fraction does not follow suit. It is exactly `0.0000` until `eps` reaches about `0.1`,
so a model can violate the criterion, carry a measurably broken identity, and still produce no
negative weight at all. **The identity is structurally fragile and the sign problem numerically
robust**, and those are statements about two different quantities -- the residual measures the
perturbation, the negative fraction measures its consequence, and only the first is linear.

So the criterion decides the **identity**, and nothing further on its own: it carries positivity
only along route A, where the weights stay real, and it does not predict the negative *fraction* on
either route. That is the mechanism behind the next point.

**The residual is not a severity measure** (`reads/expAB_coupling_controlled.py`). It grows
monotonically with `tp` while the negative fraction does not: on that experiment's own draws it
runs `0.0375`, `0.1825`, `0.0600` across `tp = 0.15, 0.30, 0.70`, rising and then falling. The
matched-filling table above (`reads/expZZ_matched_filling.py`) measures the same axis on its own
draws and finds the same shape at `0.0500`, `0.1850`, `0.0800`; the two are separate samples of one
quantity and agree on the ordering, which is the claim. The identity's failure permits the signs to
differ; it does not say
by how much they will.

**The scope of this axis.** The rows above are an `if and only if` on it: at matched filling,
bipartite means the identity holds and there is no sign problem, and non-bipartite means both fail.
Positivity has other sources. Wei et al. [7] give a unified account of the known sign-free models
and
identify sign-free models with *repulsive* interactions and *without* particle-hole symmetry, and
Wu and Zhang [6] derive positivity from a conjugate-pairing property tied to time-reversal rather
than to a lattice bipartition. §7 measures one of those other places directly -- the attractive
charge channel, sign-free at every filling -- and finds the read saturated there too, at the
opposite sign. So the symmetric point is one source of positivity that this rig can read, and the
scope of §5 is the axis it was measured on.

The criterion above is the standard one. That the repulsive Hubbard model is sign-free at half
filling on a bipartite lattice has been known since the method's early years, and the usual
statement of the mechanism -- the two spin determinants related by particle-hole symmetry -- is
what §4's identity writes as an equality and §5's `S K S^-1 = -K` writes as a condition on `K`.
[6] and [7] give the modern account of that landscape and are cited above for it. For real `K` the
two-colouring is bipartiteness and the zero diagonal is the absence of a staggered potential, so
the agreement on 69 of 69 one-body matrices is a check that this rig computes the known criterion
correctly. What the rig is for begins at §6: the same verdict taken from the weights a simulation
already holds, with no `K` in hand -- a question about what is recoverable from data.

## 6. The criterion performed on output, with no Hamiltonian

Everything above decides the identity from `K`: build the support graph, look for a diagonal
unitary with `S K S^-1 = -K`, two-colour it, check the flux on every odd cycle. That is a decision
about a matrix. A running simulation does not have a clean matrix to hand -- it has configurations
and determinants -- and the terms most likely to have broken the identity, a doping term or an
added staggered potential, are exactly the ones a criterion is rarely re-derived for.

Read as a statement about the data rather than about `K`, §4's identity says something a read can
act on. Across configurations,

    ln|det_up|(x) - ln|det_dn|(x)   is exactly affine in   sum(x)

with slope `lambda` and offset `-dtau L tr(K)`. That is one channel against another on a shared
index, which is what `coupling` reads, and the read is invariant to both the offset and the scale --
so neither constant has to be known. An exact affine relation must saturate a normalised alignment
at 1. A broken identity cannot (`reads/expAW_criterion_from_output.py`, 2x4 and rings, `U = 4`,
`beta = 6`, 400 configurations):

| | `1 - \|strength\|` |
|---|---|
| identity holds (8 lattices) | `0` to `2.2e-16` -- machine epsilon |
| identity fails (7 lattices) | `1.0e-3` to `2.9e-2` |

**The read agrees with the measured identity residual on 15 of 15 lattices, and with §5's algebraic
criterion on 15 of 15, using no Hamiltonian.** Neither side is given the other's input: `coupling`
sees two columns and never sees `K`; the criterion sees `K` and never sees a configuration. The
agreement is counted against a cut on `1 - |strength|`, and the two populations are thirteen orders
apart -- `2.2e-16` against `1.0e-3` -- so every cut in that gap returns the same 15 of 15. No cut is
chosen here because none of them differ.

*The departure is systematic rather than sampling noise.* Across a factor of four in sample size
and three seeds, the saturated lattices stay at machine zero and the broken ones stay at `9.8e-4`
to `1.4e-2` without shrinking. A departure that fell as `n` grew would be the read reporting its
own variance.

*What it costs and what it is for.* Two columns a simulation already has, and one read -- `O(n)`
after determinants that were computed anyway, against a criterion that otherwise needs the
Hamiltonian's structure. It turns "does this model still satisfy §4's identity" from a question
asked once on paper into a check a run can perform on itself.

### 6.1 How it is used

Log two scalars per configuration. Both are already computed by any determinant QMC code:

  * `ln|det_up| - ln|det_dn|`, which comes out of the stable propagation -- `slogdet` is formed
    every sweep to build the weight;
  * `sum(x)`, a reduction over the auxiliary field.

Then one call, `coupling(A, B)`, and read whether it saturates. Three places to do it:

  1. **As a pilot, before committing to a production run.** A few hundred configurations at small
     `beta` decide whether §4's identity holds for the model *as assembled*. That is the identity
     and not positivity: on §5's route B the identity holds and the weights still carry a phase, and
     §7.1 measures four such lattices where the read is exactly saturated and the sign deficit runs
     from `0.08482` to `0.36442`. A saturated pilot rules out a broken pairing, not a sign problem.
  2. **As a regression check.** A sign-free setup acquires a term -- `t'`, a staggered field, a
     boundary twist, a chemical potential. The two-colouring is rarely re-derived when a term is
     added; this catches it from the output.
  3. **As a build check.** The read sees what the code *did*, not what the model was meant to be.
     Where an assembly routine and its documentation disagree, the read follows the code.

### 6.2 The build check, measured

The third use is the one the other two cannot cover, because it is the case where the criterion of
§5 is *unavailable by construction*: if the code may not be running the model its author believes,
then the `K` and the `lambda` one would feed to a direct test of §4's identity are the suspect
quantities themselves. The read needs neither -- it is invariant to the offset and the scale --
so it stays available (`reads/expAZ_build_check.py`).

Intended: an **open** 7-site chain, bipartite, and §5's criterion computed on it reports sign-free.
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
and between `5.5` and `91` times its own seed spread. From `beta = 4` the sign fires too and the
read is no longer needed.

**The scope has two provable limits: one blind spot, and one degenerate input the read reports.**
A wrong decoupling constant -- the continuous Gaussian
`sqrt(dtau U)` used in the discrete field -- is **not** detected: `1 - |strength|` is `-2.2e-16`,
exactly saturated, because rescaling an affine relation leaves it affine. Scale invariance is why
the read needs no `lambda` and why it cannot see a wrong one; the two are the same fact, and the
direct residual catches that case at `1.25`, so the two checks are complementary. Applying the
field with one sign to both channels leaves the two determinants
identical, so the logged difference is identically zero and the read returns **unresolved** rather
than a departure -- a degenerate input, visible before any read is taken.

### 6.3 Compared with watching the average sign

Every DQMC code already reports `<sgn>`. The read fires earlier. On `2x4` with `t' = 0.3`, at
`beta = 6` over 400 configurations:

| quantity | value |
|---|---|
| negative-weight fraction | **0.0000** -- not one negative weight |
| §4 identity residual | **10.53** -- broken by an order of magnitude |
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

## 7. Reading the weights: which problem is it, how far from the symmetric point, and at what cost

An observer standing at the output holds a cloud of configuration weights and nothing else -- no
`K`, no knowledge of the decoupling, and no constant to supply. This section is what can be
decided from there, and it has three parts, in the order a practitioner meets them.

**First, which problem the run has.** `concentration` on the weight cloud's `(Re, Im)` frame
returns a *directional* statistic and an *axial* one, and only the pair classifies: `focus = 1`
with `resultant = 1` is sign-free; `focus = 1` with `resultant < 1` is a real sign problem, because
rank one in the plane means the phase takes two values `pi` apart and a phase confined to `Z2` is
what a sign *is*; `focus < 1` is a phase problem no rotation reaches. That triage is §7.1, and it
decides what the rest of the section measures.

**Second, how far the run sits from the protecting symmetry**, which is the coupling read below --
exactly calibrated where the lockstep is perfect, and moving while the average sign is still
identically 1.

**Third, what any of it can cost**, which §9 bounds.

The coupling read that follows is a statement about a *real* determinantal weight, so the
classification comes first.

### 7.1 The triage: which problem is it?

**The triage recovers §5's positivity verdict from the weights alone**
(`reads/expAV_monomial_conjugation.py`). §5 establishes from `K` that route B buys the identity
*without* positivity. That same verdict is readable from the output, where an observer actually
stands -- holding a finite sample of weights and no knowledge of the mechanism. On four route-B
lattices the identity holds to machine precision, so every comparison between the channels is
saturated and blind (`beta = 6`, 120 draws -- a shallower point and a smaller sample than §5's
table, so the deficits are not the same numbers):

| lattice | §4 identity residual | `focus` | `\|Im\|` after de-rotation | sign deficit |
|---|---|---|---|---|
| ring 5, flux `pi/2` | `2.04e-14` | 0.75254 | 1.000 | 0.10998 |
| ring 7, flux `pi/2` | `2.22e-14` | 0.83339 | 0.999 | 0.08482 |
| tri ladder 6, flux `pi/2` | `2.13e-14` | 0.54294 | 1.000 | 0.31001 |
| tri ladder 8, flux `pi/2` | `3.91e-14` | 0.57340 | 1.000 | 0.36442 |

Every row satisfies the identity and still carries a phase. `focus` below 1 says the cloud is not
rank one, so no global rotation makes those weights real -- which is §5's statement that route B
carries the identity without positivity, obtained with no `K` and no mechanism. Nothing here is
thresholded: `focus` is compared to 1, the definitional value of a rank-one cloud.

This is the case that separates the two halves of §7. The channel comparison of §7.2 is exactly
saturated on all four rows and reports nothing; the one-sided triage reads the answer off the
cloud's own geometry.

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
`concentration` reports a directional statistic and an axial one, and they differ exactly where it
matters: an antipodal cloud reads `resultant ~ 0` and `focus ~ 1`. A real sign problem *is*
antipodal -- the weights sit at `+-1`, on one line through the origin -- while a phase problem is
spread around the circle. Measured, `focus` is **1.0000** on every real-weight row of that
experiment and `0.5734` to `0.7525` on its three genuinely complex ones; across the complex rows of
§7.1's table (`reads/expAV_monomial_conjugation.py`) as well it runs `0.543` to `0.833`.

`focus = 1` is rank one in the `(Re, Im)` plane, which is the statement that **a single global
rotation makes every weight real**. The rotation is read off the cloud's own leading direction, no
angle chosen, and applying it returns a real sign
problem carrying the same `|<sgn>|`: rows rotated by `0.7` and `1.9` present a maximum imaginary
part of `0.64` and `0.95`, de-rotate to `9e-16` and `4e-16`, and recover `0.93500` and `0.92500`
exactly. Where `focus < 1` no rotation helps and the residual stays at `1.0`.

**A global phase cancels.** A phase common to every configuration multiplies numerator and
denominator of `<O> = sum(O w)/sum(w)` alike, so it cancels exactly and costs nothing. Measured:
rotating the weights by `0.3` to `1.3` leaves the mean phase `|<u>|` at **0.93500** throughout,
unmoved (`reads/expBN_a_global_phase_and_the_wrong_estimator.py`; `u = w/|w|`, the frame `resultant`
is read on).

It is only visible to an estimator that takes the real part. `mean(Re u)/mean(|u|)` -- which is
correct for real weights and is the natural thing to carry over -- reads `cos(theta)` too small,
falling from `0.93500` to `0.25011` across those same rotations, and would overstate a cost that
goes as `1/<sgn>^2` by **13.98** times. `focus` therefore identifies when the real-part estimator
has stopped being the right one.

*What that leaves.* `focus` and `resultant` answer
different questions, and only the pair classifies. `focus = 1` says the cloud is rank one in the
`(Re, Im)` plane, so a single global rotation makes every weight real -- it does **not** say the
run is sign-free. What survives that rotation is a real problem whose severity is `resultant`:
`1.00000` on the clean lattice and on the inert twist, `0.93500` and `0.92500` on the staggered and
doped rows, which are sign problems and are exactly the ones §5's criterion speaks to. `focus < 1`
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
it: `focus = 1` says a phase that a rotation removes, leaving the sign problem §5's criterion
speaks to; `focus < 1` says a phase that no rotation removes, which is what route B produces.

So the phase is readable, and the blindness above is a statement about *comparisons* rather than
about what can be measured. The two halves fit together: a common mode is invisible to a read of the
relation and visible to a read of one side. And what the one-sided read returns is `|<sgn>|` itself,
so it carries the ceiling of the paragraph below -- seeing the phase exactly and paying the same
`O(1/<sgn>^2)` to resolve it.

**A second collision, with the channels exactly related rather than identical.** §5's
route B restores the identity on an odd cycle by threading it with flux, and the two channels then
satisfy `G_dn = 1 - G_up` to machine precision -- while the weight carries a phase. Four systems
whose channel relation is exact and whose sign behaviour is not
(`reads/expBD_route_calibration.py`, `beta = 8`, 250 draws; the deficit is `1 - |<w/|w|>|`):

| K | relation residual | sign deficit | coupling |
|---|---|---|---|
| 2x4 real, half filled | 1.1e-11 | **0.00000** | -1.0000 |
| ring 5, flux `pi/2` | **7.7e-15** | **0.15581** | -1.0000 |
| ring 7, flux `pi/2` | **5.0e-15** | **0.09241** | -1.0000 |
| triangular ladder, flux `pi/2` | **1.2e-14** | **0.46110** | -1.0000 |

The relation is exact on every row, both reads return the same saturated value on every row, and
the sign deficit runs from `0` to `0.478`. **No read of the relation between the two channels can
see a phase problem**, because the relation does not carry it: `G_dn = 1 - G_up` holds identically
whether the weight is positive or spread across the circle.

That is a second impossibility of the same kind as the one below and it is not the same statement.
Below, the two channels are identical *as data*, so nothing remains to compare. Here they are
distinct and exactly related, and the comparison is well posed and still blind.

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
while the frame the read is given is the Green's function's diagonal. Whether a read of a single
channel could recover that phase is not measured here.

*Which twists are live.* The total phase round the x-cycle is `Lx * phi`, so a twist that is a
multiple of `pi` is gauge-equivalent to a real boundary condition and the determinants stay real;
on `Lx = 4` that makes `pi/4` and `pi/2` inert. A periodic lattice of width 2 is inert at every
`phi`, because its x-direction has one distinct bond per row reached from both ends and the two
Peierls phases add to `-2t cos(phi)`. Every row therefore reports `max |Im G|`, and the width is
asserted, so a row cannot be read unless its flux is doing something.

Three quantities have now been checked for whether they measure severity -- the identity residual
(§5), the coupling strength, and the coupling tightness -- and none does. The severity is the
negative fraction, and the negative fraction is a count of parity disagreements. §8 gives the
reason none of the three reaches it: each has already summed that count away.

### 7.2 The coupling read, and its exact calibration

The lockstep of §3 is produced by the identity of §4, and the identity holds exactly at the
symmetric point of §5. That suggests a continuous measure of how far a system sits from that
point, and there is one: the alignment between the two channels, read as a coupling.

**The calibration is exact.** At half filling on a bipartite lattice, particle-hole symmetry gives
`G_up[i,i](x) + G_dn[i,i](x) = 1` configuration by configuration -- verified over 2000
configurations a row at a maximum residual of **4.9e-15** at `beta = 2`, rising to `1.2e-12` at
`beta = 8` (`reads/expTT_order_parameter.py`) -- so the two centred channels are exact
negatives and their coupling must read `-1`. It reads **-1.0000**, at every beta
(`reads/expQQ_coupling_vs_sign.py`), with the instrument's own exact re-pairing null placing the
permuted control at `|z| <= 1.7` (`reads/expAL_capability_across_representations.py`). On the
controlled axis of §5 it reads `-1.0000` with `tightness = 1.000`
(`reads/expAB_coupling_controlled.py`): the entire coupling in a single mode.

**What the deficit measures, exactly.** The read is a normalised alignment, so its deficit from
saturation is an angle:

    deficit = 1 - cos(angle between the two centred channel frames)

Write the violation of the particle-hole relation as `E = B~ + A~`, and split it into the part
along `A~` and the part perpendicular to it. Two consequences follow, and both are measured:

Both are measured on this section's own chains, at the same model, betas and seeds as the
capability table below (`reads/expBH_perpendicular_violation.py`):

* **The parallel part is invisible.** Rescaling one channel takes `|E|/|A~|` from `0` to `1` to
  `4` -- `B = 1 - A`, then `1 - 2A`, then `3 - 5A` -- and leaves the deficit at **exactly**
  `0.00000000` at every one. The read has no notion of the two channels' relative size, only of
  their alignment.
* **The perpendicular part is what registers**, and to leading order
  `deficit = |E_perp|^2 / (2 |A~| |B~|)`:

| `beta` | deficit | `\|E_perp\|^2 / (2\|A~\|\|B~\|)` | relative miss |
|---|---|---|---|
| 1.0 | 0.03574 | 0.03514 | **1.68%** |
| 1.5 | 0.04893 | 0.04769 | 2.55% |
| 2.0 | 0.06141 | 0.05956 | 3.01% |
| 3.0 | 0.10788 | 0.10135 | 6.06% |
| 4.0 | 0.12512 | 0.11618 | 7.14% |
| 6.0 | 0.20127 | 0.17326 | **13.92%** |

  The quadratic form is the leading term of `1 - cos`, so it is close where the angle is small and
  low once it is not: the miss grows monotonically from `1.68%` to `13.92%` as the deficit grows by
  `5.6x`. Both columns come from the same run, so the relative miss is exact within it
  and carries the digits shown; the deficit's own reproducibility across seeds is the capability
  table's (`reads/expAN_capability_table.py`) -- `0.6%` at the top and `10.4%` at `beta = 6`, so
  the last row's deficit is `0.20` as a measurement, not `0.20127`. Past the shallow rows the
  exact cosine is what holds, and the exact cosine is what the read computes. The deficit column is the capability table's own, which is
  what makes the comparison a statement about these chains rather than about a synthetic frame.

So the order parameter is the **angle between the two centred channel frames**, and what it tracks
as filling or temperature moves is the growth of the component of the particle-hole violation that
cannot be absorbed into a rescaling. That is why it is an onset detector: the angle moves as soon
as the violation appears, which is before `<sgn>` has left 1.

**The calibration comes from one of §5's two routes, not from the identity.** §5 shows the identity
holds when either `S K S^-1 = -K` (route A) or `S K S^-1 = -conj(K)` (route B) is available. Only
route B gives the exact particle-hole relation, and with it the exact `-1`
(`reads/expBD_route_calibration.py`, 250 configurations per row; the criterion itself is
`reads/expAO_spectral_criterion.py`, and the separation is gated in
`tests/test_spectral_criterion.py`):

| K | route A | route B | `max\|G_up + G_dn - 1\|` | strength |
|---|---|---|---|---|
| 2x4 real | yes | yes | 1.1e-11 | **-1.0000** |
| ring 8, real | yes | yes | 4.6e-13 | **-1.0000** |
| ring 6, flux `pi` | yes | yes | 1.5e-10 | **-1.0000** |
| ring 5, flux `pi/2` | no | **yes** | **7.7e-15** | **-1.0000** |
| ring 7, flux `pi/2` | no | **yes** | **5.0e-15** | **-1.0000** |
| triangular 4x4, flux `pi/2` | no | **yes** | **4.5e-12** | **-1.0000** |
| ring 6, flux `pi/4` | **yes** | no | **1.4e-01** | -0.9925 |
| ring 6, flux `pi/3` | **yes** | no | **1.9e-01** | -0.9863 |
| ring 8, flux `pi/4` | **yes** | no | **6.4e-01** | **-0.9453** |

Route B is the anti-similarity, and it is what forces `G_dn = 1 - G_up` configuration by
configuration -- the identity this section's calibration rests on. Where only route A is open the
lattice is still sign-free and the identity of §4 still holds exactly, but the two channels are no
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

and it holds to **4.4e-16** across six flux-threaded rings
(`reads/expBL_route_A_is_derived.py`; every row is route A only, and sign-free):

| K | r | `-1 + 2r` | measured |
|---|---|---|---|
| ring 6, flux `pi/8` | 0.00091 | -0.99817 | **-0.99817** |
| ring 6, flux `pi/4` | 0.00374 | -0.99251 | **-0.99251** |
| ring 6, flux `pi/3` | 0.00683 | -0.98634 | **-0.98634** |
| ring 8, flux `pi/4` | 0.02736 | -0.94529 | **-0.94529** |
| ring 8, flux `pi/3` | 0.02215 | -0.95570 | **-0.95570** |
| ring 10, flux `pi/4` | 0.00216 | -0.99568 | **-0.99568** |

The departure from `-1` under route A is the imaginary weight of the frame, exactly; the sign
problem is absent throughout.

`-1.0000` therefore comes from the particle-hole structure route B supplies, which the half-filled
bipartite case happens to have -- not from positivity, and not from the §4 identity. The three
coincide in every measurement in §§3-5 and come apart under flux.

**It is readable where the average sign is not.** Both columns come from the same
importance-sampled chains at each beta (`reads/expAN_capability_table.py`; `StableChains` is
gated against brute-force enumeration of every auxiliary field at 5e-15, which
`reads/expQQ_coupling_vs_sign.py` states at the head of its own output; `N = 8`, `U = 4`, `dtau =
0.125`, `t2 = 0.7`, `mu = 1.0`,
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
locked it is identically 1 with zero variance and no derivative. The coupling has already moved:
`0.03574 +- 0.00022` across four seeds, which is 162 times its own seed spread away from
saturation, on a row where `<sgn>` has no derivative at all.

*What `|z|` is, and what it is not.* `|z|` is the instrument's own resolution against its exact
re-pairing null, and that null is centred at `strength = 0`. For a single pair of columns it is
exactly `|strength| sqrt(n - 1)` -- pinned at nine settings in
`tests/test_the_reads_are_the_statistics_named.py` -- so it answers whether the two channels are
coupled at all, which is not the claim of this section. It is reported because `resolved` gates on it: an unresolved read returns `strength = 0`
rather than a number. It scales with the sampling budget rather than with the physics -- `50` at
`R = 24, n_meas = 15` against `137` at `R = 64, n_meas = 40` on the same rows -- and it falls down
the table because `|strength|` falls, which is the deficit rising. The claim that the deficit has
moved is carried by the seed spread in the table, not by `|z|`.

*The re-pairing null assumes exchangeability, and these are Markov chains.* §9.2c measures `2 tau`
for the influence function at `2.60x` at `beta = 2` and `4.24x` at `beta = 4`, so successive sweeps
are not independent and the permutation null understates the spread a re-pairing would really have.
`|z|` is optimistic by roughly that factor's square root. This does not reach the seed-spread
errors, which are taken across independent chains and carry the autocorrelation already.

**The rows are quoted to the precision they reproduce to, which is not the same precision.** The
deficit is reproducible to under 1% where the claim lives and to 10% at `beta = 6`, so the deep
rows carry fewer digits. That difference is itself the reason §7 claims onset and not severity: the
deficit is a converged estimate on the sign-free rows, and on the deepest row its error does not
shrink with sampling in the way a converged estimate's must
(`reads/expBB_deficit_convergence.py`, three seeds a cell): the relative seed spread falls from
**7.6% to 3.3%** where the sign is free as the sampling is doubled, and *rises* from **3.3% to
33.7%** where it is not, with the central value moving from `0.12081` to `0.20154` on the same
rows. A converged estimator cannot do either.

**The capability does not depend on the field distribution.** Run on the same
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
of the other -- and they agree to about 10% wherever the negative fraction is zero.

**The last row is a limitation, and it is the first row where the sign problem appears.** The Ising
deficit there spans `0.09336` to `0.95636` across four seeds -- a spread ten times its own lower
value. The deficit is reproducible while the negative fraction is zero and stops being so at the
onset, which bounds the claim to exactly the regime §7 makes it in. Reproducibility across seeds
is how that boundary is located, and it is why every row here is replicated.

**On the controlled axis of §5 the read gives a calibration and no trend.** Along `tp` at fixed
filling it is exact at `tp = 0` and unusable after: `strength` is `-1.0000`, `-0.0480`, unresolved
(`z = -0.55`), `-0.2794` -- not monotone, and the unresolved row is the one with the worst sign
problem (0.1825). The library returns zero there rather than a value, marking the read unresolved.

A monotone quantity does appear on that axis -- `tightness` falls 1.000, 0.848, 0.676, 0.564 --
and it is **not** a property of the coupling. At the symmetric point `G_dn = 1 - G_up` exactly, so
the centred channels satisfy `B~ = -A~` and the cross-covariance is `-A~' A~`, whose spectrum is
the single channel's own. Measured directly: the dominant share of channel A alone is 0.9986 where
the coupling's tightness is 1.0000, and away from the symmetric point the coupling's value sits
between the two channels' individual shares (0.770/0.963, 0.772/0.423, 0.470/0.613). `tightness`
is inherited from the channels' internal structure, which also varies with `tp`. It is a property
of each channel on its own, not of the coupling between them.

**The scope is by axis.** Along beta at fixed filling the deficit is monotone
and resolved at `|z| ~ 130` where the average sign is identically 1 -- that is the capability, and
it is what §7 claims. Along `tp` the read supplies an exact calibration at the symmetric point and
no usable signal away from it.

**A read of each channel against the system's own zero**
(`reads/expAT_balance_at_the_systems_own_zero.py`). `Screen.register` takes a lens's own
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

**The last column is an identity rather than a measurement, and that is the point.** The default
zero is the column mean, so the residual it scores is the centred sample's own mean -- identically
zero -- and the pvalue is exactly `1.00000` whatever the data, which
`tests/test_the_reads_are_the_statistics_named.py` pins on random columns at four offsets and
sizes. Withholding the system's law does not weaken the read, it
removes it, and the separation in the previous column therefore belongs entirely to the zero.

*What the read is.* `balance` scores the joint residual `||r||^2` against the
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
the six seeds rather than all six. What is asserted is therefore the reversal of ordering, not a
clean split, and the gate pins the overlap. The classical instrument of the same shape is
Hotelling's `T^2`; what the read supplies is the rank, derived from the spectrum rather than
assumed.

*Read the pvalue, not the boolean.* `closed` is a per-run decision at the reader's level and
fluctuates -- a sign-free ring fires on 9 runs of 12. The pvalue behind it is the quantity and its
median across independent runs separates with no overlap, which is the same discipline every other
sampled figure here uses.

*Scope.* This separates **real-weight** sign problems at fixed filling. It does not see a phase
problem: the route B rows, whose channels are exactly related and whose weights carry a phase,
read `0.49` and `0.38`. Route B leaves the channels exactly related, and this read asks each
channel against its own zero, which that relation leaves undisturbed.

**Why no reweighted estimator can be a severity meter**
(`closed/expAS_weighted_reads_are_capped.py`). Recovering a physical expectation from an
`|w|`-sampled chain requires the reweighted ratio `<O s> / <s>`, and correct importance sampling
draws at `|w|`, so a configuration enters that ratio carrying only its **sign**. Kish's effective
sample size of those weights is then

    ESS = (sum w)^2 / sum w^2 = (n <sgn>)^2 / n = n <sgn>^2

which is the `O(1/<sgn>^2)` cost of the sign problem written as a property of the weights, exactly
and for every sign pattern (§11, `Ceiling.kish_of_signs`).
`carriage` returns exactly that as `effective_n` -- verified at a ratio of `1.0000` across eight
`(n, <sgn>)` combinations -- and it is documented as *a property of the weights, not of the frames,
and the ceiling on what a reweighted estimator built from them can support*.

That is the sign problem restated as a property of the weights rather than of the model: **any**
estimator that reweights by the sign pays it, whatever the observable. It is a statement about
reweighted means, which is what Kish's formula is about, and it is not extended here to arbitrary
functionals of a signed sample.

*§7's coupling read is not such an estimator, and that is why it is readable.* The read is handed
the two channels' Green's-function diagonals and nothing else; the sign is collected alongside and
never enters it (`reads/expQQ_coupling_vs_sign.py`, and asserted in
`tests/test_the_reads_are_the_statistics_named.py`). It is an unweighted statistic of the
configurations the `|w|` sampler visits, so it does not pay the `1/<sgn>^2` cost -- which is
exactly why it carries a converged value on rows where `<sgn>` is `1.00000 +- 0.00000` and has no
derivative to read. What it reports is a property of the ensemble the sampler actually visits, and
the reason it cannot report severity is the coarseness below, not this ceiling.

**The read is an onset detector rather than a severity meter.**
No map from the deficit to `-ln<sgn>` is calibrated here, and on this data none can be: on the rows
where the deficit is useful, `<sgn>` is identically 1 and `-ln<sgn>` is identically 0, so there is
nothing there to calibrate against. §8 gives the reason this is not a matter of more data -- the
severity is set by the parity of a crossing count, and the deficit is a summary that has already
discarded that count.

**The reading is a synchronisation, and the sign of it is set by the mechanism.** A read that
returned `-1` wherever the model is sign-free would be indistinguishable from a read that had
learned one number, so it is put to the other place this model is provably sign-free. The two
places are structurally opposite. Repulsive `U` decoupled in the spin channel is sign-free at half
filling because `G_dn = 1 - G_up`: the channels are exact negatives, anti-synchronised. Attractive
`U` decoupled in the charge channel is sign-free at **any** filling because the field couples to
`n_up + n_dn - 1`, both spins see the identical diagonal factor, and the weight is `det^2`. Same
instrument, same call (`reads/expAE_both_mechanisms.py`, 2x4, `|U| = 4`, `beta = 10`, 300
configurations per seed, **six seeds per row reported as a range**):

| U | channel | mu | negative fraction | strength | `max\|G_up - G_dn\|` |
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

*What that last column is* (`reads/expBI_greens_diagonal_at_large_beta.py`). `G` here is
`(I + B)^-1` in the conditioned frame of §2, and its diagonal is an occupancy only while that frame
is well conditioned. `G` is not symmetric, so its diagonal is not bounded by its spectrum once the
eigenvector basis is ill-conditioned. Measured on this lattice at half filling, the diagonal stays
inside `[0, 1]` to `beta = 6` and leaves it after -- `-0.2754` to `1.1181` at `beta = 8`, `-0.5423`
to `1.2563` at `beta = 10` -- so the column is a maximum over 1800 draws of a quantity that is no
longer a filling. The relation the §7.2 calibration rests on is unaffected and exact throughout:
`max|G_up + G_dn - 1|` runs `2.7e-15` at `beta = 2` to `6.2e-14` at `beta = 12` on the same draws.
The relation *between* the channels survives where neither channel's diagonal is an occupancy, and
that relation is what §7 reads.

**The last column is why the `+1` rows are not the evidence.** On the attractive rows the two
channels come out bit-identical, so the read is being handed the same array twice; `+1` is the
sign of the reading and says nothing about the instrument's discrimination. What the table
establishes is the **contrast**. Doping is exactly what takes the repulsive read from `-1.0000` to
unresolved, and it leaves the attractive read where it was -- because the attractive model stays
sign-free under doping and the repulsive one does not. So the read tracks whether the channels move
together, which is the property the sign depends on, rather than filling or distance from a
particle-hole symmetric point.

Saturation and a zero negative fraction are **not** equivalent. At `mu = 0.8, beta = 6` the
negative fraction runs `0.0000` to `0.0100` across seeds while the read runs `-0.8665` to
`-0.5358`: the departure from saturation exceeds the negative fraction on every seed, including
the seed where the negative fraction is exactly zero. That is §7's own result -- the read departs
before the average sign does -- appearing again.

No filling is estimated in that table. These are uniform draws over the field rather than
importance samples, so no expectation value taken from them carries information; the rows are
labelled by their inputs.

**The boundary of the reading, measured.** Both saturated cases above relate the channels
*affinely*, so the test is a mechanism whose positivity comes from somewhere else. Wu and Zhang [6]
and
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

### 7.3 Where the comparison stops: a phase common to both channels

**So the correspondence between saturation and sign-freedom is a statement about real
determinantal weights**, whose sign is a product of two determinant signs. That is the setting
every row of §§3-5 was measured in, and it is where that correspondence holds.

The reading itself goes further. A complex weight is unreachable by
a **comparison** between the two channels -- for a reason given below, that its phase is common to
them -- and is read exactly by a **one-sided** read of the weight itself. What the two-mode
diagnosis further separates is whether that phase is global, in which case it cancels in
`<O> = sum(O w)/sum(w)` and costs nothing -- leaving a real weight whose own sign problem
`resultant` then measures -- or configuration-dependent, in which case it is the problem itself and
no rotation reaches it.

**The Kramers reading [8] is the same arithmetic as §5's route A, with one sign changed.** There
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

So saturation is not necessary for sign-freedom: the Kramers rows are sign-free and read `0.59`
**because** their frames are 20% imaginary.

**Is the boundary intrinsic, or an artifact of reading one scalar?** Two things were checked, and
they answer in opposite directions (`reads/expAH_boundary_is_intrinsic.py`).

*The `+1 / -1` split needed the mechanism to be known.* Handing the complex frame to the read
directly reproduces the concatenated values -- the library reduces a complex frame through the same
real embedding `iota(x) = (Re x, Im x)` -- and the experiment asserts that agreement. The read's
`phase` field is `0.0101` and `0.2120` on the two Kramers rows, so the
alignment is not purely real and the signed real part is not the whole of it; but a phase of
`0.2120` rad inflates the magnitude by `1/cos(phase) = 1.023`, taking `0.4794` to `0.490`. The
unsaturated reading is not a magnitude hidden behind a sign. Asking the instrument for the
directions itself,
via `principal_directions` on channel A, does **not** recover the split: the per-direction couplings
come out `-0.462, +0.854` and `-0.509, --, +0.232`, not `+1` and `-1`. Those are the directions of
A's own variance, not the eigendirections of the A-to-B relation. So the blockwise saturation of the
previous table was recovered using knowledge of the mechanism, and a mechanism-agnostic application
of this read stays unsaturated.

**Why the relation reads are blind to a phase: the phase is a common mode.** Placing the two
determinants on the screen rather than the two Green's functions shows what is happening. On every
route B row the unit-modulus determinants couple at `+1.000` with `z ~ 21` -- their phases are
perfectly aligned. The weight is `det_up * det_dn`, so its phase is twice a phase the two channels
*share*, and it varies configuration to configuration. Two aligned quantities moving together are
invisible to any comparison between them, which is what every two-sided read reported.

## 8. What does not carry the sign

The sign is a per-configuration fact, and a per-configuration order parameter would be worth more
than an ensemble one. Three candidates were derived and tested. All three fail, and they fail for
the same reason.

| candidate | what it is | why it fails |
|---|---|---|
| `max_i \|G_up[i,i] + G_dn[i,i] - 1\|` | the residual of the particle-hole identity | a coarse summary of the object carrying the sign. Median 1.1x to 6.4x higher on negative-weight configurations, but positive-weight ones reach 190 while negative ones go down to 0.22 |
| `min_k \|lambda_k(M)\|` | the distance to a sign flip, in the conditioned frame | a **distance is sign-blind**. Negative-weight configurations sit 2 to 8x closer to a zero in median, and the distributions overlap completely: a configuration can approach a crossing without making it, or cross and travel far past |
| the §4 identity residual | the exact failure of an exact relation | a statement about **magnitudes**. Median ratio 0.74 to 1.10 -- no discrimination at all |

The reason is structural. `det(I + B) = prod_k (1 + lambda_k)`, so the sign is the **parity** of the
number of eigenvalues past the crossing -- exactly, and for any product
(§11, `Parity.prod_pos_iff_even_neg`).

*What that rules out is narrower than it first appears.* The sign is a discontinuous function of
the spectrum, so no continuous function *equals* it (`sign_not_continuousAt_zero`,
`no_continuous_function_is_the_sign`). A continuous function can *determine* it: the product does,
which is `det_determines_sign`, and the product is continuous in the eigenvalues. Continuity alone
is therefore no obstruction, and "a parity is discontinuous" settles nothing on its own.

The obstruction is coarseness, not regularity. Each of the three candidates is a summary that has
already discarded the crossing *count* -- a maximum residual, a minimum modulus, a magnitude
comparison -- and a count modulo two cannot be recovered from a quantity that never encoded it. The
one cheap-looking scalar that does keep it is the determinant itself, and its cost is what the
question was asked to avoid. The §4 identity *holding* forces the signs to lock; the identity
*failing* only permits them to differ, and says nothing about which configurations do.

It is the same wall §9 meets from the other side, and it is why the constrained path's node is
hard: what has to be predicted is a discrete global invariant, and the available cheap observables
are continuous local ones.

**The parity argument covers real weights only, and the sweep that shows it says more.** A parity
is not continuous; a *phase* is. The model carries the instrument for testing that: both
`(n_up - 1/2)(n_dn - 1/2) = 1/4 - m^2/2` and `= rho^2/2 - 1/4` are exact identities on the four
states of a site, so splitting `U = (1-theta)U + theta U` and decoupling each piece in its own
channel gives a family in which **every theta is the same physics** -- the model's own gate --
while `theta = 0` is real and `theta = 1` is complex
(`reads/expAI_phase_is_not_a_parity.py`, 2x4, `U = 4`, `beta = 4`, 400 prior draws x 4 seeds):

| mu | theta | mean `\|Im w\| / \|w\|` | `1 - \|<w/\|w\|>\|` | neg fraction of `Re w` | strength over seeds |
|---|---|---|---|---|---|
| 0.0 | 0.00 | 0.0e+00 | **0.0000** | **0.0000** | **-1.0000 to -1.0000** |
| 0.0 | 0.50 | 3.4e-15 | **0.0000** | **0.0000** | -0.1725 to 0.0653 |
| 0.0 | 1.00 | 2.2e-14 | **0.0000** | **0.0000** | **+1.0000 to +1.0000** |
| 0.8 | 0.00 | 0.0e+00 | 0.0000 | 0.0006 | -0.8956 to -0.8895 |
| 0.8 | 0.50 | 6.4e-01 | 0.7159 | 0.4050 | -0.0790 to -0.0458 |
| 0.8 | 1.00 | 6.4e-01 | 0.6460 | 0.4956 | **+1.0000 to +1.0000** |

*The two complex-weight columns, and why neither is a count.* `|Im w| / |w|` is bounded by 1 and is
a share of the weight's magnitude; a ratio to `Re w` instead would be unbounded and would diverge
wherever `Re w` passes near zero, which is exactly what a weight crossing between positive and
negative does. `1 - |<w/|w|>|` is the mean-phase deficit, and it is zero when the weights are real
and positive whatever their spread in magnitude.

*A fraction of draws carrying a phase would need a chosen angle, and is not quoted.* Counting the
draws whose argument is neither near `0` nor near `pi` requires a value for "near"; at `0.1` radians
it undercounts, reading
`0.9350` where every draw in the row carries a phase. Removing the window does not repair it --
`Im w != 0` then reads `0.9925` to `1.0000` on the `mu = 0` rows, whose weights are real, because at
that point it is counting floating-point dust at `1e-15`. There is no cut-free form of it at finite
precision. The two columns above need no window and separate the same two populations by fourteen
orders of magnitude, so the claim is made with them.

**The phase is not a function of the field either, and this is asked on the whole field.** §4 gives
the magnitude ratio exactly from the field,
`ln|det_up| - ln|det_dn| = -dtau L tr(K) + lambda sum(x)`, so the natural question for a complex
weight is whether `arg(w)` has an analogous relation. If it did, the phase could be computed
without the determinant and the sign problem would not be a sampling problem at all.

A null on any particular functional says only that the one functional does not carry the phase, so
the question is put on the largest linear basis available
(`reads/expAX_is_the_phase_a_read_of_the_field.py`): the frame is
every one of the `L x N` Ising variables per configuration, nothing selected and nothing
summarised, read by `carriage` -- whose null is the exact re-pairing of weights with frames --
against the phase's two components. Every linear functional of the field is a vector in that
basis, so the read subsumes them all.

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
this leaves open: a nonlinear function of the field is outside a linear alignment's reach, and a
relation below the instrument's resolution at 400 configurations would not register.

**The `mu = 0` block is the result.** Same Hamiltonian, same physics, weights real and positive at
every theta -- there is no sign problem anywhere in that block, charge channel included -- and the
read runs the full range from `-1.0000` to `+1.0000`. A quantity that moves that far while both the
model and its sign structure hold still is reading the decoupling, not the model. **The read is a
property of a (Hamiltonian, decoupling) pair** -- as is the sign problem itself. So the read is
quoted about a Hamiltonian together with the decoupling used, and every claim in §§3-7 is a claim
about the *spin* decoupling named there.

The doped block supplies the complex weights the parity argument does not reach: the weights are
about 60% imaginary by magnitude -- `mean |Im w| / |w|` runs `0.568` to `0.647` there against
`3e-15` on the half-filled rows of the same family -- so there the sign problem is a *phase*
problem. The read does not reach it either: at `theta = 1` it is saturated at exactly `+1.0000`
while 49.6% of draws have negative real part -- the collision of §7 again, since `theta = 1` sets
`lambda_s = 0` and the two channels coincide.

**And in the interior of the family there is no lockstep at all.** Measuring what actually delivers
positivity at each theta (`reads/expAJ_how_positivity_arises.py`, half filling, 300 draws x 3
seeds): at `theta = 0` the determinants are real and their signs agree on **1.0000** of
configurations -- the mechanism of §§3-4. At `theta = 1` they agree on `1.0000` trivially, because
`lambda_s = 0` makes them the *same number*. In between, at `theta = 0.25, 0.50, 0.75`, the two
determinants' real parts agree on **0.5167, 0.5178, 0.5289** of configurations -- chance -- while
the weight is exactly real and positive on every draw. The lockstep is absent there and the model
is sign-free anyway. That is the mechanical reason the read decays across the family: it compares
the two channels, and in the interior the channels are not what carries positivity.

*One caution about that experiment.* The quantity `arg(charge_phase) + arg(det_up) + arg(det_dn)`
is `arg(w)`, so its vanishing is positivity restated rather than an independent invariant, and its
*maximum* over draws summarises nothing: a maximum of `pi` means one draw was negative-real, not
that all draws were real. The distribution above is what the claim rests on.

*No reweighted expectation is quoted from that table.* Fields are drawn from the prior and the
effective sample size is 1.5 to 13 of 400, which supports none. Every column is a property of the
prior ensemble: the negative fraction of `Re w` among the draws, the typical size of `Im w` against
`Re w`, and the coupling of the two channels over those draws. The §7 coupling along beta is a
different object, measured on importance-sampled chains, and the two are not compared here.

*Where the complex weights are.* Turning the decoupling toward the charge channel does not by
itself produce them: at half filling the whole family is sign-free, the charge channel included,
with the weight real to 1e-15 and no negative real part at any theta. Complex weights appear under
doping, which is why the table carries both blocks.

## 9. What any read of the sign must contend with

Each result below is a structural constraint on what can be read from sampled data, and together
they are why §7 reads a coupling between two channels and claims onset rather than severity.

**9.1 Reads that score severity from the sampled data.** Every one reduces to the average sign
under a different, cheaper measure. The anchor is where the reduction can be written down and
checked (`closed/expAQ_anchor_reduces_to_the_sign.py`): with `B = diag(sigma) A`, the read returns
the average of `sigma` **reweighted by each row's squared length**, agreeing with it to `8e-05` to
`1.2e-03` across five draws. The residual is the centring the read performs and the plain average
does not; against its own definition on the centred frames the read is exact, which is arithmetic
rather than a finding. So a read of this kind does not fail to see the sign -- it sees a reweighted
one. They fail for one reason: the sign correlates
with `|D|`, since negative weights cluster near the zeros of the determinant, so any weighting that
is not `|D|`-weighted mis-weights exactly the configurations that decide the answer.

**9.2 A magnitude read cannot see a sign, for any exponent**
(`closed/expAP_phase_blindness.py`; proved for every `q` in §11, `Blindness.sum_abs_pow_blind`).
With `B = diag(sigma) A` the magnitudes agree cell for cell,
so `sum |A|^q` and `sum |B|^q` agree for **every** `q` -- measured at `q = 0.5, 1, 2, 3` across
three frames differing by row sign flips, with a worst disagreement of **exactly 0.000e+00**, not a
tolerance. The singular spectrum is blind for the same reason, `diag(sigma)` being orthogonal:
maximum singular-value difference `0.000e+00`. The file carries its own positive control, and it is
what makes the blindness a property of magnitude reads rather than of the frames -- the coupling
*between* the two sides returns `+1.0000`, `0.0000` and `-1.0000` on the same three frames. That is
why §7 reads a coupling and not a magnitude: the sign of a determinantal weight lives in the
relation between the channels, not in the size of either.

**9.2b What can and cannot see the sign, classified.** Three families of read were put to the
question.

| family | status |
|---|---|
| reweighting by the sign, which any physical expectation needs | **capped** at `n <sgn>^2` -- §7, from Kish's effective sample size |
| a read of the relation between the two channels | **blind** -- §7, twice: identical channels, and exactly related channels |
| a read of one channel alone | **measured: no ordering across lattices** |

The third is the one neither impossibility rules out, so it was measured
(`reads/expBG_single_channel_has_no_indicator.py`, 200 draws a point, `beta = 8`, `U = 4`). Two
populations, and they answer differently.

*Along one lattice's flux sweep.* Threading the frustrated ladder through a full period takes the
sign deficit `0.07000` up to `0.46508` at `pi/2` and back to `0.07000`, so a read that carried the
sign has to turn round with it. Over those eleven points the deficit tracks itself across seeds at
`rho = +0.973`. `phase` returns `+0.159` at `p = 0.640` -- no relation. `attenuation`, `top_share`
and `dominance` return `-0.661`, `-0.770` and `-0.770`, so they do move with the deficit on this
lattice, at well under its own reproducibility.

*Across lattices.* That ordering does not survive the change of population. On **eighteen**
lattices running from sign-free to a deficit of `0.46508` -- clean `2x4` and `2x6`, open and
periodic chains, a star, four flux-threaded rings, four staggered amplitudes, a doped cell, a
next-nearest-hopping cell, and two odd rings and a ladder at `pi/2` -- no single-channel read orders
them: `phase` `+0.273`, `attenuation` `+0.007`, `top_share` `+0.017`, `dominance` `-0.017`, every
one at `p >= 0.272`.

*And that null has power.* At `n = 18` a rank correlation of `|rho| >= 0.468` would reach
`p < 0.05`; the largest observed is `0.273`. Among the **ten sign-free lattices alone** --
every one at deficit exactly `0.00000` -- `attenuation` spans `0.05161` to `0.79089`, a wider range
than between them and any lattice here that has a sign problem. A quantity that varies more among
sign-free systems than between them and a sign problem is not an indicator of one.

What survives all three is identification of an operator from a correlation sequence, which
aggregates no signed configuration and reads no channel relation. That is the family §10 points at.

**9.2c The ceiling of §7 is sound but not tight, and the slack is measurable**
(`reads/expBC_ceiling_is_not_tight.py`).
Kish's formula assumes independent draws. A DQMC run is a Markov chain, so the evidence it
actually carries is `n <sgn>^2 / (2 tau_int)`, and everything above uses the ceiling as an upper
bound -- which it remains, since the correction only makes the true figure smaller, which
strengthens the argument that no weighted read is a severity meter. The ceiling bounds the evidence
a run can carry; it is not the evidence a given run holds, and it is not an error bar.

Which `tau_int` matters is the standard question with the standard answer: there is no single
autocorrelation time for a chain, each estimator has its own, set by its influence function. The
reported quantity is the reweighted ratio `<O s> / <s>`, whose linearisation is
`z_t = (O_t s_t - r s_t) / <s>` at the pooled `r`, and `z` relaxes more slowly than `s` does. The
sign's own `tau` accounts for two thirds of the gap.
At `N = 8`, `U = 4`, `beta = 4`, `t2 = 0.7`, `mu = 1.0` (a real sign problem at `<sgn> = 0.827`),
against a reference built from the scatter of 128 independent replicas -- four seeds of 32, which
is the run `--seeds 31 32 33 34`. The one-seed default gives the same picture more coarsely and
different digits, so the rows below are that invocation and not the bare one:

| | |
|---|---|
| `tau` of the sign | `1.106 +- 0.140` |
| `tau` of the influence function, per diagonal | `1.50 +- 0.15` to `3.18 +- 0.43` |
| evidence overstated by the ceiling | `3.0x` to `6.4x`, mean `4.15x` |
| error bar implied by the ceiling | `1.98x` too small |
| corrected by the sign's `tau` | `1.33x` |
| corrected by the influence function's `tau` | `0.97x` (reference `tau`; `1.21x` from one chain -- see below) |

The influence function's `tau` is quoted per observable because that is what it is: there is one
per estimator, and the eight Green's-function diagonals of this lattice do not share it. They are
read off the same chains and are not independent of one another, so the mean carries no `1/sqrt(8)`
and the range is the result. The error-bar row is a median over the eight diagonals; the two
correction rows divide that median by `sqrt(2 tau)` at the mean `tau`, so they describe a typical
diagonal rather than any particular one.

**The slack is ordinary autocorrelation.** Holding `beta = 4` fixed and
switching the sign problem off with `t2 = 0` gives `<sgn> = 1.00000` exactly -- so the ceiling
reads `n`, the whole sample -- and the slack is still `3.66x`, against `4.15x` with the sign
problem on. A factor of `1.13x` between them, with the sign problem carrying the smaller part of
it. On that row the sign has no variance at all, so it has no autocorrelation time to correct by,
and the slack is there regardless.

**`beta` is the other axis the slack could be attributed to, and it was scanned** (`--beta`, one
seed a point). The slack is not monotone in `beta`:

| `beta` | `<sgn>` | `2 tau_infl` |
|---|---|---|
| 2 | 0.99792 | 2.60x |
| 3 | 0.95590 | 3.56x |
| 4 | 0.83056 | 4.24x |
| 5 | 0.65417 | 3.61x |

`<sgn>` falls monotonically, by a third across the scan, and the slack does not follow it: it rises
to `beta = 4` and then comes back down while `<sgn>` keeps falling. A quantity controlled by the
sign problem could not do that. What `beta` does do is lengthen the imaginary-time extent and slow
the chain, which is ordinary, and the `t2 = 0` row above is the same statement with `beta` held
fixed instead.

None of the machinery here is new. That Kish's
formula assumes independent draws is textbook, and so is the treatment of a derived quantity:
Wolff's `Gamma` method [17] propagates a function of primary observables by projecting their
fluctuations through its derivatives, `pi_F = sum_alpha (dF/da_alpha) pi_alpha`, which for
`F = <O s>/<s>` is exactly the `z_t` above. It has done so since 2004 and ships in the standard
packages. What is reported here is therefore an applied correction to *this* paper's bound --
`n <sgn>^2` is stated in §9 without the autocorrelation factor that standard error analysis would
supply, and on this rig that factor is `4.15x` -- not a new way of obtaining it.

The control identifies the chain as the cause. Drawing the same sweeps i.i.d.
with replacement from the pooled ensemble preserves the marginal, and so preserves `<sgn>` and
`n <sgn>^2`, while removing the dependence between consecutive entries: `tau_infl` falls from
`2.074` to `0.536` and the sign's from `1.106` to `0.463`, both the independent value of one half,
and the ceiling goes from `1.98x` too small to `1.02x`, which is tight. Permuting each chain's
sweeps in time is a different control and carries no information here: it leaves each chain's mean
where it was, so the scatter of chain means is invariant by construction. Residual thermalisation
is excluded separately, because a chain still relaxing would inflate the
scatter through the same channel as autocorrelation: the across-replica scatter of the first half
of each
chain and of the second differ by a median factor of `1.07` over the eight diagonals, spanning
`0.93` to `1.17`. A chain still thermalising would have the first half scattering more.

`tau_int` is read with `dynamics(z).reconstruct_decay()`, which rebuilds `C(tau)` from the
operator's modal powers and eigenvalues and sums it over all lags, so no window, truncation lag or
multiplier is chosen: moving the summation limit from 1000 to 8000 changes the answer by
`0.0e+00`. Two limits are measured. It must be `reconstruct_decay`, which
reads the **connected** operator; `rates().dominant` reads the raw one, where the constant
function is an eigenmode with `|mu| = 1`, and on a sign sequence with mean `0.83` it returns that
mode instead of the decay. And a scalar frame fits a single exponential, while this chain's
influence function has a slower tail, so the one-chain read understates `tau` by about `1.5x`: it
takes the error bar from `1.98x` to `1.21x`, not to `1.00x`. Closing the rest needs the replicas.
The estimator itself is not new -- window-free routes are established practice, `emcee` fitting a
second-order ARMA for this purpose [16] and Wallerberger's log-binning targeting lattice and Monte
Carlo chains [15] -- and each keeps a constant of its own, a model order and a binning factor
respectively. The claim made here is the measured correction to this paper's own bound, not the
means of computing it.

**9.3 The scalar decoupling family is complete** (`model2d.single_site_identity`,
`reads/expAI_phase_is_not_a_parity.py`). Using `n^2 = n`, the interaction has exactly two
quadratic forms, with the `m^2` coefficient pinned at `-U/2` and the `rho^2` coefficient at `+U/2`
for any rewriting; the "shift around alpha" freedom is one-body and is absorbed into the hopping
matrix. Both rewritings are checked by quadrature on a single site at every mixing
(`model2d.single_site_identity`, 40 Gauss-Hermite nodes): over `U` in `{2, 4, 8}`, `dtau` in
`{0.0625, 0.125, 0.25}` and `theta` in `{0, 0.25, 0.5, 0.75, 1}`, all four occupation states of
each cell, the worst relative error is **6.5e-16**; a single cell alone reads `2.1e-16`, so the
figure is a property of the whole grid.

Splitting `U` between the spin and charge channels is therefore the whole family, and the mixing
was swept across it. Which decoupling has the worse sign problem is a question about `<sgn>`, and
`<sgn>` is an expectation, so it is **summed rather than sampled**
(`reads/expBJ_decoupling_family_exact.py`): every auxiliary field is enumerated -- both of them, the
real spin field and the imaginary charge field, `4^(N L)` configurations -- and `<sgn>` comes out as
`|sum w| / sum |w|` with no sampler, no seed and no effective sample size. On a `2x1` lattice at
`L = 4`, `dtau = 0.25`, `U = 4`, that is 65536 field pairs a row:

| `theta` | `<sgn>` at `mu = 0` | `<sgn>` at `mu = 0.8` | mean-phase deficit | negative fraction of `Re w` |
|---|---|---|---|---|
| 0.00 | **1.00000000** | **1.00000000** | 0.00000000 | 0.00000000 |
| 0.25 | **1.00000000** | 0.98315847 | 0.02467001 | 0.00000000 |
| 0.50 | **1.00000000** | 0.96202358 | 0.06589094 | 0.00122070 |
| 0.75 | **1.00000000** | 0.93210857 | 0.16293791 | 0.04254150 |
| 1.00 | **1.00000000** | 0.85036742 | 0.38931226 | 0.15625000 |

At half filling every `theta` is sign-free exactly, which is §8's statement about this family
restated without a sampler. Under doping `<sgn>` falls monotonically from `theta = 0`, and no
interior `theta` beats the pure spin decoupling by more than `1e-12`: **the spin channel wins with
no interior optimum**, on an exact sum rather than on draws from the prior. The SU(2) vector
decoupling lies outside the family; it was derived and built, and is 2.2x worse at 2x2 and
unresolvable at 4x4.

*The lattice is small because exactness costs exponentially.* What the enumeration establishes is
the ordering across `theta`, which is what the claim is; it is not a statement about how severe the
sign problem becomes at production size.

**9.4 A constant contour shift buys nothing here** (`closed/expAR_contour_2d.py`). Deforming the
integration contour is the one route that can change the exponent rather than the prefactor. Two
shift shapes were tried, neither fitted -- a uniform `c`, the one-dimensional deformation carried
over, and one staggered by sublattice, which is the model's own structure and which a uniform shift
cannot see. Measured on `4x4`, `mu = 1.0`, `U = 6`, `beta = 6`, against an undeformed mean phase of
**0.4167 +- 0.0554** re-measured in the same run, every gain is below 1: the best is `0.23` and the
worst `0.08`, so the deformation makes the phase problem worse at every amplitude tried. The scan
runs to `c = 4`, far past any plausible optimum, because a thimble need not lie near the real axis
and a second regime at large shift would have falsified the reading; there is none. The point was
selected by measuring undeformed phases first -- at `mu = 0` this lattice is at §5's sign-free
point, where the undeformed phase is `1.0000` and any shift can only spoil it -- and the undeformed
value is printed beside the scan.

*What that does not close.* A constant shift is the crudest member of the deformation family. A
thimble method deforms along a flow, and nothing here measures one.

**9.5 Complex Langevin** [12, 13] (`closed/expCC_cl_fails.py`, `closed/expDD_cl_reads.py`,
`closed/expEE_no_error_bar.py`, `closed/expJJ_seed_spread.py`). Built and gated: the analytic
drift against a central finite difference of the complexified action at `7.4e-10` in the spin
channel and `1.4e-09` in the charge channel, with the wrong-convention control at `0.40` and
`0.36` (`tests/gate_main.py`). It converges to the wrong answer here, and only in the charge
channel. On the doped rows of `closed/expDD_cl_reads.py` the charge channel returns a density of
`1.05095`, `1.10220` and `1.17205` against exact values of `1.03977`, `1.08089` and `1.13979` at
`mu = 0.3, 0.6, 1.0` -- resolved at `z = 64.4`, `85.4` and `33.6` -- while the spin channel on the
same grid returns `1.03974`, `1.08084` and `1.13987` against the same exact values, matching to
better than `1e-4` on every row. The uncertainty on those figures is estimated from reproducibility
across seeds; a within-run
formula and a seed-to-seed spread differ by more than an order of magnitude on this data, and the
seed spread is the one that holds.

**9.6 The constrained path's trial wavefunction** [9, 10] (`closed/expW_trial.py`,
`closed/expX_dial.py`, `closed/expKK_multidet_bias.py`, `closed/expLL_unfitted_trial.py`,
`closed/expMM_self_consistent.py`, `closed/expNN_selection.py`). The bias is set by the node and by
nothing else, and it moves by a factor of 24 with the trial at fixed coupling -- across the
staggered-field family at `U = 8` it runs `+0.05246` at `h = 0` to `+1.25917` at `h = 4`. The
computable criterion points the wrong way: `argmin <Psi_T|H|Psi_T>` is *anti*-correlated with the
bias over a nine-candidate pool (`rho = -0.80`, `p = 0.0096`). Fitted to the exact ground state,
four to six non-orthogonal determinants cut the bias by `1.7x` to `10.4x` depending on the
coupling, so the prize is real -- and **no answer-free construction beat the plain free
determinant**, every candidate coming in 5 to 15x worse. What blocks the route is not generating
determinants but that the only objective which selects them correctly is the one that needs the
answer.

*Two things make that band a band rather than a figure.* The ceiling is not monotone in `k`.
Relative to the single-determinant walk, a run of `closed/expKK_multidet_bias.py` gives `0.970`,
`0.379`, `0.044` at `U = 4` for `k = 2, 4, 6`, and `1.165`, `0.493`, `0.079` at `U = 8`: the second
determinant helps at weak coupling and *hurts* at strong. Only the `U = 12` reversal clears its own
error bars on that run -- `3.2` standard errors from the quoted biases, against `1.0` at `U = 8`
and `0.1` at `U = 4`. A second run of the same script gives `0.777`, `0.398`, `0.072` at `U = 4`,
so the direction survives a re-run and the digits do not.

*The fit is why.* It does not reproduce across processes
(`closed/expBF_multidet_reproducibility.py`): three refits inside one process agree to all
seventeen significant figures, a spread of exactly `0.0e+00`, while four separate processes return
four distinct values spanning of order `1e-7` -- enough to move the bias by more than the walker
seed spread, so the largest reduction reads `10.43x` on one full run and `9.00x` on another. **The
error bar recorded beside each entry understates it**, because it varies walker seeds with the fit
held fixed and has no term for the fit. The low end is stable across runs (`1.74x` and `1.73x`);
the high end is quoted as a band.

## 10. Outlook

The reframing changes what a positivity-restoring representation has to achieve.

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
the bulk drifting inward. The individual channels cross on around 11% of configurations. What the
symmetry supplies is not distance but **synchrony** -- the two channels cross together.

**The criterion says what a construction has to achieve.** §5 turns the
requirement into a decidable property of the one-body matrix: a diagonal unitary `S` with either
`S K S^-1 = -K` or `S K S^-1 = -conj(K)`. It is constructive on the hopping graph -- it names the flux that
opens a route where frustration closes every flux-free one -- but only route A leaves the weights
real, so on a graph with an odd cycle the identity can be restored and the positivity cannot.

It is also a **closed door on doping**, in one line. Every diagonal matrix commutes with every
diagonal conjugation, so `S K S^-1` leaves `diag(K)` untouched; both routes then demand
`K_ii = -K_ii` -- and `K_ii` is real because `K` is Hermitian -- so `diag(K) = 0` is forced. A
chemical potential can therefore never be accommodated, by any flux, on any graph, at any size.
Measured over 84 flux values -- rings of 5, 6, 7 and 8 sites at 21 fluxes each, a full period
on every graph -- the best residual of all 84 is `1.776e-14` at `mu = 0`, and moving the diagonal
takes it to `8.926` at `mu = 0.2` and `26.63` at `mu = 0.6`
(`reads/expBK_doping_closes_every_flux.py`). Flux is the only freedom a diagonal-unitary
conjugation has on a cycle, so sweeping it over a period sweeps everything route B could use. The
measurement samples; the statement quantifies, so it is also proved -- §11,
`Doping.doping_closes_both_routes`.

So a construction seeking positivity away from half filling cannot get there by choosing a lattice,
a flux or a size: the obstruction is a diagonal term, and no conjugation of this kind removes one.
It does not need to push the spectrum away from anything either. It needs to reproduce a **pairing**
between two channels whose magnitudes are unrelated over 8 to 16 orders and whose crossings coincide
exactly. That is what the algebraic
positivity results -- Majorana, Kramers, split-orthogonal [7, 8] -- achieve when they achieve it,
by making the weight a squared modulus so the question cannot arise; it is also what a
meron-cluster decomposition [11] achieves, by making the cancelling configurations cancel exactly
rather than estimating their residue; and it is why a perturbative or
reweighting fix has nothing to expand in, since there is no small residual, only a discrete
coincidence that either holds or does not.

**What a read of this problem can carry, classified.** Three families were put to the question.
Each entry is established in the section named, and the first two are proofs rather than
measurements -- machine-checked ones, in the sense of §11.

| family | status | where | machine-checked |
|---|---|---|---|
| reweight by the sign | **capped** at `n <sgn>^2` | §7 | `Ceiling.kish_of_signs` |
| read the relation between the two channels | **blind**, two ways | §7 | `Blindness.sum_abs_pow_blind` for the magnitude half |
| read one channel alone | **no indicator**, measured | §9.2b | -- |
| identify an operator from a correlation sequence | not excluded by any of the above | -- | -- |

The ceiling is the sign problem restated: correct importance sampling draws at `|w|`, so a
configuration enters any weighted average carrying only its sign, and Kish's effective sample size
of those weights is exactly `n <sgn>^2`. Because that number is a property of the **weights**, it
bounds every estimator that reweights by them. §7's coupling read does not reweight -- it is an
unweighted statistic of the configurations the `|w|` chain visits -- which is why it is readable
where `<sgn>` is not; what stops it reporting severity is §8's coarseness -- the read has already
summed away the crossing count the severity is set by.

The blindness is two distinct statements. Where the two channels coincide as data, no comparison
between them remains. Where they are distinct and **exactly related** -- §5's route B, at `5e-15` --
the comparison is well posed and still carries nothing, because the relation holds whether the
weight is positive or spread across the circle.

What survives is the last row, for a structural reason: identifying an operator from a
correlation sequence aggregates no signed configuration and compares no channels, so neither the
ceiling nor the blindness applies to it. It is a place to look rather than a result -- what it
costs to identify such an operator from data a sign problem permits is not settled here.

**One attempt on that route is recorded, and it failed** (`reads/expAY_order_from_the_instrument.py`).
The prize is concrete: a correlation sequence `C(tau) = sum_i c_i lambda_i^tau` is measured at short
`tau`, where the sign is still mild and sampling is cheap, and the eigenvalues it identifies fix the
operator that gives the long-time behaviour one would otherwise pay `1/<sgn>^2` to sample. The
bottleneck is **model order** -- how many `lambda_i` to keep -- and every standard answer is a
chosen number: an information criterion picks a penalty, a singular-value cut picks a level, a
stability window picks a width. Under this paper's rules none is admissible.

`spectral_optics` reports `resolved_modes`, a count against a floor derived from the data rather
than supplied, which is model-order selection performed by the read. It does not work here, and
the reason is structural: on the Hankel embedding of a sum of decaying
exponentials it returns **1 on all four test cases** -- true orders 2 and 3, well separated and
close, at each of four noise levels from none to `1e-2`, five seeds apiece -- while the same matrix
has exact numerical rank 2 and 3 with singular values an order of magnitude apart (5.42 and 0.46 at
order 2). The leading mode carries a `top_share` of 0.994 to 0.996 there:
the modes are real and wildly unequal, and a floor that separates signal from a noise sea is asking
a different question from "how many modes are there".

So the row stays open, with one route into it closed by measurement.

**What the measurement supplies.** The order parameter of §7 gives an exactly
calibrated, continuously varying measure of distance from the protecting symmetry, reproducible to
under 1% across seeds in a regime where the average sign is identically 1 and has no derivative.
It does not forecast severity: §8 gives the
reason no summary of this kind reaches the severity -- the severity is set by the parity of a
crossing count, and a count modulo two cannot be recovered from a quantity that never encoded it.

**The open question this leaves.** Whether the synchrony can be restored by construction, rather
than measured. Every route in §9 that attempted it worked on the field, the contour, or the trial
wavefunction, and the identity of §4 says the object to work on is neither: it is the relation
between the two channels' determinants. Nothing in this work has attempted to build a
representation in which that relation is imposed rather than inherited from a symmetry.

## 11. What is machine-checked

Several of this paper's statements are algebra rather than measurement, and a few are
**universally quantified** -- "at any size", "on any graph", "for every exponent" -- where what
supports them in the text is a sweep. A sweep over 84 flux values reports that no flux it tried
opened a route; the claim being made is that none exists.

Those statements are machine-checked in Lean 4 / Mathlib, in `research/lean/` -- 81 theorems and
lemmas across twelve modules. The development declares no axiom of its own and contains no
`sorry`. Every theorem the table below names -- 48 of the 81 -- carries an explicit
`#print axioms`, and each elaborates against Lean's three foundational axioms alone: `propext`,
`Classical.choice`, `Quot.sound`. The other 33 are the supporting lemmas those theorems are built
from, elaborated by the same build. The footprint is emitted while a declaration elaborates, so it
is a property of the proof and not a claim made about it.

**What is not in Lean.** Every number read from a running sampler. The lockstep at
`beta = 12`, the `-1.0000` calibration, the measured deficits, the 15-of-15 agreement between the
oracle and the read -- these are properties of this model measured on this rig, and formalising them
would mean formalising the rig.

What the development covers, beyond the model's own algebra, is the read: not what it returned
here, but what it is guaranteed to return. §6 rests on the alignment being invariant to an
offset and a scale, and on saturation being equivalent to an exact affine relation, with the
deficit's `[0, 2]` range proved alongside them. Those were prose, and they are `Alignment.lean`
now. With
them the agreement on 15 lattices becomes a check that this rig computes the criterion correctly,
rather than the evidence that the criterion works.

**The theorems are about the read that is actually called.** A proof about a normalised alignment
and an implementation computing something slightly different would be two correct halves and one
wrong whole, and neither half can reveal that on its own. The three guarantees are therefore put to
`coupling.strength` itself and hold to `1e-12`: invariance to an offset and a positive scale,
saturation exactly on affine data, and the `[0, 2]` bound. Each is checked against a case that must
fail it -- an uncentred cosine breaks the invariance -- so the agreement is shown able to
discriminate (`tests/test_the_read_has_the_proved_properties.py`).

The development covers the algebraic spine and what the instrument guarantees. The result is the
measurement.

**Two claims moved from measurement to proof.**

The first is §10's **closed door on doping**. The text argues it in one line and then measures 84
flux values across four graphs, reporting a best residual of `8.926` at `mu = 0.2` and `26.63` at
`mu = 0.6` against `1.776e-14` at `mu = 0`. `Doping.doping_closes_both_routes` says that for every
finite lattice, every Hermitian
`K` with zero diagonal, every non-zero `mu` and every diagonal unitary `S`, neither route is open.
The mechanism is `conj_diagonal_eq` -- a diagonal conjugation leaves `K_ii` exactly where it was --
and route B needs one further step, that Hermiticity makes `K_ii` real, so `K_ii = -conj(K_ii)`
forces zero rather than merely forcing it imaginary.

The second is §9.2's **blindness at every exponent**. The text measures `q = 0.5, 1, 2, 3` with a
worst disagreement of exactly `0.000e+00`. `Blindness.sum_abs_pow_blind` covers every real `q` at
once, so the cube is provably as blind as the square. Its companion `gram_blind` says the same of
the singular spectrum, for every frame rather than for
the three measured. `coupling_is_not_blind` fixes the scope: the two-sided read does see the flip,
so the blindness is a property of magnitude reads.

| claim (section) | Lean | status |
|---|---|---|
| a negative weight is a channel disagreement, and the two sets coincide (§3) | `Weight.lean` | proved -- `weight_neg_iff`, `negative_set_eq_disagreement_set`, and the counted form the last table column reports as "exact" |
| `det(I + B) = det(B) det(I + B^-1)` for invertible `B` (§4) | `Identity.lean` | proved -- `det_one_add_eq`, the step that carries the diagonal factor into the ratio of the full determinants |
| the log-determinant identity `ln\|det(I+B_up)\| - ln\|det(I+B_dn)\| = -dtau * L * tr(K) + lambda * sum(x)` (§4) | `Identity.lean` | proved -- `log_det_identity`, with §5's pairing entering as a named hypothesis rather than as a lemma, so what the identity requires is visible in the statement |
| the difference cannot be constant, so no `det_up = c det_dn` exists (§4) | `Identity.lean` | proved -- `difference_not_constant`, §4's opening argument |
| the two routes, entry by entry; they coincide for real `K` (§5) | `Criterion.lean` | proved -- `routeA_iff`, `routeB_iff`, `routeA_iff_routeB_of_real` |
| the criterion IS a 2-colouring of the support graph with a zero diagonal (§5) | `Criterion.lean` | proved -- `signed_criterion` |
| the spectral criterion is strictly weaker (§5) | `Criterion.lean` | proved -- `spectral_similarity_is_weaker` exhibits a similarity to `-K` on a matrix with a non-zero diagonal, which no diagonal conjugation can have |
| a proper signing alternates along every walk; an ODD cycle admits none (§5) | `Cycles.lean` | proved -- `signing_alternates`, `no_proper_signing_of_odd_closed_walk`; the tree case is the same lemma with no odd closed walk to obstruct it |
| **a diagonal term closes both routes, at any size** (§5, §10) | `Doping.lean` | proved -- `conj_diagonal_eq`, `routeA_forces_zero_diagonal`, `routeB_forces_zero_diagonal`, `doping_closes_both_routes`; the staggered-potential case is the same theorem with a site-dependent diagonal |
| `ESS = (sum w)^2 / sum w^2 = n <sgn>^2` exactly, and it is a ceiling (§7, §9) | `Ceiling.lean` | proved -- `kish_of_signs` for every `n` and every sign pattern; `kish_le_card` (Cauchy-Schwarz) and `kish_eq_card_of_sign_free` say it is saturated exactly where there is no sign problem |
| **a magnitude read is blind at every exponent**, and so is the singular spectrum (§9.2) | `Blindness.lean` | proved -- `sum_abs_pow_blind` for every real `q`; `gram_blind`; `coupling_is_not_blind` is the positive control that fixes the scope |
| the coupling read IS the `\|x\|^2`-weighted average sign (§9.1) | `Coupling.lean` | proved -- `frobenius_inner_eq_weighted_signs`, `coupling_is_weighted_mean_sign`; `uniform_rows_give_plain_mean_sign` collapses it onto `p_neg` when the weighting disappears |
| the scalar decoupling family is complete, and its two coefficients are forced (§9.3) | `Decoupling.lean` | proved -- `mixing_is_exact_for_every_theta` (every `theta` is exact, so a sweep over `theta` is the whole family), `spin_coefficients_are_forced` |
| **the sign IS the parity of the crossing count**, for any product (§8) | `Parity.lean` | proved -- `prod_pos_iff_even_neg` and `prod_neg_iff_odd_neg`, by induction on the factors |
| what the parity argument rules out, and what it does not (§8) | `Parity.lean` | proved -- `sign_not_continuousAt_zero` and `no_continuous_function_is_the_sign` (no continuous function equals the sign) against `det_determines_sign` (a continuous one determines it). The obstruction is coarseness rather than regularity |
| **the alignment read is invariant to an offset and a positive scale** (§6) | `Alignment.lean` | proved -- `strength_affine_invariant`, with `centre_add_const` and `centre_smul` underneath it. This is what lets the criterion run on output with no `K`: §4's identity carries an offset `-dtau L tr(K)` and a slope `lambda`, and neither survives to the read |
| **saturation at 1 is exactly an affine relation between the two columns** (§6) | `Alignment.lean` | proved -- `abs_strength_eq_one_iff`, an equivalence via Cauchy-Schwarz's equality case, with `abs_strength_eq_one_of_affine` as the direction the criterion runs in and `no_affine_relation_of_abs_strength_ne_one` as the one the build check uses |
| the read lies in `[-1, 1]` and the deficit in `[0, 2]` (§7.2) | `Alignment.lean` | proved -- `abs_strength_le_one` (Cauchy-Schwarz, no hypothesis at all) and `deficit_mem_Icc` |
| **§7.2's calibration is forced, not measured**: `G_dn = 1 - G_up` makes the read exactly `-1` | `Alignment.lean` | proved -- `strength_eq_neg_one_of_reflected` and `deficit_eq_zero_of_reflected`. A reflected column centres to the negative of the original, so the read has no other value available to it; the paper's `-1.0000` is an entailment, not a coincidence |
| the criterion is not vacuous | `Alignment.lean` | proved -- `centre_sample_ne_zero` exhibits a column the read is defined on, and `not_affinely_related` exhibits two columns that are not affinely related, so neither side of the equivalence is empty |
| a sign is the rank-one case of a phase; de-rotation is exact and preserves `\|<w>\|` (§7.1) | `RankOne.lean` | proved -- `derotation_is_exact`, `derotation_preserves_mean_modulus`, `rank_one_phase_is_two_valued`; `real_part_estimator_is_not_rotation_invariant` is the mechanism behind the `0.93500 -> 0.25011` fall |

**What the Lean development does not close.** It does not prove the criterion is necessary for
positivity, only that it decides the identity; §5 is explicit that the identity delivers positivity
along route A alone, and that scope is measured rather than derived. It does not touch §7's reading,
§2's conditioning, or any of §9's closed routes beyond the two algebraic ones above. And it says
nothing about the open question of §10, which is a question about constructions that do not yet
exist.

## References

Every entry below was checked against the publisher's or arXiv's own record, and each is annotated
with where this paper leans on it.

**The method.**

[1] R. Blankenbecler, D. J. Scalapino and R. L. Sugar, *Monte Carlo calculations of coupled
boson-fermion systems. I*, Phys. Rev. D **24**, 2278-2286 (1981).
doi:10.1103/PhysRevD.24.2278 — the determinantal formulation this paper measures in.

[2] J. E. Hirsch, *Discrete Hubbard-Stratonovich transformation for fermion lattice models*,
Phys. Rev. B **28**, 4059(R) (1983). doi:10.1103/PhysRevB.28.4059 — the decoupling whose own
constant `lambda = arccosh(exp(dtau U / 2))` is the coefficient in §4's identity, and whose two
channels (spin and charge) are the two mechanisms compared in §7.

[3] Z. Bai, C.-R. Lee, R.-C. Li and S. Xu, *Stable solutions of linear systems involving long
chain of matrix multiplications*, Linear Algebra Appl. **435**, 659-673 (2011) — the conditioning
problem of §2, and why `I + B` has to be inverted through a stratified factorisation rather than
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
which is why §10 claims a reading and not a solution.

[19] R. Mondaini, S. Tarat and R. T. Scalettar, *Quantum critical points and the sign problem*,
Science **375**, 418-424 (2022). doi:10.1126/science.abg9299; arXiv:2108.08974 — the average sign
read as a diagnostic rather than only an obstacle, linked quantitatively to quantum critical
behaviour. Cited in §1 to mark the boundary of this paper's claim: where `<sgn>` varies it is
informative, and the reading offered here is for the regime where it does not vary at all.

**Where positivity comes from.**

[6] C. Wu and S.-C. Zhang, *Sufficient condition for absence of the sign problem in the fermionic
quantum Monte Carlo algorithm*, Phys. Rev. B **71**, 155115 (2005).
doi:10.1103/PhysRevB.71.155115; arXiv:cond-mat/0407272 — positivity from the eigenvalues of the
fermion matrix pairing into complex conjugates, tied to time-reversal symmetry.

[7] Z. C. Wei, C. Wu, Y. Li, S. Zhang and T. Xiang, *Majorana positivity and the fermion sign
problem of quantum Monte Carlo simulations*, Phys. Rev. Lett. **116**, 250601 (2016).
doi:10.1103/PhysRevLett.116.250601; arXiv:1601.01994 — a unified account of the known sign-free
models, and the reason §5's scope is stated as a measured boundary rather than a general one:
sign-free models exist *with repulsive interactions and without particle-hole symmetry*, so the
symmetric point this paper reads is one place positivity comes from and not the only one.

[8] Z.-X. Li, Y.-F. Jiang and H. Yao, *Majorana-time-reversal symmetries: a fundamental principle
for sign-problem-free quantum Monte Carlo simulations*, Phys. Rev. Lett. **117**, 267002 (2016).
doi:10.1103/PhysRevLett.117.267002; arXiv:1601.05780 — the classification into Majorana and
Kramers classes.

[18] I. Hen, *Determining quantum Monte Carlo simulability with geometric phases*, Phys. Rev.
Research **3**, 023080 (2021). doi:10.1103/PhysRevResearch.3.023080 — a QMC simulation is
sign-problem-free exactly when the total phases along the chordless cycles of the weighted graph
whose adjacency matrix is the Hamiltonian vanish. Cited in §5 as the nearest statement of the same
shape as route B's cycle condition, and to mark the two differences: the graph there is built from
the Hamiltonian and the conclusion is positivity, where route B's graph is the support graph of the
one-body matrix and the conclusion is §4's identity, which §5 shows is a weaker thing to obtain.

**Routes measured and closed (§9).**

[9] S. Zhang, J. Carlson and J. E. Gubernatis, *Constrained path Monte Carlo method for fermion
ground states*, Phys. Rev. B **55**, 7464-7477 (1997). doi:10.1103/PhysRevB.55.7464 — the
constrained-path method benchmarked in §9.

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
doi:10.1103/PhysRevLett.102.131601 — the complex-Langevin route attempted in §9.

**The instrument.**

[E] Entroptics, version 0.2.3, [doi:10.5281/zenodo.22687899](https://doi.org/10.5281/zenodo.22687899)
(released 2026-09-09). That is the version DOI, not the concept DOI
[10.5281/zenodo.21273400](https://doi.org/10.5281/zenodo.21273400), which resolves to whatever the
latest release is; every figure here was read through version 0.2.3.
Every read is reached through `research/code/entroptics_adapter.py`, which names each one after the
question this paper asks of it. The counts below are how many of the 54 cited experiments call
each. `denoise` is listed in the note below rather than here, because no figure in this paper is
read with it.

| adapter | library read | § | calls |
|---|---|---|---|
| `channel_alignment` | `reads.coupling` | 7.2 | 18 |
| `weight_cloud` | `reads.concentration` | 7.1 | 2 |
| `cloud_axes` | `reads.principal_directions` | 7.1 | 2 |
| `evidence_ceiling`, `frame_carriage` | `carriage` | 9, 8 | 1 each |
| `autocorrelation_time` | `dynamics(...).reconstruct_decay` | 9.2c | 1 |
| `balance_at_own_zero` | `Screen().balance` | 7 | 1 |
| `single_channel_optics` | `reads.spectral_optics` | 9.2b, 10 | 2 |
| `dynamics` | `dynamics(...)`, for `rates` and `forgetting` | 9.2c, 9.5 | 2 |

The coupling read is standardised against its own exact re-pairing null -- exact by construction
rather than asymptotic, which is what lets §7 quote a sigma with no constant supplied.
`carriage.effective_n` is Kish's effective sample size and gives §9's ceiling;
`reconstruct_decay` rebuilds the autocorrelation from the one-step operator's spectrum and supplies
the `tau_int` that shows the same ceiling is not tight. Where a Tracy-Widom comparison is made,
the external reference is [14].

One read is named above and used by no figure here. `denoise` (`Aperture.extract`) was scored as
an estimator of a mean in `reads/expRR_natural_noise.py` and is the wrong tool for this problem:
optimal shrinkage recovers a low-rank signal and biases a mean, and the sign problem's difficulty
is entirely in a mean.

[14] M. Chiani, *Distribution of the largest eigenvalue for real Wishart and Gaussian random
matrices and a simple approximation for the Tracy-Widom distribution*, J. Multivariate Anal.
**129**, 69-81 (2014); arXiv:1209.3394.

[15] M. Wallerberger, *Efficient estimation of autocorrelation spectra*, arXiv:1810.05079 (2018)
— log-binning of the autocorrelation function, aimed at lattice and Monte Carlo chains. Cited in
§9 for the point that window-free estimation of `tau_int` is established practice; its binning
factor is the constant it keeps.

[16] D. Foreman-Mackey et al., *Autocorrelation time estimation*, `emcee` documentation,
<https://emcee.readthedocs.io/en/stable/tutorials/autocorr/>. Cited in §9 for the same point: it
fits a second-order ARMA (via `celerite`) to the chain and sums the model's autocorrelation
analytically, so no window is chosen -- the model order is. Consulted for the method it
describes, not for a numerical result.

[17] U. Wolff (ALPHA Collaboration), *Monte Carlo errors with less errors*, Comput. Phys. Commun.
**156**, 143-153 (2004); erratum **176**, 383 (2007); arXiv:hep-lat/0306017 — the `Gamma` method.
Cited in §9 because it already does what that section's correction needs: the autocorrelation time
of a derived observable is computed by projecting the primary fluctuations through the function's
derivatives, `pi_F = sum_alpha (dF/da_alpha) pi_alpha`, which is the linearisation used there.
Consulted through its documented formulation in the `pyobs` implementation
(<https://mbruno46.github.io/pyobs/intro/gamma.html>) rather than the printed paper, and cited
for that formulation only.

The identity of §4 and the lockstep of §3 are derived and verified here, and no source is cited
for them. Every entry above was read before being cited.
