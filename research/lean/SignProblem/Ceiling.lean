/-
Section 8 -- the ceiling, and why no weighted read can be a severity meter.

Correct importance sampling draws at `|w|`, so a configuration enters any weighted average carrying
only its SIGN. Kish's effective sample size of those weights is then

    ESS = (sum w)^2 / sum w^2 = (n <sgn>)^2 / n = n <sgn>^2

exactly -- the `O(1/<sgn>^2)` cost of the sign problem written as a property of the WEIGHTS. The
paper verifies the arithmetic against the instrument at a ratio of 1.0000 across eight
`(n, <sgn>)` combinations. What a verification across eight combinations cannot say is that the
equality is exact for every `n` and every sign pattern, which is what makes it a CEILING on every
aggregation rather than a coincidence at the points tried. That is proved here.

Three statements, and the third is the one section 5 leans on:

  * `kish_of_signs` -- the equality, exactly, for any `±1` weights;
  * `kish_le_card` -- the ceiling never exceeds the sample size (Cauchy-Schwarz), with equality
    exactly when every sign agrees, so a sign-free run loses nothing;
  * `kish_collapses` -- it falls as the square of the average sign, which is the exponential cost.

READ IT AS AN UPPER BOUND. Kish's formula assumes INDEPENDENT draws and a DQMC run is a Markov
chain, so the evidence a run actually carries is `n <sgn>^2 / (2 tau_int)` -- smaller still, measured
at 4.15x on this rig (`reads/expBC_ceiling_is_not_tight.py`). Nothing here is weakened by that; the
argument that no weighted read can be a severity meter is strengthened, because the true figure is
smaller than the bound. What the ceiling must not be read as is the evidence a given run HAS.
-/
import Mathlib.Tactic
import Mathlib.Analysis.MeanInequalities
import Mathlib.Algebra.Order.Chebyshev

namespace SignProblem

open Finset

variable {ι : Type*} [Fintype ι]

/-- A sign weight: every configuration enters carrying `+1` or `-1`. -/
def SignWeights (w : ι → ℝ) : Prop := ∀ i, w i = 1 ∨ w i = -1

theorem sq_eq_one_of_sign {w : ι → ℝ} (hw : SignWeights w) (i : ι) : w i ^ 2 = 1 := by
  rcases hw i with h | h <;> rw [h] <;> norm_num

/-- The sum of squares of sign weights is the sample size. This is the step that makes Kish's
    formula collapse to something with no free parameters in it. -/
theorem sum_sq_eq_card {w : ι → ℝ} (hw : SignWeights w) :
    ∑ i, w i ^ 2 = (Fintype.card ι : ℝ) := by
  simp [sq_eq_one_of_sign hw, Finset.card_univ]

/-- Kish's effective sample size of the sign weights, `ESS = (sum w)^2 / sum w^2`. -/
noncomputable def kish (w : ι → ℝ) : ℝ := (∑ i, w i) ^ 2 / ∑ i, w i ^ 2

/-- The average sign. -/
noncomputable def meanSign (w : ι → ℝ) : ℝ := (∑ i, w i) / (Fintype.card ι : ℝ)

/-- **`ESS = n <sgn>^2`, exactly.**

    Not to a tolerance and not at the points tried: for every finite ensemble and every pattern of
    signs. This is the ceiling section 8 derives, and it is a property of the weights alone -- the
    frame does not appear in the statement, which is why it bounds EVERY aggregation of them,
    section 5's own coupling read included. -/
theorem kish_of_signs {w : ι → ℝ} (hw : SignWeights w) (hn : 0 < Fintype.card ι) :
    kish w = (Fintype.card ι : ℝ) * meanSign w ^ 2 := by
  have hcard : (Fintype.card ι : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr hn.ne'
  rw [kish, meanSign, sum_sq_eq_card hw, div_pow]
  field_simp

/-- The ceiling never exceeds the sample size: `n <sgn>^2 ≤ n`, because `|<sgn>| ≤ 1`.

    Cauchy-Schwarz against the constant vector. This is what makes it a CEILING rather than an
    estimate -- there is no sign pattern that buys evidence the ensemble does not have. -/
theorem kish_le_card {w : ι → ℝ} (hw : SignWeights w) (hn : 0 < Fintype.card ι) :
    kish w ≤ (Fintype.card ι : ℝ) := by
  have hcard : (0 : ℝ) < (Fintype.card ι : ℝ) := Nat.cast_pos.mpr hn
  rw [kish, sum_sq_eq_card hw, div_le_iff₀ hcard]
  have h := sq_sum_le_card_mul_sum_sq (s := (univ : Finset ι)) (f := w)
  simpa [Finset.card_univ, sum_sq_eq_card hw] using h

/-- And it collapses as the square of the average sign, which is the exponential cost written out:
    with `<sgn> = exp(-beta N df)` the ceiling is `n exp(-2 beta N df)`. -/
theorem kish_collapses {w v : ι → ℝ} (hw : SignWeights w) (hv : SignWeights v)
    (hn : 0 < Fintype.card ι) (h : |meanSign v| ≤ |meanSign w|) :
    kish v ≤ kish w := by
  rw [kish_of_signs hw hn, kish_of_signs hv hn]
  have hcard : (0 : ℝ) ≤ (Fintype.card ι : ℝ) := Nat.cast_nonneg _
  have hsq : meanSign v ^ 2 ≤ meanSign w ^ 2 := by
    nlinarith [abs_nonneg (meanSign v), abs_nonneg (meanSign w),
               sq_abs (meanSign v), sq_abs (meanSign w), h]
  exact mul_le_mul_of_nonneg_left hsq hcard

/-- A sign-free ensemble loses nothing: every weight `+1` gives `ESS = n`.

    This is the other end of the same statement, and it is what makes the ceiling informative rather
    than merely true -- it is saturated exactly where there is no sign problem. -/
theorem kish_eq_card_of_sign_free (hn : 0 < Fintype.card ι) :
    kish (fun _ : ι => (1 : ℝ)) = (Fintype.card ι : ℝ) := by
  have hcard : (Fintype.card ι : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr hn.ne'
  simp [kish, Finset.card_univ]
  field_simp

end SignProblem
