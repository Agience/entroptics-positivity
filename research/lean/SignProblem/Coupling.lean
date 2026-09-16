/-
Section 8.1 -- every cheap read of the severity is the average sign under a different measure.

This is the statement that collapses the whole family of "score the severity from the sampled data"
proposals, and the paper states it as a table: `p_neg` averages the sign under the uniform measure,
the coupling read averages it weighted by `|x|^2`, and `<sgn>` itself averages it weighted by `|D|`.
They all fail for one reason -- the sign correlates with `|D|`, so any weighting that is not
`|D|`-weighted mis-weights exactly the configurations that decide the answer.

What is provable here is the middle row, which is the one that looks like it might be different: the
two-sided coupling between `A` and `diag(sigma) A` IS the `|x|^2`-weighted mean of `sigma`. The paper
verifies it numerically to six decimals (0.915433 against 0.915432); the identity behind it is two
lines of algebra and is proved below for every frame at once.

That matters for how a new proposal gets assessed. Section 8.1's rule is: before chasing a new
correlate, ask which measure it averages the sign under. If it is not `|D|`-weighted, it is one of
the ones already refuted, whatever it is called. Having the reduction as a theorem rather than as a
measured agreement is what makes that rule usable on a read nobody has measured yet.
-/
import Mathlib.Tactic
import Mathlib.Data.Matrix.Basic
import SignProblem.Blindness

namespace SignProblem

open Matrix Finset

variable {m n : Type*} [Fintype m] [Fintype n] [DecidableEq m]

/-- The squared length of row `i` -- the `|x|^2` the measure is built from. -/
noncomputable def rowSq (A : Matrix m n ℝ) (i : m) : ℝ := ∑ j, (A i j) ^ 2

/-- **The Frobenius inner product between the two sides IS the `|x|^2`-weighted sum of the signs.**

    `<A, diag(sigma) A>_F = sum_i sigma_i |x_i|^2`. That is the whole reduction: the two-sided read
    never sees a configuration's sign except multiplied by that configuration's squared length. -/
theorem frobenius_inner_eq_weighted_signs (σ : m → ℝ) (A : Matrix m n ℝ) :
    ∑ i, ∑ j, A i j * (diagonal σ * A) i j = ∑ i, σ i * rowSq A i := by
  refine Finset.sum_congr rfl fun i _ => ?_
  rw [rowSq, Finset.mul_sum]
  refine Finset.sum_congr rfl fun j _ => ?_
  rw [flip_apply]
  ring

/-- The two sides have the same Frobenius norm, since a row sign flip changes no magnitude. So the
    normalisation the read divides by is the same on both sides, and cannot rescue anything. -/
theorem frobenius_norm_eq (σ : m → ℝ) (hσ : Flip σ) (A : Matrix m n ℝ) :
    ∑ i, ∑ j, ((diagonal σ * A) i j) ^ 2 = ∑ i, ∑ j, (A i j) ^ 2 := by
  refine Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun j _ => ?_
  rw [flip_apply, mul_pow]
  have hone : σ i ^ 2 = 1 := by
    have := mul_self_of_flip hσ i
    nlinarith [this]
  rw [hone, one_mul]

/-- **So the normalised coupling is exactly the `|x|^2`-weighted MEAN of the signs.**

    `strength = (sum_i sigma_i |x_i|^2) / (sum_i |x_i|^2)`, which is an average of `sigma` under the
    measure `|x|^2`, and nothing else. It is `<sigma>` in different clothing -- a quantity already
    shown not to bound the free-energy difference, because the measure that decides the answer is
    `|D|` and this one is not it. -/
theorem coupling_is_weighted_mean_sign (σ : m → ℝ) (hσ : Flip σ) (A : Matrix m n ℝ)
    (hA : ∑ i, rowSq A i ≠ 0) :
    (∑ i, ∑ j, A i j * (diagonal σ * A) i j) / (∑ i, rowSq A i)
      = (∑ i, σ i * rowSq A i) / (∑ i, rowSq A i) := by
  rw [frobenius_inner_eq_weighted_signs]

/-- With all rows of equal length the weighting disappears and the read is the PLAIN average sign.

    This is the degenerate case that makes the reduction legible: the `|x|^2` weighting is the only
    thing separating this read from `p_neg`, so a frame with uniform row lengths collapses one into
    the other exactly. -/
theorem uniform_rows_give_plain_mean_sign (σ : m → ℝ) (A : Matrix m n ℝ) (c : ℝ) (hc : c ≠ 0)
    (hrow : ∀ i, rowSq A i = c) (hn : 0 < Fintype.card m) :
    (∑ i, ∑ j, A i j * (diagonal σ * A) i j) / (∑ i, rowSq A i)
      = (∑ i, σ i) / (Fintype.card m : ℝ) := by
  have hcard : (Fintype.card m : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr hn.ne'
  rw [frobenius_inner_eq_weighted_signs]
  simp only [hrow, Finset.sum_const, Finset.card_univ, nsmul_eq_mul]
  rw [← Finset.sum_mul]
  field_simp

end SignProblem
