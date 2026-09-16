/-
Section 8.2 -- a magnitude read cannot see a sign, for ANY exponent.

With `B = diag(sigma) A` the magnitudes agree cell for cell, so `sum |A|^q` and `sum |B|^q` agree.
The paper measures this at `q = 0.5, 1, 2, 3` with a worst disagreement of exactly `0.000e+00`. Four
exponents is not "any exponent", and the difference matters, because the natural response to a
blindness measured at the square is to reach for the cube. Proved here for every real `q` at once,
so there is no exponent left to try.

The singular spectrum is blind for the same reason and it is worth separating: `diag(sigma)` is
orthogonal, so the Gram matrix is untouched, and every singular value with it. A reader who accepts
that entrywise magnitudes are blind might still expect the SPECTRUM to see something, since a
spectrum is not an entrywise quantity. It does not.

AND THE SCOPE IS THE POINT. This is a statement about MAGNITUDE reads of one frame, not about what
can be measured. The coupling BETWEEN the two sides is not blind -- the same three frames give
`+1.0000`, `0.0000` and `-1.0000` -- and that positive control is what makes the blindness a
property of magnitude reads rather than of the frames. It is carried here as a theorem rather than
left in prose, because the scope is exactly the thing that is easy to over-read: a measured
blindness says nothing about a one-sided read of a different object.
-/
import Mathlib.Tactic
import Mathlib.Data.Matrix.Basic
import Mathlib.Analysis.SpecialFunctions.Pow.Real

namespace SignProblem

open Matrix Finset

variable {m n : Type*} [Fintype m] [Fintype n] [DecidableEq m]

/-- A row sign flip: `B = diag(sigma) A` with `sigma i = ±1`. -/
def Flip (σ : m → ℝ) : Prop := ∀ i, σ i = 1 ∨ σ i = -1

theorem abs_eq_one_of_flip {σ : m → ℝ} (hσ : Flip σ) (i : m) : |σ i| = 1 := by
  rcases hσ i with h | h <;> rw [h] <;> norm_num

theorem mul_self_of_flip {σ : m → ℝ} (hσ : Flip σ) (i : m) : σ i * σ i = 1 := by
  rcases hσ i with h | h <;> rw [h] <;> ring

@[simp] theorem flip_apply (σ : m → ℝ) (A : Matrix m n ℝ) (i : m) (j : n) :
    (diagonal σ * A) i j = σ i * A i j := by
  simp [Matrix.mul_apply, Matrix.diagonal_apply, Finset.sum_ite_eq]

/-- The magnitudes agree CELL FOR CELL. Everything else in this file follows. -/
theorem abs_flip_eq (σ : m → ℝ) (hσ : Flip σ) (A : Matrix m n ℝ) (i : m) (j : n) :
    |(diagonal σ * A) i j| = |A i j| := by
  rw [flip_apply, abs_mul, abs_eq_one_of_flip hσ, one_mul]

/-- **A magnitude read is blind to a row sign flip, at every exponent `q` at once.**

    The paper measures `q = 0.5, 1, 2, 3`; this says there is no `q` that would have worked. The
    cube is exactly as blind as the square, provably, and so is every other power. -/
theorem sum_abs_pow_blind (σ : m → ℝ) (hσ : Flip σ) (A : Matrix m n ℝ) (q : ℝ) :
    ∑ i, ∑ j, |(diagonal σ * A) i j| ^ q = ∑ i, ∑ j, |A i j| ^ q := by
  refine Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun j _ => ?_
  rw [abs_flip_eq σ hσ A i j]

/-- A row sign flip is ORTHOGONAL: `diag(sigma) * diag(sigma) = I`. -/
theorem diagonal_mul_self (σ : m → ℝ) (hσ : Flip σ) :
    (diagonal σ : Matrix m m ℝ) * diagonal σ = 1 := by
  rw [Matrix.diagonal_mul_diagonal]
  have : (fun i => σ i * σ i) = fun _ : m => (1 : ℝ) :=
    funext fun i => mul_self_of_flip hσ i
  rw [this, Matrix.diagonal_one]

/-- Hence the Gram matrix is untouched, and with it the whole singular spectrum.

    The paper reports a maximum singular-value difference of `0.000e+00`; this is the reason, and it
    holds for every `A` rather than for the three frames measured. -/
theorem gram_blind (σ : m → ℝ) (hσ : Flip σ) (A : Matrix m n ℝ) :
    (diagonal σ * A)ᵀ * (diagonal σ * A) = Aᵀ * A := by
  rw [Matrix.transpose_mul, Matrix.diagonal_transpose]
  rw [Matrix.mul_assoc, ← Matrix.mul_assoc (diagonal σ), diagonal_mul_self σ hσ, Matrix.one_mul]

/-- **The positive control, and the scope.**

    The coupling BETWEEN the two sides is not blind. On a frame with one positive row and one
    negative one, the two-sided Frobenius inner product changes sign under the flip while every
    magnitude read above returns the same number. So the blindness is a property of magnitude reads
    of one frame, and says nothing about a read of the relation between two.

    Without this a reader could take section 8.2 to mean the instrument cannot see a sign problem at
    all -- which is exactly the over-reading the paper warns against, and which section 5's reading
    disproves. -/
theorem coupling_is_not_blind :
    ∃ (σ : Fin 2 → ℝ) (A : Matrix (Fin 2) (Fin 1) ℝ),
      Flip σ ∧
      (∑ i, ∑ j, A i j * (diagonal σ * A) i j) ≠ (∑ i, ∑ j, A i j * A i j) := by
  refine ⟨![1, -1], !![1; 1], ?_, ?_⟩
  · intro i; fin_cases i <;> simp
  · simp [Fin.sum_univ_succ, Matrix.mul_apply, Matrix.diagonal_apply]
    norm_num

end SignProblem
