/-
Section 3 -- the identity that produces the lockstep.

    ln|det(I + B_up(x))| - ln|det(I + B_dn(x))| = -dtau * L * tr(K) + lambda * sum(x)

Two facts about the two determinants look incompatible: their signs agree on every configuration,
and their log-magnitudes differ by 8 to 16 orders. A relation `det_up = c det_dn` for a constant `c`
would hold the log-difference fixed, and it is not fixed. So the relation must preserve the sign and
leave the magnitude free -- which is what an exponential factor does.

The derivation has exactly two moving parts and this file separates them, because section 4 is about
the second one:

  * `det_one_add_eq` -- `det(I + B) = det(B) * det(I + B⁻¹)` for invertible `B`. Pure algebra, true
    of every invertible matrix, nothing to do with the model.
  * the PAIRING `det(I + B_up⁻¹) = det(I + B_dn)`. This is not automatic; it is exactly what
    section 4's criterion on `K` decides, and it enters here as a hypothesis rather than as a
    lemma. Everything section 4 does is establish when that hypothesis holds.

Keeping them apart is the point. The identity is an equality between two logarithms and it is
`hpair` that carries all the physics; a reader who wants to know what the identity REQUIRES should
be able to see it in the statement, not reconstruct it from a proof.
-/
import Mathlib.Tactic
import Mathlib.LinearAlgebra.Matrix.NonsingularInverse
import Mathlib.Analysis.SpecialFunctions.Log.Basic

namespace SignProblem

open Matrix

variable {n : Type*} [Fintype n] [DecidableEq n]

/-- `det(I + B) = det(B) * det(I + B⁻¹)` for invertible `B`.

    The whole content is that `B * (I + B⁻¹) = B + I`; the determinant is multiplicative and the
    rest is bookkeeping. This is the step that "carries one factor into the ratio of the full
    determinants", and it is where the model's own `det(B_up)/det(B_dn)` becomes a statement about
    `det(I + B)`, which is what a simulation actually computes. -/
theorem det_one_add_eq (B : Matrix n n ℝ) (hB : IsUnit B.det) :
    (1 + B).det = B.det * (1 + B⁻¹).det := by
  rw [← det_mul]
  congr 1
  rw [mul_add, mul_one, Matrix.mul_nonsing_inv B hB, add_comm]

/-- Section 3's identity.

    `hpair` is section 4's condition, as data: the two channels' determinants pair up, which is what
    `S K S⁻¹ = -K` (or the anti-similarity) delivers. `hup` is the decoupling's own arithmetic --
    the diagonal factor contributes `exp(lambda * sum x)` and the `L` hopping slices contribute
    `exp(-dtau * L * tr K)` -- and it carries the decoupling's OWN constant
    `lambda = arccosh(exp(dtau U / 2))`, which is the coefficient the paper verifies to `1e-14` and
    shows a neighbouring interaction's constant fails by fourteen orders.

    Note what the conclusion does NOT say. It fixes the log-DIFFERENCE and leaves each magnitude
    free, which is exactly the shape section 3 argues the relation has to have. -/
theorem log_det_identity
    (Bup Bdn : Matrix n n ℝ) (hBup : IsUnit Bup.det)
    (hpair : (1 + Bup⁻¹).det = (1 + Bdn).det)
    (hdn : (1 + Bdn).det ≠ 0)
    (lam dtau trK sx : ℝ) (L : ℕ)
    (hup : Bup.det = Real.exp (lam * sx - dtau * L * trK)) :
    Real.log |(1 + Bup).det| - Real.log |(1 + Bdn).det|
      = -(dtau * L * trK) + lam * sx := by
  have hexp : (0 : ℝ) < Real.exp (lam * sx - dtau * L * trK) := Real.exp_pos _
  have hfac : (1 + Bup).det = Bup.det * (1 + Bdn).det := by
    rw [det_one_add_eq Bup hBup, hpair]
  have habs : |Bup.det| = Real.exp (lam * sx - dtau * L * trK) := by
    rw [hup, abs_of_pos hexp]
  rw [hfac, abs_mul, Real.log_mul (by rw [habs]; exact hexp.ne') (abs_ne_zero.mpr hdn),
      habs, Real.log_exp]
  ring

/-- Read as a statement about data, the identity says the log-difference is an exact AFFINE function
    of `sum(x)` -- offset `-dtau * L * tr(K)`, slope `lambda`.

    That is the form section 4.1 reads from two logged columns with no Hamiltonian in hand: a
    normalised alignment saturates at 1 on an affine relation and cannot on a broken one, and
    because the alignment is invariant to both the offset and the scale it needs neither `lambda`
    nor `tr(K)`. -/
theorem log_det_is_affine_in_field_sum
    (lam dtau trK : ℝ) (L : ℕ) :
    ∃ a b : ℝ, ∀ sx : ℝ, -(dtau * L * trK) + lam * sx = a + b * sx :=
  ⟨-(dtau * L * trK), lam, fun _ => rfl⟩

/-- Section 3's opening argument, made precise: a relation `det_up = c det_dn` with a CONSTANT `c`
    would hold the log-difference fixed across configurations, and the measured difference is not
    fixed -- it ranges. So no such `c` exists, unless the decoupling constant vanishes.

    `lambda = arccosh(exp(dtau U / 2))` is zero only at `U = 0`, i.e. with no interaction at all, so
    on any interacting model the difference genuinely moves with the field. -/
theorem difference_not_constant (lam dtau trK : ℝ) (L : ℕ) (hlam : lam ≠ 0) :
    ¬ ∃ c : ℝ, ∀ sx : ℝ, -(dtau * L * trK) + lam * sx = c := by
  rintro ⟨c, hc⟩
  have h0 := hc 0
  have h1 := hc 1
  rw [← h0] at h1
  simp at h1
  exact hlam h1

end SignProblem
