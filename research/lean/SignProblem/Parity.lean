/-
Section 7 -- the sign is a PARITY, and what that does and does not rule out.

§7 tests three per-configuration candidates for predicting the sign, finds all three fail, and gives
a structural reason:

    det(I + B) = prod_k (1 + lambda_k)

so the sign is the parity of the number of eigenvalues past the crossing. That much is exact and is
`prod_pos_iff_even_neg` below.

WHAT THE PARITY ARGUMENT ACTUALLY RULES OUT, which is narrower than §7's prose claimed. The section
said "no continuous summary of a spectrum determines it" and then, in the next sentence, "the
determinant itself is of course a scalar that settles the parity exactly" -- and the determinant is
continuous. Both cannot stand. Formalising forces the distinction:

  * a continuous function can DETERMINE the sign. The product does, and `det_determines_sign` says
    so. Continuity is therefore not the obstruction, and an argument resting on it proves nothing.
  * a continuous function cannot BE the sign. `sign` jumps at a crossing, and no continuous function
    equals a function that jumps -- `no_continuous_function_is_the_sign`.

So the correct reading of §7 is the second: the cheap observables fail not because they are
continuous but because each is a COARSE summary that has already discarded the crossing count, and
the one continuous scalar that keeps it is the determinant, whose cost the question was asked to
avoid. The measurement in §7 is what establishes that the three specific candidates discard it; this
file establishes what shape of argument is available at all.
-/
import Mathlib.Tactic
import Mathlib.Data.Real.Sign
import Mathlib.Analysis.SpecialFunctions.Pow.Real

namespace SignProblem

open Finset

variable {ι : Type*} [DecidableEq ι]

/-- **The sign of a product is the parity of its negative factors.**

    `det(I + B) = prod_k (1 + lambda_k)`, so this is §7's structural fact: whether the weight is
    positive is settled by whether an EVEN number of eigenvalues have gone past the crossing. It is
    a count, and a count modulo two -- which is why no amount of knowing *how far* the eigenvalues
    have moved settles it. -/
theorem prod_pos_iff_even_neg (s : Finset ι) (f : ι → ℝ) (hf : ∀ i ∈ s, f i ≠ 0) :
    0 < ∏ i ∈ s, f i ↔ Even (s.filter (fun i => f i < 0)).card := by
  classical
  induction s using Finset.induction with
  | empty => simp
  | insert a s ha ih =>
      have hfa : f a ≠ 0 := hf a (mem_insert_self a s)
      have hrest : ∀ i ∈ s, f i ≠ 0 := fun i hi => hf i (mem_insert_of_mem hi)
      have hP : ∏ i ∈ s, f i ≠ 0 := prod_ne_zero_iff.mpr hrest
      rw [prod_insert ha, filter_insert]
      rcases lt_or_gt_of_ne hfa with hneg | hpos
      · -- a negative factor flips the parity: `0 < f a * P` exactly when `P < 0`
        rw [if_pos hneg, card_insert_of_notMem (fun h => ha (mem_filter.mp h).1),
            Nat.even_add_one, ← ih hrest]
        constructor
        · intro h
          rcases lt_trichotomy (∏ i ∈ s, f i) 0 with hl | he | hg
          · exact not_lt.mpr hl.le
          · exact absurd he hP
          · exact absurd h (not_lt.mpr (mul_neg_of_neg_of_pos hneg hg).le)
        · intro h
          have hl : ∏ i ∈ s, f i < 0 := by
            rcases lt_trichotomy (∏ i ∈ s, f i) 0 with hl | he | hg
            · exact hl
            · exact absurd he hP
            · exact absurd hg h
          exact mul_pos_of_neg_of_neg hneg hl
      · -- a positive factor leaves it alone
        rw [if_neg (not_lt.mpr hpos.le), ← ih hrest]
        constructor
        · intro h
          rcases lt_trichotomy (∏ i ∈ s, f i) 0 with hl | he | hg
          · exact absurd h (not_lt.mpr (mul_neg_of_pos_of_neg hpos hl).le)
          · exact absurd he hP
          · exact hg
        · exact fun h => mul_pos hpos h

/-- The same statement read as the sign itself: negative weight, odd count. -/
theorem prod_neg_iff_odd_neg (s : Finset ι) (f : ι → ℝ) (hf : ∀ i ∈ s, f i ≠ 0) :
    (∏ i ∈ s, f i) < 0 ↔ Odd (s.filter (fun i => f i < 0)).card := by
  have hne : (∏ i ∈ s, f i) ≠ 0 := prod_ne_zero_iff.mpr hf
  rw [← Nat.not_even_iff_odd, ← prod_pos_iff_even_neg s f hf]
  constructor
  · exact fun h hp => absurd h (not_lt.mpr hp.le)
  · intro h
    rcases lt_trichotomy (∏ i ∈ s, f i) 0 with hlt | heq | hgt
    · exact hlt
    · exact absurd heq hne
    · exact absurd hgt h

/-! ### What the parity argument does and does not rule out -/

/-- **A continuous scalar CAN determine the sign**, and the determinant is one.

    Stated because §7's prose claimed the opposite in passing. Continuity is not the obstruction:
    the product is continuous in the eigenvalues and settles the parity exactly. What makes it
    useless for the question §7 asks is its cost, not its regularity. -/
theorem det_determines_sign (s : Finset ι) (f g : ι → ℝ)
    (hf : ∀ i ∈ s, f i ≠ 0) (hg : ∀ i ∈ s, g i ≠ 0)
    (h : ∏ i ∈ s, f i = ∏ i ∈ s, g i) :
    (Even (s.filter (fun i => f i < 0)).card ↔ Even (s.filter (fun i => g i < 0)).card) := by
  rw [← prod_pos_iff_even_neg s f hf, ← prod_pos_iff_even_neg s g hg, h]

/-- **The sign jumps.** `Real.sign` is not continuous at zero: it is `0` there and `1` immediately
    to the right. This is the crossing, in one line. -/
theorem sign_not_continuousAt_zero : ¬ ContinuousAt Real.sign 0 := by
  intro h
  -- an epsilon-delta argument, kept concrete: at `eps = 1/2` any `delta` still admits a positive
  -- point, where the sign is `1` and therefore `1` away from `sign 0 = 0`.
  rw [Metric.continuousAt_iff] at h
  obtain ⟨d, hd, hball⟩ := h (1 / 2) (by norm_num)
  have hx : dist (d / 2) (0 : ℝ) < d := by
    rw [Real.dist_eq, sub_zero, abs_of_pos (by linarith)]
    linarith
  have := hball hx
  rw [Real.sign_of_pos (by linarith : (0 : ℝ) < d / 2), Real.sign_zero, Real.dist_eq] at this
  norm_num at this

/-- **No continuous function IS the sign.**

    This is what the parity argument rules out, stated at the strength it actually has: a summary
    that equals the sign has to jump where the sign jumps, and a continuous one cannot. It does not
    rule out a continuous function that determines the sign without equalling it -- see
    `det_determines_sign` -- which is why §7's evidence against the three candidates is the
    MEASUREMENT that each discards the crossing count, not their continuity. -/
theorem no_continuous_function_is_the_sign (g : ℝ → ℝ) (hg : Continuous g) :
    ¬ (∀ x, g x = Real.sign x) := by
  intro h
  exact sign_not_continuousAt_zero ((hg.continuousAt).congr (Filter.Eventually.of_forall h))

end SignProblem
