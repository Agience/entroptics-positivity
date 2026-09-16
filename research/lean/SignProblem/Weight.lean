/-
Section 2 -- the sign is a disagreement between two channels.

The paper opens by saying that `sign w(x) < 0` exactly when the two channels' determinant signs
disagree, and calls that "arithmetic, immediate from the product". This file is that arithmetic,
stated once so the rest of the development can name it, together with the form section 2 actually
measures: not a statement about one configuration but the equality of two SETS of configurations --
the ones carrying a negative weight, and the ones on which the channels disagree. The paper's last
table column, "disagree = neg fraction ... exact", is that equality counted.
-/
import Mathlib.Tactic
import Mathlib.Data.Fintype.Card
import Mathlib.Data.Finset.Card

namespace SignProblem

variable {ι : Type*}

/-- A configuration's weight: the product of the two spin channels' determinants.

    `w(x) = det(I + B_up(x)) * det(I + B_dn(x))`, with each factor abstracted to the real number it
    is. Nothing below needs the determinants' internal structure -- which is the point of section 2,
    and the reason the statement is about a product and not about a matrix. -/
def weight (du dd : ℝ) : ℝ := du * dd

/-- The two channels AGREE on a configuration when their determinants have the same sign. -/
def Agree (du dd : ℝ) : Prop := (0 < du ∧ 0 < dd) ∨ (du < 0 ∧ dd < 0)

/-- A negative weight is exactly a disagreement between the channels.

    Section 2: "`sign w(x) < 0` **if and only if** the two channels' determinant signs disagree.
    That is arithmetic, immediate from the product." -/
theorem weight_neg_iff (du dd : ℝ) :
    weight du dd < 0 ↔ (0 < du ∧ dd < 0) ∨ (du < 0 ∧ 0 < dd) :=
  mul_neg_iff

/-- Agreement gives a positive weight. This is the direction the lockstep uses: the individual
    channels may cross zero as often as they like, and while they cross TOGETHER the product's sign
    never changes. -/
theorem weight_pos_of_agree {du dd : ℝ} (h : Agree du dd) : 0 < weight du dd := by
  rcases h with ⟨hu, hd⟩ | ⟨hu, hd⟩
  · exact mul_pos hu hd
  · exact mul_pos_of_neg_of_neg hu hd

/-- And conversely, on configurations where neither determinant vanishes, a positive weight IS
    agreement. Together with `weight_neg_iff` this is the "if and only if" section 2 claims. -/
theorem agree_of_weight_pos {du dd : ℝ} (hu : du ≠ 0) (hd : dd ≠ 0)
    (h : 0 < weight du dd) : Agree du dd := by
  rcases lt_trichotomy du 0 with hu' | hu' | hu'
  · rcases lt_trichotomy dd 0 with hd' | hd' | hd'
    · exact Or.inr ⟨hu', hd'⟩
    · exact absurd hd' hd
    · exact absurd h (by simpa [weight] using not_lt.mpr (le_of_lt (mul_neg_of_neg_of_pos hu' hd')))
  · exact absurd hu' hu
  · rcases lt_trichotomy dd 0 with hd' | hd' | hd'
    · exact absurd h (by simpa [weight] using not_lt.mpr (le_of_lt (mul_neg_of_pos_of_neg hu' hd')))
    · exact absurd hd' hd
    · exact Or.inl ⟨hu', hd'⟩

section Ensemble

variable [Fintype ι] [DecidableEq ι]

/-- Over a whole ensemble: the configurations carrying a negative weight are EXACTLY the ones on
    which the channels disagree.

    This is the equality section 2's last column reports as "exact" -- an identity checked
    configuration by configuration, not a fitted correspondence. Stated as sets rather than as
    fractions because the fractions are these sets counted, and the counting adds nothing. -/
theorem negative_set_eq_disagreement_set
    (du dd : ι → ℝ) (hu : ∀ i, du i ≠ 0) (hd : ∀ i, dd i ≠ 0) :
    {i | weight (du i) (dd i) < 0} = {i | ¬ Agree (du i) (dd i)} := by
  ext i
  simp only [Set.mem_setOf_eq]
  constructor
  · intro h hagree
    exact absurd h (not_lt.mpr (le_of_lt (weight_pos_of_agree hagree)))
  · intro h
    rcases lt_trichotomy (weight (du i) (dd i)) 0 with hlt | heq | hgt
    · exact hlt
    · exact absurd heq (mul_ne_zero (hu i) (hd i))
    · exact absurd (agree_of_weight_pos (hu i) (hd i) hgt) h

/-- The same statement as the paper counts it: the disagreement fraction IS the negative-weight
    fraction, on doped rows as well as half-filled ones. -/
theorem negative_count_eq_disagreement_count
    (du dd : ι → ℝ) (hu : ∀ i, du i ≠ 0) (hd : ∀ i, dd i ≠ 0)
    [DecidablePred fun i => weight (du i) (dd i) < 0]
    [DecidablePred fun i => ¬ Agree (du i) (dd i)] :
    (Finset.univ.filter fun i => weight (du i) (dd i) < 0).card
      = (Finset.univ.filter fun i => ¬ Agree (du i) (dd i)).card := by
  congr 1
  ext i
  simpa using Set.ext_iff.mp (negative_set_eq_disagreement_set du dd hu hd) i

/-- The lockstep, as section 2 states it: if the channels agree on EVERY configuration then no
    configuration carries a negative weight, however often the individual channels cross zero.

    The hypothesis is about agreement and says nothing about how often a channel changes sign, which
    is what makes the measured 44.67% flip rate at `4x4` compatible with a zero negative fraction --
    and what makes the half-filled rows evidence rather than a vacuous check. -/
theorem no_negative_weight_of_lockstep
    (du dd : ι → ℝ) (h : ∀ i, Agree (du i) (dd i)) :
    ∀ i, 0 < weight (du i) (dd i) :=
  fun i => weight_pos_of_agree (h i)

end Ensemble

end SignProblem
