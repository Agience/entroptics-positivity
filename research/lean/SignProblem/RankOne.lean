/-
Section 5.3 -- the sign problem is the RANK-ONE case of a phase problem.

The triage reads two statistics off the weight cloud's `(Re, Im)` frame and sorts a run into three
regimes:

    focus = 1, resultant = 1    sign-free
    focus = 1, resultant < 1    a real sign problem
    focus < 1                   a phase problem no rotation reaches

The middle line carries the argument, and the paper states its content as: "rank one in the plane
means the phase takes two values `pi` apart, and a phase confined to `Z2` is what a sign IS." That
is a claim about complex numbers, not a measurement, and it is proved here.

Two consequences follow and both are things the paper reports as measured:

  * DE-ROTATION IS EXACT. A cloud lying on a line through the origin becomes real when multiplied by
    the conjugate of the line's direction -- a maximum imaginary part of `0.95` goes to `4e-16`, and
    the angle is read off the cloud's own leading direction rather than chosen.
  * `|<sgn>|` IS RECOVERED EXACTLY. De-rotating changes no magnitude, so the modulus of the mean is
    what it was; the reals left behind carry the sign and nothing else.

WHY THIS IS WORTH PROVING RATHER THAN MEASURING. The failure it guards against is silent and
expensive. A global phase cancels in `<O> = sum(O w)/sum(w)` and costs nothing, but the estimator
carried over from real weights, `mean(Re w)/mean(|w|)`, reads `cos(theta)` too small across rotations
that leave `|<w>|` unmoved -- and would overstate a cost that goes as `1/<sgn>^2` by fourteen times.
The theorem below is the statement that there was nothing there to lose.
-/
import Mathlib.Tactic
import Mathlib.Analysis.SpecialFunctions.Complex.Circle


namespace SignProblem

open Finset

variable {ι : Type*} [Fintype ι]

/-- A weight cloud is RANK ONE in the plane when every weight is a real multiple of one common unit
    direction. The multiple is signed, which is where the two values `pi` apart come from. -/
def RankOne (w : ι → ℂ) : Prop :=
  ∃ u : ℂ, ‖u‖ = 1 ∧ ∃ r : ι → ℝ, ∀ i, w i = (r i : ℂ) * u

/-- **De-rotation is exact.** Multiplying by the conjugate of the cloud's own direction leaves every
    weight real -- not small, zero.

    The direction is the cloud's, not one supplied: `u` comes from the cloud in `RankOne`, which is
    what the leading principal direction returns. No angle is chosen, which is what the canon
    requires. -/
theorem derotation_is_exact {w : ι → ℂ} (u : ℂ) (hu : ‖u‖ = 1) (r : ι → ℝ)
    (hw : ∀ i, w i = (r i : ℂ) * u) (i : ι) :
    (w i * (starRingEnd ℂ) u).im = 0 := by
  rw [hw i, mul_assoc]
  have hconj : u * (starRingEnd ℂ) u = 1 := by
    rw [RCLike.mul_conj, hu]
    norm_num
  rw [hconj, mul_one]
  simp

/-- **And it recovers the mean's modulus exactly**, because a unit rotation changes no magnitude.

    So `|<w>|` -- and with it `|<sgn>|` -- is what it was before the cloud was de-rotated. That is
    the sense in which a global phase is harmless, stated as an equality rather than as a
    reassurance. -/
theorem derotation_preserves_mean_modulus {w : ι → ℂ} (u : ℂ) (hu : ‖u‖ = 1)
    (r : ι → ℝ) (hw : ∀ i, w i = (r i : ℂ) * u) :
    ‖∑ i, w i‖ = |∑ i, r i| := by
  have hsum : ∑ i, w i = ((∑ i, r i : ℝ) : ℂ) * u := by
    rw [Finset.sum_congr rfl fun i _ => hw i, ← Finset.sum_mul]
    norm_cast
  rw [hsum, norm_mul, hu, mul_one, Complex.norm_real, Real.norm_eq_abs]

/-- **A phase confined to `Z2` is what a sign is.**

    On a rank-one cloud with no zero weight, each weight's phase is one of exactly two values `pi`
    apart -- `u` or `-u` -- and which one it is IS the configuration's sign. There is nothing else
    in the cloud to read. -/
theorem rank_one_phase_is_two_valued {w : ι → ℂ} (u : ℂ) (hu : ‖u‖ = 1) (r : ι → ℝ)
    (hw : ∀ i, w i = (r i : ℂ) * u) (hne : ∀ i, w i ≠ 0) (i : ι) :
    ∃ ρ : ℝ, 0 < ρ ∧ (w i = (ρ : ℂ) * u ∨ w i = (ρ : ℂ) * (-u)) := by
  have hr : r i ≠ 0 := by
    intro h
    exact hne i (by rw [hw i, h]; simp)
  rcases lt_trichotomy (r i) 0 with h | h | h
  · exact ⟨-r i, by linarith, Or.inr (by rw [hw i]; push_cast; ring)⟩
  · exact absurd h hr
  · exact ⟨r i, h, Or.inl (hw i)⟩

/-- The estimator carried over from real weights is the one that goes wrong, and this exhibits it.

    `mean(Re w) / mean(|w|)` is not rotation-invariant: on a cloud of positive weights rotated by
    `theta` it reads `cos(theta)` rather than `1`, while `|<w>| / mean|w|` is unmoved. Section 5.3
    measures the damage as a fall from `0.93500` to `0.25011`; here it is the mechanism, exact. -/
theorem real_part_estimator_is_not_rotation_invariant
    {ι : Type*} [Fintype ι] [Nonempty ι] (θ : ℝ) :
    (∑ _i : ι, (Complex.exp ((θ : ℂ) * Complex.I)).re) / (Fintype.card ι : ℝ)
      = Real.cos θ := by
  have hcard : (0 : ℝ) < (Fintype.card ι : ℝ) := by
    exact_mod_cast Fintype.card_pos
  rw [Complex.exp_mul_I]
  simp only [Complex.add_re, Complex.mul_re, Complex.I_re, Complex.I_im,
             Complex.cos_ofReal_re, Complex.sin_ofReal_re, Complex.sin_ofReal_im,
             Finset.sum_const, Finset.card_univ, nsmul_eq_mul]
  field_simp
  ring

end SignProblem
