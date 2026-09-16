/-
Sections 4.1 and 5.1 -- what the alignment read GUARANTEES, as opposed to what it measured here.

Everything else in this development is about the model. This file is about the INSTRUMENT, and it
exists because three of the paper's load-bearing sentences are claims about the read rather than
about the Hubbard model, and were carried in prose:

  * §4.1: "the read is invariant to both the offset and the scale -- so neither constant has to be
    known." That is what lets the criterion run on a simulation's output with no `K` in hand,
    needing neither `lambda` nor `dtau L tr(K)`. If it were false, §4.1 would need both constants
    and would not be a read on output at all.
  * §4.1: "An exact affine relation must saturate a normalised alignment at 1. A broken identity
    cannot." That is the criterion itself, and it is an equivalence rather than a correlation.
  * §5.1: the deficit `1 + strength` lies in `[0, 2]` while `-ln<sgn>` is unbounded, which is why
    §5 claims onset and not severity, and why no fixed map between the two can exist.

AT ONE COLUMN PER SIDE THE READ IS PEARSON'S `r`. The coupling read on two single-column frames is
the centred, normalised alignment below -- which is what makes these theorems statements about the
read and not merely about a quantity resembling it. What the instrument supplies beyond this
arithmetic is the decision procedure around it: an exact re-pairing null, a resolved/unresolved
verdict, a mode count that can return zero. The arithmetic is what the paper's claims quantify over,
and it is what is proved here.

THE CENTRING IS NOT DECORATION. An uncentred cosine is not offset-invariant, so on data carrying an
offset -- which `ln|det_up| - ln|det_dn|` does, by `-dtau L tr(K)` -- it would read high for the
wrong reason. `strength` below is the centred quantity, which is the one the paper reports.
-/
import Mathlib.Tactic
import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.Analysis.InnerProductSpace.PiL2

namespace SignProblem

variable {n : ℕ}

/-- The all-ones column.

    Built with `WithLp.toLp` rather than written as a bare `fun _ => 1`, which elaborates at the
    function type: `EuclideanSpace` is a type synonym for it and the ascription does not survive. -/
noncomputable def ones (n : ℕ) : EuclideanSpace ℝ (Fin n) := WithLp.toLp 2 (fun _ => (1 : ℝ))

@[simp] theorem ones_apply (i : Fin n) : ones n i = 1 := rfl

/-- The mean of a finite column. -/
noncomputable def mean (a : EuclideanSpace ℝ (Fin n)) : ℝ := (∑ i, a i) / n

/-- A column with its mean removed -- the frame the read actually works on. -/
noncomputable def centre (a : EuclideanSpace ℝ (Fin n)) : EuclideanSpace ℝ (Fin n) :=
  a - (mean a) • ones n

@[simp] theorem centre_apply (a : EuclideanSpace ℝ (Fin n)) (i : Fin n) :
    centre a i = a i - mean a := by
  simp [centre]

/-- **The read**: the normalised alignment of two centred columns.

    At one column per side this is Pearson's `r`, and it is what the paper calls `strength`. -/
noncomputable def strength (a b : EuclideanSpace ℝ (Fin n)) : ℝ :=
  (inner ℝ (centre a) (centre b) : ℝ) / (‖centre a‖ * ‖centre b‖)

/-- **The read is bounded**, which is Cauchy-Schwarz and needs no hypothesis at all: not that the
    columns are non-degenerate, not that they are related in any way. -/
theorem abs_strength_le_one (a b : EuclideanSpace ℝ (Fin n)) : |strength a b| ≤ 1 :=
  abs_real_inner_div_norm_mul_norm_le_one _ _

theorem neg_one_le_strength (a b : EuclideanSpace ℝ (Fin n)) : -1 ≤ strength a b :=
  neg_le_of_abs_le (abs_strength_le_one a b)

theorem strength_le_one (a b : EuclideanSpace ℝ (Fin n)) : strength a b ≤ 1 :=
  le_of_abs_le (abs_strength_le_one a b)

/-- §5.1's deficit, `1 + strength`. -/
noncomputable def deficit (a b : EuclideanSpace ℝ (Fin n)) : ℝ := 1 + strength a b

/-- **The deficit is bounded in `[0, 2]`.**

    Section 5 rests its onset-not-severity limit on exactly this: a quantity confined to `[0, 2]`
    cannot be a monotone function of `-ln<sgn>`, which is unbounded above, so no fixed map between
    the read and the severity exists -- not as a limitation of this read, but as arithmetic. -/
theorem deficit_mem_Icc (a b : EuclideanSpace ℝ (Fin n)) : deficit a b ∈ Set.Icc (0 : ℝ) 2 := by
  have h₁ := neg_one_le_strength a b
  have h₂ := strength_le_one a b
  exact ⟨by unfold deficit; linarith, by unfold deficit; linarith⟩

/-- The deficit is zero exactly at perfect anti-alignment, which is where §5's calibration sits:
    particle-hole symmetry gives `G_dn = 1 - G_up`, the centred channels are exact negatives, and
    the read returns `-1`. -/
theorem deficit_eq_zero_iff (a b : EuclideanSpace ℝ (Fin n)) :
    deficit a b = 0 ↔ strength a b = -1 := by
  unfold deficit
  constructor <;> intro h <;> linarith

/-- Centring removes an added constant: the offset is gone before the read sees the data. -/
theorem centre_add_const (a : EuclideanSpace ℝ (Fin n)) (p : ℝ) (hn : 0 < n) :
    centre (a + p • ones n) = centre a := by
  have hcast : (n : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr hn.ne'
  have hmean : mean (a + p • ones n) = mean a + p := by
    unfold mean
    simp only [PiLp.add_apply, PiLp.smul_apply, ones_apply, smul_eq_mul, mul_one,
               Finset.sum_add_distrib, Finset.sum_const, Finset.card_univ, Fintype.card_fin,
               nsmul_eq_mul]
    field_simp
  ext i
  simp only [centre_apply, PiLp.add_apply, PiLp.smul_apply, ones_apply, smul_eq_mul, mul_one,
             hmean]
  ring

/-- Centring is homogeneous: a rescaling passes straight through it. -/
theorem centre_smul (a : EuclideanSpace ℝ (Fin n)) (q : ℝ) : centre (q • a) = q • centre a := by
  have hmean : mean (q • a) = q * mean a := by
    unfold mean
    simp only [PiLp.smul_apply, smul_eq_mul, ← Finset.mul_sum]
    ring
  ext i
  simp only [centre_apply, PiLp.smul_apply, smul_eq_mul, hmean]
  ring

/-- **The read is invariant to the offset and to a positive scale.**

    This is §4.1's licence to run the criterion on output with no `K`: §3's identity carries an
    offset `-dtau L tr(K)` and a slope `lambda`, and neither survives to the read. A read that
    needed them would need the Hamiltonian, and §4.1 would not exist. -/
theorem strength_affine_invariant (a b : EuclideanSpace ℝ (Fin n)) (p q : ℝ) (hq : 0 < q)
    (hn : 0 < n) :
    strength (q • a + p • ones n) b = strength a b := by
  have hcentre : centre (q • a + p • ones n) = q • centre a := by
    rw [centre_add_const _ p hn, centre_smul]
  unfold strength
  rw [hcentre, real_inner_smul_left, norm_smul, Real.norm_eq_abs, abs_of_pos hq]
  rcases eq_or_ne (‖centre a‖) 0 with h | h
  · rw [norm_eq_zero] at h
    simp [h]
  · rcases eq_or_ne (‖centre b‖) 0 with h' | h'
    · rw [norm_eq_zero] at h'
      simp [h']
    · field_simp

/-- **Saturation is exactly an affine relation.**

    `|strength| = 1` holds precisely when one centred column is a non-zero multiple of the other,
    which -- undoing the centring -- is precisely an exact affine relation between the two columns.
    Section 3's identity says `ln|det_up| - ln|det_dn|` is an exact affine function of `sum(x)`, so
    an unbroken identity forces the read to saturate and a broken one forbids it. -/
theorem abs_strength_eq_one_iff (a b : EuclideanSpace ℝ (Fin n)) :
    |strength a b| = 1 ↔ centre a ≠ 0 ∧ ∃ r : ℝ, r ≠ 0 ∧ centre b = r • centre a :=
  abs_real_inner_div_norm_mul_norm_eq_one_iff _ _

/-- The forward half, in the form §4.1 uses it: an exact affine relation saturates the read.

    This is the direction the criterion runs in -- the identity holds, therefore the read must
    saturate -- and it admits no counterexample, which is what makes the measured agreement on 15
    lattices a check on the rig rather than the evidence for the claim. -/
theorem abs_strength_eq_one_of_affine (a b : EuclideanSpace ℝ (Fin n)) (r : ℝ) (hr : r ≠ 0)
    (hca : centre a ≠ 0) (h : centre b = r • centre a) :
    |strength a b| = 1 :=
  (abs_strength_eq_one_iff a b).mpr ⟨hca, r, hr, h⟩

/-- And the contrapositive, which is what the build check of §4.1 relies on: if the read does not
    saturate then no affine relation exists, so the identity is broken -- read off two logged
    columns, with no Hamiltonian and no knowledge of the decoupling. -/
theorem no_affine_relation_of_abs_strength_ne_one (a b : EuclideanSpace ℝ (Fin n))
    (h : |strength a b| ≠ 1) :
    ¬ ∃ r : ℝ, r ≠ 0 ∧ centre a ≠ 0 ∧ centre b = r • centre a := by
  rintro ⟨r, hr, hca, hb⟩
  exact h (abs_strength_eq_one_of_affine a b r hr hca hb)

/-! ### The calibration of §5.1

Particle-hole symmetry gives `G_dn = 1 - G_up` configuration by configuration. That is an affine
relation with slope `-1`, so the read does not merely happen to come back at `-1` on the sign-free
rows -- it cannot come back at anything else. The paper reports `-1.0000` as a measurement; this is
why the measurement had no other option.
-/

/-- Centring a reflected column reflects its centring: `centre (c - a) = - centre a`. -/
theorem centre_const_sub (a : EuclideanSpace ℝ (Fin n)) (c : ℝ) (hn : 0 < n) :
    centre (c • ones n - a) = - centre a := by
  have hcast : (n : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr hn.ne'
  have hmean : mean (c • ones n - a) = c - mean a := by
    unfold mean
    simp only [PiLp.sub_apply, PiLp.smul_apply, ones_apply, smul_eq_mul, mul_one,
               Finset.sum_sub_distrib, Finset.sum_const, Finset.card_univ, Fintype.card_fin,
               nsmul_eq_mul]
    field_simp
  ext i
  simp only [centre_apply, PiLp.neg_apply, PiLp.sub_apply, PiLp.smul_apply, ones_apply,
             smul_eq_mul, mul_one, hmean]
  ring

/-- **The calibration is forced, not measured.**

    Where the two channels satisfy `b = c - a` -- which is what particle-hole symmetry delivers at
    `c = 1`, `G_dn = 1 - G_up` -- the centred columns are exact negatives and the read returns
    exactly `-1`. Section 5.1 reports `-1.0000`; no other value was available to it. -/
theorem strength_eq_neg_one_of_reflected (a : EuclideanSpace ℝ (Fin n)) (c : ℝ) (hn : 0 < n)
    (hca : centre a ≠ 0) :
    strength a (c • ones n - a) = -1 := by
  have hb : centre (c • ones n - a) = - centre a := centre_const_sub a c hn
  unfold strength
  rw [hb, inner_neg_right, norm_neg, real_inner_self_eq_norm_mul_norm]
  have hn0 : ‖centre a‖ ≠ 0 := fun h => hca (norm_eq_zero.mp h)
  field_simp

/-- And the deficit is therefore exactly zero there, which is the quantity §5 reports as the order
    parameter's floor. -/
theorem deficit_eq_zero_of_reflected (a : EuclideanSpace ℝ (Fin n)) (c : ℝ) (hn : 0 < n)
    (hca : centre a ≠ 0) :
    deficit a (c • ones n - a) = 0 := by
  rw [deficit_eq_zero_iff]
  exact strength_eq_neg_one_of_reflected a c hn hca

/-! ### The criterion is not vacuous

A theorem about a read is worth nothing if the read cannot discriminate, and an `iff` is worth
nothing if one side is never satisfiable. Both risks are real here and both are closed by exhibiting
a witness, in the spirit of the negative control every gate in this repository carries: the sweep
that finds nothing and the sweep that cannot find anything look identical from the outside.
-/

/-- The three-point column `(0, 1, 2)`, whose centring is `(-1, 0, 1)`. -/
noncomputable def sample : EuclideanSpace ℝ (Fin 3) := WithLp.toLp 2 ![0, 1, 2]

/-- The three-point column `(0, 1, 0)`, whose centring is `(-1/3, 2/3, -1/3)`. -/
noncomputable def other : EuclideanSpace ℝ (Fin 3) := WithLp.toLp 2 ![0, 1, 0]

theorem mean_sample : mean sample = 1 := by
  simp [mean, sample, Fin.sum_univ_three]
  norm_num

/-- **The hypothesis `centre a ≠ 0` is satisfiable**, so the saturation criterion is not quantifying
    over an empty set. A constant column centres to zero and the read is undefined on it; this one
    does not. -/
theorem centre_sample_ne_zero : centre sample ≠ 0 := by
  intro h
  have h1 : centre sample 0 = 0 := by rw [h]; rfl
  rw [centre_apply, mean_sample] at h1
  simp [sample] at h1

/-- **Two columns that are NOT affinely related exist**, so the criterion can fail -- which is what
    makes it a criterion. Read at index 1: the centring of `other` is `2/3` there while the centring
    of `sample` is `0`, and no multiple of zero is two thirds. -/
theorem not_affinely_related : ¬ ∃ r : ℝ, r ≠ 0 ∧ centre other = r • centre sample := by
  rintro ⟨r, _hr, h⟩
  have h1 : centre other 1 = r * centre sample 1 := by rw [h]; rfl
  have hms : mean sample = 1 := mean_sample
  have hmo : mean other = 1 / 3 := by
    simp [mean, other, Fin.sum_univ_three]
  rw [centre_apply, centre_apply, hms, hmo] at h1
  norm_num [sample, other] at h1

end SignProblem
