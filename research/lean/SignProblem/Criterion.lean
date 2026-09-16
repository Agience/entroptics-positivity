/-
Section 4 -- what the identity requires, as a decidable property of the one-body matrix.

Section 3's identity needs the pairing `det(I + B_up⁻¹) = det(I + B_dn)`, which needs `expmK⁻¹`
similar to `expmK` by a similarity that COMMUTES with the interaction's diagonal factor. That
restricts the similarity to a diagonal one, and section 4 turns the requirement into two routes:

    route A    S K S⁻¹ = -K            every cycle even, at any flux
    route B    S K S⁻¹ = -conj(K)      odd cycles need flux pi/2 mod pi

with `S` a diagonal unitary. This file is the elementwise content of both: a diagonal conjugation
acts entry by entry, so each route is a condition on the entries of `K` and nothing more. That is
what makes the criterion `O(N^2)` with no eigenvalues, no determinants and no field.

NO NEW CRITERION IS CLAIMED, and the paper says so in its own words: for real `K` the two-colouring
is bipartiteness and the zero diagonal is the absence of a staggered potential, which is the
long-known particle-hole route to positivity. What is formalised here is that this rig computes the
known criterion correctly -- the yardstick the reading of section 5 is scored against, verified
before it is used.
-/
import Mathlib.Tactic
import Mathlib.Data.Matrix.Basic
import Mathlib.LinearAlgebra.Matrix.Hermitian

namespace SignProblem

open Matrix

variable {n : Type*} [Fintype n] [DecidableEq n]

/-- A diagonal conjugation, written with the inverse diagonal explicitly so that no invertibility
    instance is needed to state it. -/
noncomputable def conj (s : n → ℂ) (K : Matrix n n ℂ) : Matrix n n ℂ :=
  diagonal s * K * diagonal (fun j => (s j)⁻¹)

/-- A diagonal conjugation acts ENTRY BY ENTRY. Everything in section 4 follows from this one line:
    the conjugation cannot mix entries, so each route becomes a condition on each entry separately,
    and a condition on each entry separately is a statement about the support graph. -/
@[simp] theorem conj_apply (s : n → ℂ) (K : Matrix n n ℂ) (i j : n) :
    conj s K i j = s i * K i j * (s j)⁻¹ := by
  simp [conj, Matrix.mul_apply, Matrix.diagonal_apply, Finset.sum_ite_eq,
        Finset.sum_ite_eq', mul_comm, mul_assoc, mul_left_comm]

/-- Route A: `S K S⁻¹ = -K`. -/
def RouteA (s : n → ℂ) (K : Matrix n n ℂ) : Prop := conj s K = -K

/-- Route B: `S K S⁻¹ = -conj(K)`, the ANTI-similarity. It admits odd cycles carrying half-odd-integer
    flux, and it is the route that leaves a phase behind -- so it delivers the identity without
    delivering positivity. -/
def RouteB (s : n → ℂ) (K : Matrix n n ℂ) : Prop :=
  conj s K = -(K.map (starRingEnd ℂ))

/-- `S` is a diagonal UNITARY: every diagonal entry has modulus one. -/
def Unimodular (s : n → ℂ) : Prop := ∀ i, ‖s i‖ = 1

theorem ne_zero_of_unimodular {s : n → ℂ} (hs : Unimodular s) (i : n) : s i ≠ 0 := by
  intro h
  have := hs i
  rw [h] at this
  simp at this

/-- Route A, entry by entry. -/
theorem routeA_iff (s : n → ℂ) (K : Matrix n n ℂ) :
    RouteA s K ↔ ∀ i j, s i * K i j * (s j)⁻¹ = -K i j := by
  constructor
  · intro h i j
    have := congrFun (congrFun h i) j
    simpa using this
  · intro h
    ext i j
    simpa using h i j

/-- Route B, entry by entry. -/
theorem routeB_iff (s : n → ℂ) (K : Matrix n n ℂ) :
    RouteB s K ↔ ∀ i j, s i * K i j * (s j)⁻¹ = -(starRingEnd ℂ) (K i j) := by
  constructor
  · intro h i j
    have := congrFun (congrFun h i) j
    simpa using this
  · intro h
    ext i j
    simpa using h i j

section RealCase

/-- For a REAL one-body matrix the two routes coincide, because conjugation does nothing.

    This is why the signed-diagonal colouring is the whole story there -- the paper checks it on 11
    real matrices and the general criterion and the colouring return the same answer on every one. -/
theorem routeA_iff_routeB_of_real (s : n → ℂ) (K : Matrix n n ℂ)
    (hK : ∀ i j, (starRingEnd ℂ) (K i j) = K i j) :
    RouteA s K ↔ RouteB s K := by
  rw [routeA_iff, routeB_iff]
  constructor <;> intro h i j <;> rw [h i j] <;> rw [hK]

/-- A SIGNED diagonal: `s i = ±1`, which is what route A reduces to on a real `K`. -/
def Signed (s : n → ℝ) : Prop := ∀ i, s i = 1 ∨ s i = -1

theorem signed_sq {s : n → ℝ} (hs : Signed s) (i : n) : s i * s i = 1 := by
  rcases hs i with h | h <;> rw [h] <;> ring

theorem signed_ne_zero {s : n → ℝ} (hs : Signed s) (i : n) : s i ≠ 0 := by
  rcases hs i with h | h <;> rw [h] <;> norm_num

/-- The criterion, for a real `K` and a signed diagonal: `S K S = -K` holds exactly when every
    non-zero entry joins two sites of OPPOSITE colour.

    That is a 2-colouring of `K`'s support graph, and it costs `O(N^2)` to check -- no eigenvalues,
    no determinants, no field. The `i = j` case is separated out below because it is the one the
    colouring language hides, and it is the case that closes the door on doping. -/
theorem signed_criterion (s : n → ℝ) (hs : Signed s) (K : Matrix n n ℝ) :
    (diagonal s * K * diagonal s = -K) ↔ ∀ i j, K i j ≠ 0 → s i * s j = -1 := by
  have hentry : ∀ i j, (diagonal s * K * diagonal s) i j = s i * K i j * s j := by
    intro i j
    simp [Matrix.mul_apply, Matrix.diagonal_apply, Finset.sum_ite_eq, Finset.sum_ite_eq',
          mul_comm, mul_assoc, mul_left_comm]
  constructor
  · intro h i j hij
    have hthis := congrFun (congrFun h i) j
    rw [hentry] at hthis
    simp only [Matrix.neg_apply] at hthis
    -- `s i * K i j * s j = -K i j` factors as `K i j * (s i * s j + 1) = 0`, and `K i j ≠ 0`
    have hfac : K i j * (s i * s j + 1) = 0 := by linear_combination hthis
    rcases mul_eq_zero.mp hfac with h0 | h0
    · exact absurd h0 hij
    · linarith
  · intro h
    ext i j
    rw [hentry]
    simp only [Matrix.neg_apply]
    by_cases hij : K i j = 0
    · rw [hij]; ring
    · linear_combination K i j * h i j hij

/-- The spectral criterion is the obvious one and it is WRONG, and this says why in one line.

    For Hermitian `K`, a spectrum symmetric about zero is exactly similarity to `-K`. But the
    derivation needs a similarity that COMMUTES with `diag(exp(sigma lambda x))`, and only a
    diagonal one does. The staggered rows of section 4 have spectra symmetric to `1e-15` and break
    the identity by up to `21`: the similarity exists and is not a signed diagonal.

    Formally: a similarity `P K P⁻¹ = -K` carries no information about the DIAGONAL of `K`, whereas
    a diagonal one forces it to vanish (`Doping.lean`). So the two conditions are not the same
    condition, and the gap between them is exactly a staggered potential. -/
theorem spectral_similarity_is_weaker :
    ∃ (K : Matrix (Fin 2) (Fin 2) ℝ) (P : Matrix (Fin 2) (Fin 2) ℝ),
      P * K = -(K * P) ∧ IsUnit P.det ∧ K 0 0 ≠ 0 := by
  refine ⟨!![1, 0; 0, -1], !![0, 1; 1, 0], ?_, ?_, ?_⟩
  · ext i j; fin_cases i <;> fin_cases j <;> simp [Matrix.mul_apply, Fin.sum_univ_succ]
  · simp [Matrix.det_fin_two_of]
  · simp

end RealCase

end SignProblem
