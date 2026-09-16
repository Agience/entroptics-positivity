/-
Section 9 -- the closed door on doping, proved rather than swept.

The paper states it in one line:

    "Every diagonal matrix commutes with every diagonal conjugation, so `S K S⁻¹` leaves `diag(K)`
     untouched; both routes then demand `K_ii = -K_ii` -- and `K_ii` is real because `K` is
     Hermitian -- so `diag(K) = 0` is forced. A chemical potential can therefore never be
     accommodated, by any flux, on any graph, at any size."

and then supports it with 84 measured flux values across four graphs. Measurement over 84 points is
not the same claim as "at any size": the sweep can only report that no flux it TRIED opened a route.
This file proves the statement the sweep was standing in for, with nothing quantified away -- for
every finite index type, every `K`, every diagonal unitary, both routes.

The whole argument is that a diagonal conjugation fixes the diagonal, which is one line of algebra.
Its force comes from being universally quantified, which is exactly what a sweep cannot supply.
-/
import Mathlib.Tactic
import SignProblem.Criterion

namespace SignProblem

open Matrix

variable {n : Type*} [Fintype n] [DecidableEq n]

/-- A diagonal conjugation leaves the diagonal of `K` untouched, whatever `s` is.

    `s i * K i i * (s i)⁻¹ = K i i`: the two factors cancel because they are scalars and scalars
    commute. This is the whole mechanism, and everything below is a consequence. -/
theorem conj_diagonal_eq (s : n → ℂ) (hs : ∀ i, s i ≠ 0) (K : Matrix n n ℂ) (i : n) :
    conj s K i i = K i i := by
  rw [conj_apply, mul_comm (s i) (K i i), mul_assoc, mul_inv_cancel₀ (hs i), mul_one]

/-- Route A forces a zero diagonal. -/
theorem routeA_forces_zero_diagonal {s : n → ℂ} {K : Matrix n n ℂ}
    (hs : ∀ i, s i ≠ 0) (h : RouteA s K) (i : n) : K i i = 0 := by
  have h1 : conj s K i i = K i i := conj_diagonal_eq s hs K i
  have h2 : conj s K i i = -K i i := by
    have := congrFun (congrFun h i) i
    simpa using this
  have : K i i = -K i i := h1 ▸ h2
  linear_combination this / 2

/-- Route B forces a zero diagonal too, once `K` is Hermitian -- and `K` is Hermitian because it is
    a one-body Hamiltonian.

    The extra step over route A is that the anti-similarity demands `K_ii = -conj(K_ii)`, which on
    its own says only that `K_ii` is imaginary. Hermiticity says it is real. Real and imaginary
    together is zero. -/
theorem routeB_forces_zero_diagonal {s : n → ℂ} {K : Matrix n n ℂ}
    (hs : ∀ i, s i ≠ 0) (hherm : K.IsHermitian) (h : RouteB s K) (i : n) : K i i = 0 := by
  have h1 : conj s K i i = K i i := conj_diagonal_eq s hs K i
  have h2 : conj s K i i = -(starRingEnd ℂ) (K i i) := by
    have := congrFun (congrFun h i) i
    simpa using this
  have hreal : (starRingEnd ℂ) (K i i) = K i i := by
    have := hherm.apply i i
    simpa using this
  rw [h1, hreal] at h2
  linear_combination h2 / 2

/-- **Neither route is open on a matrix with a non-zero diagonal entry.**

    This is the statement the paper's `mu` sweep was evidence for, with the quantifier restored: not
    "no flux we tried worked" but "no `S` exists", for every finite lattice and every `K`. -/
theorem no_route_of_nonzero_diagonal {K : Matrix n n ℂ} (hherm : K.IsHermitian)
    {i : n} (hii : K i i ≠ 0) :
    ¬ ∃ s : n → ℂ, (∀ j, s j ≠ 0) ∧ (RouteA s K ∨ RouteB s K) := by
  rintro ⟨s, hs, hA | hB⟩
  · exact hii (routeA_forces_zero_diagonal hs hA i)
  · exact hii (routeB_forces_zero_diagonal hs hherm hB i)

/-- **A chemical potential can never be accommodated, by any flux, on any graph, at any size.**

    `K - mu * I` is the one-body matrix with a uniform chemical potential, and it has `-mu` on every
    diagonal entry. So on any lattice with at least one site and any `mu ≠ 0`, neither route is
    open -- whatever the hopping graph is, whatever flux threads it, however large it is.

    Section 9 draws the consequence: "a construction seeking positivity away from half filling
    cannot get there by choosing a lattice, a flux or a size: the obstruction is a diagonal term,
    and no conjugation of this kind removes one." -/
theorem doping_closes_both_routes
    [Nonempty n] (K : Matrix n n ℂ) (hherm : K.IsHermitian) (hK : ∀ i, K i i = 0)
    (mu : ℝ) (hmu : mu ≠ 0) :
    ¬ ∃ s : n → ℂ, (∀ j, s j ≠ 0) ∧
        (RouteA s (K - (mu : ℂ) • 1) ∨ RouteB s (K - (mu : ℂ) • 1)) := by
  obtain ⟨i⟩ := ‹Nonempty n›
  have hsmul : ((mu : ℂ) • (1 : Matrix n n ℂ)).IsHermitian := by
    unfold Matrix.IsHermitian
    rw [Matrix.conjTranspose_smul, Matrix.conjTranspose_one]
    simp [Complex.conj_ofReal]
  have hherm' : (K - (mu : ℂ) • (1 : Matrix n n ℂ)).IsHermitian := hherm.sub hsmul
  have hii : (K - (mu : ℂ) • (1 : Matrix n n ℂ)) i i ≠ 0 := by
    simp [Matrix.sub_apply, Matrix.smul_apply, hK i]
    exact_mod_cast hmu
  exact no_route_of_nonzero_diagonal hherm' hii

/-- The same obstruction as section 4's "case neither condition names": a STAGGERED POTENTIAL.

    It leaves the lattice bipartite and the filling at exactly one per site, and it breaks the
    identity anyway -- "because a diagonal term cannot be negated by signs". Here that is the
    theorem above with a site-dependent diagonal rather than a uniform one, which is why the same
    proof covers both and why the paper can report the staggered rows and the doped rows as one
    phenomenon. -/
theorem staggered_potential_closes_both_routes
    (K : Matrix n n ℂ) (hherm : K.IsHermitian)
    {i : n} (hii : K i i ≠ 0) :
    ¬ ∃ s : n → ℂ, (∀ j, s j ≠ 0) ∧ (RouteA s K ∨ RouteB s K) :=
  no_route_of_nonzero_diagonal hherm hii

end SignProblem
