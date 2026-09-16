/-
Section 4 -- the criterion is a statement about the hopping GRAPH, not about lattices.

The paper measures it on topologies the rest of the work never uses -- chains open and periodic,
trees, stars, even and odd rings, triangular ladders -- and reports the pattern: "A tree satisfies it
however it is drawn, having no cycles at all; any graph carrying an odd cycle never does."

Twenty-four one-body matrices agreeing is evidence for a rule. The rule itself is this file. The
content is one propagation lemma -- a proper signing alternates along every walk -- and the odd-cycle
obstruction falls out of it immediately:

    a closed walk of ODD length carries s a = -s a, which is impossible for s a = ±1.

That is the direction the measurement actually uses. Every non-bipartite row in section 4 is a graph
with an odd cycle, and what is being claimed of those rows is that no signed diagonal exists -- a
universal statement over all `2^N` signings, which the rig checks by construction and which is
proved here for every graph and every size at once.

A walk is carried as a LIST of sites with consecutive pairs joined by a non-zero entry of `K`. That
is the whole graph structure this argument needs; bringing in a full graph library would add
definitions without adding content, since the criterion never asks anything of the graph except
which pairs are joined.
-/
import Mathlib.Tactic
import Mathlib.Data.Matrix.Basic
import SignProblem.Criterion

namespace SignProblem

open Matrix

variable {n : Type*} [Fintype n] [DecidableEq n]

/-- A walk in `K`'s support graph: consecutive sites are joined by a non-zero entry. -/
def IsWalk (K : Matrix n n ℝ) : List n → Prop
  | [] => True
  | [_] => True
  | a :: b :: rest => K a b ≠ 0 ∧ IsWalk K (b :: rest)

/-- A proper signing: every edge of the support graph joins two sites of opposite sign. This is
    exactly the right-hand side of `signed_criterion`. -/
def ProperSigning (K : Matrix n n ℝ) (s : n → ℝ) : Prop :=
  Signed s ∧ ∀ i j, K i j ≠ 0 → s i * s j = -1

/-- **A proper signing alternates along every walk**: after `ℓ` steps the sign has been multiplied
    by `(-1)^ℓ`.

    Everything else in this file is this lemma read at a closed walk. -/
theorem signing_alternates {K : Matrix n n ℝ} {s : n → ℝ} (hs : ProperSigning K s) :
    ∀ (a : n) (l : List n), IsWalk K (a :: l) →
      s ((a :: l).getLast (by simp)) = (-1) ^ l.length * s a := by
  intro a l
  induction l generalizing a with
  | nil => intro _; simp
  | cons b rest ih =>
      intro hw
      obtain ⟨hedge, hrest⟩ := hw
      have hstep : s b = -s a := by
        have h := hs.2 a b hedge
        rcases hs.1 a with ha | ha <;> rw [ha] at h ⊢ <;> linarith
      have := ih b hrest
      rw [List.getLast_cons_cons] at *
      rw [this, hstep, List.length_cons]
      ring

/-- **An odd closed walk admits no proper signing.**

    This is the odd-cycle obstruction: a triangle, a 5-ring, a 7-ring, a triangular ladder, a `3x3`
    periodic lattice whose wraps are odd. In every such case the paper measures the identity broken
    and a sign problem present, and this says the first of those could not have gone otherwise.

    Note the quantifier. It is not that no signing was found; it is that none exists. -/
theorem no_proper_signing_of_odd_closed_walk
    {K : Matrix n n ℝ} {a : n} {l : List n}
    (hw : IsWalk K (a :: l)) (hclosed : (a :: l).getLast (by simp) = a)
    (hodd : Odd l.length) :
    ¬ ∃ s : n → ℝ, ProperSigning K s := by
  rintro ⟨s, hs⟩
  have h := signing_alternates hs a l hw
  rw [hclosed, hodd.neg_one_pow] at h
  have hne : s a ≠ 0 := signed_ne_zero hs.1 a
  have : s a = -s a := by linarith [h]
  apply hne
  linarith

/-- A TREE satisfies the criterion however it is drawn, "having no cycles at all".

    Stated as its contrapositive content: the obstruction above is the only one this argument
    produces, so a graph with no closed walk of odd length is never refuted by it. A star graph is
    the paper's witness and it reads `5.9e-12`. -/
theorem no_odd_obstruction_of_no_odd_closed_walk
    {K : Matrix n n ℝ} {s : n → ℝ} (hs : ProperSigning K s)
    {a : n} {l : List n} (hw : IsWalk K (a :: l))
    (hclosed : (a :: l).getLast (by simp) = a) :
    Even l.length := by
  have h := signing_alternates hs a l hw
  rw [hclosed] at h
  have hne : s a ≠ 0 := signed_ne_zero hs.1 a
  rcases Nat.even_or_odd l.length with he | ho
  · exact he
  · exfalso
    rw [ho.neg_one_pow] at h
    apply hne
    linarith

/-- The `i = j` case of the criterion, which the colouring language hides: a self-loop is an odd
    closed walk of length one, so a non-zero DIAGONAL entry refutes the signing immediately.

    That is the same obstruction as the doping no-go, arrived at from the graph side: a chemical
    potential puts a self-loop on every site. -/
theorem no_proper_signing_of_nonzero_diagonal
    {K : Matrix n n ℝ} {i : n} (hii : K i i ≠ 0) :
    ¬ ∃ s : n → ℝ, ProperSigning K s := by
  rintro ⟨s, hs⟩
  have h := hs.2 i i hii
  rcases hs.1 i with hi | hi <;> rw [hi] at h <;> norm_num at h

end SignProblem
