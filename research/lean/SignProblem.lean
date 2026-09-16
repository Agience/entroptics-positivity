/-
The machine-checked algebraic core of `research/PAPER.md`.

WHAT IS HERE AND WHAT IS NOT. This paper's results are measurements, and a measurement is not a
theorem. What it also has is an algebraic spine -- the identity of section 3, the criterion of
section 4, the ceiling of section 8 -- and several of the paper's statements are universally
quantified claims that a sweep can only sample. Those are the ones formalised here. The division is
deliberate and it is stated in the paper's section 10: a claim that reads "at any size, on any
graph, at any flux" belongs in Lean, and a claim that reads "measured at 137 sigma" does not.

Two statements in particular were carried by measurement where a proof was available, and both are
now proved:

  * the CLOSED DOOR ON DOPING (`Doping.lean`) -- the paper supports "a chemical potential can never
    be accommodated, by any flux, on any graph, at any size" with 84 measured flux values across
    four graphs. A sweep reports that no flux it TRIED opened a route. `doping_closes_both_routes`
    says none exists.
  * MAGNITUDE BLINDNESS AT EVERY EXPONENT (`Blindness.lean`) -- the paper measures `q = 0.5, 1, 2, 3`
    with a worst disagreement of exactly zero. `sum_abs_pow_blind` covers every real `q`, so there
    is no exponent left to reach for.

  import SignProblem.Weight      -- sec 2: a negative weight IS a disagreement between channels
  import SignProblem.Identity    -- sec 3: the log-determinant identity, and what it requires
  import SignProblem.Criterion   -- sec 4: the two routes, entry by entry; the spectral one is weaker
  import SignProblem.Cycles      -- sec 4: the criterion as a property of the hopping GRAPH
  import SignProblem.Doping      -- sec 9: a diagonal term closes both routes, at any size
  import SignProblem.Ceiling     -- sec 8: ESS = n <sgn>^2, exactly, and it is a ceiling
  import SignProblem.Blindness   -- sec 8.2: magnitude reads are blind at EVERY exponent
  import SignProblem.Coupling    -- sec 8.1: the coupling read IS the |x|^2-weighted average sign
  import SignProblem.Decoupling  -- sec 8.3: the scalar decoupling family is complete
  import SignProblem.RankOne     -- sec 5.3: a sign is the rank-one case of a phase
  import SignProblem.Alignment   -- sec 4.1, 5.1: what the alignment READ guarantees
  import SignProblem.Parity      -- sec 7: the sign is a parity, and what that rules out
-/
import SignProblem.Weight
import SignProblem.Identity
import SignProblem.Criterion
import SignProblem.Cycles
import SignProblem.Doping
import SignProblem.Ceiling
import SignProblem.Blindness
import SignProblem.Coupling
import SignProblem.Decoupling
import SignProblem.RankOne
import SignProblem.Alignment
import SignProblem.Parity

namespace SignProblem

/-!
## The axiom footprint

Every theorem below is checked against Lean's three foundational axioms and nothing else:
`propext`, `Classical.choice`, `Quot.sound`. There is no `sorry` and no axiom of this
development's own. `#print axioms` is an info message emitted while a declaration is elaborated;
measured here, `lake` replays those messages from its cache, so a warm build reports the same
twenty-six lines as a cold one. `lean_build.py --clear-dry-run` shows what a cold run would clear,
for the case where it does not.
-/

#print axioms weight_neg_iff
#print axioms negative_set_eq_disagreement_set
#print axioms log_det_identity
#print axioms det_one_add_eq
#print axioms signed_criterion
#print axioms routeA_iff
#print axioms routeB_iff
#print axioms routeA_iff_routeB_of_real
#print axioms spectral_similarity_is_weaker
#print axioms no_proper_signing_of_odd_closed_walk
#print axioms signing_alternates
#print axioms conj_diagonal_eq
#print axioms routeA_forces_zero_diagonal
#print axioms routeB_forces_zero_diagonal
#print axioms doping_closes_both_routes
#print axioms kish_of_signs
#print axioms kish_le_card
#print axioms sum_abs_pow_blind
#print axioms gram_blind
#print axioms coupling_is_not_blind
#print axioms frobenius_inner_eq_weighted_signs
#print axioms coupling_is_weighted_mean_sign
#print axioms mixing_is_exact_for_every_theta
#print axioms spin_coefficients_are_forced
#print axioms derotation_is_exact
#print axioms derotation_preserves_mean_modulus
#print axioms abs_strength_le_one
#print axioms deficit_mem_Icc
#print axioms strength_affine_invariant
#print axioms abs_strength_eq_one_iff
#print axioms abs_strength_eq_one_of_affine
#print axioms strength_eq_neg_one_of_reflected
#print axioms deficit_eq_zero_of_reflected
#print axioms centre_sample_ne_zero
#print axioms not_affinely_related
#print axioms prod_pos_iff_even_neg
#print axioms prod_neg_iff_odd_neg
#print axioms det_determines_sign
#print axioms sign_not_continuousAt_zero
#print axioms no_continuous_function_is_the_sign

-- The remaining theorems section 10's table names. Every declaration the paper cites has its
-- axiom footprint printed, so the CI check covers the whole of what the table claims.
#print axioms difference_not_constant
#print axioms kish_eq_card_of_sign_free
#print axioms uniform_rows_give_plain_mean_sign
#print axioms rank_one_phase_is_two_valued
#print axioms real_part_estimator_is_not_rotation_invariant
#print axioms centre_add_const
#print axioms centre_smul
#print axioms no_affine_relation_of_abs_strength_ne_one

end SignProblem
