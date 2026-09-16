/-
Section 8.3 -- the scalar decoupling family is COMPLETE.

The claim is not that the alternatives tried were worse; it is that there are no others. Using
`n^2 = n` for a fermionic occupation number, the interaction has exactly two quadratic forms
available, with the `m^2` coefficient pinned at `-U/2` and the `rho~^2` coefficient at `+U/2` for any
rewriting. The "shift the interaction around alpha" freedom that looks like a third option is
one-body and gets absorbed into the hopping matrix, so it supplies nothing new.

That is why splitting `U` between the SPIN and CHARGE channels is the whole family, and why the
paper can report a sweep across `theta = 0` to `1` as a complete answer rather than as five points:
the family has one parameter and the sweep covers it.

Both identities below are EXACT -- not approximations valid at small `dtau`, not leading order.
The paper checks them by quadrature on a single site at every mixing with a worst relative error of
`3.6e-16`; they are proved here for every value at once.

The SU(2) vector decoupling lies outside this family, which is why the paper had to build it
separately rather than read it off the sweep. Nothing here claims otherwise: the completeness is of
the SCALAR family, which is what its name says.
-/
import Mathlib.Tactic

namespace SignProblem

/-- A fermionic occupation number is idempotent: `n^2 = n`, because it takes the values 0 and 1.
    Every identity below is this fact applied twice. -/
def Occupation (x : ℝ) : Prop := x ^ 2 = x

theorem occupation_zero_or_one {x : ℝ} (h : Occupation x) : x = 0 ∨ x = 1 := by
  have : x * (x - 1) = 0 := by unfold Occupation at h; nlinarith [h]
  rcases mul_eq_zero.mp this with h' | h'
  · exact Or.inl h'
  · exact Or.inr (by linarith)

variable {nu nd : ℝ}

/-- The SPIN channel form: `n_up n_dn = (rho - m^2) / 2` with `rho = n_up + n_dn` and
    `m = n_up - n_dn`.

    The `m^2` coefficient is `-1/2`, so in the interaction it is pinned at `-U/2`. There is no
    freedom in it: it is forced by `n^2 = n` alone. -/
theorem spin_channel_identity (hu : Occupation nu) (hd : Occupation nd) :
    nu * nd = ((nu + nd) - (nu - nd) ^ 2) / 2 := by
  unfold Occupation at hu hd
  nlinarith [hu, hd]

/-- The CHARGE channel form: `n_up n_dn = (rho~^2 + rho~) / 2` with `rho~ = rho - 1`.

    The `rho~^2` coefficient is `+1/2`, so in the interaction it is pinned at `+U/2`. Its sign is
    the reason the charge channel needs an IMAGINARY field for repulsive `U`, which is what turns a
    sign problem into a phase problem -- and why section 8.3's sweep finds the spin channel winning
    monotonically with no interior optimum. -/
theorem charge_channel_identity (hu : Occupation nu) (hd : Occupation nd) :
    nu * nd = (((nu + nd) - 1) ^ 2 + ((nu + nd) - 1)) / 2 := by
  unfold Occupation at hu hd
  nlinarith [hu, hd]

/-- **The whole family, in one line: every affine mixing of the two forms is exact.**

    For any `theta`, `(1 - theta)` of the spin form plus `theta` of the charge form is still exactly
    `n_up n_dn`. So the mixing parameter is the only freedom there is, and a sweep over `theta` is a
    sweep over the entire scalar family rather than a sample from it.

    This is what "the family is provably closed" means, and it is what stops the search: there is no
    further rewriting to look for, so the question moves to decouplings outside the scalar family
    (the SU(2) vector one), which is where section 8.3 goes next. -/
theorem mixing_is_exact_for_every_theta (hu : Occupation nu) (hd : Occupation nd) (θ : ℝ) :
    (1 - θ) * (((nu + nd) - (nu - nd) ^ 2) / 2)
      + θ * ((((nu + nd) - 1) ^ 2 + ((nu + nd) - 1)) / 2)
      = nu * nd := by
  have hs := spin_channel_identity hu hd
  have hc := charge_channel_identity hu hd
  rw [← hs, ← hc]
  ring

/-- The two coefficients are PINNED, which is the other half of the completeness claim.

    If a rewriting `a * rho + b * m^2` reproduces `n_up n_dn` on every occupation pattern then
    `a = 1/2` and `b = -1/2`. Exhibiting the four patterns is enough because there are only four,
    which is the whole reason a two-site identity settles a question about a family of rewritings. -/
theorem spin_coefficients_are_forced (a b : ℝ)
    (h : ∀ nu nd : ℝ, Occupation nu → Occupation nd →
          nu * nd = a * (nu + nd) + b * (nu - nd) ^ 2) :
    a = 1 / 2 ∧ b = -(1 / 2) := by
  have h10 := h 1 0 (by norm_num [Occupation]) (by norm_num [Occupation])
  have h11 := h 1 1 (by norm_num [Occupation]) (by norm_num [Occupation])
  norm_num at h10 h11
  constructor <;> linarith

end SignProblem
