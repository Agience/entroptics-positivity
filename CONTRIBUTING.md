# Contributing to Entroptics Sign Problem

The fermion sign problem read before the censoring: the paper, the Entroptics reads, and the
determinantal quantum Monte Carlo machinery they run on. **entroptics is the engine, not a
dependency.**

## Rule zero: check entroptics first

    ../entroptics/src/entroptics/

Never reimplement a primitive the library has. If a library read does not fit, **measure** that and
show the number.

## Rule one: in this rig a defect does not look like a crash, it looks like a result

That is the finding this repository was built by, and it is not a figure of speech. Every one of
these printed as a physical claim before it was caught, and none announced itself:

| what it printed | what it was |
|---|---|
| a 2.7% method error | the finite-temperature Green's-function convention, in a ground-state code |
| *"every walker was killed by the constraint"* | float64 weight underflow |
| *"every walker was killed by the constraint"* | the arbitrary global sign of the trial wavefunction |
| a 13% error where the constraint cannot fire | half filling is an OPEN shell under periodic boundaries |
| a 12x worse bias at k = 2 | a two-body factorisation that holds only for a single determinant |
| an average sign that GROWS with beta | an estimator reset by population control |
| a negative-weight fraction at half filling | the naive matrix product, whose condition number is exp(beta) |

So:

* **Every gate carries a negative control.** A gate that cannot be made to fail is not a gate. Each
  one in `research/code/tests/` states the perturbation that breaks it and the value it then reads.
* **Compare against something exact**, not against a tolerance. Exact sector arithmetic, brute-force
  enumeration of every auxiliary field, a known symmetry identity. A precise failure value is worth
  more than pass/fail — a relative error of exactly 2.0 identified a sign flip in one step where a
  generic disagreement would have needed a bisection.
* **A gate that exercises only the reduced case cannot see the general one.** The `k = 1` identity
  gate passed bit-for-bit over a two-body error of 6 to 31% at `k > 1`, because at `k = 1` the two
  expressions coincide.

## Rule two: no fit, no force, no constant

Nothing here is tuned to a target, and every threshold is derived from the arithmetic rather than
chosen:

* the numerical rank floor is `eps * k * lambda_max`, the resolution float64 carries on a Gram
  matrix, not a preferred tolerance;
* drift tolerances are relative to the quantity's own scale, not absolute;
* the coupling's null is the instrument's own exact re-pairing null — no constant is supplied;
* a *basis* is not a fit. Adding vectors can only lower a variational energy, so a set of field
  strengths entering a basis is not a tuned parameter, and the monotone fall of the bound is what
  demonstrates it.

Where a constant was invented it was swept, and one of them was found to fabricate a failure in the
CONTROL — an adaptive-step reference biased a known-exact calculation to 6.5 sigma. Sweep yours.

## Rule three: estimate uncertainty from reproducibility

A within-run formula was wrong by 13x in one direction and 40x in the other on the same study. Run
the same point at several seeds and take the spread of the answers. If that disagrees with the
internal formula, believe the seeds — and do it before anything is built on the significance.

## Running

    pip install -r research/requirements.txt
    python -m pytest research/code/tests -q

The sampler is gated against brute-force enumeration of every auxiliary field at 5e-15. If that
gate does not pass, nothing downstream of it means anything.
