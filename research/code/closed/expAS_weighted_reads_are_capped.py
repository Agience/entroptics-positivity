"""Experiment AS -- every weighted read is capped by the sign problem, exactly.

Section 5 records that the coupling read detects ONSET and not SEVERITY.  This file gives the
reason, and it is structural rather than a property of that particular read.

For correctly importance-sampled configurations the sampling is done at `|w|`, so each
configuration enters a weighted average carrying only its SIGN: `w_i = +-1`.  Kish's effective
sample size of those weights is then

    ESS = (sum_i w_i)^2 / sum_i w_i^2 = (n <s>)^2 / n = n <s>^2

which is the `O(1/<s>^2)` cost of the sign problem, written as a property of the weights.

`carriage` reports that quantity as `effective_n`, and its own description is what makes this an
argument rather than a coincidence: it is "a property of the WEIGHTS, not of the frames, and the
ceiling on what any aggregation of them can support".  So the bound binds every read that
aggregates signed configurations -- `coupling` among them -- and not merely the ones tried here.

The consequence for this paper: no weighted read can escape the bottleneck, and a route that could
must not aggregate signed configurations at all.
"""
from __future__ import annotations

import numpy as np

import entroptics as E


def sign_weights(n, p_plus, seed):
    """The weights an importance-sampled ensemble actually carries: signs alone."""
    rng = np.random.default_rng(seed)
    return np.where(rng.random(n) < p_plus, 1.0, -1.0)


def kish(w):
    """Kish's effective sample size, written out."""
    return float(w.sum() ** 2 / (w ** 2).sum())


if __name__ == "__main__":
    print("=" * 92)
    print("EVERY WEIGHTED READ IS CAPPED AT n <sgn>^2.")
    print("Sampling at |w| leaves the sign as the whole weight, and the ceiling follows.")
    print()
    print(f"{'n':>7} {'<sgn>':>9} | {'carriage effective_n':>21} {'n <sgn>^2':>12} {'ratio':>8}")
    rng = np.random.default_rng(0)
    for n in (400, 2000):
        for p in (0.999, 0.75, 0.55, 0.505):
            w = sign_weights(n, p, seed=int(1000 * p) + n)
            X = rng.standard_normal((n, 8))
            c = E.carriage(X, w)
            pred = n * float(w.mean()) ** 2
            print(f"{n:7d} {float(w.mean()):9.4f} | {c.effective_n:21.4f} {pred:12.4f} "
                  f"{c.effective_n / pred if pred > 0 else float('nan'):8.4f}", flush=True)
        print(flush=True)
    print("The ratio is 1 by construction, and that is the point: the read does not estimate the")
    print("cost, it IS the cost.  A ceiling on the weights binds every aggregation of them, so no")
    print("weighted read -- including the coupling of section 5 -- can be a severity meter.")
