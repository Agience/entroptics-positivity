"""Experiment AQ -- on the anchor, a coupling read IS a reweighted average sign.

Section 8.1 records that reads which score severity from the sampled data reduce to the average
sign under a different, cheaper measure.  The anchor is the case where that reduction can be
written down and checked exactly: take the two sides to differ by the configuration's own sign,

    B = diag(sigma) A

which is what the two channels look like when the second is the first with the weight's sign
applied.  Then the read returns the average of `sigma` REWEIGHTED by each row's squared length --
not the plain average sign, and not something new.

Two numbers are separated here, because conflating them overstates the result:

  * the read against its OWN definition, `<Ac, Bc> / (|Ac| |Bc|)` on the centred frames, agrees to
    six decimals -- but that is the definition, not a finding;
  * the read against the `|x|^2`-WEIGHTED average sign agrees to about `1e-3`, the residual being
    the centring the read performs and the plain average does not.

So the reduction is real and it is approximate, and the size of the approximation is the centring.
The plain unweighted average sign is further off still, which is the point: a read of this kind
does not fail to see the sign, it sees a REWEIGHTED one, and a weighting that is not the weight
`|D|` mis-weights exactly the configurations that decide the answer.
"""
from __future__ import annotations

import numpy as np

import entroptics as E


def anchor(T=500, N=8, p_plus=0.65, seed=0):
    """A frame and the same frame with row signs applied -- the anchor of section 8.1."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((T, N))
    sigma = np.where(rng.random(T) < p_plus, 1.0, -1.0)
    return A, sigma[:, None] * A, sigma


def weighted_sign(A, sigma):
    """The average of sigma weighted by each row's squared length."""
    w = (A ** 2).sum(axis=1)
    return float((sigma * w).sum() / w.sum())


def centred_alignment(A, B):
    """<Ac, Bc> / (|Ac| |Bc|) -- the read's own definition, formed independently."""
    Ac = A - A.mean(axis=0, keepdims=True)
    Bc = B - B.mean(axis=0, keepdims=True)
    return float((Ac * Bc).sum() / (np.linalg.norm(Ac) * np.linalg.norm(Bc)))


if __name__ == "__main__":
    print("=" * 100)
    print("ON THE ANCHOR, THE READ IS A REWEIGHTED AVERAGE SIGN.")
    print()
    print(f"{'seed':>5} {'frac +1':>8} | {'plain <s>':>10} {'|x|^2-weighted':>15} "
          f"{'read':>10} | {'read - weighted':>16} {'read - definition':>18}")
    for seed in (0, 3, 7, 11, 19):
        A, B, sigma = anchor(seed=seed)
        read = float(E.reads.coupling(A, B).strength)
        wsgn = weighted_sign(A, sigma)
        defn = centred_alignment(A, B)
        print(f"{seed:5d} {float((sigma > 0).mean()):8.4f} | {float(sigma.mean()):10.5f} "
              f"{wsgn:15.5f} {read:10.5f} | {abs(read - wsgn):16.3e} "
              f"{abs(read - defn):18.3e}", flush=True)
    print()
    print("The last column is the read against its own definition and is machine precision, which")
    print("is a check on the arithmetic and not a finding.  The column before it is the finding:")
    print("the read tracks the |x|^2-weighted average sign to about 1e-3, the residual being the")
    print("centring.  A read of this kind does not fail to see the sign -- it sees a REWEIGHTED")
    print("one, and a weighting that is not |D| mis-weights the configurations that decide the")
    print("answer.")
