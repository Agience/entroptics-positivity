"""Is `arg(w)` a read of the auxiliary field?  Asked on the WHOLE field, not on chosen functionals.

WHY THIS IS THE CORE QUESTION.  §4 gives the magnitude ratio exactly from the field,

    ln|det_up| - ln|det_dn|  =  -dtau L tr(K)  +  lambda sum(x)

so the magnitudes cost nothing: they are a closed form in a quantity the sampler already holds. If
the ARGUMENT had any comparable relation, the phase could be computed without the determinant and
the sign problem would not be a sampling problem at all. So the question is not decorative.

WHY IT IS ASKED AGAIN.  §7 already reports that it does not, and answers it against three chosen
functionals -- `exp(i sum x)`, the staggered field sum, and the nearest-slice product -- returning
`z` between `-1.3` and `+1.8`. Three functionals is a GUESS AT A BASIS. A negative result there
says those three do not carry it; it does not say nothing does, and the difference matters for a
claim this load-bearing.

WHAT IS DIFFERENT HERE.  §6's method: hand the instrument the whole object and let it find the
directions. The frame is the FIELD ITSELF -- every one of the `L x N` Ising variables per
configuration, flattened, with nothing selected and nothing summarised -- read against the phase on
the (cos, sin) plane. `coupling` scores the relation against its own exact re-pairing null, and
`principal_directions` says which field directions carry it if any do.

This is the largest linear basis available: any functional §7 could have chosen is a vector in it,
so a null here subsumes a null there. Two things it still does not decide, and neither is claimed:
a NONLINEAR function of the field is outside a linear alignment's reach, and a relation weaker than
the instrument's resolution at this sample size would not register. What a null does establish is
that the phase carries no linear information from the field that this instrument can resolve -- on
the full basis rather than on three guesses.

THE CONTROL IS THE POINT.  The same read is run on `ln|det_up| - ln|det_dn|`, which §4 says IS an
exact affine function of the field. That must saturate. A rig where the magnitude relation failed
to register would prove nothing about the phase.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import expm

import entroptics_adapter as EA
from stable import slogdet_one_plus_block, udt_product


def sample(K, beta=6.0, U=4.0, dtau=0.125, n=400, seed=5):
    """Fields, log-magnitude ratio, and phase -- one pass, so all three describe the same draws."""
    K = np.asarray(K, dtype=complex)
    L = int(round(beta / dtau))
    N = K.shape[0]
    eK = expm(-dtau * K)
    lam = float(np.arccosh(np.exp(dtau * U / 2.0)))
    rng = np.random.default_rng(seed)
    X, ratio, phase = [], [], []
    for _ in range(n):
        x = rng.choice([-1.0, 1.0], size=(L, N))
        sg, lg = {}, {}
        for s in (+1, -1):
            d = np.exp(s * lam * x)
            Bl = eK[None, :, :] * d[:, None, :]
            Uu, D, T = udt_product(Bl[None], 4)
            sgn, la = slogdet_one_plus_block(Uu, D, T)
            sg[s], lg[s] = complex(sgn[0]), float(la[0])
        w = sg[+1] * sg[-1]
        X.append(x.ravel())
        ratio.append(lg[+1] - lg[-1])
        phase.append(w / abs(w))
    ph = np.asarray(phase)
    return (np.asarray(X), np.asarray(ratio)[:, None],
            np.stack([ph.real, ph.imag], axis=1))


def read_against_the_field(K, **kw):
    """`carriage`, not `coupling`.

    `coupling` compares two sides on ONE shared basis and refuses a 384-wide field against a
    2-wide phase, which is right: a signed alignment across two different bases is not defined.
    The question here is a different shape -- does this frame carry this scalar -- and `carriage`
    is the read for it: it nulls the CONTRIBUTION OF THE WEIGHTS to an aggregation over the
    frames, against the exact null that weights and frames are re-paired at random.

    So the field configurations are the frames and the phase's components are the weights.
    """
    X, ratio, ph = sample(K, **kw)
    rng = np.random.default_rng(3)

    def scored(w, draws=30):
        """`carried` against the spread of `carried` under the read's own re-pairing.

        `carried` is 1 when the weights carry nothing, so the frame-level question is how far the
        true pairing sits from that. The null is DRAWN rather than assumed, because the count of
        per-coordinate `resolved` hits is not the statistic here: at `far = 0.05` and 384
        coordinates about 19 clear it by chance, which is the range those counts come out in.
        """
        if float(np.ptp(w)) == 0.0:
            return None                     # a real-weight row: sin is identically zero
        true = float(EA.frame_carriage(X, w).carried)
        nulls = [float(EA.frame_carriage(X, w[rng.permutation(len(w))]).carried) for _ in range(draws)]
        m, s = float(np.mean(nulls)), float(np.std(nulls, ddof=1))
        return dict(carried=true, null_mean=m, null_sd=s, null_max=float(np.max(nulls)),
                    sigma=(true - m) / s)

    return dict(
        cos=scored(ph[:, 0]), sin=scored(ph[:, 1]), control=scored(ratio[:, 0]),
        n=len(X), width=X.shape[1],
    )


if __name__ == "__main__":
    from reads.expAO_spectral_criterion import build, ring, tri_ladder

    CASES = [
        ("2x4 clean (real w)", build(2, 4, 0.0, 0.0, 0.0)),
        ("2x4 mu=0.4 (real w)", build(2, 4, 0.0, 0.4, 0.0)),
        ("ring 5 flux pi/2", ring(5, np.pi / 2)),
        ("ring 7 flux pi/2", ring(7, np.pi / 2)),
        ("tri ladder 6 flux pi/2", tri_ladder(6, np.pi / 2)),
        ("tri ladder 8 flux pi/2", tri_ladder(8, np.pi / 2)),
    ]

    print("=" * 118)
    print("IS arg(w) A READ OF THE FIELD?  The frame is every Ising variable, not a functional.")
    print()
    print("The CONTROL column is ln|det_up| - ln|det_dn| read against the same frame. Section 4")
    print("says that is an exact affine function of sum(x), so it must saturate. If it does not,")
    print("the rig cannot see a relation of this kind and the phase column means nothing.")
    print()
    def cell(d):
        if d is None:
            return f"{'--':>9} {'--':>8}"
        return f"{d['carried']:9.4f} {d['sigma']:+7.1f}s"

    print(f"{'lattice':>24} {'cells':>6} | {'cos carried':>9} {'vs null':>8} | "
          f"{'sin carried':>9} {'vs null':>8} | {'CONTROL':>9} {'vs null':>8}")
    for name, K in CASES:
        r = read_against_the_field(np.asarray(K, dtype=complex))
        print(f"{name:>24} {r['width']:6d} | {cell(r['cos'])} | {cell(r['sin'])} | "
              f"{cell(r['control'])}", flush=True)

    print()
    print("=" * 118)
    print("A phase column that does not resolve, beside a control column that does, says the")
    print("phase carries no LINEAR information from the field that this instrument can resolve --")
    print("on the full field basis, which contains every functional section 7 chose.")
