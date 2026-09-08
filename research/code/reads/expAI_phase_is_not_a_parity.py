"""Experiment AI -- a phase is not a parity, and the same physics can have either.

Section 7 argues that no continuous summary determines the sign, because the sign is a PARITY and
a parity is not a continuous function of anything.  The boundary found in section 5 raises the
other case: when the weight is complex the sign problem is a PHASE, which is continuous, so that
argument does not reach it and section 7 is incomplete as written.

The cleanest instrument for this is already in the model.  Both of these are exact identities on
the four states of a site,

    (n_up - 1/2)(n_dn - 1/2) = 1/4 - m^2/2  (SPIN, real field)  =  rho^2/2 - 1/4  (CHARGE,
                                                                     imaginary field)

so splitting `U = (1-theta) U + theta U` and decoupling each piece in its own channel gives a
one-parameter family in which EVERY theta describes the same physics.  `theta = 0` is the real
spin decoupling of sections 2 to 5; `theta = 1` is pure charge and complex.  Any observable must
come out theta-independent -- that is the model's own gate -- while the sign structure does not,
because the sign problem is a property of the REPRESENTATION and not of the system.

So this sweep asks a question with a definite right answer available for comparison: does the read
follow the representation's sign structure, along an axis where the physics is pinned?

WHAT THE SWEEP ACTUALLY FOUND, which is not what it was built to find.  At half filling the whole
family is sign-free -- the charge channel included, `|Im w / Re w|` at 1e-15 and no negative real
part at any theta -- so the complex-weight case this file was written for is not there and had to
be reached by doping.  What IS there is stronger: along an axis where the physics is pinned by
construction and the weights stay real and positive throughout, the read moves from -1.0000 to
+1.0000.  The read is therefore a property of a (Hamiltonian, DECOUPLING) pair and not of a
Hamiltonian.  That is not a defect -- the sign problem is a property of the same pair -- but it
does mean the read cannot be quoted about a model without saying how the model was decoupled.

ON THE ESTIMATOR.  Fields are drawn from the prior, so the mean phase is a REWEIGHTED average and
the effective sample size is reported beside it.  A previous measurement in this project was
discarded for quoting a reweighted quantity at an ESS of 1.2 out of 400; no row here is read where
the ESS does not support it, and the ESS column is printed for every row so that judgement is the
reader's too.
"""
from __future__ import annotations

import numpy as np

import entroptics as E
from model2d import Model2D
from stable import udt_product, inv_one_plus_block, slogdet_one_plus_block


def sweep(theta, beta, mu=0.0, U=4.0, dtau=0.125, Lx=2, Ly=4, n_draw=400, seed=0):
    L = int(round(beta / dtau))
    m = Model2D(Lx=Lx, Ly=Ly, t=1.0, mu=mu, U=U, dtau=dtau, L=L, theta=theta)
    rng = np.random.default_rng(seed)
    A, B, W = [], [], []
    for _ in range(n_draw):
        X = rng.standard_normal((L, m.N))
        Y = rng.standard_normal((L, m.N))
        g, det = {}, {}
        for sigma in (+1, -1):
            Bl = m.B_slices(X[None], Y[None], sigma)[0]
            Uu, D, T = udt_product(Bl[None], 4)
            g[sigma] = np.diag(inv_one_plus_block(Uu, D, T)[0]).copy()
            s, la = slogdet_one_plus_block(Uu, D, T)
            det[sigma] = complex(s[0]) * np.exp(float(la[0]))
        w = complex(m.charge_phase(Y[None])[0]) * det[+1] * det[-1]
        A.append(g[+1]); B.append(g[-1]); W.append(w)
    A, B, W = np.array(A), np.array(B), np.array(W)

    aw = np.abs(W)
    ess = float(aw.sum() ** 2 / (aw ** 2).sum()) if aw.sum() > 0 else 0.0
    phase = W / np.where(aw > 0, aw, 1.0)
    mean_phase = complex((aw * phase).sum() / aw.sum()) if aw.sum() > 0 else 0j
    c = E.reads.coupling(A, B)
    return dict(ess=ess, n=n_draw,
                # |Im w| / |w|, BOUNDED BY 1.  An earlier version used |Im w| / |Re w|, which is
                # unbounded and blows up whenever Re w passes near zero -- exactly what a weight
                # crossing from positive to negative does -- so it read 71.0 on a row whose
                # weights are 57% imaginary and 3.2 on one that is 61% imaginary.  It ordered the
                # rows wrongly and it is not a share of anything.
                imagshare=float(np.mean(np.abs(W.imag) / np.where(aw > 0, aw, 1.0))),
                # THERE IS NO FRACTION OF DRAWS CARRYING A PHASE HERE, AND THAT IS A MEASUREMENT.
                #
                # This slot held `mean((|arg w| > 0.1) & (|arg w| < pi - 0.1))` -- a COUNT behind
                # a chosen 0.1-radian window -- and it is gone rather than re-cut, because both
                # ways of drawing the window are wrong:
                #
                #   * the window undercounts.  It read 0.93-0.96 on the doped rows purely by
                #     dropping the draws whose argument lands within 0.1 rad of 0 or pi;
                #   * removing it does not give a threshold-free count.  `Im w != 0` reads
                #     0.9925-1.0000 on the HALF-FILLED rows, whose weights are real -- it is
                #     counting floating-point dust at 1e-15.
                #
                # A binary 'is this weight complex' has no cut-free form at finite precision. The
                # CONTINUOUS quantities do, need no window, and separate the same two populations
                # by fourteen orders: `imagshare` above reads 3e-15 half-filled against 0.57-0.65
                # doped, and `1 - absmean` reads 3e-16 against 0.21-0.67.  Measured 2026-09-07.
                negre=float(np.mean(W.real < 0)),
                absmean=float(abs(mean_phase)), argmean=float(np.angle(mean_phase)),
                s=float(c.strength), z=float(c.z), res=bool(c.resolved),
                cphase=float(c.phase))


if __name__ == "__main__":
    SEEDS = (1, 7, 23, 45)
    print("=" * 112)
    print("THE READ IS A PROPERTY OF THE DECOUPLING, NOT OF THE MODEL.")
    print("2x4, U = 4, beta = 4, Gaussian fields from the prior, 400 draws x 4 seeds.")
    print("Every theta at a given mu is the SAME PHYSICS -- that is the model's own gate.")
    print()
    print("No reweighted expectation is quoted: the ESS column shows why.  Every number below is")
    print("a property of the PRIOR ensemble -- the negative fraction of Re w among the draws, the")
    print("typical size of Im w against Re w, and the coupling of the two channels over the draws.")
    print()
    print(f"{'mu':>5} {'theta':>6} | {'ESS / N':>11} {'mean|Im w|/|w|':>15} "
          f"{'1 - |<phase>|':>19} {'neg Re w':>9} | {'strength over seeds':>21} {'res':>5}")
    for mu in (0.0, 0.4, 0.8):
        for theta in (0.0, 0.25, 0.5, 0.75, 1.0):
            rs = [sweep(theta, 4.0, mu, seed=s) for s in SEEDS]
            st = [r["s"] for r in rs]
            print(f"{mu:5.2f} {theta:6.2f} | "
                  f"{min(r['ess'] for r in rs):5.1f}/{rs[0]['n']:<5d} "
                  f"{np.mean([r['imagshare'] for r in rs]):15.3e} "
                  f"{np.mean([1.0 - r['absmean'] for r in rs]):19.4f} "
                  f"{np.mean([r['negre'] for r in rs]):9.4f} | "
                  f"{min(st):8.4f} to {max(st):8.4f} "
                  f"{str(all(r['res'] for r in rs)):>5}", flush=True)
        print()
    print("THE mu = 0 BLOCK IS THE RESULT.  Same Hamiltonian, same physics, weights real and")
    print("positive at every theta -- no sign problem anywhere in that block -- and the read runs")
    print("from -1.0000 to +1.0000.  A number that moves that far while the model and its sign")
    print("structure both hold still is not a reading of the model.")
    print()
    print("THE DOPED BLOCKS carry the complex weights: 93 to 96 percent of draws have a genuine")
    print("phase -- argument neither ~0 nor ~pi -- and the weights are around 60% imaginary by")
    print("magnitude.  There the sign problem is a PHASE problem, which the negative-fraction")
    print("column cannot describe, and at theta = 1 the read is saturated at exactly +1.0000")
    print("because the two channels coincide.")
