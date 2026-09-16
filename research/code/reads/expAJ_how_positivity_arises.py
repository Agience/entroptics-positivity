"""Experiment AJ -- the family is sign-free at every theta, but not for the same reason.

AI found that at half filling the whole decoupling family is sign-free: weights real and positive
at every theta, charge channel included.  That cannot be one mechanism holding throughout, because
at `theta = 1` the spin coupling `lam_s` is ZERO, so both spins see the identical slice matrix and
there are not two determinant signs to lock.  Whatever makes the weight positive there is not the
lockstep of sections 2 and 3.

So the question is what each end actually does, measured rather than argued:

    w = charge_phase(Y) * det(I + B_up) * det(I + B_dn)

  theta = 0   `lam_c = 0`, so `charge_phase = 1` and both determinants are REAL.  Positivity is a
              SIGN LOCKSTEP: sgn(det_up) == sgn(det_dn) configuration by configuration.
  theta = 1   `lam_s = 0`, so `B_up = B_dn` and `det_up = det_dn = d`, both complex.  The weight is
              `charge_phase * d^2`, and positivity can only be a PHASE CANCELLATION:
              `arg(charge_phase) + 2 arg(d) = 0 mod 2pi`.

Both are measured on the same draws, and each is measured at BOTH ends, so each end is checked
against the mechanism that does not apply there as well as the one that does.  A file that only
measured the applicable mechanism at each end would be describing its own case split.

WHAT THE PHASE RESIDUAL IS, AND IS NOT.  `arg(charge_phase) + arg(det_up) + arg(det_dn)` is
`arg(w)`, so a residual of zero is positivity RESTATED and not an independent invariant.  It is
reported because it says which mechanism delivers positivity at each theta, not because it explains
anything.  Its MAXIMUM over draws is also the wrong summary and was read wrongly once: a max of pi
means at least one draw was negative-real, not that every draw was real, and on the doped rows the
distribution is in fact spread across the circle.  The fraction with a genuine phase is measured in
AI, where it belongs.

This is what makes the section 8 result mechanical rather than surprising: a read that compares the
two channels reports the lockstep, and the lockstep is not what carries positivity at the far end
of the family.
"""
from __future__ import annotations

import numpy as np

from model2d import Model2D
from stable import udt_product, slogdet_one_plus_block


def measure(theta, beta, mu=0.0, U=4.0, dtau=0.125, Lx=2, Ly=4, n_draw=300, seed=0):
    L = int(round(beta / dtau))
    m = Model2D(Lx=Lx, Ly=Ly, t=1.0, mu=mu, U=U, dtau=dtau, L=L, theta=theta)
    rng = np.random.default_rng(seed)
    imag_share, lock, cancel, wneg, wimag = [], [], [], [], []
    for _ in range(n_draw):
        X = rng.standard_normal((L, m.N))
        Y = rng.standard_normal((L, m.N))
        det = {}
        for sigma in (+1, -1):
            Bl = m.B_slices(X[None], Y[None], sigma)[0]
            Uu, D, T = udt_product(Bl[None], 4)
            s, la = slogdet_one_plus_block(Uu, D, T)
            det[sigma] = complex(s[0]) * np.exp(float(la[0]))
        cp = complex(m.charge_phase(Y[None])[0])
        w = cp * det[+1] * det[-1]

        # how complex is each determinant, relative to its own magnitude
        imag_share.append(max(abs(d.imag) / (abs(d) + 1e-300) for d in det.values()))
        # MECHANISM 1: do the two determinants' real parts share a sign
        lock.append(float(np.sign(det[+1].real) == np.sign(det[-1].real)))
        # MECHANISM 2: does the charge c-number cancel the determinants' phase
        resid = np.angle(cp) + np.angle(det[+1]) + np.angle(det[-1])
        cancel.append(float(abs(np.angle(np.exp(1j * resid)))))
        wneg.append(float(w.real < 0))
        wimag.append(abs(w.imag) / (abs(w) + 1e-300))
    return dict(imag=float(np.mean(imag_share)), lock=float(np.mean(lock)),
                cancel=float(np.max(cancel)), neg=float(np.mean(wneg)),
                wimag=float(np.max(wimag)))


if __name__ == "__main__":
    SEEDS = (1, 7, 23)
    print("=" * 112)
    print("SIGN-FREE AT EVERY THETA, BY DIFFERENT MECHANISMS AT THE TWO ENDS.")
    print("2x4, U = 4, mu = 0 (half filling), beta = 4, 300 prior draws x 3 seeds.")
    print()
    print("  `det is complex`  mean |Im det| / |det|, per configuration, worst of the two spins")
    print("  `signs lock`      fraction where sgn(Re det_up) == sgn(Re det_dn)   -- MECHANISM 1")
    print("  `phase residual`  max |arg(charge_phase) + arg(det_up) + arg(det_dn)| -- MECHANISM 2")
    print()
    print(f"{'theta':>6} | {'det is complex':>15} {'signs lock':>11} {'phase residual':>15} "
          f"| {'w has Im':>9} {'neg Re w':>9}")
    for theta in (0.0, 0.25, 0.5, 0.75, 1.0):
        rs = [measure(theta, 4.0, seed=s) for s in SEEDS]
        print(f"{theta:6.2f} | {np.mean([r['imag'] for r in rs]):15.3e} "
              f"{np.mean([r['lock'] for r in rs]):11.4f} "
              f"{max(r['cancel'] for r in rs):15.3e} | "
              f"{max(r['wimag'] for r in rs):9.3e} {np.mean([r['neg'] for r in rs]):9.4f}",
              flush=True)
    print()
    print("THE TWO ENDS DO DIFFERENT THINGS.  At theta = 0 the determinants are real and their")
    print("signs lock; the phase residual is trivially zero because there are no phases.  At")
    print("theta = 1 the determinants are fully complex and the phase residual is what vanishes,")
    print("while `signs lock` stops being the reason for anything -- there, both spins see the")
    print("same slice matrix, so the two determinants are the SAME NUMBER and agreeing costs")
    print("nothing.")
    print()
    print("That is why the read of sections 2 to 5 is representation-dependent.  It compares the")
    print("two channels, which is where positivity lives at one end of this family and not at the")
    print("other.")
