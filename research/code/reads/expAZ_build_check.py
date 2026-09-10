"""Experiment AZ -- does the code run the model you think it does?  Read from output alone.

THE RESULT.  A determinantal QMC run whose lattice is built with an off-by-one wrap -- periodic
boundaries where open were intended, on an ODD chain -- is running a non-bipartite model.  At
`beta = 1` to `3` it produces NOT ONE NEGATIVE WEIGHT across three seeds, so every standard health
check passes, and the criterion of section 4 computed on the INTENDED lattice reports `sign-free`
because it reads the model rather than the code.  The output-side read fires anyway:

    beta    negative fraction (4 seeds)    1 - |strength|      correctly-built control
       1                       0.00000           1.904e-07                    1.1e-16
       2                       0.00000           6.319e-05                   -2.2e-16
       3                       0.00000           3.336e-04                    1.1e-16

Eleven orders between the bug and the control at `beta = 2`, and the signal is 5 to 30 times the
seed spread (`6.32e-05 +- 5.8e-06` at `beta = 2`, `3.34e-04 +- 6.1e-05` at `beta = 3`, four seeds).

WHY THIS READ AND NOT THE DIRECT CHECK.  Section 3's identity can be tested directly --

    ln|det_up|(x) - ln|det_dn|(x)  ==  -dtau*L*tr(K) + lambda*sum(x)

-- and that test needs `lambda` and `tr(K)`.  A build check cannot have them.  The premise of the
exercise is that the code may not be running the model the author believes, so the `lambda` and the
`K` one would supply are the suspect quantities themselves; feeding the intended model to a test of
whether the intended model is running is circular.  `coupling` reads the same identity as a
statement about DATA -- one logged column against another on a shared index -- and it is invariant
to the offset and to the scale, so it needs neither constant.  That invariance is the whole reason
this works, and it is also the reason for the second scope limit below.

WHAT IS INJECTED.  Three bugs a working physicist writes, none of which crashes:

    B1  periodic wrap where open was intended, on an odd chain -- an odd cycle, so not bipartite
    B2  the wrong decoupling constant -- the continuous-Gaussian `sqrt(dtau U)` in the discrete
        Hirsch field, a copy from the neighbouring representation
    B3  the auxiliary field applied with the same sign to both channels -- the dropped `sigma`

THE SCOPE, MEASURED, AND BOTH LIMITS ARE PROVABLE RATHER THAN EMPIRICAL.

    B1  detected.  The relation stops being affine, which is what the read tests.
    B2  NOT detected -- `1 - |strength|` is `-2.2e-16`, exactly saturated.  A wrong `lambda`
        rescales the relation and leaves it affine.  The read is scale-invariant, which is
        precisely why it survives not knowing `lambda` and precisely why it cannot see a wrong
        one.  The two facts are the same fact.  (The direct residual DOES catch this, at 1.61 --
        so the two checks are complementary, not ranked.)
    B3  NOT detected as a departure -- the read returns UNRESOLVED.  Applying one sign to both
        channels makes the two determinants identical, so the logged difference is identically
        zero and there is no column to correlate.  A degenerate input is visible upstream of the
        read and should be caught there.

HONEST ABOUT THE INSTRUMENT.  At `D = 1` -- two logged scalars, which is this experiment's frame --
`coupling.strength` IS Pearson's r and `z` is `r*sqrt(T-1)`, so `|z|` reads 19.97 on the bug AND on
the control and does no discriminating here.  What discriminates is saturation against the seed
spread.  The contribution of this experiment is the METHOD -- that an offset-and-scale-free read of
two already-computed columns detects a class of model-implementation mismatch while every weight is
still positive -- not a claim that only this library can compute it.  The reads where the
instrument is not replaceable are the weight-cloud ones in `expAU` and `expAV`, which are
multi-dimensional and constant-free.

TO USE IT.  Log `ln|det_up| - ln|det_dn|` from the `slogdet` any DQMC code already forms each
sweep, and `sum(x)` over the auxiliary field.  Call `coupling(A, B)` and check saturation against
a two-seed spread.  `O(n)` over two vectors, no model knowledge required.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import expm

import entroptics as E
from stable import udt_product, slogdet_one_plus_block
from reads.expAO_spectral_criterion import chain, criterion

U, DTAU, NDRAW = 4.0, 0.125, 400
SEEDS = (5, 11, 23, 37)


def run(K_built, *, lam_built=None, drop_sigma=False, beta=6.0, U=U, dtau=DTAU,
        n=NDRAW, seed=5):
    """One pass over configurations, returning what an observer actually holds.

    `A` is `ln|det_up| - ln|det_dn|` per configuration, `B` is `sum(x)`, `sg` the weight's sign.
    `lam_built` and `drop_sigma` are the injection points: what the CODE does, which is not
    necessarily what the author meant.
    """
    K = np.asarray(K_built, dtype=complex)
    L = int(round(beta / dtau))
    eK = expm(-dtau * K)
    lam = float(np.arccosh(np.exp(dtau * U / 2.0))) if lam_built is None else float(lam_built)
    rng = np.random.default_rng(seed)
    A, B, sg_all = [], [], []
    for _ in range(n):
        X = rng.choice([-1.0, 1.0], size=(L, K.shape[0]))
        lg, sign = {}, 1.0
        for sigma in (+1, -1):
            eff = (+1) if drop_sigma else sigma
            d = np.exp(eff * lam * X)
            Bl = eK[None, :, :] * d[:, None, :]
            Uu, D, T = udt_product(Bl[None], 4)
            s, la = slogdet_one_plus_block(Uu, D, T)
            lg[sigma] = float(la[0])
            sign *= float(np.real(s[0]))
        A.append(lg[+1] - lg[-1])
        B.append(float(X.sum()))
        sg_all.append(sign)
    return np.array(A)[:, None], np.array(B)[:, None], np.array(sg_all)


def departure(A, B):
    """`1 - |strength|`: zero when the relation is exactly affine, positive when it is not."""
    c = E.reads.coupling(A, B)
    return (1.0 - abs(c.strength)) if c.resolved else float("nan"), abs(c.z), c.resolved


def identity_residual(A, B, K_intended, lam, L, dtau):
    """The direct check -- fed the model, which is the thing a build check cannot assume."""
    off = -dtau * L * float(np.real(np.trace(np.asarray(K_intended))))
    return float(np.max(np.abs(A.ravel() - (off + lam * B.ravel()))))


def main():
    N = 7
    K_ok = chain(N, periodic=False)            # intended: open chain, bipartite, sign-free
    K_bug = chain(N, periodic=True)            # as built: odd ring, an odd cycle
    lam = float(np.arccosh(np.exp(DTAU * U / 2.0)))
    lam_gauss = float(np.sqrt(DTAU * U))

    print("=" * 100)
    print("THE ORACLE, COMPUTED ON THE MODEL THE AUTHOR INTENDED")
    print("=" * 100)
    print(f"  criterion(intended open {N}-chain) -> sign-free = {criterion(K_ok)}")
    print(f"  criterion(as-built {N}-ring)       -> sign-free = {criterion(K_bug)}"
          "   <- needs an audit of the code, which is the thing in doubt")
    print(f"  lambda discrete = {lam:.6f}    lambda Gaussian = {lam_gauss:.6f}")

    print()
    print("=" * 100)
    print("B1 -- THE WINDOW WHERE THE SIGN IS SILENT AND THE READ IS NOT")
    print("=" * 100)
    hdr = (f"{'beta':>6}{'neg fraction':>15}{'1-|strength|':>16}{'seed spread':>14}"
           f"{'control':>13}   window")
    print(hdr)
    print("-" * len(hdr))
    for beta in (1.0, 2.0, 3.0, 4.0, 6.0):
        deps, negs = [], []
        for seed in SEEDS:
            A, B, sg = run(K_built=K_bug, beta=beta, seed=seed)
            d, _, _ = departure(A, B)
            deps.append(d)
            negs.append(float((sg < 0).mean()))
        Ac, Bc, sgc = run(K_built=K_ok, beta=beta, seed=SEEDS[0])
        ctrl, _, _ = departure(Ac, Bc)
        silent = all(n == 0.0 for n in negs)
        fires = float(np.mean(deps)) > 1e-8
        window = ("*** SIGN SILENT, READ FIRES ***" if (silent and fires)
                  else ("both silent" if silent else "sign already fired"))
        print(f"{beta:>6.1f}{float(np.mean(negs)):>15.5f}{float(np.mean(deps)):>16.3e}"
              f"{float(np.std(deps)):>14.1e}{ctrl:>13.1e}   {window}")

    print()
    print("  The control is the SAME lattice built correctly, at the same beta and seed: it must")
    print("  stay at machine zero, or the read is firing on beta rather than on the bug.")

    print()
    print("=" * 100)
    print("THE SCOPE -- WHICH BUGS THIS READ SEES, AT beta = 3 WHERE THE SIGN IS STILL SILENT")
    print("=" * 100)
    L = int(round(3.0 / DTAU))
    cases = [
        ("control -- code matches intent", dict(K_built=K_ok)),
        ("B1  periodic wrap, odd ring",    dict(K_built=K_bug)),
        ("B2  wrong lambda (Gaussian)",    dict(K_built=K_ok, lam_built=lam_gauss)),
        ("B3  sigma dropped on the field", dict(K_built=K_ok, drop_sigma=True)),
    ]
    hdr = (f"{'case':<34}{'neg frac':>10}{'direct resid':>15}{'1-|strength|':>15}"
           f"{'|z|':>9}   read")
    print(hdr)
    print("-" * len(hdr))
    for tag, kw in cases:
        A, B, sg = run(beta=3.0, seed=5, **kw)
        d, z, resolved = departure(A, B)
        resid = identity_residual(A, B, K_ok, lam, L, DTAU)
        verdict = ("FIRES" if (resolved and d > 1e-8)
                   else ("saturated" if resolved else "UNRESOLVED"))
        ds = "nan" if np.isnan(d) else f"{d:.3e}"
        print(f"{tag:<34}{float((sg < 0).mean()):>10.4f}{resid:>15.3e}{ds:>15}"
              f"{z:>9.2f}   {verdict}")

    print()
    print("  B2 is saturated because a wrong lambda rescales an affine relation and leaves it")
    print("  affine.  Scale-invariance is why the read needs no lambda and why it cannot see a")
    print("  wrong one -- the same fact twice.  The direct residual catches B2 and needs the")
    print("  model; the two checks are complementary and neither dominates.")
    print()
    print("  B3 is UNRESOLVED, not a departure: one sign on both channels makes the two")
    print("  determinants identical, so the logged difference is identically zero.  That is a")
    print("  degenerate input, visible before any read is taken.")
    print()
    print("  |z| is 19.97 on every row: at D = 1 this read is Pearson's r and z is r*sqrt(T-1),")
    print("  which saturates.  Saturation against the seed spread is what discriminates here.")


if __name__ == "__main__":
    main()
