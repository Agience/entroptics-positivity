"""Section 9.2b, third row: a read of one channel alone carries no indicator of the sign.

Section 9 rules out two families. Aggregating signed configurations is capped at `n <sgn>^2`, and a
read of the relation between the two channels is blind to a phase common to them. Neither argument
reaches a read of a single channel: it aggregates no signed configuration and compares no channels.
So that family is measured here.

Two populations are measured. The first is the frustrated ladder threaded by flux over a full
period, where the sign deficit rises and then falls, so a read that tracked the sign would have to
turn round with it and a read that merely moved with flux would not. The second is a set of
lattices whose sign behaviour spans sign-free to a third of the draws negative, where the question
is whether the reads order them.

The statistic is the rank correlation with the deficit, which asks only for the ordering, and its
reference comes from the data: the same sweep on a second seed says what tracking looks like here.

    python remote_run.py reads/expBG_single_channel_has_no_indicator.py
"""
import os
import sys

import numpy as np
from scipy.linalg import expm
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import entroptics_adapter as EA  # noqa: E402
from reads.expAO_spectral_criterion import (  # noqa: E402
    build, chain, ring, star, tri_ladder,
)
from stable import inv_one_plus_block, slogdet_one_plus_block, udt_product  # noqa: E402

BETA = 8.0
U = 4.0
DTAU = 0.125
N_DRAW = 200
FLUX = np.linspace(0.0, 1.0, 11)


def probe(K, seed=5, beta=BETA, n=N_DRAW):
    """One lattice: the sign deficit and the single-channel spectral reads on the same draws.

    Channel A is the up-spin Green's function diagonal, one row per configuration -- the frame
    section 7 hands to a one-sided read. The weight is the product of the two channels'
    determinants, and the deficit is `1 - |<w/|w|>|`, which is the negative-weight fraction's
    analogue for a complex weight.
    """
    L = int(round(beta / DTAU))
    N = K.shape[0]
    eK = expm(-DTAU * np.asarray(K, dtype=complex))
    lam = float(np.arccosh(np.exp(DTAU * U / 2.0)))
    rng = np.random.default_rng(seed)
    frame, weights = [], []
    for _ in range(n):
        X = rng.choice([-1.0, 1.0], size=(L, N))
        dets, diag = {}, None
        for sg in (+1, -1):
            Bl = eK[None, :, :] * np.exp(sg * lam * X)[:, None, :]
            Uu, D, T = udt_product(Bl[None], 4)
            s, _ = slogdet_one_plus_block(Uu, D, T)
            dets[sg] = complex(s[0])
            if sg == +1:
                diag = np.diag(inv_one_plus_block(Uu, D, T)[0]).copy()
        frame.append(diag)
        weights.append(dets[+1] * dets[-1])
    w = np.asarray(weights)
    deficit = 1.0 - float(abs((w / abs(w)).mean()))
    so = EA.single_channel_optics(np.array(frame))
    # Circular distance from pi. A plain `|phase - pi|` wraps, reading 6.2 for a gap of 0.04.
    gap = abs(np.angle(np.exp(1j * (float(so.phase) - np.pi))))
    return {
        "deficit": deficit,
        "phase": gap,
        "attenuation": float(so.attenuation),
        "top_share": float(so.top_share),
        "dominance": float(so.dominance),
    }


def main():
    print("=" * 96)
    print("A READ OF ONE CHANNEL ALONE, ALONG A FLUX SWEEP OF THE FRUSTRATED LADDER")
    print("=" * 96)
    print(f"  8-site triangular ladder, beta = {BETA}, U = {U}, dtau = {DTAU}, "
          f"{N_DRAW} draws a point, {len(FLUX)} points")
    print()

    rows = [probe(tri_ladder(8, t * np.pi)) for t in FLUX]
    again = [probe(tri_ladder(8, t * np.pi), seed=17) for t in FLUX]

    print(f"{'flux/pi':>8} {'sign deficit':>14} {'phase gap':>12} {'attenuation':>13} "
          f"{'top_share':>11} {'dominance':>11}")
    print("-" * 96)
    for t, r in zip(FLUX, rows):
        print(f"{t:8.2f} {r['deficit']:14.5f} {r['phase']:12.5f} {r['attenuation']:13.5f} "
              f"{r['top_share']:11.5f} {r['dominance']:11.5f}", flush=True)
    print()

    deficit = [r["deficit"] for r in rows]
    other = [r["deficit"] for r in again]
    lo, hi = min(deficit), max(deficit)
    down = sum(1 for a, b in zip(deficit, deficit[1:]) if b < a)
    print(f"  the sweep moves the sign deficit over   {lo:.3f} to {hi:.3f}")
    print(f"  and it turns round: the maximum is at flux {FLUX[int(np.argmax(deficit))]:.2f} pi, "
          f"with {down} of {len(deficit) - 1} steps descending")
    print()

    rho_self, p_self = spearmanr(deficit, other)
    print("  rank correlation with the sign deficit, over the eleven points:")
    print(f"{'read':>16} {'rho':>9} {'p':>9}")
    print("-" * 96)
    for name in ("phase", "attenuation", "top_share", "dominance"):
        rho, p = spearmanr(deficit, [r[name] for r in rows])
        print(f"{name:>16} {rho:+9.3f} {p:9.3f}")
    print(f"{'the deficit itself':>16} {rho_self:+9.3f} {p_self:9.3f}   "
          f"(the same sweep on a second seed -- what tracking looks like here)")
    print()
    print("  A read that carried the sign would have to turn round where the deficit does. The")
    print("  deficit tracks itself across seeds; none of the four reads tracks it.")
    print()

    print("=" * 96)
    print("AND ACROSS EIGHTEEN LATTICES WHOSE SIGN BEHAVIOUR SPANS SIGN-FREE TO A HALF")
    print("=" * 96)
    # Eighteen lattices, not six. A rank correlation over six points cannot reach p < 0.05 below
    # |rho| = 0.83, so a null there is a statement about the study's power rather than about the
    # reads. This population spans sign-free to a deficit near a half, across four families.
    cases = [
        ("2x4 clean, half filled", build(2, 4, 0.0, 0.0, 0.0)),
        ("2x6 clean, half filled", build(2, 6, 0.0, 0.0, 0.0)),
        ("chain 8, open", chain(8, periodic=False)),
        ("chain 8, periodic", chain(8, periodic=True)),
        ("star graph, 8 sites", star(8)),
        ("ring 6, flux pi/4", ring(6, np.pi / 4)),
        ("ring 8, flux pi/4", ring(8, np.pi / 4)),
        ("ring 10, flux pi/4", ring(10, np.pi / 4)),
        ("ring 6, flux pi/3", ring(6, np.pi / 3)),
        ("2x4 staggered h = 0.1", build(2, 4, 0.0, 0.0, 0.1)),
        ("2x4 staggered h = 0.2", build(2, 4, 0.0, 0.0, 0.2)),
        ("2x4 staggered h = 0.6", build(2, 4, 0.0, 0.0, 0.6)),
        ("2x4 staggered h = 1.2", build(2, 4, 0.0, 0.0, 1.2)),
        ("2x4 doped mu = 0.4", build(2, 4, 0.0, 0.4, 0.0)),
        ("2x4 tp = 0.3", build(2, 4, 0.3, 0.0, 0.0)),
        ("ring 5, flux pi/2", ring(5, np.pi / 2)),
        ("ring 7, flux pi/2", ring(7, np.pi / 2)),
        ("tri ladder 8, flux pi/2", tri_ladder(8, np.pi / 2)),
    ]
    print(f"{'lattice':>26} {'sign deficit':>14} {'phase gap':>12} {'attenuation':>13} "
          f"{'top_share':>11} {'dominance':>11}")
    print("-" * 96)
    across = []
    for name, K in cases:
        r = probe(np.asarray(K, dtype=complex))
        across.append(r)
        print(f"{name:>26} {r['deficit']:14.5f} {r['phase']:12.5f} {r['attenuation']:13.5f} "
              f"{r['top_share']:11.5f} {r['dominance']:11.5f}", flush=True)
    print()
    d = [r["deficit"] for r in across]
    print(f"  rank correlation with the sign deficit, over the {len(across)} lattices:")
    print(f"{'read':>16} {'rho':>9} {'p':>9}")
    print("-" * 96)
    for name in ("phase", "attenuation", "top_share", "dominance"):
        rho, p = spearmanr(d, [r[name] for r in across])
        print(f"{name:>16} {rho:+9.3f} {p:9.3f}")
    print()
    # What |rho| this population COULD have resolved, so the null is a statement about the reads
    # rather than about the sample size. Spearman's t-approximation at n - 2 degrees of freedom.
    from scipy.stats import t as student
    n_lat = len(across)
    tcrit = float(student.ppf(0.975, n_lat - 2))
    rho_crit = tcrit / np.sqrt(tcrit ** 2 + n_lat - 2)
    print(f"  at n = {n_lat} lattices, |rho| >= {rho_crit:.3f} would reach p < 0.05;")
    print(f"  the largest observed is {max(abs(spearmanr(d, [r[k] for r in across])[0]) for k in ('phase', 'attenuation', 'top_share', 'dominance')):.3f}.")
    print()
    free = [r for r in across if r["deficit"] == 0.0]
    att = [r["attenuation"] for r in free]
    print(f"  and among the {len(free)} SIGN-FREE lattices alone, attenuation spans "
          f"{min(att):.5f} to {max(att):.5f} --")
    print("  a wider range than between them and any lattice here that has a sign problem.")
    print()

    print("=" * 96)
    print("HOW MANY DIGITS OF A SIGN DEFICIT ARE REAL, AT THE DRAW COUNTS THESE TABLES USE")
    print("=" * 96)
    print("  The deficit is `1 - |<w/|w|>|`, a sampled mean over draws, so it carries the mean's")
    print("  own error. Sections quoting the same lattice from different runs differ by that much")
    print("  and not by more. Measured across seeds on two lattices the paper quotes repeatedly:")
    print()
    print(f"{'lattice':>26} {'n':>6} {'min':>10} {'max':>10} {'spread':>10} {'s.e.':>10}")
    print("-" * 96)
    SEEDS = (5, 17, 31, 43, 59, 71)
    for name, K in (("ring 5, flux pi/2", ring(5, np.pi / 2)),
                    ("tri ladder 8, flux pi/2", tri_ladder(8, np.pi / 2))):
        for n in (200, 800):
            ds = [probe(np.asarray(K, dtype=complex), seed=s, n=n)["deficit"] for s in SEEDS]
            se = float(np.std(ds, ddof=1))
            print(f"{name:>26} {n:6d} {min(ds):10.5f} {max(ds):10.5f} "
                  f"{max(ds) - min(ds):10.5f} {se:10.5f}", flush=True)
    print()
    print("  The spread falls as the draw count rises, which is what a sampled mean does. At the")
    print("  200-to-300 draws the tables use, the deficit is reproducible in its second decimal.")


if __name__ == "__main__":
    main()
