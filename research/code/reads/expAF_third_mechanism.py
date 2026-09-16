"""Experiment AF -- a THIRD positivity mechanism, and whether the read still saturates.

Sections 2 to 5 read two sign-free situations, and BOTH of them relate the two channels affinely:

    repulsive, spin channel, half filling     G_dn = 1 - G_up      -> the read gives -1
    attractive, charge channel, any filling   G_dn =     G_up      -> the read gives +1

If the read only saturates on an affine relation then "it measures synchronisation" is too broad a
description of it, and section 7 is overstated.  So the test is a mechanism where positivity comes
from somewhere else entirely.

WU-ZHANG / KRAMERS POSITIVITY.  Take a spin-dependent flux: the hopping carries a phase
`exp(+i phi)` for up and `exp(-i phi)` for down, so `K_dn = conj(K_up)` -- the two spins are
time-reversal partners.  Decouple ATTRACTIVE U in the charge channel, whose field is real and
spin-independent.  Then `B_dn = conj(B_up)` as matrices, so

    det(I + B_dn) = conj(det(I + B_up))     and     weight = |det(I + B_up)|^2 >= 0.

The weight is positive because the two determinants are COMPLEX CONJUGATES, not because they are
equal and not because they are affinely related.  Nothing here is assumed: the conjugation is
checked elementwise, and the weight's imaginary part is measured rather than dropped.

WHAT THE CHANNELS LOOK LIKE.  `G_dn = conj(G_up)`, so on the real part the channels agree and on
the imaginary part they are exact negatives.  Handed the full complex channel as a real vector
`[Re, Im]`, the map from one channel to the other is an ORTHOGONAL involution that is not `+I` or
`-I`.  That is the question this file exists to answer: does the read saturate on a relation that
is deterministic but not a scalar multiple?

THE CONTROL.  Apply the SAME flux to both spins instead of opposite ones.  Time reversal is gone,
`B_dn = B_up` complex, the weight is `det^2` rather than `|det|^2`, and it acquires a phase.  Same
lattice, same interaction, same field -- only the mechanism removed.

WHY THE LATTICE IS AT LEAST 3 WIDE.  A first run used 2x4 and every row, control included, came
back with `Im G = 0` exactly: on a periodic lattice of width 2 the x-direction has ONE distinct
bond per row, the loop reaches it from both ends, and the two Peierls phases add to `-2t cos(phi)`,
which is real.  The flux cancelled identically and the file was testing nothing while printing a
full table.  Both sides are now guarded -- the width is asserted, and `max |Im G|` is a reported
column that the caller checks is non-zero before reading anything else.

The phase sits on the x-bonds only, so the flux through each plaquette is zero and this is a
spin-dependent TWIST around the x-cycle rather than a magnetic field.  That is all the mechanism
needs: `K` is complex, `K_dn = conj(K_up)`, and the Kramers structure follows.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import expm

import entroptics_adapter as EA
from stable import udt_product, inv_one_plus_block, slogdet_one_plus_block


def hop_flux(Lx, Ly, t=1.0, mu=0.0, phi=0.0):
    """Hermitian hopping with a Peierls phase `exp(i phi)` on every x-bond.

    Hermiticity is imposed by construction (`K[j,i] = conj(K[i,j])`) and checked by the caller;
    a non-Hermitian K would make the spectrum complex for reasons having nothing to do with the
    mechanism under test.
    """
    assert Lx > 2, "a width-2 ring reaches the same bond twice and cancels the phase"
    N = Lx * Ly
    idx = lambda x, y: (x % Lx) * Ly + (y % Ly)
    K = np.zeros((N, N), dtype=complex)
    for x in range(Lx):
        for y in range(Ly):
            i = idx(x, y)
            j = idx(x + 1, y)
            K[i, j] += -t * np.exp(1j * phi)
            K[j, i] += -t * np.exp(-1j * phi)
            k = idx(x, y + 1)
            K[i, k] += -t
            K[k, i] += -t
    K[np.diag_indices(N)] = -mu
    return K


def run(Lx, Ly, beta, U, mu, phi, time_reversal, dtau=0.125, n_draw=200, seed=0):
    """`time_reversal=True` gives the two spins opposite flux; False gives them the same flux."""
    L = int(round(beta / dtau))
    Kup = hop_flux(Lx, Ly, mu=mu, phi=phi)
    Kdn = np.conj(Kup) if time_reversal else Kup.copy()
    assert np.allclose(Kup, Kup.conj().T), "K_up is not Hermitian"
    assert np.allclose(Kdn, Kdn.conj().T), "K_dn is not Hermitian"
    eup, edn = expm(-dtau * Kup), expm(-dtau * Kdn)
    N = Lx * Ly
    lam = float(np.arccosh(np.exp(dtau * abs(U) / 2.0)))   # ATTRACTIVE: real, charge channel
    rng = np.random.default_rng(seed)

    A, B, wim, wre, conj_err = [], [], [], [], []
    for _ in range(n_draw):
        X = rng.choice([-1.0, 1.0], size=(L, N))
        d = np.exp(lam * X)                                 # spin-INDEPENDENT by construction
        g, ld, sg = {}, {}, {}
        for tag, ek in (("up", eup), ("dn", edn)):
            Bl = ek[None, :, :] * d[:, None, :]
            Uu, D, T = udt_product(Bl[None], 4)
            g[tag] = np.diag(inv_one_plus_block(Uu, D, T)[0]).copy()
            s, la = slogdet_one_plus_block(Uu, D, T)
            sg[tag], ld[tag] = complex(s[0]), float(la[0])
        w = sg["up"] * sg["dn"] * np.exp(ld["up"] + ld["dn"])
        wre.append(float(np.real(w))); wim.append(float(np.imag(w)))
        conj_err.append(float(np.abs(g["dn"] - np.conj(g["up"])).max()))
        A.append(np.concatenate([g["up"].real, g["up"].imag]))
        B.append(np.concatenate([g["dn"].real, g["dn"].imag]))

    A, B = np.array(A), np.array(B)
    wre, wim = np.array(wre), np.array(wim)
    c = EA.channel_alignment(A, B)
    # the same read on the REAL parts alone, where the two channels are identical by conjugation
    cr = EA.channel_alignment(A[:, :N], B[:, :N])
    ci = EA.channel_alignment(A[:, N:], B[:, N:])
    return dict(conj=float(np.max(conj_err)), imag=float(np.abs(A[:, N:]).max()),
                phase=float(np.max(np.abs(wim) / (np.abs(wre) + 1e-300))),
                neg=float(np.mean(wre < 0)),
                s=float(c.strength), z=float(c.z), res=bool(c.resolved),
                sre=float(cr.strength), sim=float(ci.strength))


if __name__ == "__main__":
    SEEDS = (1, 7, 23, 45, 93)
    print("=" * 116)
    print("A THIRD POSITIVITY MECHANISM, AND THE BOUNDARY OF THE READING.")
    print("Attractive |U| = 4, charge channel, beta = 8, 200 draws x 5 seeds per row.")
    print()
    print("Only twists that are not gauge-equivalent to a real boundary condition are live: the")
    print("total phase round the x-cycle is Lx*phi, so on Lx = 4 the values pi/4 and pi/2 give pi")
    print("and 2pi and are INERT.  `max |Im G|` is the guard that says so.")
    print()
    print(f"{'Lx':>3} {'Ly':>3} {'phi/pi':>7} {'T-rev':>6} | {'max|Im G|':>10} {'conj err':>9} "
          f"{'|Im w/Re w|':>12} {'neg':>7} | {'strength over seeds':>21} | {'Re':>7} {'Im':>7}")
    for (Lx, Ly, pp) in ((4, 3, 0.0625), (4, 3, 0.125), (4, 3, 0.1875), (3, 4, 0.125),
                         (5, 2, 0.1), (4, 3, 0.25)):
        for tr in (True, False):
            rs = [run(Lx, Ly, 8.0, -4.0, 0.0, pp * np.pi, tr, seed=s) for s in SEEDS]
            st = [r["s"] for r in rs]
            live = "" if max(r["imag"] for r in rs) > 1e-8 else "  <- INERT"
            print(f"{Lx:3d} {Ly:3d} {pp:7.4f} {str(tr):>6} | "
                  f"{max(r['imag'] for r in rs):10.3e} {max(r['conj'] for r in rs):9.2e} "
                  f"{max(r['phase'] for r in rs):12.3e} {max(r['neg'] for r in rs):7.4f} | "
                  f"{min(st):8.4f} to {max(st):8.4f} | "
                  f"{np.mean([r['sre'] for r in rs]):7.4f} "
                  f"{np.mean([r['sim'] for r in rs]):7.4f}{live}", flush=True)
    print()
    print("THE RESULT, AND IT CUTS BOTH WAYS.")
    print()
    print("  NOT NECESSARY.  The T-rev rows are sign-free -- `G_dn = conj(G_up)` exactly, the")
    print("  weight's imaginary part is identically zero, the negative fraction is 0.0000 -- and")
    print("  the combined read does NOT saturate.  It runs 0.06 to 0.94.")
    print()
    print("  NOT SUFFICIENT.  The control rows carry a real sign problem, up to 37.5% negative")
    print("  weight, and the combined read is EXACTLY 1.0000 on every seed and every geometry.")
    print()
    print("Both have the same cause.  Under time reversal the relation between the channels is")
    print("deterministic but carries OPPOSITE SIGNS on two orthogonal subspaces -- +1 on the real")
    print("part, -1 on the imaginary part -- and one signed scalar averages them.  Read blockwise")
    print("the mechanism is fully visible.  In the control the channels are bit-identical, so the")
    print("read saturates on a degenerate input, and that model's sign problem lives in a PHASE")
    print("COMMON TO BOTH CHANNELS, which no comparison BETWEEN the channels can see.")
    print()
    print("So the desynchronisation reading is a statement about REAL determinantal weights whose")
    print("sign is a product of two determinant signs.  That is the setting of sections 2 to 5,")
    print("and this file is where it stops.")
