"""Experiment AH -- is the boundary of section 7 intrinsic, or an artifact of reading one scalar?

Experiment AF found that on the Kramers mechanism the coupling read does not saturate as a single
number while the real and imaginary parts saturate separately at +1 and -1.  That leaves two
questions, and they turn out to have opposite answers.

  1. COULD THE INSTRUMENT FIND THE SPLIT ITSELF?  The +1 / -1 reading was obtained by splitting the
     frame into real and imaginary parts -- which is using knowledge of the mechanism.  Asked for
     directions with no such hint, via `principal_directions` on one channel, the read gives the
     directions of THAT CHANNEL'S OWN VARIANCE, which are not the eigendirections of the relation
     between the channels.  The per-direction couplings do not come out at +-1.

  2. THE OTHER DIRECTION IS AN IMPOSSIBILITY.  In the control the two channels are bit-identical
     (`numpy.array_equal` is True, not "agree to tolerance").  A pair with `B = A` carries no
     between-channel content at all: every statistic of the pair is a statistic of one channel.
     Three systems are exhibited with `B = A`, the same saturated read, and negative fractions of
     0.0000, 0.0300 and 0.3333.  No comparison BETWEEN channels can separate them.

     This is a limit on any between-channel read, not on this one.  The control's sign problem sits
     in the PHASE of `det(I + B_up)` -- a global property of the determinant -- while the frame
     handed to the read is the Green's function's diagonal.  Whether some read of a single channel
     could recover that phase is NOT measured here and is not claimed either way.

A note on the complex frame.  `reads.coupling` reduces a complex frame through the real embedding
`iota(x) = (Re x, Im x)`, which is exactly the concatenation AF used, so the two agree; this file
checks that rather than assuming it.  The read's `phase` field is also reported, because a signed
real part could in principle be averaging away a magnitude, and it is not: `phase` is ~0.009.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import expm

import entroptics_adapter as EA
from stable import udt_product, inv_one_plus_block, slogdet_one_plus_block
from reads.expAF_third_mechanism import hop_flux


def channels(Lx, Ly, beta, U, phi, time_reversal, dtau=0.125, n_draw=300, seed=0):
    """The two channels' Green's-function diagonals (complex) and the configuration weights."""
    L = int(round(beta / dtau))
    N = Lx * Ly
    Kup = hop_flux(Lx, Ly, phi=phi)
    Kdn = np.conj(Kup) if time_reversal else Kup.copy()
    eup, edn = expm(-dtau * Kup), expm(-dtau * Kdn)
    lam = float(np.arccosh(np.exp(dtau * abs(U) / 2.0)))
    rng = np.random.default_rng(seed)
    A, B, W = [], [], []
    for _ in range(n_draw):
        X = rng.choice([-1.0, 1.0], size=(L, N))
        d = np.exp(lam * X)
        g, ld, sg = {}, {}, {}
        for tag, ek in (("up", eup), ("dn", edn)):
            Bl = ek[None, :, :] * d[:, None, :]
            Uu, D, T = udt_product(Bl[None], 4)
            g[tag] = np.diag(inv_one_plus_block(Uu, D, T)[0]).copy()
            s, la = slogdet_one_plus_block(Uu, D, T)
            sg[tag], ld[tag] = complex(s[0]), float(la[0])
        W.append(sg["up"] * sg["dn"] * np.exp(ld["up"] + ld["dn"]))
        A.append(g["up"]); B.append(g["dn"])
    return np.array(A), np.array(B), np.array(W)


def embed(M):
    """The real embedding the library itself uses for a complex frame."""
    return np.concatenate([M.real, M.imag], axis=1)


def per_direction(Ae, Be):
    """Signed coupling along each direction the instrument resolves, from channel A alone."""
    V = EA.cloud_axes(Ae)
    out = []
    for j in range(V.shape[1]):
        v = np.real(V[:, j])
        c = EA.channel_alignment((Ae @ v)[:, None], (Be @ v)[:, None])
        out.append(float(c.strength) if c.resolved else None)
    return out


if __name__ == "__main__":
    print("=" * 104)
    print("1. DOES THE INSTRUMENT FIND THE SUBSPACES ITSELF?  4x3, |U| = 4, beta = 8, 300 draws.")
    print("   Directions come from principal_directions on channel A -- chosen by the read.")
    print()
    print(f"{'case':>22} {'neg':>7} {'whole':>8} {'phase':>8} {'k':>3} | per-direction coupling")
    for name, (phi, tr) in (("Kramers pi/16", (np.pi / 16, True)),
                            ("Kramers 3pi/16", (3 * np.pi / 16, True)),
                            ("control 3pi/16", (3 * np.pi / 16, False))):
        A, B, W = channels(4, 3, 8.0, -4.0, phi, tr, seed=7)
        Ae, Be = embed(A), embed(B)
        direct = EA.channel_alignment(A, B)          # complex frame, handed in as-is
        whole = EA.channel_alignment(Ae, Be)         # the same thing, embedded by hand
        assert abs(direct.strength - whole.strength) < 1e-9, \
            "the complex frame and its real embedding disagreed"
        per = " ".join("  --  " if v is None else f"{v:+.3f}" for v in per_direction(Ae, Be))
        print(f"{name:>22} {float(np.mean(np.real(W) < 0)):7.4f} {whole.strength:8.4f} "
              f"{direct.phase:8.4f} {len(per.split()):3d} | {per}")
    print()
    print("   Not +-1.  Those are the directions of A's OWN variance, not the eigendirections of")
    print("   the A-to-B relation, so the blockwise saturation in AF used knowledge of the")
    print("   mechanism and a mechanism-agnostic application of this read stays unsaturated.")

    print()
    print("=" * 104)
    print("2. THE COLLISION.  When B == A exactly, every read of the pair is a read of A alone.")
    print()
    print(f"{'system':>26} {'B == A':>7} {'neg':>7} | {'strength':>9} {'z':>8} {'res':>5}")
    for name, (phi, tr) in (("attractive, no flux", (0.0, True)),
                            ("control pi/16", (np.pi / 16, False)),
                            ("control 3pi/16", (3 * np.pi / 16, False))):
        A, B, W = channels(4, 3, 8.0, -4.0, phi, tr, seed=7)
        Ae, Be = embed(A), embed(B)
        c = EA.channel_alignment(Ae, Be)
        print(f"{name:>26} {str(np.array_equal(Ae, Be)):>7} "
              f"{float(np.mean(np.real(W) < 0)):7.4f} | {c.strength:9.4f} {c.z:8.1f} "
              f"{str(bool(c.resolved)):>5}")
    print()
    print("   Same saturated value, negative fractions spanning 33 percentage points.  The read's")
    print("   output does not determine the sign behaviour, and with B = A there is no between-")
    print("   channel content left for any read to use.  The control's sign problem is in the")
    print("   PHASE of det(I + B_up); the frame supplied is the Green's function's diagonal.")
