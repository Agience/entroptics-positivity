"""Experiment II -- the ALGEBRAIC axis: what exactly forbids a negative weight, and what breaks it.

Every closure in this rig is of the form "within family X, every member is worse", and each proof
is conditional on its family.  The decoupling proof is airtight GIVEN a Hubbard-Stratonovich
decomposition in a fixed single-particle basis.  But every historical escape from the sign problem
came from changing the family rather than the field: Majorana positivity, fermion bags,
Kramers/split-orthogonal positivity, meron-cluster.  Those work by exhibiting an algebraic
structure in which the weight is manifestly |something|^2 -- and the completeness proof says
nothing whatever about that axis.

This measures the axis instead of citing it.  For the spin-decoupled Hubbard model the weight of a
field configuration is

    w(x) = det(I + B_up(x)) * det(I + B_dn(x)),   B_sigma = prod_l e^{-dt K} diag(e^{sigma lam x_l})

At half filling on a bipartite lattice this is provably non-negative, and away from it, it is not.
The question worth answering is not WHETHER (that is known) but WHICH RELATION carries the
positivity and HOW doping breaks it -- because a relation with a measurable residual is something
a different representation might restore, while "it is model-specific" is not actionable.

Three things are measured together at each chemical potential, over the same sampled fields:

  neg      the fraction of configurations with w(x) < 0 -- the sign problem itself
  ph_res   the residual of the particle-hole relation that forces w >= 0.  On a bipartite
           lattice with sublattice operator P (diagonal, +-1), particle-hole conjugation maps
           B_up(x) to something whose determinant pairs with B_dn(x)'s.  The residual is how far
           that pairing misses, relative to the determinants' own scale.
  gap      the smallest |det| over the sample, which says whether w is near a zero crossing at all

A residual that is zero exactly where `neg` is zero, and grows with mu as `neg` turns on, IS the
mechanism -- it names the obstruction rather than restating the symptom.
"""
from __future__ import annotations

import numpy as np

from model2d import Model2D
from stable import udt_product, slogdet_one_plus_block


def weights(m, X):
    """(sign, log|w|, sign_up, log|det_up|, sign_dn, log|det_dn|) for one field X of shape (L, N).

    STABILISED, and that is not optional.  A naive product B(L-1)...B(0) has condition number
    ~exp(beta * bandwidth), so at beta = 6 the determinant is numerical noise: a first version of
    this file used the plain product and reported a negative-weight fraction of 0.0100 at HALF
    FILLING on a bipartite lattice, where the weight is provably non-negative, with the up/down
    ratio's standard deviation reading 1e14 and then nan.  That is exactly the failure this whole
    investigation opened by finding -- the naive product fakes a severe sign problem -- and
    `stable.py` exists to fix it.  Reintroducing it in a new file is the same defect wearing a
    different filename.
    """
    lam = float(np.arccosh(np.exp(m.dtau * m.U / 2.0)))
    out = []
    for sigma in (+1, -1):
        d = np.exp(sigma * lam * X)                       # (L, N)
        Bl = m.expmK[None, :, :] * d[:, None, :]          # (L, N, N)
        U, D, T = udt_product(Bl[None], 4)
        s, la = slogdet_one_plus_block(U, D, T)
        out.append((float(np.real(s[0])), float(la[0])))
    (su, lu), (sd, ld) = out
    return su * sd, lu + ld, su, lu, sd, ld


def sublattice(Lx, Ly):
    return np.array([(-1.0) ** (x + y) for x in range(Lx) for y in range(Ly)])


def ph_residual(m, X):
    """How far the up/down pairing misses, in LOG magnitude.

    The first version asserted the pairing was `det_up == det_dn` and measured
    |du - dd| / (|du| + |dd|), which read 0.98 at half filling where positivity is provable --
    so the assumed relation was simply wrong.  The relation is not assumed here: what is reported
    is log|det_dn| - log|det_up|, whose SPREAD across fields says whether the two are locked
    together (spread 0, some fixed relation) or independent (spread large).  A relation nobody
    has guessed still shows up as a small spread.
    """
    _, _, su, lu, sd, ld = weights(m, X)
    return ld - lu, su, sd


if __name__ == "__main__":
    Lx, Ly = 2, 4
    U, dtau = 4.0, 0.125
    rng = np.random.default_rng(0)
    n_draw = 300
    print("=" * 100)
    print(f"WHAT CARRIES THE POSITIVITY, AND WHAT BREAKS IT   {Lx}x{Ly}, U = {U}, "
          f"{n_draw} fields, STABILISED determinants")
    print("mu = 0 is half filling on a bipartite lattice, where the weight is provably >= 0.")
    print("A regime with no sign problem measures nothing, so beta is swept until one appears.")
    print()
    print(f"{'beta':>5} {'mu':>5} {'neg frac':>9} {'<sgn>':>8} {'med dlog':>10} "
          f"{'sd dlog':>9} {'sign_up==sign_dn':>17}")
    for beta in (2.0, 4.0, 6.0, 8.0):
        L = int(round(beta / dtau))
        for mu in (0.0, 0.4):
            m = Model2D(Lx=Lx, Ly=Ly, t=1.0, mu=mu, U=U, dtau=dtau, L=L, theta=0.0)
            neg, dl, same = 0, [], 0
            for _ in range(n_draw):
                X = rng.choice([-1.0, 1.0], size=(L, m.N))
                sw, _, su, lu, sd, ld = weights(m, X)
                neg += int(sw < 0)
                dl.append(ld - lu)
                same += int(su == sd)
            dl = np.array(dl)
            print(f"{beta:5.1f} {mu:5.2f} {neg/n_draw:9.4f} {1-2*neg/n_draw:8.4f} "
                  f"{np.median(dl):10.4f} {dl.std():9.4f} {same/n_draw:17.4f}", flush=True)
    print()
    print("At half filling the negative fraction must be 0 at EVERY beta -- if it is not, the")
    print("determinants are not being computed stably and nothing else on the row means anything.")
    print("'sd dlog' is what says whether the two determinants are locked to each other; a small")
    print("spread is a relation, whatever its form, and a relation is what a different")
    print("representation could act on.")
