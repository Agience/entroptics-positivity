"""Experiment W -- is the constrained-path bias a tax or a dial?

The bias is set by the node of Psi_T and by nothing else.  If it moves with the quality of Psi_T,
then CPMC is not a fixed-accuracy method, it is a method whose accuracy you buy with trial-
wavefunction work -- and that is a different research programme from "the sign problem is
exponential, stop".

Swept: the free (U = 0) determinant against self-consistent UHF started from a staggered
antiferromagnetic field, at three interaction strengths, on 2x4 at (3,3) where the exact ground
state is available in a 3136-dimensional sector.

Reported for each Psi_T: the variational energy <Psi_T|H|Psi_T>, the exact overlap
|<Psi_T|Psi_0>|, and the CPMC energy.  A bias that tracks the overlap is the constraint doing
what the derivation says it does.  A bias that does not is a defect, and would be the finding.

Gates, both of which must hold before any row is read:
  * at U = 0 the free trial must have overlap exactly 1 and variational energy exactly E_0;
  * the overlap routine must reproduce a determinant's own energy through the sector basis.
"""
from __future__ import annotations

import numpy as np

from model2d import Model2D
from sector_ed import ground_energy
from cpmc_fast import run
from trial import free_trial, uhf_trial, stagger_2d, trial_overlap, trial_energy


def cpmc_energy(m, nu, nd, psi_t, beta=8.0, n_walkers=400, seeds=(1, 2, 3)):
    """Seed-to-seed spread, not the within-run block error -- see expV_cpbias.extrapolate."""
    r = [run(m, nu, nd, beta, n_walkers=n_walkers, seed=s, psi_t=psi_t) for s in seeds]
    e = np.array([q["e"] for q in r])
    return (float(e.mean()), float(e.std(ddof=1) / np.sqrt(len(seeds))),
            int(sum(q["killed"] for q in r)))


if __name__ == "__main__":
    Lx, Ly, nu, nd = 2, 4, 3, 3
    stag = stagger_2d(Lx, Ly)

    print("=" * 92)
    print("GATE 1  U = 0 : the free determinant IS the ground state")
    m0 = Model2D(Lx=Lx, Ly=Ly, t=1.0, mu=0.0, U=0.0, dtau=0.05, L=1, theta=0.0)
    ex0 = ground_energy(m0.K, 0.0, nu, nd)
    pf0 = free_trial(m0.K, nu, nd)
    ov0 = trial_overlap(pf0, m0.K, 0.0, nu, nd)
    ev0 = trial_energy(pf0, m0.K, 0.0, nu, nd)
    print(f"  overlap {ov0:.12f}   (1 - overlap) {1-ov0:+.2e}")
    print(f"  <Psi_T|H|Psi_T> {ev0:+.12f}   exact {ex0:+.12f}   diff {ev0-ex0:+.2e}")

    print()
    print("GATE 2  U = 4 : the overlap routine must reproduce the determinant's OWN energy")
    m4 = Model2D(Lx=Lx, Ly=Ly, t=1.0, mu=0.0, U=4.0, dtau=0.05, L=1, theta=0.0)
    pf4 = free_trial(m4.K, nu, nd)
    from trial import _det_amplitudes
    au, _ = _det_amplitudes(pf4[+1], m4.N, nu)
    ad, _ = _det_amplitudes(pf4[-1], m4.N, nd)
    v = np.kron(au, ad); v /= np.linalg.norm(v)
    # rebuild the sector Hamiltonian and take <v|H|v>
    from sector_ed import _hop_matrix, _states
    from scipy.sparse import kron as skron, identity, csr_matrix
    Hu, su = _hop_matrix(m4.N, nu, m4.K); Hd, sd = _hop_matrix(m4.N, nd, m4.K)
    du, dd = len(su), len(sd)
    H = skron(Hu, identity(dd), format="csr") + skron(identity(du), Hd, format="csr")
    nuo = np.array([[(s >> i) & 1 for i in range(m4.N)] for s in su], float) - 0.5
    ndo = np.array([[(s >> i) & 1 for i in range(m4.N)] for s in sd], float) - 0.5
    dg = np.concatenate([4.0 * (ndo @ nuo[a]) for a in range(du)])
    H = H + csr_matrix((dg, (np.arange(du*dd), np.arange(du*dd))), shape=H.shape)
    e_sector = float(v @ (H @ v))
    e_wick = trial_energy(pf4, m4.K, 4.0, nu, nd)
    print(f"  through the sector basis {e_sector:+.12f}")
    print(f"  by Wick on the determinant {e_wick:+.12f}   diff {e_sector-e_wick:+.2e}")

    print()
    print("=" * 92)
    print("SWEEP  2x4, (3,3), beta = 8, dtau = 0.05")
    print(f"{'U':>4} {'Psi_T':>6} {'m_stag':>8} {'<T|H|T>':>11} {'overlap':>9} "
          f"{'CPMC':>19} {'exact':>11} {'bias':>9} {'killed':>7}")
    for U in (2.0, 4.0, 8.0, 12.0):
        m = Model2D(Lx=Lx, Ly=Ly, t=1.0, mu=0.0, U=U, dtau=0.05, L=1, theta=0.0)
        ex = ground_energy(m.K, U, nu, nd)
        cands = [("free", free_trial(m.K, nu, nd), 0.0)]
        pu, conv, ms = uhf_trial(m.K, U, nu, nd, stag, h0=1.0)
        cands.append((f"UHF{'' if conv else '*'}", pu, ms))
        for name, pt, ms in cands:
            ov = trial_overlap(pt, m.K, U, nu, nd)
            ev = trial_energy(pt, m.K, U, nu, nd)
            e, se, kl = cpmc_energy(m, nu, nd, pt)
            print(f"{U:4.1f} {name:>6} {ms:8.4f} {ev:+11.5f} {ov:9.5f} "
                  f"{e:+11.5f}+-{se:.5f} {ex:+11.5f} {e-ex:+9.5f} {kl:7d}", flush=True)
    print()
    print("* means UHF did not reach the self-consistency tolerance.")
