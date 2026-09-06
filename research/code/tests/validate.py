"""Gate: the DQMC weight/propagator formulas must reproduce the exact Trotter trace
by brute-force enumeration of every auxiliary field.  If this does not match to
machine precision the sampler is wrong and nothing downstream means anything.

Also runs the negative control: a deliberately WRONG propagator formula must FAIL
this gate, so we know the gate can fail.
"""
import itertools
import numpy as np
from dqmc import Model, ExactTrotter

np.set_printoptions(precision=8, suppress=True)


def enumerate_sum(m: Model, corrupt=False):
    """Exact sum over all 2^(N L) auxiliary fields."""
    N, L = m.N, m.L
    Zs, num, sgn_sum, abs_sum = 0.0, np.zeros((L, N, N)), 0.0, 0.0
    for bits in itertools.product([-1.0, 1.0], repeat=N * L):
        s = np.array(bits).reshape(L, N)
        sign, logabs, G = m.weight_and_G(s)
        if corrupt:
            G = np.roll(G, 1, axis=0)          # negative control: wrong time origin
        w = sign * np.exp(logabs)
        Zs += w
        num += w * G
        sgn_sum += sign * np.exp(logabs)
        abs_sum += np.exp(logabs)
    C = np.exp(-m.dtau * m.U / 4.0) / 2.0
    return C ** (N * L) * Zs, num / Zs, sgn_sum / abs_sum


if __name__ == "__main__":
    m = Model(N=2, t=1.0, t2=0.7, mu=0.9, U=8.0, dtau=0.25, L=6)
    ed = ExactTrotter(m)
    Z_enum, G_enum, avg_sign = enumerate_sum(m)
    G_ed = ed.propagator()

    print(f"N={m.N} L={m.L} dtau={m.dtau} beta={m.beta}  <sign> = {avg_sign:.6f}")
    print(f"Z  exact-ED   = {ed.Z:.12e}")
    print(f"Z  enumerated = {Z_enum:.12e}")
    print(f"Z  rel diff   = {abs(Z_enum - ed.Z) / abs(ed.Z):.3e}")
    dG = np.max(np.abs(G_enum - G_ed))
    print(f"G max abs diff = {dG:.3e}   (scale |G|max = {np.max(np.abs(G_ed)):.4f})")
    print("G_ed[:2]   =", G_ed[:2].ravel())
    print("G_enum[:2] =", G_enum[:2].ravel())

    _, G_bad, _ = enumerate_sum(m, corrupt=True)
    dbad = np.max(np.abs(G_bad - G_ed))
    print(f"\nNEGATIVE CONTROL (wrong time origin): max abs diff = {dbad:.3e}")

    ok = (abs(Z_enum - ed.Z) / abs(ed.Z) < 1e-10) and (dG < 1e-10) and (dbad > 1e-3)
    print("\nGATE:", "PASS" if ok else "FAIL")
