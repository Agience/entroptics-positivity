"""Section 7.1: a global phase cancels in the ratio, and the real-part estimator does not survive it.

Two claims sit together in section 7.1 and neither had a file that produces them.

  * A phase common to every configuration multiplies numerator and denominator of
    `<O> = sum(O w)/sum(w)` alike, so it cancels exactly. Rotating every weight by a common angle
    must therefore leave `|<w>|` where it was.
  * The estimator carried over from real weights, `mean(Re w)/mean(|w|)`, is not that ratio. It
    reads `cos(theta)` times the true value, so it falls under the same rotation that changes
    nothing, and reports a sign problem that is not there.

The second is the one that costs something. `<sgn>` enters the cost as `1/<sgn>^2`, so a reading
low by `cos(theta)` overstates the cost by `1/cos(theta)^2` -- the last column.

The weights are a real staggered lattice's, drawn once and then rotated, so every row is the SAME
ensemble seen through a different global phase. That is what makes the first column's constancy a
measurement rather than an identity of the arithmetic.

    python remote_run.py reads/expBN_a_global_phase_and_the_wrong_estimator.py
"""
from __future__ import annotations

import numpy as np

from reads.expAO_spectral_criterion import build
from reads.expAU_axial_versus_directional import unit_weights

# expAU's own frame and parameters, deliberately. Section 7.1 reports `resultant` on the
# UNIT-MODULUS weights `u = w / |w|`, so the estimator has to be read on the same frame or the two
# numbers in the section would be about different quantities.
ANGLES = (0.0, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3)


def main():
    u = np.asarray(unit_weights(build(2, 4, 0.0, 0.0, 0.6)))   # staggered h = 0.6
    if u.ndim == 2:                       # (Re, Im) rows -> complex
        u = u[:, 0] + 1j * u[:, 1]
    base = abs(u.mean())
    print("=" * 96)
    print("A GLOBAL PHASE CANCELS IN THE RATIO; THE REAL-PART ESTIMATOR DOES NOT SURVIVE IT")
    print("=" * 96)
    print("  2x4 staggered h = 0.6, on expAU's unit-modulus frame and parameters")
    print("  The same weights at every row, multiplied by exp(i theta).")
    print()
    print(f"{'theta':>7} {'|<u>|':>17} {'mean(Re w)/mean(|w|)':>22} "
          f"{'cos(theta) x base':>19} {'cost overstated by':>20}")
    print("-" * 96)

    for th in ANGLES:
        r = u * np.exp(1j * th)
        ratio = abs(r.mean())
        naive = float(r.real.mean() / np.abs(r).mean())
        print(f"{th:7.2f} {ratio:17.5f} {naive:22.5f} {np.cos(th) * base:19.5f} "
              f"{(ratio / naive) ** 2 if naive > 0 else float('inf'):20.2f}", flush=True)

    print()
    print("  The first column does not move: a common phase cancels, exactly, and the run is no")
    print("  worse than it was. The second falls as cos(theta) and would be read as a collapsing")
    print("  average sign. The last column is what that costs, because the cost goes as 1/<sgn>^2.")


if __name__ == "__main__":
    main()
