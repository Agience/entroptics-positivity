"""§7, gated: the phase carries no linear information from the field, asked on the whole field.

This gates the paper's most load-bearing negative. If `arg(w)` were a function of the field the
phase could be computed without the determinant, so a weak test here would be the worst weak test
in the repo -- and the earlier one asked three chosen functionals, which is a guess at a basis.

THE CONTROL IS WHAT MAKES THE NULL MEAN ANYTHING. `ln|det_up| - ln|det_dn|` is an exact affine
function of `sum(x)` by §4, read on the SAME frames in the same call. It must stand clear of the
null. A rig where it did not would return a null for the phase for the trivial reason that it
cannot see a relation of this kind at all, and that is the failure this file exists to exclude.

No level is chosen anywhere. `carried = 1` is the definitional value of weights that carry nothing,
and every comparison is against the spread of `carried` under the read's own exact re-pairing,
drawn in the same call.
"""
from __future__ import annotations

import numpy as np
import pytest

from functools import lru_cache

from reads.expAO_spectral_criterion import build, ring, tri_ladder
from reads.expAX_is_the_phase_a_read_of_the_field import read_against_the_field

CASES = (
    ("2x4 mu=0.4", build(2, 4, 0.0, 0.4, 0.0)),
    ("ring 5 flux pi/2", ring(5, np.pi / 2)),
    ("ring 7 flux pi/2", ring(7, np.pi / 2)),
    ("tri ladder 8 flux pi/2", tri_ladder(8, np.pi / 2)),
)


@lru_cache(maxsize=None)
def scored(name):
    K = dict(CASES)[name]
    return read_against_the_field(np.asarray(K, dtype=complex))


@pytest.mark.parametrize("name", [n for n, _ in CASES])
def test_the_magnitude_ratio_stands_clear_of_the_null(name):
    """THE POSITIVE CONTROL. §4's relation must be visible to this read on these frames."""
    c = scored(name)["control"]
    assert c is not None, f"{name}: the control produced no reading"
    assert c["carried"] > c["null_max"], \
        f"{name}: the magnitude ratio does not clear its own null -- {c['carried']:.4f} against " \
        f"a null reaching {c['null_max']:.4f}. The phase result below means nothing until it does."


@pytest.mark.parametrize("name", [n for n, _ in CASES])
def test_the_phase_stays_inside_the_null(name):
    """THE RESULT. Neither component of the phase leaves the re-pairing null's own range."""
    r = scored(name)
    for part in ("cos", "sin"):
        d = r[part]
        if d is None:
            continue                      # a real-weight row: this component is identically zero
        assert d["carried"] <= d["null_max"], \
            f"{name}: {part} arg(w) left the null -- {d['carried']:.4f} against a null reaching " \
            f"{d['null_max']:.4f}. If that survives a refit and more configurations, the phase " \
            f"carries field information and §7 must be rewritten."


@pytest.mark.parametrize("name", [n for n, _ in CASES])
def test_the_control_and_the_phase_are_not_the_same_reading(name):
    """The separation, stated as a comparison so it cannot be read as two unrelated nulls."""
    r = scored(name)
    c = r["control"]
    for part in ("cos", "sin"):
        d = r[part]
        if d is None:
            continue
        assert c["sigma"] > d["sigma"], \
            f"{name}: the phase reads as strongly as the magnitude ratio ({d['sigma']:+.1f} " \
            f"against {c['sigma']:+.1f} sigma), which §4 says is an exact relation"


def test_the_field_basis_contains_the_functionals_section_5_chose():
    """The reason this subsumes the earlier test, asserted rather than asserted-about.

    The frame is every Ising variable, so `sum(x)`, a staggered sum and any slice product are
    linear combinations of its columns. Checked by construction: the frame's width is `L * N`.
    """
    r = scored("ring 5 flux pi/2")
    beta, dtau, N = 6.0, 0.125, 5
    assert r["width"] == int(round(beta / dtau)) * N, \
        f"the frame is not the full field: width {r['width']}, expected {int(round(beta / dtau)) * N}"
