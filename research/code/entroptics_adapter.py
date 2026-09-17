"""The reader: a domain adapter over the Entroptics library, in this paper's own vocabulary.

WHY A WRAPPER, AND WHAT IT IS NOT. It is not a convenience layer and it adds no arithmetic. The
library does every computation; this module's whole job is to hold, in one place, three things that
were otherwise spread across two dozen experiment files:

  1. WHICH library read answers which question of the paper, under the paper's name for it. A file
     that says `channel_alignment(G_up, G_dn)` states what is being measured; the same file saying
     `E.reads.coupling(A, B)` states which function is being called, and the reader has to know the
     mapping to see that it is section 7.2. The mapping belongs somewhere findable, and reference
     [E] of the paper is prose, not something a run can check.
  2. WHICH VERSION the numbers were read through. The pin lives in `research/requirements.txt` --
     the file pip actually installs from -- and is checked here, at import, with a reason. It was
     added after the library reported `0.2.2` from stale install metadata while the imported code
     was `0.2.3`, and nothing in the rig could tell.
  3. THE CALLS THAT LOOK RIGHT AND ARE NOT. Two of them cost real time and both are recorded at the
     read they belong to: `rates().dominant` in place of `reconstruct_decay()`, and
     `mean(Re w)/mean(|w|)` in place of the phase-aware estimator. A caution written beside the
     function is a caution that gets read; one written in a paper section is not.

THE DICTIONARY. Each domain read below maps to exactly one library entry point and to the section
that uses it:

    channel_alignment      <- reads.coupling              sec 7.2  the two channels' alignment, and its
                                                                   exact -1 at the lockstep
    weight_cloud           <- reads.concentration         sec 7.1  (resultant, focus) on the (Re, Im)
                                                                   frame: which problem is it
    cloud_axes             <- reads.principal_directions  sec 7.1  the de-rotation axis, read off the
                                                                   cloud's own leading direction
    evidence_ceiling       <- carriage(...).effective_n   sec 9    Kish's effective sample size,
                                                                   exactly n <sgn>^2 on signed weights
    carried                <- carriage(...).carried       sec 8    what a frame carries of a weight
    autocorrelation_time   <- dynamics(...).reconstruct_decay
                                                          sec 9    tau_int with no window chosen
    balance_at_own_zero    <- Screen().balance            sec 7    each channel against the system's
                                                                   own zero, not against zero
    single_channel_optics  <- reads.spectral_optics       sec 9.2b one channel alone; and sec 10's
                                                                   model-order attempt, which failed
    denoise                <- Aperture(...).extract       --       the WRONG tool for a mean here;
                                                                   used by no figure in the paper

WHAT IS RETURNED. The library's own result objects, unchanged, so a caller reads `.strength`, `.z`,
`.resolved`, `.focus` exactly as the library documents them. Returning a reduced tuple would put a
choice of which fields matter into this file, and that choice belongs to the experiment. The
consequence is that `tests/test_entroptics_adapter.py` can assert what this module claims -- that
each read IS the library call it names, value for value -- which is the gate that keeps it a naming
layer rather than a second implementation.
"""
from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np

# Named `entroptics_adapter` so that it does not shadow the installed `entroptics` package: a plain
# import resolves the library directly, with no sys.path shim.
_lib = importlib.import_module("entroptics")
_dynamics = importlib.import_module("entroptics.dynamics")
_environment = importlib.import_module("entroptics.environment")

#: The one place the version is stated -- the file pip installs from. Read rather than repeated, so
#: relaxing or tightening the requirement cannot leave a second number behind saying otherwise.
_REQUIREMENTS = Path(__file__).resolve().parent.parent / "requirements.txt"


def _pinned_version() -> str:
    """The entroptics floor from research/requirements.txt -- the single source.

    Accepts `==` or `>=`. What the number is used for here is a MINIMUM on the call surface and on
    the read values, so both spellings answer the same question.
    """
    for line in _REQUIREMENTS.read_text(encoding="utf-8").splitlines():
        stmt = line.split("#", 1)[0].strip()
        for op in ("==", ">="):
            if stmt.startswith("entroptics" + op):
                return stmt[len("entroptics" + op):].strip()
    raise ImportError(f"no 'entroptics==' or 'entroptics>=' requirement found in {_REQUIREMENTS}")


def _version_tuple(v: str) -> tuple:
    return tuple(int(x) for x in v.split(".")[:3] if x.isdigit())


#: e.g. "0.2.3" -- set in requirements.txt, not here.
REQUIRED_VERSION = _pinned_version()

#: What is actually loaded. `entroptics.__version__` comes from the INSTALLED DISTRIBUTION's
#: metadata, not from the source being imported, so an editable install whose metadata was recorded
#: at an earlier release reports that earlier number while running current code. That is the exact
#: state this check was written for, and it is why the failure below names both the version and the
#: file: the file is what tells you which copy you are running.
INSTALLED_VERSION = getattr(_lib, "__version__", "0.0.0")

if _version_tuple(INSTALLED_VERSION) < _version_tuple(REQUIRED_VERSION):
    raise ImportError(
        f"entroptics {REQUIRED_VERSION} or later is required and {INSTALLED_VERSION} is loaded, "
        f"from {getattr(_lib, '__file__', '?')}.\n"
        f"The paper's numbers were read through {REQUIRED_VERSION}; an older reader may have a "
        f"different call surface and may return different values.\n"
        f"If that path is a source checkout at the right version, the DISTRIBUTION METADATA is "
        f"stale rather than the code -- reinstall it (`pip install -e .`) so the version it "
        f"reports is the version it is."
    )


# ---------------------------------------------------------------------------------------------
# section 7.2 -- the alignment between the two spin channels
# ---------------------------------------------------------------------------------------------

def channel_alignment(a, b, **kw):
    """The normalised alignment between two channel frames -- section 7.2's order parameter.

    `a` and `b` are frames of the same shape, one row per configuration: the paper reads the two
    spin channels' Green's-function diagonals, `G_up` against `G_dn`. The result carries

        .strength   the alignment itself, exactly -1 where the lockstep is perfect, because
                    particle-hole symmetry gives `G_dn = 1 - G_up` configuration by configuration;
        .z          that value against the read's OWN exact re-pairing null -- exact by
                    construction rather than asymptotic, which is what lets section 7 quote a
                    sigma with no constant supplied;
        .resolved   whether there was an alignment to report at all.

    The deficit `1 + strength` is what section 7 calls the distance from the protecting symmetry.
    It is an onset detector rather than a severity meter, and section 8 gives the reason: the
    severity is set by the parity of a crossing count, and the deficit is a summary that has
    already discarded that count. The deficit's own range is `[0, 2]`
    (`Alignment.deficit_mem_Icc`).

    THE SIGN DOES NOT ENTER THIS READ. Callers hand it the two channels' frames; the sign of the
    weight is collected alongside and is never an argument. So this is an unweighted statistic of
    the configurations an `|w|` chain visits, not a reweighted estimator, and the `n <sgn>^2`
    ceiling of `evidence_ceiling` does not apply to it. That is why it resolves where the average
    sign is identically 1.
    """
    return _lib.reads.coupling(a, b, **kw)


# ---------------------------------------------------------------------------------------------
# section 7.1 -- the triage: which problem does this run have?
# ---------------------------------------------------------------------------------------------

def weight_cloud(w, **kw):
    """The directional and axial statistics of a complex weight cloud -- section 7.1's triage.

    `w` is the complex weights themselves, one per configuration; they are stacked into the
    `(Re, Im)` frame here so that no caller has to remember the axis convention. The result carries

        .resultant  the directional statistic -- how far the weights point one way;
        .focus      the axial one -- whether they lie on a LINE, regardless of direction.

    The pair sorts the run into three regimes, and the sorting is the result:

        focus = 1, resultant = 1   sign-free
        focus = 1, resultant < 1   a real sign problem -- rank one in the plane means the phase
                                   takes two values pi apart, and a phase confined to Z2 is what a
                                   sign IS
        focus < 1                  a genuine phase problem that no rotation reaches

    Getting this wrong is expensive rather than merely untidy. A global phase cancels in
    `<O> = sum(O w)/sum(w)` and costs nothing, but the estimator carried over from real weights,
    `mean(Re w)/mean(|w|)`, reads `cos(theta)` too small across rotations that leave `|<w>|`
    unmoved, and would overstate a cost that goes as `1/<sgn>^2` by fourteen times.
    """
    u = np.asarray(w)
    if np.iscomplexobj(u) and u.ndim == 1:
        u = np.stack([u.real, u.imag], axis=1)
    return _lib.reads.concentration(u, **kw)


def cloud_axes(w, **kw):
    """The weight cloud's own leading directions -- the de-rotation axis of section 7.1.

    Returns the axes as columns. TWO properties of this read are load-bearing and neither is
    available from a hand-rolled PCA:

      * the angle is read off the cloud's own leading direction, so no angle is chosen -- which is
        what lets section 7.1 de-rotate a maximum imaginary part of 0.95 down to 4e-16 and recover
        `|<sgn>|` exactly, under a canon that forbids a supplied constant;
      * it returns ZERO columns on a cloud with no axis, a parameter-free "there is none". A
        hand-rolled PCA always returns an axis, and saying "none" would need a chosen cut.

    So `V.shape[1]` is the mode count, and zero is a verdict rather than a degenerate case.
    """
    u = np.asarray(w)
    if np.iscomplexobj(u) and u.ndim == 1:
        u = np.stack([u.real, u.imag], axis=1)
    return _lib.reads.principal_directions(u, **kw)


# ---------------------------------------------------------------------------------------------
# section 9 -- what any read of the sign must contend with
# ---------------------------------------------------------------------------------------------

def evidence_ceiling(frame, weights, **kw):
    """Kish's effective sample size of the weights -- the ceiling of section 9.

    Correct importance sampling draws at `|w|`, so a configuration enters any weighted average
    carrying only its SIGN, and

        ESS = (sum w)^2 / sum w^2 = (n <sgn>)^2 / n = n <sgn>^2

    exactly. That is the `O(1/<sgn>^2)` cost of the sign problem written as a property of the
    WEIGHTS: every estimator that REWEIGHTS by the sign pays it, whatever the observable, because
    recovering a physical expectation from an `|w|`-sampled chain means forming `<O s> / <s>`.

    IT IS A STATEMENT ABOUT REWEIGHTED MEANS, which is what Kish's formula is about, and it is not
    extended to arbitrary functionals of a signed sample. `channel_alignment` is not reweighted and
    is not bounded by this.

    Returns the library's carriage object; `.effective_n` is the ceiling. It does not depend on the
    frame, which is the point of it being a property of the weights, and `tests/test_weighted_ceiling`
    asserts exactly that across three frame shapes.

    READ IT AS AN UPPER BOUND, which is how the paper uses it throughout. Kish's formula assumes
    INDEPENDENT draws and a DQMC run is a Markov chain, so the evidence a run actually carries is
    `n <sgn>^2 / (2 tau_int)` -- smaller still, measured at 4.15x on this rig. Converting the
    ceiling into an error bar without that factor is off by about two.
    """
    return _lib.carriage(frame, weights, **kw)


def frame_carriage(frame, weights, **kw):
    """What a frame CARRIES of a set of weights -- the `.carried` read of section 8.

    The same library call as `evidence_ceiling`, under the name of the other question it answers,
    because the two are used for opposite purposes and conflating them is how a ceiling gets read
    as a signal. `evidence_ceiling` asks how much evidence the weights can support and is a
    property of the weights alone; this asks whether a given frame carries the weight pattern at
    all, and is scored against the read's own permuted null -- which is what section 8 does when it
    asks whether the phase is a function of the field.
    """
    return _lib.carriage(frame, weights, **kw)


def autocorrelation_time(z, max_lag, **kw):
    """The integrated autocorrelation time of a chain, with no window chosen -- section 9.

    `C(tau)` is rebuilt from the one-step operator's modal powers and eigenvalues and summed over
    all lags, so there is no window, truncation lag or multiplier: moving the summation limit from
    1000 to 8000 changes the answer by 0.0e+00.

    IT MUST BE `reconstruct_decay`, WHICH READS THE CONNECTED OPERATOR. `rates().dominant` reads
    the raw one, where the constant function is an eigenmode with `|mu| = 1`; on a sign sequence
    with mean 0.83 it returns that mode instead of the decay and overstates tau by more than an
    order of magnitude. This function exists partly so that the wrong one cannot be reached by
    autocomplete.

    AND `z` IS NOT THE SIGN. There is no single autocorrelation time for a chain: each estimator
    has its own, set by its influence function. The quantity a practitioner reports is the
    reweighted ratio `<O s>/<s>`, whose linearisation is `z_t = (O_t s_t - r s_t)/<s>` at the
    pooled `r`, and `z` relaxes more slowly than `s` does. Reaching for the sign's own tau accounts
    for only two thirds of the gap.

    Returns `C(tau)` as a numpy array; the caller sums it. Summing here would put section 9's
    convention (`0.5 + sum(C[1:])`) into this file, where a different section could not see it.
    """
    return np.asarray(_environment.to_numpy(dynamics(z, **kw).reconstruct_decay(max_lag)), float)


def dynamics(w, **kw):
    """The one-step operator of a chain, for the reads around `reconstruct_decay`.

    A scalar chain is reshaped to the library's `(T, F)` frame with one feature. That reshape is
    the domain convention -- a chain of one observable is a one-feature frame, and its rows are the
    states -- and doing it here is what keeps every caller from spelling it a slightly different
    way. A frame handed in already 2-D passes through untouched.

    Exposed as well as used, because section 9 asks the operator whether a decay EXISTS before
    integrating one: `forgetting()["margin"] >= 1` is a non-decaying mode, and `tau_int` does not
    exist there. That is a refusal the read supplies, and it has to be reachable without going
    around this module.
    """
    x = np.asarray(w)
    return _dynamics.dynamics(x.reshape(-1, 1) if x.ndim == 1 else x, **kw)


# ---------------------------------------------------------------------------------------------
# section 7 / 9.2b -- reads of ONE channel
# ---------------------------------------------------------------------------------------------

def balance_at_own_zero(**kw):
    """A screen for testing each channel against the SYSTEM's own zero -- section 7.

    Returns the library's `Screen`, whose `balance` takes the `zero` the system defines rather than
    the arithmetic zero: at the particle-hole point that is `1/2`, not `0`. Withholding the
    system's law and letting the read use its own zero is the control, and it is what makes this a
    test of the system's symmetry rather than of where the numbers happen to sit.

    READ THE PVALUE, NOT THE BOOLEAN. `closed` is a per-run decision at the reader's level and
    fluctuates -- a sign-free ring fires on 9 runs of 12. The pvalue behind it is the quantity, and
    its median across independent runs separates with no overlap.
    """
    return _lib.Screen(**kw)


def single_channel_optics(data, **kw):
    """The spectral reads of ONE channel alone -- section 9.2b, and section 10's failed attempt.

    Neither of section 9's two impossibilities rules this family out: it aggregates no signed
    configuration and compares no channels, so it was measured
    (`reads/expBG_single_channel_has_no_indicator.py`). Across eighteen lattices spanning sign-free
    to a sign deficit of 0.46508, no read here orders them by it -- `phase` +0.273, `attenuation`
    +0.007, `top_share` +0.017, `dominance` -0.017, all at p >= 0.272, where |rho| >= 0.468 would
    have reached p < 0.05 at that n. `attenuation` ranges further among the ten sign-free lattices
    alone (0.05161 to 0.79089) than between them and any lattice there with a sign problem.

    `resolved_modes` is model-order selection performed by the read, against a floor derived from
    the data rather than supplied, and section 10 records that it does not work on this problem:
    on the Hankel embedding of a sum of decaying exponentials it returns 1 for every true order
    above 1, at every noise level including none, while the same matrix has exact numerical rank 2
    and 3. The reason is structural -- a floor that separates signal from a noise sea is asking a
    different question from "how many modes are there" -- so this is a closed route, recorded, not
    a read to reach for.
    """
    return _lib.reads.spectral_optics(np.asarray(data), **kw)


def denoise(W, **kw):
    """Optimal shrinkage on a weight frame -- section 9.7, where it is the wrong tool.

    `extract` recovers a low-rank signal and biases a mean, and the sign problem's difficulty is
    entirely in a mean. Kept in the adapter so a reader can run the attempt the paper reports.

    Returns `(clean, info)` with `clean` in `W`'s own units, so the caller applies no rescaling.
    `clean` is `info['centre']` plus a shrunk projection onto the resolved modes: where nothing is
    resolvable that projection is empty and `clean.mean(axis=0)` is the centre, and where modes are
    resolved the projection carries a mean of its own and the two differ (section 9.7 measures the
    gap at `6.0e-05` to `1.3e-02`).
    """
    return _lib.Aperture(np.asarray(W), **kw).extract()
