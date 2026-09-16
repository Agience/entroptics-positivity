"""The one place that knows where heavy Python runs.

WHY A WRAPPER. Seven experiments here run past ten minutes and the test suite takes about fourteen,
and minutes of CPU on a workstation are minutes the workstation is not usable. Which machine runs
them is a property of whoever is running the program, not of the source, so every caller that spelled
out ``python reads/expBC_ceiling_is_not_tight.py`` was hard-coding one answer to that question. This
module holds it once; callers ask for a run and read the output.

CONFIGURATION, and the default is deliberate. With nothing configured this runs LOCALLY, because a
reader who clones this repository must be able to reproduce every number in the paper without access
to any particular machine. A compute host is an optimisation for whoever has one, declared in the
machine-local, git-ignored ``research.local.env`` (see ``research.local.env.example``):

    COMPUTE_HOST=user@host                       # ssh destination; absent, runs are local
    COMPUTE_PORT=22                              # optional
    COMPUTE_KEY=~/.ssh/some_key                  # optional
    COMPUTE_DIR=signproblem                      # remote working tree, relative to $HOME
    COMPUTE_PY=$HOME/research-venv/bin/python    # the interpreter there, holding requirements.txt

ITS OWN KEYS, not the Lean builder's. The two roles may well be the same machine, and reading one
set of keys for both would bake that coincidence into the source: moving the reads to a bigger box
would then move the Lean build with them. They are different questions -- a Lean builder wants a warm
Mathlib, a compute host wants cores -- and they get separate answers. There is no fallback from one
to the other, for the same reason.

WHAT A REMOTE RUN DOES. It ships the Python SOURCES and runs them there, so what runs is what is on
this machine, not whatever the remote checkout drifted to. There is no data store to ship: every
experiment here generates its own configurations from a seed, which is what makes the whole tree
portable in the first place.

THE IMPORT PATH IS PART OF THE CONTRACT. Everything runs from ``research/code`` with that directory
on ``PYTHONPATH``; a bare ``python reads/<file>.py`` puts ``reads/`` there instead and fails on
``import dqmc``. This wrapper sets both, locally and remotely, so a caller cannot get it wrong in
one place and right in the other.

WHAT IT DOES NOT DO. It does not decide what to run, parse output, or write artifacts. Callers do
that; this returns the completed process.

    python remote_run.py --describe
    python remote_run.py reads/expBD_route_calibration.py
    python remote_run.py --drift
"""
from __future__ import annotations

import argparse
import glob
import os
import shlex
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import local_env  # noqa: E402  (needs HERE on sys.path first)

#: The research tree root -- the parent of ``code/``.
RESEARCH = os.path.dirname(HERE)

#: The sources a run needs, relative to ``RESEARCH``. Everything under ``code/`` plus the pin file,
#: because an experiment reaches into its siblings: the reads import the sampler, and several import
#: from ``tests/`` and ``closed/`` to reuse a gate's construction rather than restate it.
#:
#: ``code/reference/*`` is committed DATA, not source, and it ships for the same reason the Python
#: does: ``closed/expLL_unfitted_trial`` loads the multi-determinant ceiling at IMPORT, so without
#: it the suite does not fail a test, it fails to collect one -- which is a different and less
#: legible failure. It is two small files; there is no store here to leave behind.
SOURCE_GLOBS = ("requirements.txt", "code/*.py", "code/reads/*.py", "code/closed/*.py",
                "code/tests/*.py", "code/pytest.ini", "code/reference/*")


def target():
    """Where heavy runs go, in words. ``None`` means locally."""
    host = local_env.value("COMPUTE_HOST")
    if not host:
        return None
    return f"{host}:{local_env.value('COMPUTE_PORT', '22')}"


def _ssh_argv():
    argv = ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "BatchMode=yes"]
    key = local_env.value("COMPUTE_KEY")
    if key:
        argv += ["-i", os.path.expanduser(key)]
    port = local_env.value("COMPUTE_PORT")
    if port:
        argv += ["-p", str(port)]
    return argv + [local_env.value("COMPUTE_HOST")]


def _remote_dir():
    return local_env.value("COMPUTE_DIR", "signproblem")


def _remote_prelude():
    """``cd`` into the remote tree's ``code/`` with it on the import path, as one shell prefix."""
    return f'cd {shlex.quote(_remote_dir())}/code && export PYTHONPATH=.; '


# CHOSEN: a transfer budget in seconds. It bounds a hung network and nothing else -- the payload is
# well under a megabyte of Python sources, so any value that crosses a working link behaves
# identically. It caps waiting, never correctness.
def push_sources(timeout=600):
    """Ship this tree's Python sources to the remote working tree. No-op when running locally.

    Sent as a tar stream rather than file-by-file: one round trip, and the transfer either lands
    whole or not at all.
    """
    if target() is None:
        return None
    files = []
    for g in SOURCE_GLOBS:
        files += [os.path.relpath(f, RESEARCH).replace("\\", "/")
                  for f in glob.glob(os.path.join(RESEARCH, *g.split("/")))]
    if not files:
        raise SystemExit(f"no Python sources found under {RESEARCH}; refusing to push an empty tree")
    tar = subprocess.run(["tar", "czf", "-"] + sorted(files), cwd=RESEARCH,
                         capture_output=True, timeout=timeout)
    # DERIVED: zero is the POSIX convention for success, not a threshold.
    if tar.returncode != 0:
        raise SystemExit(f"tar failed: {tar.stderr.decode('utf-8', 'replace')[:400]}")
    r = subprocess.run(
        _ssh_argv() + [f"mkdir -p {shlex.quote(_remote_dir())} && "
                       f"cd {shlex.quote(_remote_dir())} && tar xzf -"],
        input=tar.stdout, capture_output=True, timeout=timeout)
    # DERIVED: zero is the POSIX convention for success, not a threshold.
    if r.returncode != 0:
        raise SystemExit(f"pushing sources failed: {r.stderr.decode('utf-8', 'replace')[:400]}")
    return len(files)


def sources_only_on_remote():
    """Python sources present on the remote but not here, as a sorted list of relative paths.

    A push adds and overwrites; it does not delete. So a file retired here survives there and would
    be run by the next remote invocation, which is how a compute host starts disagreeing with the
    tree it is supposed to be running. Reported rather than deleted: removing a file from someone
    else's machine is not a thing a run wrapper should do silently.
    """
    if target() is None:
        return []
    r = subprocess.run(
        _ssh_argv() + [f"cd {shlex.quote(_remote_dir())} 2>/dev/null && "
                       "ls code/*.py code/reads/*.py code/closed/*.py code/tests/*.py 2>/dev/null"],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
    remote = {x.strip() for x in r.stdout.split() if x.strip().endswith(".py")}
    here = set()
    for g in SOURCE_GLOBS:
        if g.endswith(".py"):
            here |= {os.path.relpath(f, RESEARCH).replace("\\", "/")
                     for f in glob.glob(os.path.join(RESEARCH, *g.split("/")))}
    return sorted(remote - here)


# CHOSEN: one hour. Long enough for the heaviest experiment here and for the full suite, short enough
# that a wedged run is noticed in one sitting. Caps waiting, never correctness.
def run(script, args=(), *, timeout=3600, push=True, unbuffered=True, module=False):
    """Run ``python <script> <args>`` where heavy work belongs; return the CompletedProcess.

    ``script`` is relative to ``research/code`` -- e.g. ``reads/expBD_route_calibration.py`` -- which
    is the spelling the README and the paper both use, so a reader can paste one into the other.

    ``module=True`` runs ``python -m <script>`` instead, which is how the suite is invoked
    (``-m pytest``).

    ``unbuffered`` passes ``-u``, so a long run's progress is visible while it runs instead of
    arriving in one block at the end.
    """
    rel = script.replace("\\", "/")
    if not module and not os.path.exists(os.path.join(HERE, *rel.split("/"))):
        raise SystemExit(f"no such script under {HERE}: {script}")

    py_args = (["-u"] if unbuffered else []) + (["-m"] if module else []) + [rel, *map(str, args)]

    if target() is None:
        env = dict(os.environ, PYTHONPATH=HERE)
        return subprocess.run([sys.executable, *py_args], cwd=HERE, capture_output=True,
                              text=True, encoding="utf-8", errors="replace", timeout=timeout,
                              env=env)
    if push:
        push_sources()
    py = local_env.value("COMPUTE_PY", "python3")
    cmd = _remote_prelude() + " ".join([py] + [shlex.quote(a) for a in py_args])
    return subprocess.run(_ssh_argv() + [cmd], capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=timeout)


def describe():
    """One line naming where heavy runs go, for a caller that wants to say so."""
    t = target()
    if t is None:
        return f"compute target: local ({HERE})"
    where = local_env.source("COMPUTE_HOST")
    return (f"compute target: {t} ({_remote_dir()}/code, "
            f"{local_env.value('COMPUTE_PY', 'python3')}) via {where}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("script", nargs="?", help="script path relative to research/code")
    # REMAINDER, so that a script's own flags reach the script instead of being claimed here. The
    # suite is invoked as `--module pytest tests -q`, and with `nargs="*"` argparse took `-q` for
    # one of this wrapper's options and refused the whole command.
    ap.add_argument("args", nargs=argparse.REMAINDER,
                    help="arguments passed through to the script, verbatim")
    ap.add_argument("--describe", action="store_true", help="print the target and exit")
    ap.add_argument("--module", action="store_true", help="run `python -m <script>` instead")
    ap.add_argument("--drift", action="store_true",
                    help="list sources present on the remote but not here, and exit")
    a = ap.parse_args()

    print(describe(), file=sys.stderr)
    if a.describe:
        return 0
    if a.drift:
        for f in sources_only_on_remote():
            print(f)
        return 0
    if not a.script:
        ap.error("a script is required unless --describe or --drift is given")
    r = run(a.script, a.args, module=a.module)
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            pass
    sys.stdout.write(r.stdout)
    sys.stderr.write(r.stderr)
    return r.returncode


if __name__ == "__main__":
    raise SystemExit(main())
