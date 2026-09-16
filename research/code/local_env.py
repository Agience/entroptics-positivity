"""The machine-local configuration file, and the only parser for it.

Two things about running this repository are properties of a MACHINE and not of the source: which
box compiles the Lean development, and which box runs a fourteen-minute sweep. Naming either in a
caller bakes one person's hardware into the tree, and a hardcoded fallback is worse than no answer,
because being wrong quietly is the failure mode -- a run that goes to the wrong place still exits 0.

So both are named in exactly two places, neither of them source:

  1. an environment variable of the same name -- a one-off override for a single run, and what CI
     or a batch job sets;
  2. ``research.local.env`` at the repository root -- git-ignored, written once per machine, so
     nobody has to remember an environment variable on every invocation.

There is deliberately NO built-in default for a host. With nothing configured every wrapper here
runs LOCALLY, because a reader who clones this repository has to be able to reproduce every number
in the paper without access to any particular machine. A compute host is an optimisation for
whoever has one, not a dependency.

The parser is deliberately small -- ``KEY=VALUE``, ``#`` comments, an optional ``export`` prefix,
optional surrounding quotes -- rather than a dotenv dependency. A format that needs a library to
read is a format nobody can fix by hand at the moment it is wrong.
"""
from __future__ import annotations

import os
from pathlib import Path

#: Machine-local, git-ignored, at the repository root -- NOT under ``research/``. The placement is
#: load-bearing: the research tree is meant to be free of machine-specific absolute paths, and the
#: one file whose job is to name such a path sits just outside it. The ``research.`` prefix is what
#: pays for that separation.
LOCAL_FILENAME = "research.local.env"

#: Committed alongside it, documenting every key this repository reads.
EXAMPLE_FILENAME = "research.local.env.example"


def repo_root() -> Path:
    """The repository root, from this file's own location.

    Not from the working directory: these scripts are run from at least two of them (the repository
    root and ``research/code``, which the import path requires), so anything cwd-relative would
    resolve differently depending on the caller.
    """
    return Path(__file__).resolve().parents[2]


def local_config_path() -> Path:
    """Full path of the git-ignored local config file, whether or not it exists."""
    return repo_root() / LOCAL_FILENAME


def example_config_path() -> Path:
    """Full path of the committed example, which does exist."""
    return repo_root() / EXAMPLE_FILENAME


def local_value(key: str) -> str | None:
    """Any ``KEY`` from the machine-local config file, or None when it is absent or unset.

    Absent and unreadable are both "not configured" rather than fatal: the common case on a fresh
    clone is that the file does not exist at all, and that has to mean "run locally", not "refuse".
    """
    try:
        text = local_config_path().read_text(encoding="utf-8")
    except OSError:
        return None
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower().startswith("export "):
            line = line[len("export "):].lstrip()
        name, sep, value = line.partition("=")
        if not sep or name.strip() != key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        if value:
            return value
    return None


def value(key: str, default: str | None = None) -> str | None:
    """``KEY`` from the environment first, then the local file, then ``default``.

    The environment wins so that a single run can be pointed somewhere else without editing
    anything, which is what a CI job or a one-off `COMPUTE_HOST= python ...` needs.
    """
    return (os.environ.get(key, "").strip() or local_value(key) or default)


def source(key: str) -> str | None:
    """Which mechanism is supplying ``key``, in words, or None if neither is.

    Reported in `--describe` output: "runs go to X" is not actionable on its own when someone is
    trying to work out which of two places to edit.
    """
    if os.environ.get(key, "").strip():
        return f"the {key} environment variable"
    if local_value(key):
        return str(local_config_path())
    return None
