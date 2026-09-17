"""The reader is published as `entroptics-positivity`, and publishing it created a second place
where the entroptics version is stated. This holds the two together.

`entroptics_adapter._pinned_version()` reads the floor from *the file pip installs from*, and which
file that is depends on how the module was reached:

  * a repo checkout      -> `research/requirements.txt`   (what a paper rerun installs)
  * an installed wheel   -> this distribution's metadata, from `[project] dependencies`

That is one rule resolved against two files, which is the correct design — but it is exactly the
shape that goes wrong silently. Relax `requirements.txt` to a newer entroptics and forget
`pyproject.toml`, and a reader who `pip install entroptics-positivity`s gets a DIFFERENT floor from
the one the paper's figures were read through, with nothing raising anywhere.

⛔ WHY A FLOOR MATTERS HERE AT ALL. `entroptics_adapter`'s import-time check exists because the
library reported `0.2.2` from stale install metadata while the imported code was `0.2.3`, and
nothing in the rig could tell. The floor is the paper's claim about which reader produced its
numbers. Two copies of a claim is one more than can be trusted.

⚠ THIS FILE IMPORTS NEITHER ENTROPTICS NOR THE ADAPTER. It reads both manifests as text, so it
states its result without needing the library installed, and so
`test_the_instrument_is_reached_through_the_adapter.py` has nothing to exempt it from.
"""
from __future__ import annotations

import re
import tomllib
from pathlib import Path

#: tests/ -> research/code -> research -> <repo>
_CODE = Path(__file__).resolve().parent.parent
_RESEARCH = _CODE.parent
_REPO = _RESEARCH.parent

_REQUIREMENTS = _RESEARCH / "requirements.txt"
_PYPROJECT = _REPO / "pyproject.toml"

_REQ = re.compile(r"^entroptics\s*(==|>=)\s*([0-9][0-9A-Za-z.\-]*)\s*$")


def _floor(statements) -> tuple[str, str] | None:
    """`(operator, version)` for the entroptics requirement in `statements`, or None."""
    for raw in statements:
        stmt = raw.split("#", 1)[0].split(";", 1)[0].strip()
        m = _REQ.match(stmt)
        if m:
            return m.group(1), m.group(2)
    return None


def test_both_manifests_exist():
    """Neither path may quietly stop resolving — a missing file makes every check below vacuous."""
    assert _REQUIREMENTS.is_file(), f"{_REQUIREMENTS} is missing; the checkout floor is unreadable"
    assert _PYPROJECT.is_file(), f"{_PYPROJECT} is missing; the installed floor is unreadable"


def test_the_two_entroptics_floors_agree():
    req = _floor(_REQUIREMENTS.read_text(encoding="utf-8").splitlines())
    assert req is not None, f"no entroptics requirement in {_REQUIREMENTS}"

    project = tomllib.load(_PYPROJECT.open("rb"))["project"]
    pyp = _floor(project.get("dependencies", []))
    assert pyp is not None, f"no entroptics dependency in {_PYPROJECT}"

    assert req == pyp, (
        f"the entroptics floor differs between the two files pip installs from:\n"
        f"  research/requirements.txt : entroptics{req[0]}{req[1]}   (a repo checkout / paper rerun)\n"
        f"  pyproject.toml            : entroptics{pyp[0]}{pyp[1]}   (pip install entroptics-positivity)\n\n"
        f"`entroptics_adapter._pinned_version()` reads whichever of these applies, so a reader and "
        f"the paper would be held to different versions of the library. Change both, in one commit.")


def test_the_packaged_module_is_the_one_the_rig_uses():
    """`py-modules` must name the file 46 experiment files import, not a copy of it.

    The whole reason this distribution packages the module in place is that the rig imports it as a
    flat sibling. If `package-dir` or `py-modules` ever points somewhere else, the published reader
    and the reader that produced the paper's numbers become two different files with one name.
    """
    cfg = tomllib.load(_PYPROJECT.open("rb"))["tool"]["setuptools"]
    modules = cfg["py-modules"]
    assert modules == ["entroptics_adapter"], f"expected the adapter alone, got {modules}"

    root = cfg["package-dir"][""]
    packaged = (_REPO / root / "entroptics_adapter.py").resolve()
    used_by_rig = (_CODE / "entroptics_adapter.py").resolve()
    assert packaged == used_by_rig, (
        f"pyproject packages {packaged}\nbut the rig imports {used_by_rig}\n\n"
        f"These must be the same file. Publishing a copy means the paper's numbers and the "
        f"published reader can drift apart with nothing to notice.")


def test_the_rig_only_dependencies_are_not_published():
    """`scipy` and `pytest` belong to the apparatus, not to the reader.

    The adapter imports numpy and entroptics and nothing else. Shipping the rig's full requirements
    would make every installer of a reading layer pull a test runner — and would state, wrongly,
    that the reader needs them.
    """
    deps = tomllib.load(_PYPROJECT.open("rb"))["project"].get("dependencies", [])
    names = {re.split(r"[<>=!\[ ]", d, 1)[0].lower() for d in deps}
    for rig_only in ("scipy", "pytest"):
        assert rig_only not in names, (
            f"{rig_only} is declared as a dependency of the published reader. The adapter does not "
            f"import it; it belongs to research/requirements.txt, which is the rig's manifest.")
    assert names == {"numpy", "entroptics"}, f"unexpected dependency set: {sorted(names)}"
