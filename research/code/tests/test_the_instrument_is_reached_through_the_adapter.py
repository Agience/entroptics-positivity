"""Every Entroptics read in this repository goes through `entroptics_adapter`. This is the gate.

The adapter holds three things in one place -- which library read answers which question of the
paper, which VERSION the numbers were read through, and the two calls that look right and are not.
None of that survives a file that imports the library directly, and a convention nothing checks is
a convention that decays one file at a time. So it is checked.

TWO FILES ARE EXEMPT, and both by name rather than by pattern, so adding a third is a deliberate
act that shows up in a diff:

  * `entroptics_adapter.py` -- it IS the import;
  * `tests/test_entroptics_adapter.py` -- it imports the library in order to assert that the
    adapter returns the library's own values, which is the comparison that keeps the adapter a
    naming layer rather than a second implementation. That test cannot do its job through the
    thing it is testing.

The scan is over the AST, not a regex over the text. A regex alternation of import spellings always
misses one -- `import entroptics`, `import entroptics as E`, `from entroptics import reads`,
`from entroptics.dynamics import dynamics`, an `importlib.import_module` -- and the one it misses is
the one that gets written. `ast.walk` enumerates every import node the file actually has.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

CODE = Path(__file__).resolve().parent.parent

#: By name, not by pattern. See the module docstring.
EXEMPT = {"entroptics_adapter.py", "test_entroptics_adapter.py"}

#: The adapter itself reaches the library through `importlib.import_module`, so a scan for that
#: call is part of the check rather than an afterthought: it is the one spelling a plain import
#: scan would not see.
_DYNAMIC = "import_module"


def _python_files():
    return sorted(p for p in CODE.rglob("*.py") if p.name not in EXEMPT)


def _library_imports(path: Path):
    """Every import of `entroptics` in one file, as (line, statement) pairs.

    Includes the dynamic form, which is how the adapter itself does it and therefore the spelling
    most likely to be copied into a second file.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "entroptics" or alias.name.startswith("entroptics."):
                    found.append((node.lineno, f"import {alias.name}"))
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod == "entroptics" or mod.startswith("entroptics."):
                names = ", ".join(a.name for a in node.names)
                found.append((node.lineno, f"from {mod} import {names}"))
        elif isinstance(node, ast.Call):
            fn = node.func
            name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
            if name == _DYNAMIC and node.args:
                arg = node.args[0]
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str) \
                        and (arg.value == "entroptics" or arg.value.startswith("entroptics.")):
                    found.append((node.lineno, f"import_module({arg.value!r})"))
    return found


@pytest.mark.parametrize("path", _python_files(), ids=lambda p: str(p.relative_to(CODE)))
def test_no_file_imports_the_instrument_directly(path):
    """Everything reaches Entroptics under the paper's name for the read, through the adapter."""
    hits = _library_imports(path)
    assert not hits, (
        f"{path.relative_to(CODE)} imports the instrument directly:\n"
        + "\n".join(f"  line {ln}: {stmt}" for ln, stmt in hits)
        + "\n\nUse `import entroptics_adapter as EA` and the domain read that names the question."
    )


def test_the_scan_can_detect_a_direct_import():
    """THE NEGATIVE CONTROL. Each import spelling must be seen, or the sweep above proves nothing.

    A scan that finds nothing looks identical whether the tree is clean or the scan is broken. The
    five spellings below are the ones this repository has actually used, including the dynamic form
    the adapter itself needs.
    """
    spellings = [
        "import entroptics",
        "import entroptics as E",
        "from entroptics import reads",
        "from entroptics.dynamics import dynamics",
        "importlib.import_module('entroptics')",
    ]
    for src in spellings:
        tree = ast.parse(src)
        seen = False
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom, ast.Call)):
                seen = True
        assert seen, src

    tmp = CODE / "tests" / "_scan_control.py"
    tmp.write_text("\n".join(spellings[:-1] + ["import importlib",
                                               "importlib.import_module('entroptics')"]),
                   encoding="utf-8")
    try:
        found = _library_imports(tmp)
        assert len(found) == 5, \
            f"the scan saw {len(found)} of 5 import spellings: {found}"
    finally:
        tmp.unlink()


def test_the_adapter_is_the_one_place_that_imports_the_library():
    """And it does import it -- an adapter that reached nothing would pass every test above."""
    hits = _library_imports(CODE / "entroptics_adapter.py")
    assert hits, "entroptics_adapter does not import entroptics at all"
