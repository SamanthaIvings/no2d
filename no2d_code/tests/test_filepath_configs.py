from __future__ import annotations

import importlib
import re
from pathlib import Path

import pytest

import no2d_code

PKG_DIR = Path(no2d_code.__file__).resolve().parent
REPO_ROOT = PKG_DIR.parent


EXCLUDE_FROM_IMPORT = {
}

OPTIONAL_DEPENDENCIES = {"geopandas", "osmnx", "contextily"}

_HARDCODED_DATA = re.compile(r"""['"]\.\./\.\./data""")


def _iter_py_files():
    for p in PKG_DIR.rglob("*.py"):
        if "__pycache__" in p.parts:
            continue
        if "tests" in p.relative_to(PKG_DIR).parts:
            continue
        yield p


def _module_name(path: Path) -> str:
    parts = list(path.relative_to(REPO_ROOT).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


_PY_FILES = sorted(_iter_py_files(), key=str)
_MODULES = sorted({_module_name(p) for p in _PY_FILES if _module_name(p)})


@pytest.mark.parametrize("modname", _MODULES)
def test_module_imports(modname):
    if modname in EXCLUDE_FROM_IMPORT:
        pytest.skip("executes work at import time; excluded from smoke import")
    try:
        importlib.import_module(modname)
    except ModuleNotFoundError as exc:
        top = (exc.name or "").split(".")[0]
        if top in OPTIONAL_DEPENDENCIES:
            pytest.skip(f"optional dependency '{top}' not installed")
        raise


def test_discovery_found_expected_packages():
    assert any(m.startswith("no2d_code.core") for m in _MODULES)
    assert any(m.startswith("no2d_code.experiments") for m in _MODULES)
    assert any(m.startswith("no2d_code.solver") for m in _MODULES)


@pytest.mark.parametrize("pyfile", _PY_FILES, ids=[str(p.relative_to(REPO_ROOT)) for p in _PY_FILES])
def test_no_hardcoded_data_path(pyfile):
    offending = [
        f"L{i}: {line.strip()}"
        for i, line in enumerate(pyfile.read_text(encoding="utf-8").splitlines(), 1)
        if _HARDCODED_DATA.search(line)
    ]
    assert not offending, (
        f"{pyfile.relative_to(REPO_ROOT)} reintroduced a hardcoded data path; "
        f"use fc.DATA_DIR / fc.data_path(...) instead:\n  " + "\n  ".join(offending)
    )
