"""Packaging metadata: package discovery and dependency hygiene."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
PYPROJECT = ROOT / "pyproject.toml"
SRC = ROOT / "src"


def _toml():
    try:
        import tomllib
    except ModuleNotFoundError:  # Python < 3.11
        tomllib = pytest.importorskip("tomli")
    return tomllib


@pytest.fixture(scope="module")
def pyproject():
    return _toml().loads(PYPROJECT.read_text())


def _requirement_name(requirement: str) -> str:
    return re.split(r"[\s<>=!~;\[]", requirement.strip(), maxsplit=1)[0].lower()


@pytest.fixture(scope="module")
def selected_packages(pyproject):
    setuptools = pytest.importorskip("setuptools")
    find = pyproject["tool"]["setuptools"]["packages"]["find"]
    where = find.get("where", ["."])
    assert len(where) == 1
    return set(
        setuptools.find_packages(
            where=str(ROOT / where[0]),
            include=find.get("include", ["*"]),
            exclude=find.get("exclude", []),
        )
    )


def test_package_discovery_includes_dicom_conversion(selected_packages):
    assert {"okapy", "okapy.dicom", "okapy.dicom.conversion"} <= selected_packages


def test_package_discovery_includes_every_okapy_subpackage(selected_packages):
    on_disk = {
        ".".join(init.parent.relative_to(SRC).parts)
        for init in (SRC / "okapy").rglob("__init__.py")
    }

    assert on_disk <= selected_packages


def test_package_discovery_excludes_legacy_packages(selected_packages):
    assert not [p for p in selected_packages if p.startswith("okapy_legacy")]
    assert not [p for p in selected_packages if p.startswith("tests")]


def test_scipy_is_not_a_core_dependency(pyproject):
    names = {_requirement_name(r) for r in pyproject["project"]["dependencies"]}

    assert "scipy" not in names


def test_pyarrow_is_in_dev_extra(pyproject):
    dev = pyproject["project"]["optional-dependencies"]["dev"]

    assert "pyarrow" in {_requirement_name(r) for r in dev}


def _imported_modules(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            yield node.module


def test_okapy_does_not_import_scipy():
    offenders = [
        f"{path.relative_to(ROOT)}: {module}"
        for path in sorted((SRC / "okapy").rglob("*.py"))
        for module in _imported_modules(path)
        if module == "scipy" or module.startswith("scipy.")
    ]

    assert offenders == []


def test_okapy_does_not_import_legacy_package():
    offenders = [
        f"{path.relative_to(ROOT)}: {module}"
        for path in sorted((SRC / "okapy").rglob("*.py"))
        for module in _imported_modules(path)
        if module == "okapy_legacy" or module.startswith("okapy_legacy.")
    ]

    assert offenders == []
