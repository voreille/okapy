"""YAML files shipped under ``parameters/``.

- Files under a ``pyradiomics`` directory are pyradiomics parameter files or
  fragments, not Okapy configs.
- A few obsolete files must have been deleted.
- Every other YAML is an Okapy config (new shape, or legacy and migratable)
  that builds.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
import yaml

from okapy.config import load_config
from okapy.pipeline import ExtractionPipeline, build_extraction_pipeline

PARAMETERS_DIR = Path(__file__).parents[2] / "parameters"
SHIPPED_YAMLS = sorted(PARAMETERS_DIR.rglob("*.yaml"))
UNSUPPORTED_EXTRACTORS = {"riesz", "zrad"}
KNOWN_BACKEND_TYPES = {"pyradiomics_local", "command", "pet"}
RESERVED_FEATURE_KEYS = {
    "common",
    "continue_on_error",
    "result_format",
    "include_backend",
}
PYRADIOMICS_TOP_LEVEL_KEYS = {"imageType", "featureClass", "setting", "voxelSetting"}

DELETED_FILES = [
    PARAMETERS_DIR / "param.yaml",
    PARAMETERS_DIR / "pyradiomics" / "param.yaml",
    PARAMETERS_DIR / "defaults" / "defaults_zrad.yaml",
]


def _is_pyradiomics_file(path: Path) -> bool:
    return "pyradiomics" in path.relative_to(PARAMETERS_DIR).parts[:-1]


PYRADIOMICS_YAMLS = [p for p in SHIPPED_YAMLS if _is_pyradiomics_file(p)]
OKAPY_YAMLS = [
    p
    for p in SHIPPED_YAMLS
    if not _is_pyradiomics_file(p) and p not in set(DELETED_FILES)
]


def _ids(paths):
    return [str(p.relative_to(PARAMETERS_DIR)) for p in paths]


class _FakeRadiomicsFeatureExtractor:
    def __init__(self, params=None):
        self.params = params


@pytest.fixture
def fake_pyradiomics(monkeypatch):
    """Stand-in for the external pyradiomics package (broken under NumPy 2)."""

    package = types.ModuleType("radiomics")
    module = types.ModuleType("radiomics.featureextractor")
    module.RadiomicsFeatureExtractor = _FakeRadiomicsFeatureExtractor
    package.featureextractor = module
    monkeypatch.setitem(sys.modules, "radiomics", package)
    monkeypatch.setitem(sys.modules, "radiomics.featureextractor", module)


def _walk_tokens(node):
    if isinstance(node, dict):
        for key, value in node.items():
            yield str(key)
            yield from _walk_tokens(value)
    elif isinstance(node, list):
        for value in node:
            yield from _walk_tokens(value)
    elif isinstance(node, str):
        yield node


def _assert_no_unsupported_extractors(raw):
    tokens = {token.lower() for token in _walk_tokens(raw)}
    assert not tokens & UNSUPPORTED_EXTRACTORS


# ---------------------------------------------------------------------------
# Inventory
# ---------------------------------------------------------------------------


def test_parameters_directory_ships_yaml_files():
    assert SHIPPED_YAMLS, f"no YAML found under {PARAMETERS_DIR}"


def test_parameters_directory_ships_okapy_configs():
    assert OKAPY_YAMLS, f"no Okapy config YAML found under {PARAMETERS_DIR}"


def test_obsolete_parameter_files_are_deleted():
    remaining = [
        str(p.relative_to(PARAMETERS_DIR)) for p in DELETED_FILES if p.exists()
    ]

    assert remaining == []


# ---------------------------------------------------------------------------
# pyradiomics parameter files and fragments
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", PYRADIOMICS_YAMLS, ids=_ids(PYRADIOMICS_YAMLS))
def test_pyradiomics_yaml_is_a_non_empty_mapping(path):
    raw = yaml.safe_load(path.read_text())

    assert isinstance(raw, dict)
    assert raw


@pytest.mark.parametrize("path", PYRADIOMICS_YAMLS, ids=_ids(PYRADIOMICS_YAMLS))
def test_pyradiomics_yaml_has_no_unsupported_extractors(path):
    _assert_no_unsupported_extractors(yaml.safe_load(path.read_text()))


@pytest.mark.parametrize("path", PYRADIOMICS_YAMLS, ids=_ids(PYRADIOMICS_YAMLS))
def test_full_pyradiomics_param_file_uses_only_pyradiomics_sections(path):
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict) or not set(raw) & PYRADIOMICS_TOP_LEVEL_KEYS:
        pytest.skip("fragment, not a full pyradiomics parameter file")

    assert set(raw) <= PYRADIOMICS_TOP_LEVEL_KEYS


# ---------------------------------------------------------------------------
# Okapy configs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", OKAPY_YAMLS, ids=_ids(OKAPY_YAMLS))
def test_okapy_yaml_has_no_unsupported_extractors(path):
    _assert_no_unsupported_extractors(yaml.safe_load(path.read_text()))


@pytest.mark.parametrize("path", OKAPY_YAMLS, ids=_ids(OKAPY_YAMLS))
def test_okapy_yaml_loads_as_new_shape_config(path):
    from okapy.config import is_legacy_config

    config = load_config(path)

    assert not is_legacy_config(config)
    assert "volume_preprocessing" not in config
    features = config.get("feature_extraction") or {}
    for selector, backends in features.items():
        if selector in RESERVED_FEATURE_KEYS - {"common"}:
            continue
        for backend in backends or []:
            assert isinstance(backend, dict), (selector, backend)
            assert backend.get("type") in KNOWN_BACKEND_TYPES, (selector, backend)


@pytest.mark.parametrize("path", OKAPY_YAMLS, ids=_ids(OKAPY_YAMLS))
def test_okapy_yaml_builds_extraction_pipeline(path, fake_pyradiomics):
    pipeline = build_extraction_pipeline(path)

    assert isinstance(pipeline, ExtractionPipeline)


@pytest.mark.parametrize("path", OKAPY_YAMLS, ids=_ids(OKAPY_YAMLS))
def test_okapy_yaml_builds_preprocessing_pipeline(path):
    from okapy.pipeline import build_preprocessing_pipeline

    pipeline = build_preprocessing_pipeline(path)

    assert pipeline is not None
