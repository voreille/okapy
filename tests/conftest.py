import os
from pathlib import Path

import pytest

from tests.helpers.external_data import iter_collections, iter_dro_case_dirs

INTEGRATION_DIR = Path(__file__).parent / "integration"


def _env_dir(name: str) -> Path | None:
    """Return the directory named by env var ``name``, or None if unset/missing."""

    value = os.getenv(name)
    if not value:
        return None

    path = Path(value)
    return path if path.is_dir() else None


def pytest_collection_modifyitems(config, items):
    for item in items:
        if INTEGRATION_DIR in Path(item.fspath).parents:
            item.add_marker(pytest.mark.integration)


@pytest.fixture(scope="session")
def external_data_root() -> Path:
    root = _env_dir("OKAPY_TEST_DATA")
    if root is None:
        pytest.skip("Set OKAPY_TEST_DATA to an existing directory to run these tests.")
    return root


@pytest.fixture(scope="session")
def update_golden() -> bool:
    return os.getenv("UPDATE_GOLDEN") == "1"


@pytest.fixture(scope="session")
def suv_computation_test_data() -> Path:
    root = _env_dir("SUV_COMPUTATION_TEST_DATA")
    if root is None:
        pytest.skip(
            "Set SUV_COMPUTATION_TEST_DATA to an existing directory to run SUV "
            "computation tests."
        )
    return root


def pytest_generate_tests(metafunc):
    if "collection" in metafunc.fixturenames:
        root = _env_dir("OKAPY_TEST_DATA")
        collections = list(iter_collections(root, version="curated-v0")) if root else []
        metafunc.parametrize(
            "collection",
            collections,
            ids=[c["collection_id"] for c in collections],
        )

    if "dro_case_dir" in metafunc.fixturenames:
        root = _env_dir("SUV_COMPUTATION_TEST_DATA")
        case_dirs = iter_dro_case_dirs(root) if root else []
        metafunc.parametrize(
            "dro_case_dir",
            case_dirs,
            ids=[p.name for p in case_dirs],
        )
