import os
from pathlib import Path

import pytest

from tests.helpers.external_data import iter_collections


@pytest.fixture(scope="session")
def external_data_root() -> Path:
    root = os.getenv("OKAPY_TEST_DATA")
    if root is None:
        pytest.skip("Set OKAPY_TEST_DATA to run external DICOM integration tests.")
    return Path(root)


@pytest.fixture(scope="session")
def update_golden() -> bool:
    return os.getenv("UPDATE_GOLDEN") == "1"


def pytest_generate_tests(metafunc):
    if "collection" not in metafunc.fixturenames:
        return

    root = os.getenv("OKAPY_TEST_DATA")
    if root is None:
        metafunc.parametrize("collection", [])
        return

    collections = list(iter_collections(Path(root), version="curated-v0"))

    metafunc.parametrize(
        "collection",
        collections,
        ids=[c["collection_id"] for c in collections],
    )


@pytest.fixture(scope="session")
def suv_computation_test_data() -> Path:
    root = os.getenv("SUV_COMPUTATION_TEST_DATA")
    if root is None:
        pytest.skip("Set SUV_COMPUTATION_TEST_DATA to run SUV computation tests.")

    path = Path(root)
    if not path.exists():
        pytest.skip(f"SUV_COMPUTATION_TEST_DATA does not exist: {path}")

    return path
