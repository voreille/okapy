import pytest

from tests.helpers.external_data import iter_collections
from tests.helpers.dicom_snapshots import image_summary, mask_summary
from tests.helpers.assertions import assert_or_update_json

# from okapy.pipelines.dicom_to_nifti import DicomToNiftiPipeline


def pytest_generate_tests(metafunc):
    if "collection" not in metafunc.fixturenames:
        return

    # Cannot access fixtures here, so use env directly.
    import os
    from pathlib import Path

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


def test_curated_collection_conversion(collection, tmp_path, update_golden):
    if not collection["checks"].get("conversion", False):
        pytest.skip("Conversion check disabled for this collection.")

    output_dir = tmp_path / "output"

    # pipeline = DicomToNiftiPipeline()
    # result = pipeline.run(
    #     input_dir=collection["dicom_dir"],
    #     output_dir=output_dir,
    # )

    # actual_image_summary = image_summary(result.image_paths[0])
    # assert_or_update_json(
    #     actual_image_summary,
    #     collection["golden_dir"] / "image_summary.json",
    #     update=update_golden,
    # )

    # for i, mask_path in enumerate(result.mask_paths):
    #     actual_mask_summary = mask_summary(mask_path)
    #     assert_or_update_json(
    #         actual_mask_summary,
    #         collection["golden_dir"] / f"mask_summary_{i}.json",
    #         update=update_golden,
    #     )