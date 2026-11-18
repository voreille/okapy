import pytest

from okapy_legacy.dicomconverter.converter import ExtractorConverter

from tests.helpers.assertions import assert_or_update_features
from tests.helpers.external_data import iter_collections


def test_legacy_feature_snapshot(collection, tmp_path, update_golden):
    if not collection["checks"].get("features", False):
        pytest.skip("Feature check disabled for this collection.")

    converter = ExtractorConverter.from_params(collection["legacy_params_path"])

    features = converter(
        input_folder=collection["dicom_dir"],
        output_folder=tmp_path / "nifti",
        labels=collection.get("labels_to_extract", None),
    )

    golden_path = collection["golden_dir"] / "legacy_features.parquet"

    assert_or_update_features(
        actual=features,
        golden_path=golden_path,
        update=update_golden,
    )
