from pathlib import Path

import pytest

pytest.importorskip("radiomics.featureextractor", exc_type=ImportError)

from okapy_legacy.dicomconverter.converter import ExtractorConverter
from tests.helpers.assertions import assert_or_update_features


def test_suv_computation_legacy_snapshot(
    suv_computation_test_data: Path,
    tmp_path: Path,
    update_golden: bool,
):
    output_dir = tmp_path / "nifti"

    tests_dir = Path(__file__).parents[1]
    params_path = tests_dir / "configs/legacy_suv_features.yaml"

    converter = ExtractorConverter.from_params(params_path)

    features = converter(
        input_folder=suv_computation_test_data,
        output_folder=output_dir,
    )

    golden_path = tests_dir / "data/golden/suv_computation_dro/legacy_features.parquet"

    assert_or_update_features(
        actual=features,
        golden_path=golden_path,
        update=update_golden,
    )
