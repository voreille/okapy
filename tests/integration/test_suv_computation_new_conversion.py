from pathlib import Path

import pytest

from okapy.dicom.collector import DicomStudyCollector
from okapy.dicom.conversion.image import SimpleITKImageSeriesConverter
from okapy.dicom.conversion.suv import PETSUVConverter

from tests.helpers.image_snapshots import converted_images_summary
from tests.helpers.assertions import assert_or_update_json
from tests.helpers.suv_assertions import dro_expects_error


def test_suv_computation_new_conversion_snapshot(
    suv_computation_test_data: Path,
    tmp_path: Path,
    update_golden: bool,
):
    output_dir = tmp_path / "nifti"

    collector = DicomStudyCollector()

    image_converter = SimpleITKImageSeriesConverter()
    pet_converter = PETSUVConverter()

    converted_images = []

    # The DROs whose expected outcome is an error are covered by
    # test_suv_dro_values.py; they have no values to snapshot.
    case_dirs = [
        case_dir
        for case_dir in sorted(suv_computation_test_data.iterdir())
        if case_dir.is_dir() and not dro_expects_error(case_dir)
    ]

    for case_dir in case_dirs:
        collection = collector.collect(case_dir)

        for study in collection.studies:
            for series in study.image_series:
                converter = (
                    pet_converter if series.modality == "PT" else image_converter
                )

                converted_images.append(
                    converter.convert(
                        series=series,
                        output_dir=output_dir,
                    )
                )

    actual = converted_images_summary(converted_images)

    golden_path = Path(
        "tests/data/golden/suv_computation_dro/new_conversion_summary.json"
    )

    assert_or_update_json(
        actual=actual,
        golden_path=golden_path,
        update=update_golden,
    )