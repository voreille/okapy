from __future__ import annotations

from pathlib import Path

import pytest

from okapy.dicom.collector import DicomStudyCollector
from okapy.dicom.conversion.suv import PETSUVConverter, SUVComputationError
from tests.helpers.suv_assertions import (
    assert_dro_suv_stats,
    compute_roi_suv_stats,
    dro_expects_error,
    find_roi_mask,
)


def test_new_suv_conversion_matches_dro_values(
    dro_case_dir: Path,
    tmp_path: Path,
):
    if dro_expects_error(dro_case_dir):
        with pytest.raises(SUVComputationError):
            _convert_pet_images(dro_case_dir, tmp_path / "nifti")
        return

    converted_pet_images = _convert_pet_images(dro_case_dir, tmp_path / "nifti")

    assert len(converted_pet_images) == 1, (
        f"Expected exactly one PET image in {dro_case_dir}, "
        f"got {len(converted_pet_images)}."
    )

    mask_path = find_roi_mask(dro_case_dir)
    image_path = converted_pet_images[0].path

    stats = compute_roi_suv_stats(
        image_path=image_path,
        mask_path=mask_path,
    )

    assert_dro_suv_stats(stats)


def _convert_pet_images(case_dir: Path, output_dir: Path):
    collector = DicomStudyCollector()
    collection = collector.collect(case_dir)

    converter = PETSUVConverter()

    return [
        converter.convert(series=series, output_dir=output_dir)
        for study in collection.studies
        for series in study.image_series
        if series.modality == "PT"
    ]
