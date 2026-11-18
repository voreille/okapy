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


def pytest_generate_tests(metafunc):
    if "dro_case_dir" not in metafunc.fixturenames:
        return

    import os

    root = os.getenv("SUV_COMPUTATION_TEST_DATA")
    if root is None:
        metafunc.parametrize("dro_case_dir", [])
        return

    case_dirs = _iter_dro_case_dirs(Path(root))

    metafunc.parametrize(
        "dro_case_dir",
        case_dirs,
        ids=[p.name for p in case_dirs],
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
    print(f"\nDRO case: {dro_case_dir}")
    print(f"Image path: {image_path}")
    print(f"Mask path: {mask_path}")
    print(f"Stats: {stats}")

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


def _iter_dro_case_dirs(root: Path) -> list[Path]:
    """Return individual DRO case directories.

    Expected layout:

    DRO/
      DRO_0_0/
      DRO_1_0/
      DRO_2_0/
      ...

    The root itself should not be treated as one case if it contains subdirectories.
    """

    subdirs = [p for p in sorted(root.iterdir()) if p.is_dir()]

    case_dirs = [
        p for p in subdirs
        if _looks_like_dro_case_dir(p)
    ]

    if case_dirs:
        return case_dirs

    # Fallback: only treat root as one case if no subdirectory case was found.
    if _looks_like_dro_case_dir(root):
        return [root]

    raise RuntimeError(f"No DRO case directories found in {root}")


def _looks_like_dro_case_dir(path: Path) -> bool:
    has_nifti_mask = (
        any(path.rglob("*.nii")) or
        any(path.rglob("*.nii.gz"))
    )

    # At least one DICOM-like file readable by the collector.
    # Keep this simple: the collector will do the real validation.
    has_files = any(p.is_file() for p in path.rglob("*"))

    return has_nifti_mask and has_files