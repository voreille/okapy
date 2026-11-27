"""Enhanced PET (multi-frame) conversion geometry."""

from __future__ import annotations

import re

import numpy as np
import pytest
import SimpleITK as sitk

from okapy.dicom.conversion.enhanced_pet import EnhancedPETSUVConverter
from okapy.dicom.conversion.suv import PETSUVConverter
from okapy.dicom.conversion.suv_common import SUVComputationError
from tests.helpers.dicom_factories import (
    enhanced_pet_dataset,
    load_series,
    write_dataset,
)


def _series(tmp_path, **kwargs):
    ds = enhanced_pet_dataset(**kwargs)
    path = write_dataset(ds, tmp_path / "dicom" / "enhanced_pet.dcm")
    return ds.SeriesInstanceUID, load_series([path])


def _slice_values(volume) -> list[float]:
    return sitk.GetArrayFromImage(volume.image)[:, 0, 0].tolist()


def test_multi_frame_enhanced_pet_is_sorted_with_correct_geometry(tmp_path):
    # Frames stored out of z order.
    _, series = _series(
        tmp_path,
        frame_positions=[(-10.0, -20.0, 6.0), (-10.0, -20.0, 0.0), (-10.0, -20.0, 3.0)],
        frame_values=[7, 5, 6],
    )

    volume = EnhancedPETSUVConverter().convert(series, tmp_path / "out")

    image = volume.image
    assert image.GetSize() == (5, 4, 3)
    np.testing.assert_allclose(image.GetSpacing(), (2.0, 2.0, 3.0))
    np.testing.assert_allclose(image.GetOrigin(), (-10.0, -20.0, 0.0))
    assert _slice_values(volume) == [5.0, 6.0, 7.0]


@pytest.mark.parametrize(
    "converter_cls",
    [EnhancedPETSUVConverter, PETSUVConverter],
    ids=["enhanced-converter", "pet-converter-dispatch"],
)
def test_single_frame_enhanced_pet_converts(tmp_path, converter_cls):
    _, series = _series(
        tmp_path,
        frame_positions=[(1.0, 2.0, 30.0)],
        frame_values=[9],
        slice_thickness=3.0,
    )

    volume = converter_cls().convert(series, tmp_path / "out")

    image = volume.image
    assert image.GetSize() == (5, 4, 1)
    np.testing.assert_array_equal(sitk.GetArrayFromImage(image), np.full((1, 4, 5), 9))
    np.testing.assert_allclose(image.GetOrigin(), (1.0, 2.0, 30.0))
    np.testing.assert_allclose(image.GetSpacing(), (2.0, 2.0, 3.0))


def test_enhanced_pet_with_duplicate_frame_positions_raises(tmp_path):
    uid, series = _series(
        tmp_path,
        frame_positions=[(0.0, 0.0, 0.0), (0.0, 0.0, 3.0), (0.0, 0.0, 3.0)],
        frame_values=[1, 2, 3],
    )

    with pytest.raises((ValueError, SUVComputationError)) as excinfo:
        EnhancedPETSUVConverter().convert(series, tmp_path / "out")

    assert uid in str(excinfo.value)
    assert re.search(r"duplicate|same position|share", str(excinfo.value), re.I)


@pytest.mark.parametrize(
    "z_positions",
    [
        pytest.param((0.0, 3.0, 7.0, 10.0), id="one-gap-too-large"),
        pytest.param((0.0, 3.0, 9.0, 12.0), id="missing-frame"),
    ],
)
def test_enhanced_pet_with_non_uniform_frame_spacing_raises(tmp_path, z_positions):
    uid, series = _series(
        tmp_path,
        frame_positions=[(0.0, 0.0, z) for z in z_positions],
        frame_values=list(range(1, len(z_positions) + 1)),
    )

    with pytest.raises((ValueError, SUVComputationError)) as excinfo:
        EnhancedPETSUVConverter().convert(series, tmp_path / "out")

    assert uid in str(excinfo.value)
