"""DICOM SEG conversion onto the reference image grid."""

from __future__ import annotations

import numpy as np
import pydicom
import pytest
import SimpleITK as sitk

from okapy.dicom.conversion.image import SimpleITKImageSeriesConverter
from okapy.dicom.conversion.seg import SegMaskConverter
from tests.helpers.dicom_factories import (
    load_series,
    seg_dataset,
    write_dataset,
    write_image_series,
)

pytest.importorskip("highdicom")

ROWS, COLUMNS = 4, 5
SPACING_Z = 2.0
REFERENCE_Z = [0.0, 2.0, 4.0, 6.0, 8.0]


def _ct(tmp_path, name, z_positions):
    uid, paths = write_image_series(
        tmp_path / name,
        positions=[(0.0, 0.0, z) for z in z_positions],
        values=[1] * len(z_positions),
        rows=ROWS,
        columns=COLUMNS,
        rescale=(1.0, 0.0),
    )
    return uid, paths


def _reference_volume(tmp_path):
    uid, paths = _ct(tmp_path, "reference", REFERENCE_Z)
    volume = SimpleITKImageSeriesConverter().convert(
        load_series(paths), tmp_path / "out" / "images"
    )
    return uid, paths, volume


def _frame_mask() -> np.ndarray:
    frame = np.zeros((ROWS, COLUMNS), dtype=np.uint8)
    frame[1:3, 2:4] = 1
    return frame


def _seg_on(tmp_path, source_paths, slice_index):
    sources = [pydicom.dcmread(str(p)) for p in source_paths]
    mask_zyx = np.zeros((len(sources), ROWS, COLUMNS), dtype=np.uint8)
    mask_zyx[slice_index] = _frame_mask()
    seg = seg_dataset(source_images=sources, mask_zyx=mask_zyx, label="Tumor")
    path = write_dataset(seg, tmp_path / "seg" / "seg.dcm")
    return load_series([path])


def _mask_zyx(mask) -> np.ndarray:
    return sitk.GetArrayFromImage(mask.image)


def test_seg_on_the_reference_grid_lands_on_its_slice(tmp_path):
    _, ref_paths, reference = _reference_volume(tmp_path)
    seg_series = _seg_on(tmp_path, ref_paths, slice_index=2)

    masks = SegMaskConverter().convert(
        series=seg_series,
        reference_image=reference,
        output_dir=tmp_path / "out" / "masks",
    )

    assert [m.label for m in masks] == ["Tumor"]
    array = _mask_zyx(masks[0])
    assert array.shape == (len(REFERENCE_Z), ROWS, COLUMNS)
    assert np.nonzero(array.any(axis=(1, 2)))[0].tolist() == [2]
    np.testing.assert_array_equal(array[2], _frame_mask())


def test_seg_frames_between_reference_slices_are_resampled(tmp_path):
    # Same in-plane grid, but the SEG source series is shifted 0.6 mm in z
    # (0.3 slices, above the slice-matching tolerance): its frames lie between
    # reference slices. Nearest-neighbour resampling puts SEG frame k on
    # reference slice k.
    _, _, reference = _reference_volume(tmp_path)
    _, shifted_paths = _ct(tmp_path, "shifted", [z + 0.6 for z in REFERENCE_Z])
    seg_series = _seg_on(tmp_path, shifted_paths, slice_index=3)

    masks = SegMaskConverter(check_reference_uid=False).convert(
        series=seg_series,
        reference_image=reference,
        output_dir=tmp_path / "out" / "masks",
    )

    assert len(masks) == 1
    mask = masks[0]
    assert mask.image.GetSize() == reference.image.GetSize()
    np.testing.assert_allclose(mask.image.GetOrigin(), reference.image.GetOrigin())
    np.testing.assert_allclose(mask.image.GetSpacing(), reference.image.GetSpacing())
    array = _mask_zyx(mask)
    assert np.nonzero(array.any(axis=(1, 2)))[0].tolist() == [3]
    np.testing.assert_array_equal(array[3], _frame_mask())
