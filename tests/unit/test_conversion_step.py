"""DicomConversionStep end-to-end on synthetic studies."""

from __future__ import annotations

import re

import numpy as np
import pydicom
import pytest
import SimpleITK as sitk

from okapy.dicom.conversion.builder import build_dicom_conversion_step
from tests.helpers.dicom_factories import (
    DEFAULT_STUDY_UID,
    rtstruct_dataset,
    square_contour,
    write_dataset,
    write_image_series,
)

ROWS, COLUMNS = 10, 12
ORIGIN = (-20.0, -10.0, 50.0)
PIXEL_SPACING = (1.5, 0.75)  # (row, column) = (y, x)
SLICE_SPACING = 2.0


def _step(continue_on_error: bool):
    return build_dicom_conversion_step(
        {"conversion": {"continue_on_error": continue_on_error}}
    )


def _ct(directory, *, positions=None, values=None, **kwargs):
    if positions is None:
        positions = [
            (ORIGIN[0], ORIGIN[1], ORIGIN[2] + SLICE_SPACING * k) for k in range(4)
        ]
    if values is None:
        values = [100 + k for k in range(len(positions))]
    return write_image_series(
        directory,
        positions=positions,
        values=values,
        rows=ROWS,
        columns=COLUMNS,
        pixel_spacing=PIXEL_SPACING,
        rescale=(1.0, -1024.0),
        **kwargs,
    )


def _gtv_contour():
    # Square spanning x in [-17, -13] mm, y in [-7, 2] mm on slice 1.
    return square_contour(-17.0, -7.0, -13.0, 2.0, ORIGIN[2] + SLICE_SPACING)


def _rtstruct(directory, referenced_series_uid, name="rtstruct.dcm"):
    ds = rtstruct_dataset(
        rois=[(1, "GTV", [_gtv_contour()])],
        referenced_series_uid=referenced_series_uid,
    )
    write_dataset(ds, directory / name)
    return ds.SeriesInstanceUID


def _images_by_uid(collection):
    return {image.series_instance_uid: image for image in collection.images}


# --------------------------------------------------------------------------
# Smoke / regression: CT + RTSTRUCT end to end
# --------------------------------------------------------------------------


def test_ct_with_rtstruct_converts_end_to_end_to_nifti(tmp_path):
    ct_uid, _ = _ct(tmp_path / "in" / "ct")
    _rtstruct(tmp_path / "in" / "rt", ct_uid)

    collection = _step(False).run(tmp_path / "in", output_dir=tmp_path / "out")

    assert [s.study_instance_uid for s in collection.studies] == [DEFAULT_STUDY_UID]
    (image_mask_set,) = collection.studies[0].image_mask_sets
    image = image_mask_set.image
    assert image.series_instance_uid == ct_uid
    assert image.path.exists()
    assert str(image.path).endswith(".nii.gz")

    on_disk = sitk.ReadImage(str(image.path))
    assert on_disk.GetSize() == (COLUMNS, ROWS, 4)
    np.testing.assert_allclose(on_disk.GetSpacing(), (0.75, 1.5, SLICE_SPACING))
    np.testing.assert_allclose(on_disk.GetOrigin(), ORIGIN)
    np.testing.assert_allclose(
        sitk.GetArrayFromImage(on_disk)[:, 0, 0], [100 - 1024 + k for k in range(4)]
    )

    (mask,) = image_mask_set.masks
    assert mask.label == "GTV"
    assert mask.reference_series_instance_uid == ct_uid
    mask_on_disk = sitk.ReadImage(str(mask.path))
    assert mask_on_disk.GetSize() == (COLUMNS, ROWS, 4)
    np.testing.assert_allclose(mask_on_disk.GetOrigin(), ORIGIN)
    mask_zyx = sitk.GetArrayFromImage(mask_on_disk)
    assert np.nonzero(mask_zyx.any(axis=(1, 2)))[0].tolist() == [1]
    # x index (-15 - -20) / 0.75 ~ 6.7, y index (-2 - -10) / 1.5 ~ 5.3:
    # the centre of the square is inside the mask.
    assert mask_zyx[1, 5, 6] == 1
    assert mask_zyx[1, 0, 0] == 0


# --------------------------------------------------------------------------
# Bad image series: error naming the series, respecting continue_on_error
# --------------------------------------------------------------------------


def _duplicate_positions_series(directory):
    positions = [(0.0, 0.0, z) for z in (0.0, 2.0, 2.0, 4.0)]
    return _ct(directory, positions=positions)


def _unsupported_dx_series(directory):
    uid, paths = write_image_series(
        directory, modality="DX", positions=[0.0], values=[5]
    )
    for path in paths:
        ds = pydicom.dcmread(str(path))
        del ds.ImagePositionPatient
        del ds.ImageOrientationPatient
        del ds.PixelSpacing
        ds.save_as(str(path), enforce_file_format=True)
    return uid, paths


def _mixed_modality_series(directory):
    uid, paths = _ct(directory)
    ds = pydicom.dcmread(str(paths[-1]))
    ds.Modality = "MR"
    ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.4"
    ds.save_as(str(paths[-1]), enforce_file_format=True)
    return uid, paths


BAD_SERIES = [
    pytest.param(_duplicate_positions_series, r"", id="duplicate-positions"),
    pytest.param(_unsupported_dx_series, r"(?i)unsupported", id="unsupported-dx"),
    pytest.param(_mixed_modality_series, r"(?i)modalit", id="mixed-modalities"),
]


@pytest.mark.parametrize(("make_bad", "keyword"), BAD_SERIES)
def test_bad_series_raises_value_error_naming_it(tmp_path, make_bad, keyword):
    _ct(tmp_path / "in" / "good")
    bad_uid, _ = make_bad(tmp_path / "in" / "bad")

    with pytest.raises(ValueError, match=re.escape(bad_uid)) as excinfo:
        _step(False).run(tmp_path / "in", output_dir=tmp_path / "out")

    assert re.search(keyword, str(excinfo.value))


@pytest.mark.parametrize(("make_bad", "keyword"), BAD_SERIES)
def test_bad_series_is_skipped_with_continue_on_error(tmp_path, make_bad, keyword):
    good_uid, _ = _ct(tmp_path / "in" / "good")
    bad_uid, _ = make_bad(tmp_path / "in" / "bad")

    collection = _step(True).run(tmp_path / "in", output_dir=tmp_path / "out")

    images = _images_by_uid(collection)
    assert set(images) == {good_uid}
    assert images[good_uid].image.GetSize() == (COLUMNS, ROWS, 4)


# --------------------------------------------------------------------------
# Mask reference resolution
# --------------------------------------------------------------------------


def _failing_ct_series(directory):
    # Two separate discontinuities: rejected by the converter.
    positions = [(ORIGIN[0], ORIGIN[1], ORIGIN[2] + z) for z in (0, 2, 6, 8, 12, 14)]
    return _ct(directory, positions=positions)


def test_mask_referencing_a_failed_image_is_not_put_on_another_image(tmp_path):
    good_uid, _ = _ct(tmp_path / "in" / "good")
    failed_uid, _ = _failing_ct_series(tmp_path / "in" / "failed")
    _rtstruct(tmp_path / "in" / "rt", failed_uid)

    collection = _step(True).run(tmp_path / "in", output_dir=tmp_path / "out")

    assert set(_images_by_uid(collection)) == {good_uid}
    assert collection.masks == []


def test_mask_referencing_a_missing_image_is_not_put_on_another_image(tmp_path):
    good_uid, _ = _ct(tmp_path / "in" / "good")
    _rtstruct(tmp_path / "in" / "rt", "1.2.826.0.1.3680043.8.498.999999")

    collection = _step(True).run(tmp_path / "in", output_dir=tmp_path / "out")

    assert set(_images_by_uid(collection)) == {good_uid}
    assert collection.masks == []


def test_mask_referencing_a_missing_image_raises_without_continue_on_error(
    tmp_path,
):
    _ct(tmp_path / "in" / "good")
    rt_uid = _rtstruct(tmp_path / "in" / "rt", "1.2.826.0.1.3680043.8.498.999999")

    with pytest.raises((RuntimeError, ValueError), match=re.escape(rt_uid)):
        _step(False).run(tmp_path / "in", output_dir=tmp_path / "out")


def test_mask_without_reference_uids_falls_back_to_the_only_image(tmp_path):
    good_uid, _ = _ct(tmp_path / "in" / "good")
    _rtstruct(tmp_path / "in" / "rt", None)

    collection = _step(False).run(tmp_path / "in", output_dir=tmp_path / "out")

    (mask,) = collection.masks
    assert mask.label == "GTV"
    assert mask.reference_series_instance_uid == good_uid
