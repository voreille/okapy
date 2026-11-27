"""RTSTRUCT rasterization onto the reference image grid."""

from __future__ import annotations

import logging
import re

import numpy as np
import SimpleITK as sitk

from okapy.dicom.conversion.rtstruct import RTStructMaskConverter
from tests.helpers.dicom_factories import (
    load_series,
    rtstruct_dataset,
    square_contour,
    write_dataset,
)
from tests.helpers.factories import image_volume

SIZE_XYZ = (12, 12, 3)
REFERENCE_UID = "1.2.826.0.1.3680043.8.498.100"
Z = 1.0  # physical z of slice index 1


def _reference():
    image = sitk.Image(SIZE_XYZ, sitk.sitkFloat32)
    image.SetOrigin((0.0, 0.0, 0.0))
    image.SetSpacing((1.0, 1.0, 1.0))
    return image_volume(image, series_instance_uid=REFERENCE_UID)


def _convert(tmp_path, contours, *, label="GTV"):
    ds = rtstruct_dataset(
        rois=[(1, label, contours)], referenced_series_uid=REFERENCE_UID
    )
    path = write_dataset(ds, tmp_path / "dicom" / "rtstruct.dcm")
    masks = RTStructMaskConverter().convert(
        series=load_series([path]),
        reference_image=_reference(),
        output_dir=tmp_path / "out",
    )
    assert len(masks) == 1
    mask = masks[0]
    assert mask.image.GetSize() == SIZE_XYZ
    # x, y, z indexing
    return mask, np.transpose(sitk.GetArrayFromImage(mask.image), (2, 1, 0))


def test_single_contour_is_filled_on_its_slice_only(tmp_path):
    mask, xyz = _convert(tmp_path, [square_contour(2, 3, 6, 8, Z)])

    assert mask.label == "GTV"
    assert mask.reference_series_instance_uid == REFERENCE_UID
    assert np.nonzero(xyz.any(axis=(0, 1)))[0].tolist() == [1]
    assert xyz[2:7, 3:9, 1].all()
    assert not xyz[7:, :, 1].any()
    assert not xyz[:, :3, 1].any()
    assert mask.path.exists()


def test_inner_contour_on_same_slice_produces_a_hole(tmp_path):
    outer = square_contour(1, 1, 10, 10, Z)
    inner = square_contour(4, 4, 7, 7, Z)

    _, xyz = _convert(tmp_path, [outer, inner])

    plane = xyz[:, :, 1]
    # Inside the inner contour: hole.
    assert not plane[5:7, 5:7].any()
    # The ring between the two contours stays filled.
    assert plane[1:3, 1:11].all()
    assert plane[9:11, 1:11].all()
    assert plane[1:11, 1:3].all()
    assert plane[1:11, 9:11].all()
    # Outside the outer contour: empty.
    assert not plane[0, :].any()
    assert not plane[11, :].any()


def test_hole_does_not_depend_on_contour_order(tmp_path):
    outer = square_contour(1, 1, 10, 10, Z)
    inner = square_contour(4, 4, 7, 7, Z)

    _, xyz = _convert(tmp_path, [inner, outer])

    plane = xyz[:, :, 1]
    assert not plane[5:7, 5:7].any()
    assert plane[1:3, 1:11].all()


def test_separate_contours_on_same_slice_are_both_filled(tmp_path):
    left = square_contour(1, 1, 4, 4, Z)
    right = square_contour(7, 7, 10, 10, Z)

    _, xyz = _convert(tmp_path, [left, right])

    plane = xyz[:, :, 1]
    assert plane[1:5, 1:5].all()
    assert plane[7:11, 7:11].all()
    assert not plane[5:7, :].any()


def _assert_clip_warning(caplog):
    warnings = [
        r
        for r in caplog.records
        if r.levelno >= logging.WARNING and r.name.startswith("okapy")
    ]
    assert any(
        re.search(r"clip|outside|out of bounds|grid", r.getMessage(), re.I)
        for r in warnings
    ), [r.getMessage() for r in warnings]


def test_contour_partly_left_of_the_grid_is_clipped_with_a_warning(tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        _, xyz = _convert(tmp_path, [square_contour(-5, 2, 5, 8, Z)])

    plane = xyz[:, :, 1]
    assert plane[0:6, 2:9].all()
    assert not plane[6:, :].any()
    assert not plane[:, :2].any()
    assert not plane[:, 9:].any()
    _assert_clip_warning(caplog)


def test_contour_partly_beyond_the_grid_end_is_clipped_with_a_warning(tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        _, xyz = _convert(tmp_path, [square_contour(8, 6, 20, 25, Z)])

    plane = xyz[:, :, 1]
    assert plane[8:12, 6:12].all()
    assert not plane[:8, :].any()
    assert not plane[:, :6].any()
    _assert_clip_warning(caplog)
