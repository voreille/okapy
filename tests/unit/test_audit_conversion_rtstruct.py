"""Audit: RTSTRUCT holes, nesting, touching contours and out-of-grid contours."""

from __future__ import annotations

import logging

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
REFERENCE_UID = "1.2.826.0.1.3680043.8.498.700"
Z = 1.0  # physical z of slice index 1


def _reference():
    image = sitk.Image(SIZE_XYZ, sitk.sitkFloat32)
    image.SetOrigin((0.0, 0.0, 0.0))
    image.SetSpacing((1.0, 1.0, 1.0))
    return image_volume(image, series_instance_uid=REFERENCE_UID)


def _convert_rois(tmp_path, rois):
    ds = rtstruct_dataset(rois=rois, referenced_series_uid=REFERENCE_UID)
    path = write_dataset(ds, tmp_path / "dicom" / "rtstruct.dcm")
    masks = RTStructMaskConverter().convert(
        series=load_series([path]),
        reference_image=_reference(),
        output_dir=tmp_path / "out",
    )
    return {
        mask.label: np.transpose(sitk.GetArrayFromImage(mask.image), (2, 1, 0))
        for mask in masks
    }


def _convert(tmp_path, contours):
    return _convert_rois(tmp_path, [(1, "GTV", contours)])["GTV"]


# --------------------------------------------------------------------------
# Nesting
# --------------------------------------------------------------------------


def _three_levels():
    outer = square_contour(1, 1, 10, 10, Z)
    hole = square_contour(3, 3, 8, 8, Z)
    island = square_contour(5, 5, 6, 6, Z)
    return outer, hole, island


def _assert_three_levels(plane):
    # Island inside the hole is filled.
    assert plane[5:7, 5:7].all()
    # Ring between the hole boundary and the island is empty.
    assert not plane[4, 4:8].any()
    assert not plane[4:8, 4].any()
    # Ring between the outer contour and the hole is filled.
    assert plane[1:3, 1:11].all()
    assert plane[9:11, 1:11].all()
    # Outside: empty.
    assert not plane[0, :].any()
    assert not plane[11, :].any()


def test_island_inside_a_hole_is_filled(tmp_path):
    outer, hole, island = _three_levels()
    xyz = _convert(tmp_path, [outer, hole, island])
    _assert_three_levels(xyz[:, :, 1])


def test_three_level_nesting_does_not_depend_on_contour_order(tmp_path):
    outer, hole, island = _three_levels()
    xyz = _convert(tmp_path, [island, outer, hole])
    _assert_three_levels(xyz[:, :, 1])


def test_holes_on_one_slice_do_not_affect_other_slices(tmp_path):
    outer, hole, _ = _three_levels()
    other = square_contour(1, 1, 10, 10, 2.0)
    xyz = _convert(tmp_path, [outer, hole, other])

    assert xyz[1:11, 1:11, 2].all()
    assert not xyz[5, 5, 1]


# --------------------------------------------------------------------------
# Touching contours
# --------------------------------------------------------------------------


def test_two_contours_sharing_an_edge_fill_their_union(tmp_path):
    """Two side-by-side parts of one ROI that share the edge x = 4.

    Neither is inside the other, so there is no hole: the union of both is
    filled, including the voxels on the shared edge.
    """

    left = square_contour(1, 1, 4, 8, Z)
    right = square_contour(4, 1, 8, 8, Z)

    plane = _convert(tmp_path, [left, right])[:, :, 1]

    assert plane[1:9, 1:9].all(), np.argwhere(~plane[1:9, 1:9].astype(bool)) + 1


def test_two_contours_touching_at_a_corner_are_both_filled(tmp_path):
    a = square_contour(1, 1, 4, 4, Z)
    b = square_contour(4, 4, 8, 8, Z)

    plane = _convert(tmp_path, [a, b])[:, :, 1]

    assert plane[1:4, 1:4].all()
    assert plane[5:9, 5:9].all()
    assert plane[4, 4]


# --------------------------------------------------------------------------
# Contours outside the grid
# --------------------------------------------------------------------------


def test_contour_entirely_outside_in_plane_skips_the_roi_with_warning(tmp_path, caplog):
    # The converter skips ROIs whose mask is empty (skip_empty=True default).
    with caplog.at_level(logging.WARNING):
        masks = _convert_rois(
            tmp_path, [(1, "GTV", [square_contour(20, 20, 30, 30, Z)])]
        )

    assert masks == {}
    assert any(r.levelno >= logging.WARNING for r in caplog.records)


def test_contour_entirely_outside_in_z_is_skipped_with_warning(tmp_path, caplog):
    inside = square_contour(2, 2, 5, 5, Z)
    beyond = square_contour(2, 2, 5, 5, 10.0)
    before = square_contour(2, 2, 5, 5, -4.0)

    with caplog.at_level(logging.WARNING):
        xyz = _convert(tmp_path, [inside, beyond, before])

    assert xyz[2:6, 2:6, 1].all()
    assert int(xyz.sum()) == 16
    assert any(r.levelno >= logging.WARNING for r in caplog.records)


def test_roi_outside_the_grid_does_not_abort_other_rois(tmp_path):
    masks = _convert_rois(
        tmp_path,
        [
            (1, "outside", [square_contour(-30, -30, -20, -20, Z)]),
            (2, "inside", [square_contour(2, 2, 5, 5, Z)]),
        ],
    )

    assert masks["inside"][2:6, 2:6, 1].all()
    assert "outside" not in masks


def test_contour_straddling_the_grid_corner_is_clipped(tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        xyz = _convert(tmp_path, [square_contour(-5, -5, 2, 2, Z)])

    assert xyz[0:3, 0:3, 1].all()
    assert int(xyz.sum()) == 9
    assert any("clip" in r.getMessage().lower() for r in caplog.records)
