"""Audit: slice gap / duplicate / tilt detection at its boundaries.

Agreed decisions (ISSUE_37.md): after gap filling, any gap deviating more than
1 % from the slice spacing raises ``ValueError``; the in-plane component of
consecutive ImagePositionPatient differences above 1 % of the slice spacing
raises ``ValueError``.
"""

from __future__ import annotations

import re

import numpy as np
import pytest
import SimpleITK as sitk

from okapy.dicom.conversion.image import SimpleITKImageSeriesConverter
from tests.helpers.dicom_factories import (
    load_series,
    positions_along_normal,
    write_image_series,
)

# Row (1, 0, 0), column (0, -1, 0): normal is (0, 0, -1).
FLIPPED_ORIENTATION = (1.0, 0.0, 0.0, 0.0, -1.0, 0.0)


def _convert(tmp_path, **series_kwargs):
    series_kwargs.setdefault("rescale", (1.0, 0.0))
    uid, paths = write_image_series(tmp_path / "dicom", **series_kwargs)
    volume = SimpleITKImageSeriesConverter().convert(
        load_series(paths), tmp_path / "out"
    )
    return uid, volume


def _expect_error(tmp_path, **series_kwargs):
    series_kwargs.setdefault("rescale", (1.0, 0.0))
    uid, paths = write_image_series(tmp_path / "dicom", **series_kwargs)
    with pytest.raises(ValueError, match=re.escape(uid)) as excinfo:
        SimpleITKImageSeriesConverter().convert(load_series(paths), tmp_path / "out")
    return excinfo


def _slice_values(volume) -> list[float]:
    return sitk.GetArrayFromImage(volume.image)[:, 0, 0].tolist()


# --------------------------------------------------------------------------
# Spacing tolerance: just under / just over 1 %
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "positions",
    [
        pytest.param([0, 1, 2, 3, 4.009], id="last-gap-0.9pct-large"),
        pytest.param([0, 0.991, 1.991, 2.991, 3.991], id="first-gap-0.9pct-small"),
        pytest.param([0, 1, 2, 3.009, 4.009], id="middle-gap-0.9pct-large"),
    ],
)
def test_gap_just_under_tolerance_converts(tmp_path, positions):
    _, volume = _convert(tmp_path, positions=positions, values=[1, 2, 3, 4, 5])

    assert volume.image.GetSize()[2] == 5
    assert _slice_values(volume) == [1, 2, 3, 4, 5]


@pytest.mark.parametrize(
    "positions",
    [
        pytest.param([0, 1, 2, 3, 4.011], id="last-gap-1.1pct-large"),
        pytest.param([0, 0.989, 1.989, 2.989, 3.989], id="first-gap-1.1pct-small"),
        pytest.param([0, 1, 2, 3.011, 4.011], id="middle-gap-1.1pct-large"),
    ],
)
def test_gap_just_over_tolerance_raises(tmp_path, positions):
    _expect_error(tmp_path, positions=positions, values=[1, 2, 3, 4, 5])


def test_missing_slice_gap_with_residual_over_tolerance_raises(tmp_path):
    # A 2-spacing gap of 2.03 mm: each filled step would be 1.015 mm (1.5 %).
    _expect_error(tmp_path, positions=[0, 1, 2, 4.03, 5.03], values=[1, 2, 3, 4, 5])


def test_gap_at_first_slice_spanning_several_slices_is_filled(tmp_path):
    _, volume = _convert(
        tmp_path, positions=[0, 4, 5, 6, 7], values=[10, 50, 60, 70, 80]
    )

    assert _slice_values(volume) == [10, -1000, -1000, -1000, 50, 60, 70, 80]
    np.testing.assert_allclose(volume.image.GetSpacing()[2], 1.0)


def test_gap_at_last_slice_spanning_several_slices_is_filled(tmp_path):
    _, volume = _convert(tmp_path, positions=[0, 1, 2, 3, 6], values=[1, 2, 3, 4, 7])

    assert _slice_values(volume) == [1, 2, 3, 4, -1000, -1000, 7]
    np.testing.assert_allclose(volume.image.GetSpacing()[2], 1.0)


def test_two_discontinuities_raise(tmp_path):
    _expect_error(tmp_path, positions=[0, 2, 3, 4, 6], values=[1, 2, 3, 4, 5])


def test_unique_jittered_gaps_each_within_tolerance_convert(tmp_path):
    """Every gap is within 0.6 % of the nominal 2.5 mm spacing.

    No two gaps are equal, so the "most frequent" gap is arbitrary; picking
    the first one (2.515) makes the 2.485 gap look 1.2 % off. The spacing is
    uniform within tolerance and must convert.
    """

    gaps = [2.515, 2.485, 2.505, 2.495, 2.51, 2.49]
    positions = np.concatenate([[0.0], np.cumsum(gaps)]).tolist()

    _, volume = _convert(tmp_path, positions=positions, values=list(range(7)))

    assert volume.image.GetSize()[2] == 7
    np.testing.assert_allclose(volume.image.GetSpacing()[2], 2.5, atol=1e-2)


# --------------------------------------------------------------------------
# Two-slice series
# --------------------------------------------------------------------------


def test_two_slice_series_converts_with_their_distance_as_spacing(tmp_path):
    _, volume = _convert(tmp_path, positions=[0, 3.0], values=[1, 2])

    assert volume.image.GetSize()[2] == 2
    assert _slice_values(volume) == [1, 2]
    np.testing.assert_allclose(volume.image.GetSpacing()[2], 3.0)


def test_two_slices_at_the_same_position_raise(tmp_path):
    _expect_error(tmp_path, positions=[1.0, 1.0], values=[1, 2])


def test_two_tilted_slices_raise(tmp_path):
    _expect_error(tmp_path, positions=[(0, 0, 0), (0, 0.5, 1.0)], values=[1, 2])


# --------------------------------------------------------------------------
# Negative spacing direction / descending positions
# --------------------------------------------------------------------------


def test_flipped_normal_orders_slices_along_the_normal(tmp_path):
    # Normal (0, 0, -1): z = 0, -1, -2, -3 are 0, 1, 2, 3 mm along the normal.
    positions = [(0.0, 0.0, -float(k)) for k in range(4)]

    _, volume = _convert(
        tmp_path,
        positions=positions,
        orientation=FLIPPED_ORIENTATION,
        values=[1, 2, 3, 4],
    )

    image = volume.image
    np.testing.assert_allclose(image.GetSpacing()[2], 1.0)
    np.testing.assert_allclose(image.GetOrigin(), (0.0, 0.0, 0.0))
    np.testing.assert_allclose(
        image.GetDirection(), (1, 0, 0, 0, -1, 0, 0, 0, -1), atol=1e-12
    )
    assert _slice_values(volume) == [1, 2, 3, 4]
    # Physical z of slice k is -k.
    for k in range(4):
        assert image.TransformIndexToPhysicalPoint((0, 0, k))[2] == pytest.approx(-k)


def test_flipped_normal_missing_slice_is_interpolated_in_place(tmp_path):
    positions = [(0.0, 0.0, -z) for z in (0.0, 1.0, 3.0, 4.0)]

    _, volume = _convert(
        tmp_path,
        positions=positions,
        orientation=FLIPPED_ORIENTATION,
        values=[10, 20, 40, 50],
    )

    assert _slice_values(volume) == [10, 20, 30, 40, 50]
    np.testing.assert_allclose(volume.image.GetSpacing()[2], 1.0)


def test_identity_orientation_with_negative_z_positions(tmp_path):
    # Ascending normal (0, 0, 1) but positions all negative and given unsorted.
    _, volume = _convert(
        tmp_path, positions=[-10.0, -12.0, -11.0, -13.0], values=[4, 2, 3, 1]
    )

    np.testing.assert_allclose(volume.image.GetOrigin(), (0.0, 0.0, -13.0))
    assert _slice_values(volume) == [1, 2, 3, 4]


# --------------------------------------------------------------------------
# Tilt tolerance: just under / just over 1 %
# --------------------------------------------------------------------------


def test_in_plane_drift_just_under_tolerance_converts(tmp_path):
    positions = [(0.0, 0.009 * k, 2.0 * k) for k in range(5)]
    _, volume = _convert(tmp_path, positions=positions, values=[1, 2, 3, 4, 5])
    assert volume.image.GetSize()[2] == 5


def test_in_plane_drift_just_over_tolerance_raises(tmp_path):
    # 0.011 mm per 1 mm slice step: 1.1 % in-plane.
    positions = [(0.0, 0.011 * k, 1.0 * k) for k in range(5)]
    _expect_error(tmp_path, positions=positions, values=[1, 2, 3, 4, 5])


def test_tilt_only_on_the_last_step_raises(tmp_path):
    positions = [(0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 0.0, 2.0), (0.0, 0.05, 3.0)]
    _expect_error(tmp_path, positions=positions, values=[1, 2, 3, 4])


def test_oblique_series_with_one_missing_slice_keeps_geometry(tmp_path):
    angle = np.deg2rad(20.0)
    orientation = (1.0, 0.0, 0.0, 0.0, float(np.cos(angle)), float(np.sin(angle)))
    origin = (1.0, 2.0, 3.0)
    positions = positions_along_normal(
        [0.0, 2.0, 6.0, 8.0], orientation=orientation, origin=origin
    )

    _, volume = _convert(
        tmp_path,
        positions=positions,
        orientation=orientation,
        values=[10, 20, 40, 50],
    )

    assert _slice_values(volume) == [10, 20, 30, 40, 50]
    np.testing.assert_allclose(volume.image.GetSpacing(), (1.0, 1.0, 2.0), atol=1e-6)
    np.testing.assert_allclose(volume.image.GetOrigin(), origin, atol=1e-6)
