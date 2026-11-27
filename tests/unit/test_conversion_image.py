"""Single-frame image series conversion (CT / MR / PT / unsupported IODs)."""

from __future__ import annotations

import re

import numpy as np
import pytest
import SimpleITK as sitk

from okapy.dicom.conversion.image import SimpleITKImageSeriesConverter
from okapy.dicom.conversion.suv import PETSUVConverter
from tests.helpers.dicom_factories import (
    load_series,
    positions_along_normal,
    write_image_series,
)


def _convert(tmp_path, *, converter=None, **series_kwargs):
    series_kwargs.setdefault("rescale", (1.0, 0.0))
    uid, paths = write_image_series(tmp_path / "dicom", **series_kwargs)
    converter = converter or SimpleITKImageSeriesConverter()
    volume = converter.convert(load_series(paths), tmp_path / "out")
    return uid, volume


def _slice_values(volume) -> list[float]:
    """Value of voxel (x=0, y=0) for each slice, in z order."""

    array = sitk.GetArrayFromImage(volume.image)  # z, y, x
    # Every slice of these series is constant: check it, then summarise.
    for z in range(array.shape[0]):
        np.testing.assert_array_equal(array[z], array[z, 0, 0])
    return array[:, 0, 0].tolist()


def _expect_value_error_naming(tmp_path, **series_kwargs):
    """Convert the series and expect a ValueError naming its series UID."""

    series_kwargs.setdefault("rescale", (1.0, 0.0))
    uid, paths = write_image_series(tmp_path / "dicom", **series_kwargs)
    series = load_series(paths)
    with pytest.raises(ValueError, match=re.escape(uid)) as excinfo:
        SimpleITKImageSeriesConverter().convert(series, tmp_path / "out")
    return excinfo


# --------------------------------------------------------------------------
# Regression: well-formed series
# --------------------------------------------------------------------------


def test_normal_ct_series_keeps_exact_geometry_and_hounsfield_values(tmp_path):
    rows, columns = 3, 4
    base = np.arange(rows * columns, dtype=np.uint16).reshape(rows, columns)
    pixels = [base + 100 * k for k in range(6)]
    origin = (-50.5, -30.25, -100.0)
    positions = [(origin[0], origin[1], origin[2] + 2.5 * k) for k in range(6)]

    uid, volume = _convert(
        tmp_path,
        positions=positions,
        pixels=pixels,
        rows=rows,
        columns=columns,
        pixel_spacing=(0.7, 0.8),
        slice_thickness=2.5,
        rescale=(1.0, -1024.0),
    )

    image = volume.image
    assert image.GetSize() == (columns, rows, 6)
    # PixelSpacing is (row spacing, column spacing) = (y, x).
    np.testing.assert_allclose(image.GetSpacing(), (0.8, 0.7, 2.5))
    np.testing.assert_allclose(image.GetOrigin(), origin)
    np.testing.assert_allclose(image.GetDirection(), np.eye(3).ravel())

    expected = np.stack(pixels).astype(np.float64) - 1024.0
    np.testing.assert_array_equal(sitk.GetArrayFromImage(image), expected)
    assert volume.series_instance_uid == uid
    assert volume.path.exists()


def test_written_nifti_matches_returned_image(tmp_path):
    _, volume = _convert(tmp_path, positions=[0, 2, 4, 6], values=[1, 2, 3, 4])

    on_disk = sitk.ReadImage(str(volume.path))
    assert on_disk.GetSize() == volume.image.GetSize()
    np.testing.assert_allclose(on_disk.GetSpacing(), (1.0, 1.0, 2.0))
    np.testing.assert_allclose(on_disk.GetOrigin(), (0.0, 0.0, 0.0))
    np.testing.assert_array_equal(
        sitk.GetArrayFromImage(on_disk)[:, 0, 0], [1, 2, 3, 4]
    )


def test_large_gap_is_still_filled_with_air_for_ct(tmp_path):
    # Two missing slices at 3 and 4: existing behaviour fills them with -1000.
    _, volume = _convert(
        tmp_path,
        positions=[0, 1, 2, 5, 6],
        values=[10, 20, 30, 60, 70],
    )

    assert _slice_values(volume) == [10, 20, 30, -1000, -1000, 60, 70]
    np.testing.assert_allclose(volume.image.GetSpacing()[2], 1.0)


# --------------------------------------------------------------------------
# One missing slice: interpolation order
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("positions", "values", "expected"),
    [
        pytest.param(
            [0, 1, 2, 3, 5, 6],
            [10, 20, 30, 40, 60, 70],
            [10, 20, 30, 40, 50, 60, 70],
            id="gap-in-the-middle",
        ),
        pytest.param(
            [0, 2, 3, 4, 5, 6],
            [10, 30, 40, 50, 60, 70],
            [10, 20, 30, 40, 50, 60, 70],
            id="gap-after-first-slice",
        ),
        pytest.param(
            [0, 1, 2, 3, 4, 6],
            [10, 20, 30, 40, 50, 70],
            [10, 20, 30, 40, 50, 60, 70],
            id="gap-before-last-slice",
        ),
        pytest.param(
            [0, 2, 3, 4],
            [10, 30, 40, 50],
            [10, 20, 30, 40, 50],
            id="short-series-gap-after-first",
        ),
        pytest.param(
            [0, 1, 2, 4],
            [10, 20, 30, 50],
            [10, 20, 30, 40, 50],
            id="short-series-gap-before-last",
        ),
    ],
)
def test_one_missing_slice_is_interpolated_in_place(
    tmp_path, positions, values, expected
):
    _, volume = _convert(tmp_path, positions=positions, values=values)

    assert volume.image.GetSize()[2] == len(expected)
    assert _slice_values(volume) == expected
    np.testing.assert_allclose(volume.image.GetSpacing()[2], 1.0)
    np.testing.assert_allclose(volume.image.GetOrigin(), (0.0, 0.0, 0.0))


# --------------------------------------------------------------------------
# Single-slice series
# --------------------------------------------------------------------------


@pytest.mark.parametrize("modality", ["CT", "MR"])
def test_single_slice_series_converts_to_one_slice_volume(tmp_path, modality):
    _, volume = _convert(
        tmp_path,
        modality=modality,
        positions=[(1.0, 2.0, 3.0)],
        values=[42],
        slice_thickness=2.5,
    )

    assert volume.image.GetSize() == (5, 4, 1)
    assert _slice_values(volume) == [42]
    np.testing.assert_allclose(volume.image.GetOrigin(), (1.0, 2.0, 3.0))
    np.testing.assert_allclose(volume.image.GetSpacing(), (1.0, 1.0, 2.5))


def test_single_slice_pet_series_converts_to_one_slice_volume(tmp_path):
    _, volume = _convert(
        tmp_path,
        converter=PETSUVConverter(),
        modality="PT",
        positions=[(0.0, 0.0, 7.0)],
        values=[8],
        rescale=(0.5, 0.0),
        slice_thickness=3.0,
    )

    assert volume.image.GetSize() == (5, 4, 1)
    # Units=GML / SUVType=BW: output is the rescaled stored value.
    assert _slice_values(volume) == [4.0]
    np.testing.assert_allclose(volume.image.GetSpacing()[2], 3.0)


def test_multi_slice_pet_series_still_converts(tmp_path):
    _, volume = _convert(
        tmp_path,
        converter=PETSUVConverter(),
        modality="PT",
        positions=[0, 3, 6],
        values=[2, 4, 6],
        rescale=(0.5, 1.0),
    )

    assert _slice_values(volume) == [2.0, 3.0, 4.0]
    np.testing.assert_allclose(volume.image.GetSpacing()[2], 3.0)


# --------------------------------------------------------------------------
# Several slices at one position (4D / dynamic / multi-echo)
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "positions",
    [
        pytest.param([0, 1, 1, 2, 3], id="one-duplicated-position"),
        pytest.param([0, 0, 1, 1, 2, 2], id="two-echoes-per-position"),
    ],
)
def test_slices_sharing_a_position_raise_value_error(tmp_path, positions):
    _expect_value_error_naming(
        tmp_path, positions=positions, values=list(range(len(positions)))
    )


# --------------------------------------------------------------------------
# Slice spacing consistency
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "positions",
    [
        pytest.param([0, 1, 2, 3.5, 4.5, 5.5], id="one-gap-50pct-too-large"),
        pytest.param([0, 1, 2, 3, 4.2, 5.2], id="one-gap-20pct-too-large"),
        pytest.param([0, 1, 2, 2.95, 3.95, 4.95], id="one-gap-5pct-too-small"),
    ],
)
def test_irregular_slice_spacing_raises_value_error(tmp_path, positions):
    _expect_value_error_naming(
        tmp_path, positions=positions, values=list(range(len(positions)))
    )


def test_near_duplicate_slices_raise_value_error(tmp_path):
    # Gap of 0.05 against a 1 mm spacing: "missing slices" would be negative.
    _expect_value_error_naming(
        tmp_path,
        positions=[0, 1, 2, 2.05, 3.05, 4.05],
        values=[1, 2, 3, 4, 5, 6],
    )


def test_tiny_spacing_jitter_still_converts_with_correct_spacing(tmp_path):
    rng = np.random.default_rng(0)
    # +-0.002 mm on a 2.5 mm spacing: < 0.1 %.
    offsets = 2.5 * np.arange(10) + rng.uniform(-0.002, 0.002, size=10)
    offsets[0] = 0.0

    _, volume = _convert(tmp_path, positions=offsets.tolist(), values=list(range(10)))

    assert volume.image.GetSize()[2] == 10
    assert _slice_values(volume) == list(range(10))
    np.testing.assert_allclose(volume.image.GetSpacing()[2], 2.5, atol=1e-3)


# --------------------------------------------------------------------------
# Tilted / oblique stacking
# --------------------------------------------------------------------------


def test_gantry_tilted_stacking_raises_value_error(tmp_path):
    # Axial slices whose positions drift 0.2 mm in y per 1 mm slice step:
    # positions are not along the slice normal (20 % in-plane component).
    positions = [(0.0, 0.2 * k, 1.0 * k) for k in range(5)]

    excinfo = _expect_value_error_naming(
        tmp_path, positions=positions, values=[1, 2, 3, 4, 5]
    )

    assert re.search(r"tilt|oblique|normal", str(excinfo.value), re.IGNORECASE)


def test_oblique_but_consistent_series_still_converts(tmp_path):
    angle = np.deg2rad(30.0)
    orientation = (1.0, 0.0, 0.0, 0.0, float(np.cos(angle)), float(np.sin(angle)))
    origin = (5.0, -3.0, 10.0)
    positions = positions_along_normal(
        [0.0, 2.0, 4.0, 6.0], orientation=orientation, origin=origin
    )

    _, volume = _convert(
        tmp_path,
        positions=positions,
        orientation=orientation,
        values=[1, 2, 3, 4],
    )

    image = volume.image
    assert image.GetSize() == (5, 4, 4)
    np.testing.assert_allclose(image.GetSpacing(), (1.0, 1.0, 2.0), atol=1e-6)
    np.testing.assert_allclose(image.GetOrigin(), origin, atol=1e-6)
    normal = np.cross(orientation[:3], orientation[3:])
    expected_direction = np.stack([orientation[:3], orientation[3:], normal], axis=1)
    np.testing.assert_allclose(
        image.GetDirection(), expected_direction.ravel(), atol=1e-6
    )
    assert _slice_values(volume) == [1, 2, 3, 4]


# --------------------------------------------------------------------------
# MR rescale
# --------------------------------------------------------------------------


def test_mr_applies_rescale_slope_and_intercept(tmp_path):
    _, volume = _convert(
        tmp_path,
        modality="MR",
        positions=[0, 1, 2],
        values=[10, 20, 30],
        rescale=(2.0, -10.0),
    )

    assert _slice_values(volume) == [10.0, 30.0, 50.0]


def test_mr_without_rescale_tags_keeps_stored_values(tmp_path):
    _, volume = _convert(
        tmp_path,
        modality="MR",
        positions=[0, 1, 2],
        values=[10, 20, 30],
        rescale=None,
    )

    assert _slice_values(volume) == [10.0, 20.0, 30.0]


def test_ct_rescale_still_applied(tmp_path):
    _, volume = _convert(
        tmp_path,
        positions=[0, 1, 2],
        values=[0, 1000, 2000],
        rescale=(1.0, -1024.0),
    )

    assert _slice_values(volume) == [-1024.0, -24.0, 976.0]


# --------------------------------------------------------------------------
# Unsupported IODs
# --------------------------------------------------------------------------


@pytest.mark.parametrize("modality", ["DX", "CR"])
@pytest.mark.parametrize("n_slices", [1, 2])
def test_projection_images_without_geometry_are_rejected_as_unsupported(
    tmp_path, modality, n_slices
):
    uid, paths = write_image_series(
        tmp_path / "dicom",
        modality=modality,
        positions=[0.0] * n_slices,
        values=list(range(n_slices)),
    )
    # Strip the volumetric geometry: projection radiographs have none.
    import pydicom

    for path in paths:
        ds = pydicom.dcmread(str(path))
        del ds.ImagePositionPatient
        del ds.ImageOrientationPatient
        del ds.PixelSpacing
        ds.save_as(str(path), enforce_file_format=True)

    series = load_series(paths)

    with pytest.raises(ValueError, match=r"(?i)unsupported") as excinfo:
        SimpleITKImageSeriesConverter().convert(series, tmp_path / "out")

    assert uid in str(excinfo.value)
