"""Target-grid and resampling behavior of the geometry preprocessing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

from okapy.core.geometry import (
    make_reference_image_on_aligned_grid,
    resample_to_reference,
)
from okapy.preprocessing.geometry import (
    make_reference_grid,
    resample_image_volume_to_reference,
    resample_mask_volume_to_reference,
)
from okapy.preprocessing.models import GeometryConfig
from tests.helpers import factories


def _image(size, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0)) -> sitk.Image:
    rng = np.random.default_rng(7)
    image = sitk.GetImageFromArray(rng.random(size[::-1]).astype(np.float32))
    image.SetSpacing(spacing)
    image.SetOrigin(origin)
    return image


def _box_mask(image: sitk.Image, lo, hi) -> sitk.Image:
    array = np.zeros(image.GetSize()[::-1], dtype=np.uint8)
    array[lo[2] : hi[2] + 1, lo[1] : hi[1] + 1, lo[0] : hi[0] + 1] = 1
    mask = sitk.GetImageFromArray(array)
    mask.CopyInformation(image)
    return mask


def _crop(image: sitk.Image, mask: sitk.Image, config: dict):
    geometry_config = GeometryConfig.from_dict(config)
    image_volume = factories.image_volume(image)
    mask_volume = factories.mask_volume(mask)
    reference = make_reference_grid(
        image=image_volume, masks=[mask_volume], geometry_config=geometry_config
    )
    cropped_image = resample_image_volume_to_reference(
        image_volume,
        reference,
        geometry_config=geometry_config,
        output_path=Path("image_out.nii.gz"),
    )
    cropped_mask = resample_mask_volume_to_reference(
        mask_volume,
        reference,
        target_image=cropped_image,
        geometry_config=geometry_config,
        output_path=Path("mask_out.nii.gz"),
    )
    return reference, cropped_image.image, cropped_mask.image


# ---------------------------------------------------------------------------
# Tie positions at the crop boundary
# ---------------------------------------------------------------------------


def test_crop_keeps_boundary_voxel_at_tie_position():
    # 10 slices of 2.5 mm, resampled to 1 mm: the full aligned grid has 25
    # slices whose first centre is at continuous image index -0.3, i.e. target
    # voxel k sits at index -0.3 + 0.4 k.  The mask covers slices 1..5, so its
    # extent is [0.5, 5.5] in image index (12.5 mm).  The low boundary 0.5 is
    # exactly the centre of target voxel 2 (tie), the high one falls between
    # target voxels 14 and 15: the crop must keep target voxels 2..14 (13).
    image = _image((8, 8, 10), spacing=(1.0, 1.0, 2.5))
    mask = _box_mask(image, (2, 2, 1), (4, 4, 5))

    reference, _, cropped_mask = _crop(
        image, mask, {"spacing": [-1, -1, 1.0], "padding_voxels": 0}
    )

    assert reference.GetSize()[2] == 13
    # Target voxel 2: image index 0.5 -> z = 1.25 mm.
    assert reference.GetOrigin()[2] == pytest.approx(1.25, abs=1e-6)
    mask_array = sitk.GetArrayFromImage(cropped_mask)
    nonzero_slices = np.unique(np.nonzero(mask_array)[0])
    np.testing.assert_array_equal(nonzero_slices, np.arange(13))


def test_crop_at_tie_matches_full_grid_mask_slices():
    # Same set-up: the crop must contain every slice where the mask is
    # non-zero on the full (uncropped) aligned grid.
    image = _image((8, 8, 10), spacing=(1.0, 1.0, 2.5))
    mask = _box_mask(image, (2, 2, 1), (4, 4, 5))

    full = make_reference_image_on_aligned_grid(image=image, spacing=(1.0, 1.0, 1.0))
    full_mask = sitk.Resample(
        mask, full, sitk.Transform(), sitk.sitkLinear, 0.0, sitk.sitkFloat32
    )
    full_slices = np.unique(np.nonzero(sitk.GetArrayFromImage(full_mask) >= 0.5)[0])
    assert full_slices.size == 13

    _, _, cropped_mask = _crop(
        image, mask, {"spacing": [-1, -1, 1.0], "padding_voxels": 0}
    )
    cropped_slices = np.unique(np.nonzero(sitk.GetArrayFromImage(cropped_mask))[0])
    assert cropped_slices.size == full_slices.size


# ---------------------------------------------------------------------------
# padding per axis
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("padding", [[1, 2, 3], (1, 2, 3)])
def test_padding_voxels_per_axis(padding):
    image = _image((30, 30, 30))
    mask = _box_mask(image, (10, 10, 10), (14, 14, 14))

    reference, cropped_image, cropped_mask = _crop(
        image, mask, {"spacing": [-1, -1, -1], "padding_voxels": padding}
    )

    assert reference.GetSize() == (7, 9, 11)
    np.testing.assert_allclose(reference.GetOrigin(), (9.0, 8.0, 7.0), atol=1e-9)
    expected = sitk.GetArrayFromImage(image)[7:18, 8:17, 9:16]
    np.testing.assert_allclose(
        sitk.GetArrayFromImage(cropped_image), expected, rtol=1e-6
    )
    assert sitk.GetArrayFromImage(cropped_mask).sum() == 125


def test_scalar_padding_voxels_still_pads_every_axis():
    image = _image((30, 30, 30))
    mask = _box_mask(image, (10, 10, 10), (14, 14, 14))

    reference, _, _ = _crop(image, mask, {"spacing": [-1, -1, -1], "padding_voxels": 2})

    assert reference.GetSize() == (9, 9, 9)
    np.testing.assert_allclose(reference.GetOrigin(), (8.0, 8.0, 8.0), atol=1e-9)


# ---------------------------------------------------------------------------
# native spacing when ``spacing`` is omitted
# ---------------------------------------------------------------------------


def test_reference_grid_without_spacing_uses_native_spacing():
    image = _image((16, 18, 12), spacing=(0.8, 0.9, 2.5), origin=(-5.0, 3.0, 7.0))
    mask = _box_mask(image, (2, 3, 4), (6, 8, 7))

    reference, cropped_image, cropped_mask = _crop(image, mask, {})

    np.testing.assert_allclose(reference.GetSpacing(), image.GetSpacing())
    assert reference.GetSize() == (5, 6, 4)
    np.testing.assert_allclose(
        reference.GetOrigin(), image.TransformIndexToPhysicalPoint((2, 3, 4))
    )
    expected = sitk.GetArrayFromImage(image)[4:8, 3:9, 2:7]
    np.testing.assert_allclose(
        sitk.GetArrayFromImage(cropped_image), expected, rtol=1e-6
    )
    assert sitk.GetArrayFromImage(cropped_mask).sum() == 5 * 6 * 4


# ---------------------------------------------------------------------------
# resample_to_reference output pixel type
# ---------------------------------------------------------------------------


def _reference_like(image: sitk.Image) -> sitk.Image:
    reference = sitk.Image(image.GetSize(), sitk.sitkFloat32)
    reference.CopyInformation(image)
    return reference


def test_resample_to_int8_returns_int8_image():
    # sitk.sitkInt8 == 0 must not be treated as "no output type given".
    assert sitk.sitkInt8 == 0
    image = _image((6, 6, 6))
    image = sitk.Cast(image * 100.0, sitk.sitkFloat32)

    result = resample_to_reference(
        image,
        _reference_like(image),
        interpolator=sitk.sitkNearestNeighbor,
        output_pixel_type=sitk.sitkInt8,
    )

    assert result.GetPixelID() == sitk.sitkInt8
    np.testing.assert_array_equal(
        sitk.GetArrayFromImage(result),
        sitk.GetArrayFromImage(image).astype(np.int8),
    )


def test_resample_without_output_type_keeps_input_type():
    image = sitk.Cast(_image((6, 6, 6)) * 10.0, sitk.sitkInt16)

    result = resample_to_reference(
        image, _reference_like(image), interpolator=sitk.sitkNearestNeighbor
    )

    assert result.GetPixelID() == sitk.sitkInt16


@pytest.mark.parametrize("pixel_type", [sitk.sitkUInt8, sitk.sitkFloat32])
def test_resample_honours_other_output_types(pixel_type):
    image = sitk.Cast(_image((6, 6, 6)) * 10.0, sitk.sitkInt16)

    result = resample_to_reference(
        image,
        _reference_like(image),
        interpolator=sitk.sitkNearestNeighbor,
        output_pixel_type=pixel_type,
    )

    assert result.GetPixelID() == pixel_type


# ---------------------------------------------------------------------------
# regression: normal crop + resample
# ---------------------------------------------------------------------------


def test_resampled_crop_regression_geometry():
    image = _image((20, 20, 20))
    mask = _box_mask(image, (5, 5, 5), (9, 9, 9))

    reference, cropped_image, cropped_mask = _crop(
        image, mask, {"spacing": [2.0, 2.0, 2.0], "padding_voxels": 1}
    )

    # Full 2 mm grid centres at 0.5 + 2k mm; extent [4.5, 9.5] keeps k=2..4,
    # padded by one voxel -> k=1..5.
    assert reference.GetSize() == (5, 5, 5)
    np.testing.assert_allclose(reference.GetSpacing(), (2.0, 2.0, 2.0))
    np.testing.assert_allclose(reference.GetOrigin(), (2.5, 2.5, 2.5), atol=1e-6)
    assert cropped_image.GetPixelID() == sitk.sitkFloat32
    assert cropped_mask.GetPixelID() == sitk.sitkUInt8
    # Linear interpolation of the image at physical point (4.5, 4.5, 4.5).
    expected = image.EvaluateAtPhysicalPoint((4.5, 4.5, 4.5), sitk.sitkLinear)
    assert sitk.GetArrayFromImage(cropped_image)[1, 1, 1] == pytest.approx(
        expected, rel=1e-5
    )
