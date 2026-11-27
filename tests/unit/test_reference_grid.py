from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

from okapy.core.geometry import make_reference_image_on_aligned_grid, mask_index_extent
from okapy.preprocessing.geometry import (
    make_reference_grid,
    resample_image_volume_to_reference,
    resample_mask_volume_to_reference,
)
from okapy.preprocessing.models import GeometryConfig
from tests.helpers import factories

FLIPPED_XY = (-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0)


def _image(
    size=(40, 36, 30),
    spacing=(3.90625, 3.90625, 4.25),
    origin=(-250.0, -250.0, -908.0),
    direction=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
) -> sitk.Image:
    rng = np.random.default_rng(0)
    array = rng.random(size[::-1]).astype(np.float32)
    image = sitk.GetImageFromArray(array)
    image.SetSpacing(spacing)
    image.SetOrigin(origin)
    image.SetDirection(direction)
    return image


def _mask_like(image: sitk.Image, lo=(10, 12, 8), hi=(15, 16, 13)) -> sitk.Image:
    """Binary box on index ranges [lo, hi] (x, y, z), on the grid of ``image``."""

    array = np.zeros(image.GetSize()[::-1], dtype=np.uint8)
    array[lo[2] : hi[2] + 1, lo[1] : hi[1] + 1, lo[0] : hi[0] + 1] = 1
    mask = sitk.GetImageFromArray(array)
    mask.CopyInformation(image)
    return mask


def _crop(image: sitk.Image, mask: sitk.Image, config: dict):
    geometry_config = GeometryConfig.from_dict(config)
    image_volume = factories.image_volume(image, modality="PT")
    reference = make_reference_grid(
        image=image_volume,
        masks=[factories.mask_volume(mask, reference_modality="PT")],
        geometry_config=geometry_config,
    )
    cropped_image = resample_image_volume_to_reference(
        image_volume,
        reference,
        geometry_config=geometry_config,
        output_path=Path("image_out.nii.gz"),
    )
    cropped_mask = resample_mask_volume_to_reference(
        factories.mask_volume(mask, reference_modality="PT"),
        reference,
        target_image=cropped_image,
        geometry_config=geometry_config,
        output_path=Path("mask_out.nii.gz"),
    )
    return reference, cropped_image.image, cropped_mask.image


@pytest.mark.parametrize("direction", [None, FLIPPED_XY])
@pytest.mark.parametrize("padding_voxels", [0, 1, 3])
def test_native_spacing_crop_keeps_voxels_unchanged(direction, padding_voxels):
    image = _image() if direction is None else _image(direction=direction)
    mask = _mask_like(image)

    reference, cropped_image, cropped_mask = _crop(
        image,
        mask,
        {"spacing": [-1, -1, -1], "padding_voxels": padding_voxels},
    )

    # The crop is the index sub-range [lo - padding, hi + padding] of the image.
    lo = np.array([10, 12, 8]) - padding_voxels
    hi = np.array([15, 16, 13]) + padding_voxels
    assert reference.GetSize() == tuple(int(x) for x in hi - lo + 1)
    np.testing.assert_allclose(
        reference.GetOrigin(),
        image.TransformIndexToPhysicalPoint(tuple(int(x) for x in lo)),
    )
    assert reference.GetDirection() == image.GetDirection()

    expected = sitk.GetArrayFromImage(image)[
        lo[2] : hi[2] + 1, lo[1] : hi[1] + 1, lo[0] : hi[0] + 1
    ]
    np.testing.assert_allclose(
        sitk.GetArrayFromImage(cropped_image), expected, rtol=1e-6
    )

    mask_array = sitk.GetArrayFromImage(cropped_mask)
    assert mask_array.sum() == sitk.GetArrayFromImage(mask).sum()


def test_padding_is_clipped_to_the_image():
    image = _image()
    mask = _mask_like(image, lo=(0, 1, 0), hi=(3, 4, 2))

    reference, _, cropped_mask = _crop(
        image, mask, {"spacing": [-1, -1, -1], "padding_voxels": 5}
    )

    assert reference.GetSize() == (9, 10, 8)
    np.testing.assert_allclose(reference.GetOrigin(), image.GetOrigin())
    assert (
        sitk.GetArrayFromImage(cropped_mask).sum() == sitk.GetArrayFromImage(mask).sum()
    )


def test_resampled_grid_is_centred_on_the_image_grid():
    image = _image(size=(40, 36, 30), spacing=(3.0, 3.0, 4.0))

    reference = make_reference_image_on_aligned_grid(
        image=image, spacing=(2.0, 2.0, 2.0)
    )

    assert reference.GetSize() == (60, 54, 60)
    image_centre = image.TransformContinuousIndexToPhysicalPoint(
        tuple((n - 1) / 2 for n in image.GetSize())
    )
    reference_centre = reference.TransformContinuousIndexToPhysicalPoint(
        tuple((n - 1) / 2 for n in reference.GetSize())
    )
    np.testing.assert_allclose(reference_centre, image_centre)


def test_resampled_crop_lies_on_the_full_aligned_grid():
    image = _image(spacing=(3.0, 3.0, 4.0))
    mask = _mask_like(image)

    full = make_reference_image_on_aligned_grid(image=image, spacing=(2.0, 2.0, 2.0))
    reference, _, cropped_mask = _crop(
        image, mask, {"spacing": [2.0, 2.0, 2.0], "padding_voxels": 2}
    )

    # The crop origin is a voxel centre of the full aligned grid.
    index = np.asarray(
        full.TransformPhysicalPointToContinuousIndex(reference.GetOrigin())
    )
    np.testing.assert_allclose(index, np.round(index), atol=1e-6)
    assert reference.GetSpacing() == (2.0, 2.0, 2.0)
    assert sitk.GetArrayFromImage(cropped_mask).sum() > 0


def test_mask_index_extent_on_another_grid():
    image = _image(direction=FLIPPED_XY)
    mask = _mask_like(image)

    # The same mask resampled onto a finer grid with the default orientation.
    fine = sitk.Image((200, 200, 150), sitk.sitkUInt8)
    fine.SetSpacing((1.0, 1.0, 1.0))
    fine.SetOrigin((-450.0, -450.0, -910.0))
    fine_mask = sitk.Resample(mask, fine, sitk.Transform(), sitk.sitkNearestNeighbor, 0)

    lo, hi = mask_index_extent(fine_mask, image)

    np.testing.assert_allclose(lo, [9.5, 11.5, 7.5], atol=0.3)
    np.testing.assert_allclose(hi, [15.5, 16.5, 13.5], atol=0.3)
