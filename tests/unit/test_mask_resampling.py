from pathlib import Path

import numpy as np
import SimpleITK as sitk

from okapy.core.geometry import (
    MASK_THRESHOLD,
    PhysicalBox,
    make_reference_image_from_physical_box,
)
from okapy.core.models import ImageVolume, MaskIdentity, MaskVolume, SeriesIdentity
from okapy.preprocessing.geometry import resample_mask_volume_to_reference
from okapy.preprocessing.models import GeometryConfig


def _cube_mask(size: int = 16, inner: tuple[int, int] = (4, 12)) -> sitk.Image:
    """A 1 mm binary cube: ones on indices [inner[0], inner[1]) along each axis."""

    array = np.zeros((size, size, size), dtype=np.uint8)
    lo, hi = inner
    array[lo:hi, lo:hi, lo:hi] = 1
    return sitk.GetImageFromArray(array)


def _mask_volume(image: sitk.Image) -> MaskVolume:
    return MaskVolume.from_sitk(
        path=Path("mask.nii.gz"),
        image=image,
        identity=MaskIdentity(
            label="GTV",
            modality="RTSTRUCT",
            reference_modality="CT",
            patient_id="P1",
            study_instance_uid="1.2.3",
            reference_series_instance_uid="1.2.3.4",
        ),
    )


def _image_volume(image: sitk.Image) -> ImageVolume:
    return ImageVolume.from_sitk(
        path=Path("image.nii.gz"),
        image=image,
        identity=SeriesIdentity(
            modality="CT",
            patient_id="P1",
            study_instance_uid="1.2.3",
            series_instance_uid="1.2.3.4",
        ),
    )


def _resample(source: sitk.Image, reference: sitk.Image, interpolator: int) -> np.ndarray:
    resampled = sitk.Resample(
        source, reference, sitk.Transform(), interpolator, 0.0, sitk.sitkFloat32
    )
    return sitk.GetArrayFromImage(resampled)


def test_mask_is_resampled_linear_and_thresholded_at_half():
    source = _cube_mask()
    mask = _mask_volume(source)

    # Same 1 mm grid shifted by 0.4 mm on every axis: near the cube corner,
    # nearest neighbour rounds inside while trilinear averaging (0.6**3) does not.
    box = PhysicalBox(min_xyz=(0.4, 0.4, 0.4), max_xyz=(16.4, 16.4, 16.4))
    reference = make_reference_image_from_physical_box(
        box=box,
        spacing=(1.0, 1.0, 1.0),
        direction_source=source,
    )
    target = _image_volume(reference)
    geometry_config = GeometryConfig.from_dict({"spacing": [1.0, 1.0, 1.0]})

    result = resample_mask_volume_to_reference(
        mask,
        reference,
        target_image=target,
        geometry_config=geometry_config,
        output_path=Path("out.nii.gz"),
    )

    assert result.image.GetPixelID() == sitk.sitkUInt8
    actual = sitk.GetArrayFromImage(result.image)
    assert set(np.unique(actual).tolist()) <= {0, 1}

    expected = _resample(source, reference, sitk.sitkLinear) >= MASK_THRESHOLD
    np.testing.assert_array_equal(actual.astype(bool), expected)

    nearest = _resample(source, reference, sitk.sitkNearestNeighbor) > 0
    assert not np.array_equal(actual.astype(bool), nearest)

    # Reference voxel 11 sits at 11.4 mm: trilinear gives 0.6**3 = 0.216 < 0.5.
    assert actual[11, 11, 11] == 0
    assert nearest[11, 11, 11]

    provenance = result.metadata["geometry_preprocessing"]
    assert provenance["mask_interpolator"] == "linear"
    assert provenance["mask_threshold"] == MASK_THRESHOLD
