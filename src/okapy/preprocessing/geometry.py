from __future__ import annotations

from pathlib import Path

import numpy as np
import SimpleITK as sitk

from okapy.core.geometry import (
    MASK_INTERPOLATOR,
    MASK_INTERPOLATOR_NAME,
    MASK_THRESHOLD,
    PhysicalBox,
    box_index_extent,
    image_physical_box,
    intersect_boxes,
    interpolator_from_name,
    make_reference_image_on_aligned_grid,
    mask_index_extent,
    resample_to_reference,
)
from okapy.core.models import ImageVolume, MaskVolume, VolumeStage
from okapy.preprocessing.models import GeometryConfig


def masks_for_image(
    image: ImageVolume,
    masks: list[MaskVolume],
    *,
    combine_segmentation: bool,
) -> list[MaskVolume]:
    """Return masks that should be used for a given image.

    If combine_segmentation=True, all masks in the study are reused for every
    image. This is useful for PET/CT where RTSTRUCTs are drawn on CT but also
    needed on PT.
    """

    if combine_segmentation:
        return list(masks)

    return [
        mask
        for mask in masks
        if mask.reference_series_instance_uid == image.series_instance_uid
    ]


def compute_common_fov(images: list[ImageVolume]) -> PhysicalBox | None:
    if not images:
        return None
    return intersect_boxes([image_physical_box(image.image) for image in images])


def _resolve_target_spacing(
    configured_spacing: tuple[float, float, float] | None,
    native_spacing: tuple[float, float, float],
) -> tuple[float, float, float]:
    if configured_spacing is None:
        return native_spacing

    if len(configured_spacing) != 3:
        raise ValueError(f"Expected 3 spacing values, got {configured_spacing}.")

    resolved = tuple(
        native_spacing[axis] if value == -1 else float(value)
        for axis, value in enumerate(configured_spacing)
    )

    if any(value <= 0 for value in resolved):
        raise ValueError(f"Resolved spacing must be positive, got {resolved}.")

    return resolved


def make_reference_grid(
    *,
    image: ImageVolume,
    masks: list[MaskVolume],
    geometry_config: GeometryConfig,
    common_fov: PhysicalBox | None = None,
    pixel_id: int = sitk.sitkFloat32,
) -> sitk.Image:
    """Return the target grid for ``image``.

    The grid is aligned with the image grid (see
    :func:`okapy.core.geometry.make_reference_image_on_aligned_grid`), so with
    native spacing the image and masks are cropped without interpolation. With
    ``crop_to_masks`` it covers the union of the masks plus ``padding_voxels``
    voxels of the target grid on each side.
    """

    spacing = _resolve_target_spacing(
        geometry_config.spacing,
        image.geometry.spacing,
    )

    extents = []

    if geometry_config.crop_to_masks:
        if not masks:
            raise ValueError(
                f"Cannot crop image {image.path} to masks because no masks were provided."
            )

        mask_extents = [mask_index_extent(mask.image, image.image) for mask in masks]
        extents.append(
            (
                np.min([extent[0] for extent in mask_extents], axis=0),
                np.max([extent[1] for extent in mask_extents], axis=0),
            )
        )

    if geometry_config.crop_to_common_fov and common_fov is not None:
        extents.append(box_index_extent(common_fov, image.image))

    return make_reference_image_on_aligned_grid(
        image=image.image,
        spacing=spacing,
        extents=extents,
        padding_voxels=geometry_config.padding_voxels if geometry_config.crop_to_masks else 0,
        pixel_id=pixel_id,
    )


def resample_image_volume_to_reference(
    image: ImageVolume,
    reference: sitk.Image,
    *,
    geometry_config: GeometryConfig,
    output_path: Path,
) -> ImageVolume:
    resampled = resample_to_reference(
        image.image,
        reference,
        interpolator=interpolator_from_name(geometry_config.image_interpolator),
        default_value=geometry_config.default_image_value,
        output_pixel_type=sitk.sitkFloat32,
    )

    return image.with_image(
        resampled,
        path=output_path,
        stage=VolumeStage.PREPROCESSED,
        source_path=image.path,
        metadata={
            "geometry_preprocessing": {
                "spacing": tuple(float(x) for x in resampled.GetSpacing()),
                "image_interpolator": geometry_config.image_interpolator,
                "default_image_value": geometry_config.default_image_value,
            }
        },
    )


def resample_mask_volume_to_reference(
    mask: MaskVolume,
    reference: sitk.Image,
    *,
    target_image: ImageVolume,
    geometry_config: GeometryConfig,
    output_path: Path,
) -> MaskVolume:
    """Resample a binary mask onto ``reference``.

    Masks are always resampled with linear interpolation and thresholded at
    0.5 (see ``okapy.core.geometry.MASK_INTERPOLATOR``). This is not
    configurable.
    """

    # Preserve the interpolated fractions until thresholding.
    resampled_float = resample_to_reference(
        mask.image,
        reference,
        interpolator=MASK_INTERPOLATOR,
        default_value=float(geometry_config.default_mask_value),
        output_pixel_type=sitk.sitkFloat32,
    )

    resampled_binary = sitk.BinaryThreshold(
        resampled_float,
        lowerThreshold=MASK_THRESHOLD,
        upperThreshold=float("inf"),
        insideValue=1,
        outsideValue=0,
    )

    resampled_binary = sitk.Cast(
        resampled_binary,
        sitk.sitkUInt8,
    )

    return mask.with_image(
        resampled_binary,
        path=output_path,
        stage=VolumeStage.PREPROCESSED,
        target_identity=target_image.identity,
        source_path=mask.path,
        metadata={
            "geometry_preprocessing": {
                "target_series_instance_uid": (target_image.series_instance_uid),
                "target_modality_key": target_image.modality_key,
                "mask_interpolator": MASK_INTERPOLATOR_NAME,
                "mask_threshold": MASK_THRESHOLD,
                "default_mask_value": (geometry_config.default_mask_value),
            }
        },
    )


def resolve_target_spacing(
    image: ImageVolume,
    geometry_config: GeometryConfig,
) -> tuple[float, float, float]:
    native_spacing = image.geometry.spacing

    return tuple(
        native_spacing[i]
        if geometry_config.spacing[i] == -1
        else geometry_config.spacing[i]
        for i in range(3)
    )
