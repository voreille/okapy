from __future__ import annotations

import numpy as np
import SimpleITK as sitk

from okapy.core.models import ImageVolume, MaskVolume
from okapy.preprocessing.registry import register_processor


class IdentityProcessor:
    def __call__(self, item, **kwargs):
        return item


class Standardizer:
    """Z-score standardization computed on voxels above threshold."""

    def __init__(self, threshold: float = 0.0, output_dtype: str = "float32") -> None:
        self.threshold = float(threshold)
        self.output_dtype = output_dtype

    def __call__(self, image: ImageVolume, **kwargs) -> ImageVolume:
        array = sitk.GetArrayFromImage(image.image).astype(np.float64)
        values = array[array > self.threshold]

        if values.size == 0:
            raise ValueError(
                f"Cannot standardize image {image.path}: no voxels above "
                f"threshold={self.threshold}."
            )

        mean = float(np.mean(values))
        std = float(np.std(values))

        if std == 0:
            raise ValueError(f"Cannot standardize image {image.path}: std is zero.")

        standardized = (array - mean) / std
        standardized = _cast_array(standardized, self.output_dtype)

        out = sitk.GetImageFromArray(standardized)
        out.CopyInformation(image.image)

        return image.with_image(
            out,
            metadata={
                "standardizer": {
                    "threshold": self.threshold,
                    "mean": mean,
                    "std": std,
                    "output_dtype": self.output_dtype,
                }
            },
        )


class MaskedStandardizer:
    """Z-score standardization computed inside selected masks.

    Masks must already be aligned with the image grid. The study-level geometry
    pipeline guarantees this before local image processors are called.
    """

    def __init__(
        self,
        label: str | None = None,
        labels: list[str] | tuple[str, ...] | None = None,
        output_dtype: str = "float32",
    ) -> None:
        if label is not None and labels is not None:
            raise ValueError("Use either 'label' or 'labels', not both.")

        if label is not None:
            labels = [label]

        self.labels = tuple(labels) if labels is not None else None
        self.output_dtype = output_dtype

    def __call__(
        self,
        image: ImageVolume,
        *,
        masks: list[MaskVolume] | None = None,
        **kwargs,
    ) -> ImageVolume:
        if not masks:
            raise ValueError(f"MaskedStandardizer requires masks for image {image.path}.")

        selected_masks = masks
        if self.labels is not None:
            selected_masks = [mask for mask in masks if mask.label in self.labels]

        if not selected_masks:
            raise ValueError(
                f"No mask matching labels={self.labels!r} found for image {image.path}."
            )

        image_array = sitk.GetArrayFromImage(image.image).astype(np.float64)
        combined_mask = np.zeros_like(image_array, dtype=bool)

        for mask in selected_masks:
            mask_array = sitk.GetArrayFromImage(mask.image) != 0
            if mask_array.shape != image_array.shape:
                raise ValueError(
                    f"Mask/image shape mismatch for image {image.path}: "
                    f"image={image_array.shape}, mask={mask_array.shape}, label={mask.label!r}."
                )
            combined_mask |= mask_array

        values = image_array[combined_mask]
        if values.size == 0:
            raise ValueError(f"MaskedStandardizer got an empty mask for {image.path}.")

        mean = float(np.mean(values))
        std = float(np.std(values))
        if std == 0:
            raise ValueError(f"MaskedStandardizer std is zero for image {image.path}.")

        standardized = (image_array - mean) / std
        standardized = _cast_array(standardized, self.output_dtype)

        out = sitk.GetImageFromArray(standardized)
        out.CopyInformation(image.image)

        return image.with_image(
            out,
            metadata={
                "masked_standardizer": {
                    "labels": self.labels,
                    "mean": mean,
                    "std": std,
                    "output_dtype": self.output_dtype,
                }
            },
        )


class ClipIntensity:
    def __init__(self, lower: float | None = None, upper: float | None = None) -> None:
        self.lower = lower
        self.upper = upper

    def __call__(self, image: ImageVolume, **kwargs) -> ImageVolume:
        array = sitk.GetArrayFromImage(image.image)
        clipped = np.clip(array, self.lower, self.upper)

        out = sitk.GetImageFromArray(clipped.astype(array.dtype, copy=False))
        out.CopyInformation(image.image)

        return image.with_image(
            out,
            metadata={"clip_intensity": {"lower": self.lower, "upper": self.upper}},
        )


class CastImage:
    def __init__(self, pixel_type: str = "float32") -> None:
        self.pixel_type = pixel_type

    def __call__(self, image: ImageVolume, **kwargs) -> ImageVolume:
        casted = sitk.Cast(image.image, _sitk_pixel_type(self.pixel_type))
        return image.with_image(
            casted,
            metadata={"cast_image": {"pixel_type": self.pixel_type}},
        )


class CastMask:
    def __init__(self, pixel_type: str = "uint8") -> None:
        self.pixel_type = pixel_type

    def __call__(self, mask: MaskVolume, **kwargs) -> MaskVolume:
        casted = sitk.Cast(mask.image, _sitk_pixel_type(self.pixel_type))
        return mask.with_image(
            casted,
            metadata={"cast_mask": {"pixel_type": self.pixel_type}},
        )


class BinarizeMask:
    def __init__(self, threshold: float = 0.5, pixel_type: str = "uint8") -> None:
        self.threshold = float(threshold)
        self.pixel_type = pixel_type

    def __call__(self, mask: MaskVolume, **kwargs) -> MaskVolume:
        array = sitk.GetArrayFromImage(mask.image)
        binary = (array >= self.threshold).astype(_numpy_dtype(self.pixel_type))

        out = sitk.GetImageFromArray(binary)
        out.CopyInformation(mask.image)

        return mask.with_image(
            out,
            metadata={
                "binarize_mask": {
                    "threshold": self.threshold,
                    "pixel_type": self.pixel_type,
                }
            },
        )


def _cast_array(array: np.ndarray, dtype: str) -> np.ndarray:
    return array.astype(_numpy_dtype(dtype), copy=False)


def _numpy_dtype(dtype: str):
    normalized = dtype.lower()
    if normalized in {"float", "float32", "single"}:
        return np.float32
    if normalized in {"float64", "double"}:
        return np.float64
    if normalized in {"uint8", "uchar"}:
        return np.uint8
    if normalized in {"int16", "short"}:
        return np.int16
    if normalized in {"uint16", "ushort"}:
        return np.uint16
    raise ValueError(f"Unsupported numpy dtype {dtype!r}.")


def _sitk_pixel_type(pixel_type: str) -> int:
    normalized = pixel_type.lower()
    if normalized in {"float", "float32", "single"}:
        return sitk.sitkFloat32
    if normalized in {"float64", "double"}:
        return sitk.sitkFloat64
    if normalized in {"uint8", "uchar"}:
        return sitk.sitkUInt8
    if normalized in {"int16", "short"}:
        return sitk.sitkInt16
    if normalized in {"uint16", "ushort"}:
        return sitk.sitkUInt16
    raise ValueError(f"Unsupported SimpleITK pixel type {pixel_type!r}.")


register_processor("identity_processor", IdentityProcessor)
register_processor("standardizer", Standardizer)
register_processor("masked_standardizer", MaskedStandardizer)
register_processor("clip_intensity", ClipIntensity)
register_processor("cast_image", CastImage)
register_processor("cast_mask", CastMask)
register_processor("binarize_mask", BinarizeMask)
