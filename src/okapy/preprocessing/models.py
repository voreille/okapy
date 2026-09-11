from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import SimpleITK as sitk

from okapy.core.geometry import PhysicalBox
from okapy.core.models import ImageVolume, MaskVolume


Metadata = dict[str, Any]

#: Keys that used to select the mask interpolator and threshold. Masks are now
#: always resampled with linear interpolation and thresholded at 0.5 (see
#: ``okapy.core.geometry.MASK_INTERPOLATOR``), so these keys are rejected.
_REMOVED_MASK_KEYS = ("mask_interpolator", "mask_threshold")


@dataclass(frozen=True)
class GeometryConfig:
    spacing: tuple[float, float, float]
    image_interpolator: str = "linear"
    crop_to_masks: bool = True
    crop_to_common_fov: bool = False
    padding_mm: float = 0.0
    default_image_value: float = 0.0
    default_mask_value: int = 0

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> GeometryConfig:
        removed = [key for key in _REMOVED_MASK_KEYS if key in config]
        if removed:
            raise ValueError(
                f"{removed} are no longer configurable: masks are always "
                "resampled with linear interpolation and thresholded at 0.5. "
                "Remove them from 'geometry_preprocessing'."
            )

        if "spacing" not in config:
            raise ValueError("Geometry preprocessing config requires 'spacing'.")

        spacing = tuple(float(x) for x in config["spacing"])

        if len(spacing) != 3:
            raise ValueError(f"'spacing' must have length 3, got {spacing}.")

        for value in spacing:
            if value <= 0 and value != -1:
                raise ValueError(
                    "'spacing' values must be > 0, or -1 to keep the native "
                    f"spacing along that axis. Got {spacing}."
                )

        return cls(
            spacing=spacing,
            image_interpolator=str(config.get("image_interpolator", "linear")),
            crop_to_masks=bool(config.get("crop_to_masks", True)),
            crop_to_common_fov=bool(config.get("crop_to_common_fov", False)),
            padding_mm=float(config.get("padding_mm", 0.0)),
            default_image_value=float(config.get("default_image_value", 0.0)),
            default_mask_value=int(config.get("default_mask_value", 0)),
        )

@dataclass(frozen=True)
class StudyPreprocessingConfig:
    """Study-level preprocessing options."""

    combine_segmentation: bool = False
    skip_images_without_masks: bool = False
    write_outputs: bool = True
    image_output_subdir: str = "images"
    mask_output_subdir: str = "masks"

    @classmethod
    def from_dict(cls, config: dict[str, Any] | None) -> StudyPreprocessingConfig:
        config = config or {}
        general = config.get("general", config)
        return cls(
            combine_segmentation=bool(general.get("combine_segmentation", False)),
            skip_images_without_masks=bool(general.get("skip_images_without_masks", False)),
            write_outputs=bool(general.get("write_outputs", True)),
            image_output_subdir=str(general.get("image_output_subdir", "images")),
            mask_output_subdir=str(general.get("mask_output_subdir", "masks")),
        )


@dataclass(frozen=True)
class PreprocessingPlan:
    """Concrete geometry plan for one image and its selected masks."""

    image: ImageVolume
    masks: list[MaskVolume]
    modality_key: str
    geometry_config: GeometryConfig
    physical_roi: PhysicalBox
    reference_grid: sitk.Image
    output_image_path: Path
    output_mask_dir: Path


@dataclass(frozen=True)
class PreprocessingDiagnostics:
    """Optional information useful for debugging and tests."""

    plans: list[PreprocessingPlan] = field(default_factory=list)
    skipped_images: list[ImageVolume] = field(default_factory=list)
