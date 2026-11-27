from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import SimpleITK as sitk

# ``parse_bool`` is re-exported here: it used to live in this module.
from okapy.core.config_values import get_bool, parse_bool
from okapy.core.geometry import PhysicalBox
from okapy.core.models import ImageVolume, MaskVolume

Metadata = dict[str, Any]

#: Keys that used to select the mask interpolator and threshold. Masks are now
#: always resampled with linear interpolation and thresholded at 0.5 (see
#: ``okapy.core.geometry.MASK_INTERPOLATOR``), so these keys are rejected.
_REMOVED_MASK_KEYS = ("mask_interpolator", "mask_threshold")

_GEOMETRY_KEYS = frozenset(
    {
        "spacing",
        "image_interpolator",
        "crop_to_masks",
        "crop_to_common_fov",
        "padding_voxels",
        "default_image_value",
        "default_mask_value",
    }
)

#: Keys of the ``general`` section read by the preprocessing step.
_STUDY_KEYS = frozenset(
    {
        "combine_segmentation",
        "skip_images_without_masks",
        "write_outputs",
        "image_output_subdir",
        "mask_output_subdir",
    }
)

#: Keys of the ``general`` section read by other parts of okapy (DICOM
#: conversion identity/walker options).
_OTHER_GENERAL_KEYS = frozenset(
    {
        "submodalities",
        "submodality_separator",
        "submodality_modalities",
        "additional_dicom_tags",
    }
)


def _reject_unknown_keys(
    config: dict[str, Any], allowed: frozenset[str], *, section: str
) -> None:
    unknown = sorted(str(key) for key in config if key not in allowed)
    if unknown:
        raise ValueError(
            f"Unknown key(s) {unknown} in {section!r}. Allowed keys: {sorted(allowed)}."
        )


def _parse_padding_voxels(value: Any) -> int | tuple[int, int, int]:
    def as_voxels(item: Any) -> int:
        if (
            isinstance(item, bool)
            or not isinstance(item, int | float)
            or not math.isfinite(item)
            or float(item) != int(item)
            or int(item) < 0
        ):
            raise ValueError(
                "'padding_voxels' must be a non-negative integer or a list of "
                f"3 non-negative integers, got {value!r}."
            )
        return int(item)

    if isinstance(value, list | tuple):
        if len(value) != 3:
            raise ValueError(
                "'padding_voxels' must have 3 values (x, y, z), got "
                f"{len(value)}: {value!r}."
            )
        x, y, z = (as_voxels(item) for item in value)
        return (x, y, z)

    return as_voxels(value)


@dataclass(frozen=True)
class GeometryConfig:
    #: Target spacing in mm; ``None`` keeps the native spacing and ``-1`` on an
    #: axis keeps the native spacing along that axis.
    spacing: tuple[float, float, float] | None = None
    image_interpolator: str = "linear"
    crop_to_masks: bool = True
    crop_to_common_fov: bool = False
    #: Voxels of the target grid added around the masks on each side, either
    #: one value for every axis or one per axis (x, y, z).
    padding_voxels: int | tuple[int, int, int] = 0
    default_image_value: float = 0.0
    default_mask_value: int = 0

    def __post_init__(self) -> None:
        # Single source of padding validation, also for configs built directly
        # (from_dict already passes a parsed value, which re-parses unchanged).
        object.__setattr__(
            self, "padding_voxels", _parse_padding_voxels(self.padding_voxels)
        )

    @classmethod
    def from_dict(cls, config: dict[str, Any] | None) -> GeometryConfig:
        config = config or {}

        removed = [key for key in _REMOVED_MASK_KEYS if key in config]
        if removed:
            raise ValueError(
                f"{removed} are no longer configurable: masks are always "
                "resampled with linear interpolation and thresholded at 0.5. "
                "Remove them from 'geometry_preprocessing'."
            )

        if "padding_mm" in config:
            raise ValueError(
                "'padding_mm' was replaced by 'padding_voxels': the padding is "
                "now a number of voxels of the target grid."
            )

        _reject_unknown_keys(config, _GEOMETRY_KEYS, section="geometry_preprocessing")

        padding_voxels = _parse_padding_voxels(config.get("padding_voxels", 0))

        spacing = config.get("spacing")
        if spacing is not None:
            spacing = tuple(float(x) for x in spacing)

            if len(spacing) != 3:
                raise ValueError(f"'spacing' must have length 3, got {spacing}.")

            for value in spacing:
                if not math.isfinite(value):
                    raise ValueError(
                        f"'spacing' values must be finite numbers. Got {spacing}."
                    )
                if value <= 0 and value != -1:
                    raise ValueError(
                        "'spacing' values must be > 0, or -1 to keep the native "
                        f"spacing along that axis. Got {spacing}."
                    )

        return cls(
            spacing=spacing,
            image_interpolator=str(config.get("image_interpolator", "linear")),
            crop_to_masks=get_bool(config, "crop_to_masks", True),
            crop_to_common_fov=get_bool(config, "crop_to_common_fov", False),
            padding_voxels=padding_voxels,
            default_image_value=float(config.get("default_image_value", 0.0)),
            default_mask_value=int(config.get("default_mask_value", 0)),
        )


@dataclass(frozen=True)
class StudyPreprocessingConfig:
    """Study-level preprocessing options."""

    combine_segmentation: bool = False
    #: ``None`` skips images without masks (with a warning) only when their
    #: geometry config crops to masks; ``True`` always skips them; ``False``
    #: never does, which is an error together with ``crop_to_masks``.
    skip_images_without_masks: bool | None = None
    write_outputs: bool = True
    image_output_subdir: str = "images"
    mask_output_subdir: str = "masks"

    @classmethod
    def from_dict(cls, config: dict[str, Any] | None) -> StudyPreprocessingConfig:
        config = config or {}
        general = config.get("general") or {}

        misplaced = sorted(str(key) for key in config if key in _STUDY_KEYS)
        if misplaced:
            raise ValueError(
                f"Study option(s) {misplaced} must be set under 'general', not "
                "at the top level of the config."
            )

        if "result_format" in general:
            raise ValueError(
                "'result_format' in 'general' is a legacy option and is ignored "
                "in a new-shape config: set 'feature_extraction.result_format' "
                "instead."
            )

        if "padding" in general:
            raise ValueError(
                "'padding' in 'general' is a legacy option: load the config "
                "with okapy.config.load_config to migrate it, or set "
                "'geometry_preprocessing.<selector>.padding_voxels'."
            )

        _reject_unknown_keys(
            general, _STUDY_KEYS | _OTHER_GENERAL_KEYS, section="general"
        )

        skip = general.get("skip_images_without_masks")

        return cls(
            combine_segmentation=get_bool(general, "combine_segmentation", False),
            skip_images_without_masks=(
                None
                if skip is None
                else parse_bool(skip, key="skip_images_without_masks")
            ),
            write_outputs=get_bool(general, "write_outputs", True),
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
    #: ``None`` when outputs are not written.
    output_image_path: Path | None
    output_mask_dir: Path | None


@dataclass(frozen=True)
class PreprocessingDiagnostics:
    """Optional information useful for debugging and tests."""

    plans: list[PreprocessingPlan] = field(default_factory=list)
    skipped_images: list[ImageVolume] = field(default_factory=list)
