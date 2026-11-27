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
#: conversion identity/walker options, feature result format).
_OTHER_GENERAL_KEYS = frozenset(
    {
        "submodalities",
        "submodality_separator",
        "submodality_modalities",
        "additional_dicom_tags",
        "result_format",
    }
)

_TRUE_STRINGS = frozenset({"true", "yes", "1", "on"})
_FALSE_STRINGS = frozenset({"false", "no", "0", "off"})


def parse_bool(value: Any, *, key: str) -> bool:
    """Parse a boolean config value strictly.

    Real booleans pass through; the strings ``true/yes/1/on`` and
    ``false/no/0/off`` (case-insensitive) are accepted. Anything else raises
    ``ValueError`` naming ``key``.
    """

    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in _TRUE_STRINGS:
            return True
        if lowered in _FALSE_STRINGS:
            return False

    raise ValueError(f"{key!r} must be a boolean (true/false), got {value!r}.")


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
        return tuple(as_voxels(item) for item in value)

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
                if value <= 0 and value != -1:
                    raise ValueError(
                        "'spacing' values must be > 0, or -1 to keep the native "
                        f"spacing along that axis. Got {spacing}."
                    )

        return cls(
            spacing=spacing,
            image_interpolator=str(config.get("image_interpolator", "linear")),
            crop_to_masks=parse_bool(
                config.get("crop_to_masks", True), key="crop_to_masks"
            ),
            crop_to_common_fov=parse_bool(
                config.get("crop_to_common_fov", False), key="crop_to_common_fov"
            ),
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
            combine_segmentation=parse_bool(
                general.get("combine_segmentation", False),
                key="combine_segmentation",
            ),
            skip_images_without_masks=(
                None
                if skip is None
                else parse_bool(skip, key="skip_images_without_masks")
            ),
            write_outputs=parse_bool(
                general.get("write_outputs", True), key="write_outputs"
            ),
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
