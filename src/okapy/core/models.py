from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import SimpleITK as sitk

Metadata = dict[str, Any]


class VolumeStage(str, Enum):
    CONVERTED = "converted"
    PREPROCESSED = "preprocessed"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class SeriesIdentity:
    """Identity of an image series.

    This stores semantic/DICOM identity, not voxel data.
    """

    modality: str
    patient_id: str | None
    study_instance_uid: str | None
    series_instance_uid: str

    series_description: str | None = None
    submodality: str | None = None
    extra_dicom_tags: dict[str, object] = field(default_factory=dict)

    @property
    def modality_key(self) -> str:
        if self.submodality is None:
            return self.modality
        return f"{self.modality}_{self.submodality}"


@dataclass(frozen=True)
class MaskIdentity:
    """Identity of a mask/segmentation object.

    reference_series_instance_uid is the image series on which the mask was
    originally defined, e.g. CT for an RTSTRUCT.
    """

    label: str
    modality: str
    reference_modality: str

    patient_id: str | None
    study_instance_uid: str | None
    reference_series_instance_uid: str

    metadata: Metadata = field(default_factory=dict)


@dataclass(frozen=True)
class ImageGeometry:
    """Image grid and physical-space mapping.

    This mirrors the SimpleITK geometry:
    index -> physical mm is defined by origin, spacing, and direction.
    """

    size: tuple[int, int, int]
    spacing: tuple[float, float, float]
    origin: tuple[float, float, float]
    direction: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.size) != 3:
            raise ValueError(f"size must have length 3, got {self.size}.")
        if len(self.spacing) != 3:
            raise ValueError(f"spacing must have length 3, got {self.spacing}.")
        if len(self.origin) != 3:
            raise ValueError(f"origin must have length 3, got {self.origin}.")
        if len(self.direction) != 9:
            raise ValueError(
                f"direction must have length 9 for 3D images, got {self.direction}."
            )
        if any(s <= 0 for s in self.spacing):
            raise ValueError(f"spacing values must be > 0, got {self.spacing}.")
        if any(s <= 0 for s in self.size):
            raise ValueError(f"size values must be > 0, got {self.size}.")

    @classmethod
    def from_sitk(cls, image: sitk.Image) -> ImageGeometry:
        if image.GetDimension() != 3:
            raise ValueError(
                f"Only 3D images are supported, got dimension={image.GetDimension()}."
            )

        return cls(
            size=tuple(int(x) for x in image.GetSize()),
            spacing=tuple(float(x) for x in image.GetSpacing()),
            origin=tuple(float(x) for x in image.GetOrigin()),
            direction=tuple(float(x) for x in image.GetDirection()),
        )

    def copy_to(self, image: sitk.Image) -> sitk.Image:
        """Copy this geometry to an existing SimpleITK image."""

        if image.GetDimension() != 3:
            raise ValueError(
                f"Only 3D images are supported, got dimension={image.GetDimension()}."
            )

        if tuple(image.GetSize()) != self.size:
            raise ValueError(
                "Cannot copy geometry to image with different size: "
                f"geometry size={self.size}, image size={tuple(image.GetSize())}."
            )

        image.SetSpacing(self.spacing)
        image.SetOrigin(self.origin)
        image.SetDirection(self.direction)
        return image

    def as_dict(self) -> dict[str, object]:
        return {
            "size": self.size,
            "spacing": self.spacing,
            "origin": self.origin,
            "direction": self.direction,
        }


@dataclass(frozen=True)
class ImageVolume:
    """Image volume used across conversion, preprocessing, and extraction."""

    path: Path
    image: sitk.Image
    identity: SeriesIdentity
    geometry: ImageGeometry

    stage: VolumeStage = VolumeStage.CONVERTED
    source_path: Path | None = None
    metadata: Metadata = field(default_factory=dict)

    @classmethod
    def from_sitk(
        cls,
        *,
        path: Path,
        image: sitk.Image,
        identity: SeriesIdentity,
        stage: VolumeStage = VolumeStage.CONVERTED,
        source_path: Path | None = None,
        metadata: Metadata | None = None,
    ) -> ImageVolume:
        return cls(
            path=path,
            image=image,
            identity=identity,
            geometry=ImageGeometry.from_sitk(image),
            stage=stage,
            source_path=source_path,
            metadata=metadata or {},
        )

    @property
    def modality(self) -> str:
        return self.identity.modality

    @property
    def modality_key(self) -> str:
        return self.identity.modality_key

    @property
    def patient_id(self) -> str | None:
        return self.identity.patient_id

    @property
    def study_instance_uid(self) -> str | None:
        return self.identity.study_instance_uid

    @property
    def series_instance_uid(self) -> str:
        return self.identity.series_instance_uid

    @property
    def submodality(self) -> str | None:
        return self.identity.submodality

    def with_image(
        self,
        image: sitk.Image,
        *,
        path: Path | None = None,
        stage: VolumeStage | None = None,
        source_path: Path | None = None,
        metadata: Metadata | None = None,
    ) -> ImageVolume:
        return ImageVolume.from_sitk(
            path=path or self.path,
            image=image,
            identity=self.identity,
            stage=stage or self.stage,
            source_path=source_path if source_path is not None else self.source_path,
            metadata={
                **self.metadata,
                **(metadata or {}),
            },
        )


@dataclass(frozen=True)
class MaskVolume:
    """Mask volume used across conversion, preprocessing, and extraction.

    For converted masks, target_identity is usually None.

    For preprocessed masks, target_identity is the image series/grid onto which
    the mask has been resampled. This is important for PET/CT when a CT
    RTSTRUCT is reused on PT.
    """

    path: Path
    image: sitk.Image
    identity: MaskIdentity
    geometry: ImageGeometry

    stage: VolumeStage = VolumeStage.CONVERTED
    target_identity: SeriesIdentity | None = None
    source_path: Path | None = None
    metadata: Metadata = field(default_factory=dict)

    @classmethod
    def from_sitk(
        cls,
        *,
        path: Path,
        image: sitk.Image,
        identity: MaskIdentity,
        stage: VolumeStage = VolumeStage.CONVERTED,
        target_identity: SeriesIdentity | None = None,
        source_path: Path | None = None,
        metadata: Metadata | None = None,
    ) -> MaskVolume:
        return cls(
            path=path,
            image=image,
            identity=identity,
            geometry=ImageGeometry.from_sitk(image),
            stage=stage,
            target_identity=target_identity,
            source_path=source_path,
            metadata=metadata or {},
        )

    @property
    def label(self) -> str:
        return self.identity.label

    @property
    def modality(self) -> str:
        return self.identity.modality

    @property
    def reference_modality(self) -> str:
        return self.identity.reference_modality

    @property
    def patient_id(self) -> str | None:
        return self.identity.patient_id

    @property
    def study_instance_uid(self) -> str | None:
        return self.identity.study_instance_uid

    @property
    def reference_series_instance_uid(self) -> str:
        return self.identity.reference_series_instance_uid

    @property
    def target_series_instance_uid(self) -> str | None:
        if self.target_identity is None:
            return None
        return self.target_identity.series_instance_uid

    @property
    def target_modality(self) -> str | None:
        if self.target_identity is None:
            return None
        return self.target_identity.modality

    @property
    def target_modality_key(self) -> str | None:
        if self.target_identity is None:
            return None
        return self.target_identity.modality_key

    def with_image(
        self,
        image: sitk.Image,
        *,
        path: Path | None = None,
        stage: VolumeStage | None = None,
        target_identity: SeriesIdentity | None = None,
        source_path: Path | None = None,
        metadata: Metadata | None = None,
    ) -> MaskVolume:
        return MaskVolume.from_sitk(
            path=path or self.path,
            image=image,
            identity=self.identity,
            stage=stage or self.stage,
            target_identity=target_identity
            if target_identity is not None
            else self.target_identity,
            source_path=source_path if source_path is not None else self.source_path,
            metadata={
                **self.metadata,
                **(metadata or {}),
            },
        )


@dataclass(frozen=True)
class ImageMaskSet:
    """One image volume and the masks aligned to it."""

    image: ImageVolume
    masks: list[MaskVolume] = field(default_factory=list)

    def __post_init__(self) -> None:
        for mask in self.masks:
            if mask.target_identity is not None:
                if (
                    mask.target_identity.series_instance_uid
                    != self.image.series_instance_uid
                ):
                    raise ValueError(
                        "Mask target identity does not match image identity: "
                        f"mask target={mask.target_identity.series_instance_uid}, "
                        f"image={self.image.series_instance_uid}."
                    )

            if mask.geometry != self.image.geometry:
                raise ValueError(
                    f"Mask geometry does not match image geometry for label "
                    f"{mask.label!r}."
                )


@dataclass(frozen=True)
class StudyVolumes:
    """Image and mask volumes for one study."""

    study_instance_uid: str | None
    image_mask_sets: list[ImageMaskSet] = field(default_factory=list)

    @property
    def images(self) -> list[ImageVolume]:
        return [item.image for item in self.image_mask_sets]

    @property
    def masks(self) -> list[MaskVolume]:
        return [mask for item in self.image_mask_sets for mask in item.masks]


@dataclass(frozen=True)
class VolumeCollection:
    """Volumes for a complete run, potentially multiple studies."""

    studies: list[StudyVolumes] = field(default_factory=list)

    @property
    def images(self) -> list[ImageVolume]:
        return [image for study in self.studies for image in study.images]

    @property
    def masks(self) -> list[MaskVolume]:
        return [mask for study in self.studies for mask in study.masks]
