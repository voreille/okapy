from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


IMAGE_MODALITIES = {
    "CT",
    "MR",
    "PT",
    "NM",
    "US",
    "XA",
    "CR",
    "DX",
    "MG",
    "IO",
}

SPECIAL_MODALITIES = {
    "RTSTRUCT",
    "SEG",
}

ALLOWED_MODALITIES = IMAGE_MODALITIES | SPECIAL_MODALITIES


@dataclass(frozen=True)
class DicomFileRecord:
    """Lightweight metadata record for one DICOM file.

    This object is produced by DicomWalker and consumed by collectors/groupers.
    It should contain only cheap-to-read metadata, not pixel data.
    """

    path: Path
    sop_class_uid: str | None
    modality: str
    patient_id: str | None
    study_instance_uid: str | None
    series_instance_uid: str | None
    sop_instance_uid: str | None
    instance_number: int | None
    series_description: str | None
    image_position_patient: tuple[float, float, float] | None
    image_orientation_patient: (
        tuple[
            float,
            float,
            float,
            float,
            float,
            float,
        ]
        | None
    )
    referenced_series_uids: tuple[str, ...] = field(default_factory=tuple)
    extra_dicom_tags: dict[str, object] = field(default_factory=dict)

    @property
    def is_image(self) -> bool:
        return self.modality in IMAGE_MODALITIES

    @property
    def is_rtstruct(self) -> bool:
        return self.modality == "RTSTRUCT"

    @property
    def is_seg(self) -> bool:
        return self.modality == "SEG"

    @property
    def is_mask(self) -> bool:
        return self.is_rtstruct or self.is_seg


@dataclass(frozen=True)
class DicomSeries:
    """A DICOM series grouped from file records."""

    series_instance_uid: str
    modality: str
    records: tuple[DicomFileRecord, ...]

    @property
    def extra_dicom_tags(self) -> dict[str, object]:
        if not self.records:
            return {}

        # Use first record because series-level tags should be identical.
        return dict(self.records[0].extra_dicom_tags)

    @property
    def series_description(self) -> str | None:
        for record in self.records:
            if record.series_description is not None:
                return record.series_description
        return None

    @property
    def paths(self) -> tuple[Path, ...]:
        return tuple(record.path for record in self.records)

    @property
    def patient_id(self) -> str | None:
        return self.records[0].patient_id if self.records else None

    @property
    def study_instance_uid(self) -> str | None:
        return self.records[0].study_instance_uid if self.records else None

    @property
    def is_image(self) -> bool:
        return self.modality in IMAGE_MODALITIES

    @property
    def is_rtstruct(self) -> bool:
        return self.modality == "RTSTRUCT"

    @property
    def is_seg(self) -> bool:
        return self.modality == "SEG"

    @property
    def is_mask(self) -> bool:
        return self.is_rtstruct or self.is_seg


@dataclass(frozen=True)
class DicomStudy:
    """A DICOM study containing image series and mask objects."""

    study_instance_uid: str
    patient_id: str | None
    series: tuple[DicomSeries, ...] = field(default_factory=tuple)

    @property
    def image_series(self) -> tuple[DicomSeries, ...]:
        return tuple(s for s in self.series if s.is_image)

    @property
    def rtstruct_series(self) -> tuple[DicomSeries, ...]:
        return tuple(s for s in self.series if s.is_rtstruct)

    @property
    def seg_series(self) -> tuple[DicomSeries, ...]:
        return tuple(s for s in self.series if s.is_seg)

    @property
    def mask_series(self) -> tuple[DicomSeries, ...]:
        return self.rtstruct_series + self.seg_series


@dataclass(frozen=True)
class DicomCollection:
    """Top-level grouped DICOM collection."""

    studies: tuple[DicomStudy, ...]

    @property
    def image_series(self) -> tuple[DicomSeries, ...]:
        return tuple(series for study in self.studies for series in study.image_series)

    @property
    def rtstruct_series(self) -> tuple[DicomSeries, ...]:
        return tuple(
            series for study in self.studies for series in study.rtstruct_series
        )

    @property
    def seg_series(self) -> tuple[DicomSeries, ...]:
        return tuple(series for study in self.studies for series in study.seg_series)

    @property
    def mask_series(self) -> tuple[DicomSeries, ...]:
        return self.rtstruct_series + self.seg_series
