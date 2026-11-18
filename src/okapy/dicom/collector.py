# src/okapy/dicom/collector.py

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
import logging

from okapy.dicom.walker import DicomWalker

from okapy.dicom.models import (
    DicomCollection,
    DicomFileRecord,
    DicomSeries,
    DicomStudy,
    IMAGE_MODALITIES,
)

logger = logging.getLogger(__name__)


class DicomStudyCollector:
    """Collect DICOM records into studies and series.

    The collector is responsible for grouping and ordering records.

    It does not:
      - read pixel data
      - convert images
      - convert RTSTRUCT/SEG
      - preprocess images or masks
    """

    def __init__(
        self,
        walker: DicomWalker | None = None,
        *,
        drop_duplicate_sop_instances: bool = True,
        require_image_series: bool = True,
        additional_dicom_tags: Iterable[str] | None = None,
    ) -> None:
        self.walker = walker or DicomWalker(additional_dicom_tags=additional_dicom_tags)
        self.drop_duplicate_sop_instances = drop_duplicate_sop_instances
        self.require_image_series = require_image_series
        self.additional_dicom_tags = (
            tuple(additional_dicom_tags) if additional_dicom_tags else ()
        )

    def collect(
        self,
        root: Path | str | Iterable[Path | str],
    ) -> DicomCollection:
        records = self.walker.collect_records(root)

        if not records:
            raise RuntimeError(f"No valid DICOM files found in {root}")

        return self.collect_from_records(records)

    def collect_from_records(
        self,
        records: Iterable[DicomFileRecord],
    ) -> DicomCollection:
        records = list(records)

        if self.drop_duplicate_sop_instances:
            records = _drop_duplicate_sop_instances(records)

        studies = _build_studies(records)

        if self.require_image_series:
            studies = tuple(study for study in studies if len(study.image_series) > 0)

        if not studies:
            raise RuntimeError("No DICOM studies containing image series were found.")

        return DicomCollection(studies=studies)


def _build_studies(records: list[DicomFileRecord]) -> tuple[DicomStudy, ...]:
    records_by_study: dict[str, list[DicomFileRecord]] = defaultdict(list)

    for record in records:
        study_uid = record.study_instance_uid

        if study_uid is None:
            logger.warning(
                "Skipping DICOM file without StudyInstanceUID: %s",
                record.path,
            )
            continue

        records_by_study[study_uid].append(record)

    studies = []

    for study_uid, study_records in sorted(records_by_study.items()):
        series = _build_series(study_records)

        patient_ids = sorted(
            {
                record.patient_id
                for record in study_records
                if record.patient_id is not None
            }
        )

        patient_id = patient_ids[0] if patient_ids else None

        if len(patient_ids) > 1:
            logger.warning(
                "Study %s contains multiple PatientID values: %s. Using first one: %s",
                study_uid,
                patient_ids,
                patient_id,
            )

        studies.append(
            DicomStudy(
                study_instance_uid=study_uid,
                patient_id=patient_id,
                series=series,
            )
        )

    return tuple(studies)


def _build_series(
    study_records: list[DicomFileRecord],
) -> tuple[DicomSeries, ...]:
    records_by_series: dict[str, list[DicomFileRecord]] = defaultdict(list)

    for record in study_records:
        series_uid = record.series_instance_uid

        if series_uid is None:
            logger.warning(
                "Skipping DICOM file without SeriesInstanceUID: %s",
                record.path,
            )
            continue

        records_by_series[series_uid].append(record)

    series_list = []

    for series_uid, series_records in sorted(records_by_series.items()):
        sorted_records = _sort_series_records(series_records)

        modalities = sorted({record.modality for record in sorted_records})
        modality = modalities[0]

        if len(modalities) > 1:
            logger.warning(
                "Series %s contains multiple modalities: %s. Using first one: %s",
                series_uid,
                modalities,
                modality,
            )

        series_list.append(
            DicomSeries(
                series_instance_uid=series_uid,
                modality=modality,
                records=tuple(sorted_records),
            )
        )

    return tuple(
        sorted(
            series_list,
            key=lambda s: (
                _series_sort_group(s.modality),
                s.modality,
                s.series_instance_uid,
            ),
        )
    )


def _sort_series_records(
    records: list[DicomFileRecord],
) -> list[DicomFileRecord]:
    """Sort records inside a series.

    For image series, prefer geometry-based sorting when possible.
    Fall back to InstanceNumber and then path.

    For RTSTRUCT/SEG, there is usually one file, but sorting by path is stable.
    """

    if not records:
        return []

    modality = records[0].modality

    if modality in IMAGE_MODALITIES:
        return sorted(records, key=_image_sort_key)

    return sorted(records, key=lambda r: str(r.path))


def _image_sort_key(record: DicomFileRecord):
    # Prefer ImagePositionPatient z coordinate when available.
    # This is not perfect for oblique acquisitions, but it is a reasonable
    # first pass. A later geometry module can sort using the slice normal.
    if record.image_position_patient is not None:
        position_key = record.image_position_patient[2]
    else:
        position_key = float("inf")

    instance_key = (
        record.instance_number if record.instance_number is not None else float("inf")
    )

    sop_key = record.sop_instance_uid or ""

    return (
        position_key,
        instance_key,
        sop_key,
        str(record.path),
    )


def _series_sort_group(modality: str) -> int:
    if modality in IMAGE_MODALITIES:
        return 0

    if modality == "RTSTRUCT":
        return 1

    if modality == "SEG":
        return 2

    return 99


def _drop_duplicate_sop_instances(
    records: list[DicomFileRecord],
) -> list[DicomFileRecord]:
    """Drop duplicate SOPInstanceUID records.

    Duplicates can happen when the same DICOM file appears more than once
    in an unsorted input folder. We keep the first path in deterministic
    path order.
    """

    records = sorted(records, key=lambda r: str(r.path))

    seen: set[str] = set()
    deduplicated: list[DicomFileRecord] = []

    for record in records:
        sop_uid = record.sop_instance_uid

        if sop_uid is None:
            deduplicated.append(record)
            continue

        if sop_uid in seen:
            logger.warning(
                "Skipping duplicate SOPInstanceUID %s at %s",
                sop_uid,
                record.path,
            )
            continue

        seen.add(sop_uid)
        deduplicated.append(record)

    return deduplicated
