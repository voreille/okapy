from __future__ import annotations

from dataclasses import dataclass

from okapy.core.models import SeriesIdentity
from okapy.dicom.models import DicomSeries


@dataclass(frozen=True)
class SeriesIdentityConfig:
    use_submodalities: bool = False
    submodality_separator: str = " --- "
    submodality_modalities: tuple[str, ...] = ("MR",)


def build_series_identity(
    series: DicomSeries,
    *,
    config: SeriesIdentityConfig | None = None,
) -> SeriesIdentity:
    config = config or SeriesIdentityConfig()

    submodality = get_series_submodality(
        series,
        config=config,
    )

    return SeriesIdentity(
        modality=series.modality,
        patient_id=series.patient_id,
        study_instance_uid=series.study_instance_uid,
        series_instance_uid=series.series_instance_uid,
        series_description=series.series_description,
        submodality=submodality,
        extra_dicom_tags=series.extra_dicom_tags,
    )


def get_series_submodality(
    series: DicomSeries,
    *,
    config: SeriesIdentityConfig,
) -> str | None:
    if not config.use_submodalities:
        return None

    if series.modality not in config.submodality_modalities:
        return None

    return parse_submodality_from_series_description(
        series.series_description,
        separator=config.submodality_separator,
    )


def parse_submodality_from_series_description(
    series_description: str | None,
    *,
    separator: str = " --- ",
) -> str | None:
    if series_description is None:
        return None

    parts = series_description.split(separator, maxsplit=1)

    if len(parts) != 2:
        return None

    submodality = parts[1].strip()

    if not submodality:
        return None

    return normalize_submodality(submodality)


def normalize_submodality(value: str) -> str:
    return value.strip().replace(" ", "_").replace("-", "_").replace("/", "_")
