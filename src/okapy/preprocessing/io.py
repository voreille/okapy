from __future__ import annotations

from okapy.dicom.conversion.utils import (
    fit_filename,
    make_filename,
    safe_name,
    short_uid,
)


def image_output_name(
    *, patient_id: str | None, modality_key: str, series_instance_uid: str
) -> str:
    return make_filename(
        patient_id, modality_key, short_uid(series_instance_uid), extension="nii.gz"
    )


def mask_output_name(
    *,
    patient_id: str | None,
    label: str,
    target_modality_key: str | None,
    target_series_instance_uid: str | None,
    reference_series_instance_uid: str,
) -> str:
    # Not make_filename: it would re-truncate "ref-<uid>" to 128 characters.
    return fit_filename(
        f"{safe_name(patient_id)}__"
        f"{safe_name(target_modality_key)}__"
        f"{short_uid(target_series_instance_uid)}__"
        f"{safe_name(label)}__"
        f"ref-{short_uid(reference_series_instance_uid)}",
        "nii.gz",
    )
