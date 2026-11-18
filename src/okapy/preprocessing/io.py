from __future__ import annotations

from pathlib import Path

import SimpleITK as sitk


def safe_name(value: object | None) -> str:
    if value is None:
        return "unknown"
    return (
        str(value)
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "-")
    )


def short_uid(uid: str | None) -> str:
    if uid is None:
        return "unknown"
    return str(uid).split(".")[-1]


def write_image_unique(image: sitk.Image, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not path.exists():
        sitk.WriteImage(image, str(path))
        return path

    stem = path.name
    if stem.endswith(".nii.gz"):
        base = stem.removesuffix(".nii.gz")
        suffix = ".nii.gz"
    else:
        base = path.stem
        suffix = path.suffix

    counter = 1
    while True:
        candidate = path.with_name(f"{base}({counter}){suffix}")
        if not candidate.exists():
            sitk.WriteImage(image, str(candidate))
            return candidate
        counter += 1


def image_output_name(*, patient_id: str | None, modality_key: str, series_instance_uid: str) -> str:
    return (
        f"{safe_name(patient_id)}__"
        f"{safe_name(modality_key)}__"
        f"{short_uid(series_instance_uid)}.nii.gz"
    )


def mask_output_name(
    *,
    patient_id: str | None,
    label: str,
    target_modality_key: str | None,
    target_series_instance_uid: str | None,
    reference_series_instance_uid: str,
) -> str:
    return (
        f"{safe_name(patient_id)}__"
        f"{safe_name(target_modality_key)}__"
        f"{short_uid(target_series_instance_uid)}__"
        f"{safe_name(label)}__"
        f"ref-{short_uid(reference_series_instance_uid)}.nii.gz"
    )
