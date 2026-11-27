"""Shared synthetic factories for unit tests."""

from __future__ import annotations

from pathlib import Path

import SimpleITK as sitk

from okapy.core.models import ImageVolume, MaskIdentity, MaskVolume, SeriesIdentity


def image_volume(
    image: sitk.Image,
    *,
    modality: str = "CT",
    path: Path | str = "image.nii.gz",
    patient_id: str = "P1",
    study_instance_uid: str = "1.2.3",
    series_instance_uid: str = "1.2.3.4",
) -> ImageVolume:
    return ImageVolume.from_sitk(
        path=Path(path),
        image=image,
        identity=SeriesIdentity(
            modality=modality,
            patient_id=patient_id,
            study_instance_uid=study_instance_uid,
            series_instance_uid=series_instance_uid,
        ),
    )


def mask_volume(
    mask: sitk.Image,
    *,
    label: str = "ROI",
    modality: str = "SEG",
    reference_modality: str = "CT",
    path: Path | str = "mask.nii.gz",
    patient_id: str = "P1",
    study_instance_uid: str = "1.2.3",
    reference_series_instance_uid: str = "1.2.3.4",
) -> MaskVolume:
    return MaskVolume.from_sitk(
        path=Path(path),
        image=mask,
        identity=MaskIdentity(
            label=label,
            modality=modality,
            reference_modality=reference_modality,
            patient_id=patient_id,
            study_instance_uid=study_instance_uid,
            reference_series_instance_uid=reference_series_instance_uid,
        ),
    )
