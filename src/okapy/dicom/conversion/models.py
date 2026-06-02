from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import SimpleITK as sitk


@dataclass(frozen=True)
class ConvertedImage:
    path: Path
    image: sitk.Image
    modality: str
    patient_id: str | None
    study_instance_uid: str | None
    series_instance_uid: str
    series_description: str | None = None
    submodality: str | None = None
    extra_dicom_tags: dict[str, object] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    @property
    def modality_key(self) -> str:
        if self.submodality is None:
            return self.modality
        return f"{self.modality}_{self.submodality}"

@dataclass(frozen=True)
class ConvertedMask:
    path: Path
    image: sitk.Image
    label: str
    modality: str
    reference_modality: str
    patient_id: str | None
    study_instance_uid: str | None
    reference_series_instance_uid: str
    metadata: dict = field(default_factory=dict)
