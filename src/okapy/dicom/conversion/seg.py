from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pydicom

from okapy.dicom.conversion.models import ConvertedImage, ConvertedMask
from okapy.dicom.conversion.utils import (
    image_metadata,
    safe_name,
    sitk_mask_like_reference,
    write_image_unique,
)
from okapy.dicom.models import DicomSeries

from .seg_highdicom_adapter import read_seg_with_highdicom

logger = logging.getLogger(__name__)


class SegMaskConverter:
    """Convert DICOM SEG to binary masks on the reference image grid."""

    def __init__(self, *, extension: str = "nii.gz") -> None:
        self.extension = extension

    def convert(
        self,
        series: DicomSeries,
        reference_image: ConvertedImage,
        output_dir: Path,
        labels: list[str] | None = None,
    ) -> list[ConvertedMask]:
        if not series.is_seg:
            raise ValueError(f"Expected SEG series, got {series.modality}.")

        if len(series.paths) != 1:
            raise RuntimeError(f"Expected one SEG file, got {len(series.paths)}.")

        dcm = pydicom.dcmread(str(series.paths[0]))
        raw_volume = read_seg_with_highdicom(dcm)

        requested_labels = set(labels) if labels is not None else None
        masks = []

        for segment_number in raw_volume.available_segments:
            label = raw_volume.segment_infos[segment_number]["label"]

            if requested_labels is not None and label not in requested_labels:
                continue

            # highdicom adapter currently returns z, y, x, matching old pydicom_seg.
            # Old code converted it to x, y, z with transpose (2, 1, 0).
            mask_xyz = np.transpose(
                raw_volume.segment_data(segment_number),
                (2, 1, 0),
            ).astype(np.uint8)

            mask_image = sitk_mask_like_reference(mask_xyz, reference_image.image)

            filename = self._make_filename(
                reference_image=reference_image,
                label=label,
            )
            path = write_image_unique(mask_image, output_dir / filename)

            masks.append(
                ConvertedMask(
                    path=path,
                    image=mask_image,
                    label=label,
                    modality="SEG",
                    reference_modality=reference_image.modality,
                    patient_id=reference_image.patient_id,
                    study_instance_uid=reference_image.study_instance_uid,
                    reference_series_instance_uid=reference_image.series_instance_uid,
                    metadata=image_metadata(mask_image),
                )
            )

        return masks

    def _make_filename(
        self,
        *,
        reference_image: ConvertedImage,
        label: str,
    ) -> str:
        patient = safe_name(reference_image.patient_id)
        label = safe_name(label)
        reference_modality = safe_name(reference_image.modality)

        return f"{patient}__{label}__SEG__{reference_modality}.{self.extension}"
