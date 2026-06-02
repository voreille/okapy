from __future__ import annotations

from pathlib import Path
import logging

import numpy as np
import pydicom
import SimpleITK as sitk
from skimage.draw import polygon

from okapy.dicom.models import DicomSeries
from okapy.dicom.conversion.models import ConvertedImage, ConvertedMask
from okapy.dicom.conversion.utils import (
    image_metadata,
    safe_name,
    sitk_mask_like_reference,
    write_image_unique,
)

logger = logging.getLogger(__name__)


class EmptyContourError(RuntimeError):
    pass


class RTStructMaskConverter:
    """Convert RTSTRUCT contours to binary masks on a reference image grid."""

    def __init__(self, *, extension: str = "nii.gz") -> None:
        self.extension = extension

    def convert(
        self,
        series: DicomSeries,
        reference_image: ConvertedImage,
        output_dir: Path,
        labels: list[str] | None = None,
    ) -> list[ConvertedMask]:
        if not series.is_rtstruct:
            raise ValueError(f"Expected RTSTRUCT series, got {series.modality}.")

        datasets = [pydicom.dcmread(str(path)) for path in series.paths]

        if len(datasets) > 1:
            logger.warning("Multiple RTSTRUCT files found in same series.")

        dcm = datasets[0]
        roi_number_to_name = _roi_number_to_name(dcm)
        roi_number_to_contours = _roi_number_to_contours(dcm)

        requested_labels = set(labels) if labels is not None else None
        masks = []

        for roi_number, label in roi_number_to_name.items():
            if requested_labels is not None and label not in requested_labels:
                continue

            contour_sequence = roi_number_to_contours.get(roi_number)

            if not contour_sequence:
                logger.warning("Skipping empty RTSTRUCT ROI: %s", label)
                continue

            mask_xyz = self._compute_mask(
                contour_sequence=contour_sequence,
                reference_image=reference_image.image,
                label=label,
            )

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
                    modality="RTSTRUCT",
                    reference_modality=reference_image.modality,
                    patient_id=reference_image.patient_id,
                    study_instance_uid=reference_image.study_instance_uid,
                    reference_series_instance_uid=reference_image.series_instance_uid,
                    metadata=image_metadata(mask_image),
                )
            )

        return masks

    def _compute_mask(
        self,
        *,
        contour_sequence,
        reference_image: sitk.Image,
        label: str,
    ) -> np.ndarray:
        size = reference_image.GetSize()
        mask_xyz = np.zeros(size, dtype=np.uint8)

        for contour in contour_sequence:
            nodes = np.asarray(contour.ContourData, dtype=float).reshape((-1, 3))

            voxel_indices = np.asarray(
                [
                    reference_image.TransformPhysicalPointToIndex(tuple(point))
                    for point in nodes
                ],
                dtype=float,
            )

            rr, cc = polygon(voxel_indices[:, 0], voxel_indices[:, 1])

            if len(rr) == 0 or len(cc) == 0:
                continue

            z_index = int(round(voxel_indices[0, 2]))

            if (
                np.min(rr) < 0
                or np.min(cc) < 0
                or np.max(rr) >= mask_xyz.shape[0]
                or np.max(cc) >= mask_xyz.shape[1]
                or z_index < 0
                or z_index >= mask_xyz.shape[2]
            ):
                raise RuntimeError(
                    f"RTSTRUCT contour for label {label!r} is out of bounds."
                )

            mask_xyz[rr, cc, z_index] = 1

        return mask_xyz

    def _make_filename(
        self,
        *,
        reference_image: ConvertedImage,
        label: str,
    ) -> str:
        patient = safe_name(reference_image.patient_id)
        label = safe_name(label)
        reference_modality = safe_name(reference_image.modality)

        return f"{patient}__{label}__RTSTRUCT__{reference_modality}.{self.extension}"


def _roi_number_to_name(dcm) -> dict[int, str]:
    return {
        int(roi.ROINumber): str(roi.ROIName)
        for roi in getattr(dcm, "StructureSetROISequence", [])
    }


def _roi_number_to_contours(dcm) -> dict[int, object]:
    mapping = {}

    for roi_contour in getattr(dcm, "ROIContourSequence", []):
        roi_number = int(roi_contour.ReferencedROINumber)

        if not hasattr(roi_contour, "ContourSequence"):
            mapping[roi_number] = []
            continue

        mapping[roi_number] = roi_contour.ContourSequence

    return mapping