from __future__ import annotations

from pathlib import Path
import logging
from typing import Iterable

import numpy as np
import pydicom
import SimpleITK as sitk
from skimage.draw import polygon

from okapy.core.models import (
    ImageVolume,
    MaskIdentity,
    MaskVolume,
    VolumeStage,
)
from okapy.dicom.conversion.utils import (
    image_metadata,
    safe_name,
    sitk_mask_like_reference,
    write_image_unique,
)
from okapy.dicom.models import DicomSeries

logger = logging.getLogger(__name__)


class EmptyContourError(RuntimeError):
    """Raised when an RTSTRUCT ROI does not contain any usable contour."""


class RTStructMaskConverter:
    """Convert RTSTRUCT contours to binary masks on a reference image grid.

    The converter assumes that ``reference_image`` is the image volume on which
    the RTSTRUCT should be rasterized, usually the CT series referenced by the
    RTSTRUCT. In PET/CT workflows, reuse of CT masks on PT should happen later
    in preprocessing by resampling the resulting MaskVolume onto the PT grid.
    """

    def __init__(
        self,
        *,
        extension: str = "nii.gz",
        skip_empty: bool = True,
        check_reference: bool = True,
    ) -> None:
        self.extension = extension
        self.skip_empty = skip_empty
        self.check_reference = check_reference

    def convert(
        self,
        series: DicomSeries,
        reference_image: ImageVolume,
        output_dir: Path,
        labels: Iterable[str] | None = None,
    ) -> list[MaskVolume]:
        if not series.is_rtstruct:
            raise ValueError(f"Expected RTSTRUCT series, got {series.modality}.")

        datasets = [pydicom.dcmread(str(path)) for path in series.paths]

        if not datasets:
            raise ValueError(f"RTSTRUCT series {series.series_instance_uid} is empty.")

        if len(datasets) > 1:
            logger.warning(
                "Multiple RTSTRUCT files found in series %s. Using the first one.",
                series.series_instance_uid,
            )

        dcm = datasets[0]

        if self.check_reference:
            self._check_reference_series(
                dcm=dcm,
                reference_image=reference_image,
                rtstruct_series=series,
            )

        requested_labels = set(labels) if labels is not None else None
        roi_number_to_name = _roi_number_to_name(dcm)
        roi_number_to_contours = _roi_number_to_contours(dcm)

        output_dir = Path(output_dir)
        masks: list[MaskVolume] = []

        for roi_number, label in roi_number_to_name.items():
            if requested_labels is not None and label not in requested_labels:
                continue

            contour_sequence = roi_number_to_contours.get(roi_number, [])

            if not contour_sequence:
                message = f"RTSTRUCT ROI {label!r} has no contour sequence."
                if self.skip_empty:
                    logger.warning("Skipping %s", message)
                    continue
                raise EmptyContourError(message)

            mask_xyz = self._compute_mask(
                contour_sequence=contour_sequence,
                reference_image=reference_image.image,
                label=label,
            )

            if not np.any(mask_xyz):
                message = (
                    f"RTSTRUCT ROI {label!r} produced an empty mask on reference "
                    f"series {reference_image.series_instance_uid}."
                )
                if self.skip_empty:
                    logger.warning("Skipping %s", message)
                    continue
                raise EmptyContourError(message)

            mask_image = sitk_mask_like_reference(mask_xyz, reference_image.image)

            filename = self._make_filename(
                reference_image=reference_image,
                label=label,
            )
            path = write_image_unique(mask_image, output_dir / filename)

            masks.append(
                MaskVolume.from_sitk(
                    path=path,
                    image=mask_image,
                    identity=MaskIdentity(
                        label=label,
                        modality="RTSTRUCT",
                        reference_modality=reference_image.modality,
                        patient_id=reference_image.patient_id,
                        study_instance_uid=reference_image.study_instance_uid,
                        reference_series_instance_uid=reference_image.series_instance_uid,
                        metadata={
                            "rtstruct_series_instance_uid": series.series_instance_uid,
                            "roi_number": roi_number,
                        },
                    ),
                    stage=VolumeStage.CONVERTED,
                    target_identity=None,
                    metadata={
                        **image_metadata(mask_image),
                        "source": "dicom",
                        "converter": self.__class__.__name__,
                    },
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
        if reference_image.GetDimension() != 3:
            raise ValueError(
                f"RTSTRUCT conversion expects a 3D reference image, got "
                f"dimension={reference_image.GetDimension()}."
            )

        size_xyz = reference_image.GetSize()
        mask_xyz = np.zeros(size_xyz, dtype=np.uint8)

        for contour in contour_sequence:
            contour_data = getattr(contour, "ContourData", None)

            if contour_data is None:
                continue

            nodes = np.asarray(contour_data, dtype=float).reshape((-1, 3))

            if nodes.shape[0] < 3:
                logger.debug(
                    "Skipping contour with <3 points for label %s: shape=%s",
                    label,
                    nodes.shape,
                )
                continue

            voxel_indices = np.asarray(
                [
                    reference_image.TransformPhysicalPointToContinuousIndex(
                        tuple(float(x) for x in point)
                    )
                    for point in nodes
                ],
                dtype=float,
            )

            z_values = voxel_indices[:, 2]
            z_index = int(round(float(np.mean(z_values))))

            if z_index < 0 or z_index >= mask_xyz.shape[2]:
                raise RuntimeError(
                    f"RTSTRUCT contour for label {label!r} has out-of-bounds "
                    f"z index {z_index}; mask z size is {mask_xyz.shape[2]}."
                )

            # mask_xyz is indexed as [x, y, z]. skimage.draw.polygon returns
            # coordinates for the first and second array dimensions, so we pass
            # x and y coordinates in that order.
            rr, cc = polygon(voxel_indices[:, 0], voxel_indices[:, 1])

            if len(rr) == 0 or len(cc) == 0:
                continue

            if (
                np.min(rr) < 0
                or np.min(cc) < 0
                or np.max(rr) >= mask_xyz.shape[0]
                or np.max(cc) >= mask_xyz.shape[1]
            ):
                raise RuntimeError(
                    f"RTSTRUCT contour for label {label!r} is out of bounds. "
                    f"x range=({np.min(rr)}, {np.max(rr)}), "
                    f"y range=({np.min(cc)}, {np.max(cc)}), "
                    f"mask shape={mask_xyz.shape}."
                )

            mask_xyz[rr, cc, z_index] = 1

        return mask_xyz

    def _check_reference_series(
        self,
        *,
        dcm,
        reference_image: ImageVolume,
        rtstruct_series: DicomSeries,
    ) -> None:
        referenced_uids = set(_get_referenced_series_uids_from_rtstruct(dcm))

        if not referenced_uids:
            logger.warning(
                "Could not find referenced series UID in RTSTRUCT series %s.",
                rtstruct_series.series_instance_uid,
            )
            return

        if reference_image.series_instance_uid not in referenced_uids:
            logger.warning(
                "RTSTRUCT series %s references image series %s, but it is being "
                "rasterized on series %s.",
                rtstruct_series.series_instance_uid,
                sorted(referenced_uids),
                reference_image.series_instance_uid,
            )

    def _make_filename(
        self,
        *,
        reference_image: ImageVolume,
        label: str,
    ) -> str:
        patient = safe_name(reference_image.patient_id)
        label = safe_name(label)
        reference_modality = safe_name(reference_image.modality)
        reference_uid = safe_name(reference_image.series_instance_uid)

        return (
            f"{patient}__{label}__RTSTRUCT__{reference_modality}__"
            f"{reference_uid}.{self.extension}"
        )


def _roi_number_to_name(dcm) -> dict[int, str]:
    return {
        int(roi.ROINumber): str(roi.ROIName)
        for roi in getattr(dcm, "StructureSetROISequence", [])
    }


def _roi_number_to_contours(dcm) -> dict[int, object]:
    mapping = {}

    for roi_contour in getattr(dcm, "ROIContourSequence", []):
        roi_number = int(roi_contour.ReferencedROINumber)
        mapping[roi_number] = getattr(roi_contour, "ContourSequence", [])

    return mapping


def _get_referenced_series_uids_from_rtstruct(dcm) -> tuple[str, ...]:
    """Extract image series UIDs referenced by an RTSTRUCT dataset."""

    refs: list[str] = []

    for frame_ref in getattr(dcm, "ReferencedFrameOfReferenceSequence", []):
        for study_ref in getattr(frame_ref, "RTReferencedStudySequence", []):
            for series_ref in getattr(study_ref, "RTReferencedSeriesSequence", []):
                uid = getattr(series_ref, "SeriesInstanceUID", None)
                if uid is not None:
                    refs.append(str(uid))

    # Some objects may also contain the direct sequence.
    for series_ref in getattr(dcm, "ReferencedSeriesSequence", []):
        uid = getattr(series_ref, "SeriesInstanceUID", None)
        if uid is not None:
            refs.append(str(uid))

    return tuple(dict.fromkeys(refs))
