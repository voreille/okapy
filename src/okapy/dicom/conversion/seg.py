from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pydicom

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

from .seg_highdicom_adapter import read_seg_with_highdicom

logger = logging.getLogger(__name__)


class SegMaskConverter:
    """Convert a DICOM SEG object to binary masks on a reference image grid.

    The converter returns MaskVolume objects with stage=VolumeStage.CONVERTED.
    At this stage, target_identity is None: the mask is still represented on the
    native reference image grid. The preprocessing pipeline may later resample
    the same mask onto another image grid, for example PT in PET/CT.
    """

    def __init__(
        self,
        *,
        extension: str = "nii.gz",
        check_reference_uid: bool = True,
    ) -> None:
        self.extension = extension
        self.check_reference_uid = check_reference_uid

    def convert(
        self,
        series: DicomSeries,
        reference_image: ImageVolume,
        output_dir: Path,
        labels: list[str] | None = None,
    ) -> list[MaskVolume]:
        if not series.is_seg:
            raise ValueError(f"Expected SEG series, got {series.modality}.")

        if len(series.paths) != 1:
            raise RuntimeError(f"Expected one SEG file, got {len(series.paths)}.")

        dcm = pydicom.dcmread(str(series.paths[0]))

        if self.check_reference_uid:
            _check_reference_uid(
                dcm=dcm,
                reference_image=reference_image,
                series=series,
            )

        raw_volume = read_seg_with_highdicom(dcm)
        requested_labels = set(labels) if labels is not None else None

        masks: list[MaskVolume] = []

        for segment_number in raw_volume.available_segments:
            segment_info = dict(raw_volume.segment_infos[segment_number])
            label = str(segment_info.get("label", f"segment_{segment_number}"))

            if requested_labels is not None and label not in requested_labels:
                continue

            mask_xyz = _segment_data_to_xyz(
                raw_volume.segment_data(segment_number),
                reference_image=reference_image,
                label=label,
            )

            mask_image = sitk_mask_like_reference(mask_xyz, reference_image.image)

            filename = self._make_filename(
                reference_image=reference_image,
                label=label,
                segment_number=segment_number,
            )
            path = write_image_unique(mask_image, output_dir / filename)

            masks.append(
                MaskVolume.from_sitk(
                    path=path,
                    image=mask_image,
                    identity=MaskIdentity(
                        label=label,
                        modality="SEG",
                        reference_modality=reference_image.modality,
                        patient_id=reference_image.patient_id,
                        study_instance_uid=reference_image.study_instance_uid,
                        reference_series_instance_uid=reference_image.series_instance_uid,
                        metadata={
                            "seg_series_instance_uid": series.series_instance_uid,
                            "segment_number": int(segment_number),
                            "segment_info": _to_jsonable(segment_info),
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

        if not masks:
            logger.warning(
                "No SEG masks were converted for series %s. Requested labels: %s",
                series.series_instance_uid,
                labels,
            )

        return masks

    def _make_filename(
        self,
        *,
        reference_image: ImageVolume,
        label: str,
        segment_number: int,
    ) -> str:
        patient = safe_name(reference_image.patient_id)
        label = safe_name(label)
        reference_modality = safe_name(reference_image.modality)
        reference_uid = safe_name(_short_uid(reference_image.series_instance_uid))

        return (
            f"{patient}__{label}__seg{int(segment_number)}__SEG__"
            f"{reference_modality}__{reference_uid}.{self.extension}"
        )


def _segment_data_to_xyz(
    segment_data: np.ndarray,
    *,
    reference_image: ImageVolume,
    label: str,
) -> np.ndarray:
    """Convert SEG segment data to x, y, z array convention.

    The highdicom adapter currently returns z, y, x, matching the old
    pydicom_seg-based code. The previous implementation converted it to x, y, z
    with transpose (2, 1, 0), so we keep the same convention here.
    """

    mask_xyz = np.transpose(segment_data, (2, 1, 0)).astype(np.uint8)

    expected_size = tuple(int(x) for x in reference_image.image.GetSize())

    if mask_xyz.shape != expected_size:
        raise RuntimeError(
            f"SEG mask for label {label!r} does not match reference image size. "
            f"mask_xyz.shape={mask_xyz.shape}, reference size={expected_size}."
        )

    return mask_xyz


def _check_reference_uid(
    *,
    dcm,
    reference_image: ImageVolume,
    series: DicomSeries,
) -> None:
    referenced_uids = set(series.referenced_series_uids)
    referenced_uids.update(_referenced_series_uids_from_dataset(dcm))

    if not referenced_uids:
        logger.warning(
            "Could not find referenced series UID in SEG series %s.",
            series.series_instance_uid,
        )
        return

    if reference_image.series_instance_uid not in referenced_uids:
        raise RuntimeError(
            "SEG reference mismatch. "
            f"SEG references series {sorted(referenced_uids)}, but the provided "
            f"reference image is {reference_image.series_instance_uid}."
        )


def _referenced_series_uids_from_dataset(dcm) -> set[str]:
    """Best-effort extraction of referenced series UIDs from a DICOM SEG."""

    uids: set[str] = set()

    # Common direct path.
    for item in getattr(dcm, "ReferencedSeriesSequence", []):
        uid = getattr(item, "SeriesInstanceUID", None)
        if uid is not None:
            uids.add(str(uid))

    # Common nested path through DerivationImageSequence /
    # SourceImageSequence. This often contains referenced SOPs rather than a
    # series UID, but keep this best-effort logic for datasets that include it.
    for frame_group in getattr(dcm, "PerFrameFunctionalGroupsSequence", []):
        derivation_sequence = getattr(frame_group, "DerivationImageSequence", [])
        for derivation in derivation_sequence:
            for source in getattr(derivation, "SourceImageSequence", []):
                uid = getattr(source, "SeriesInstanceUID", None)
                if uid is not None:
                    uids.add(str(uid))

    # Shared functional groups can also carry derivation information.
    for shared_group in getattr(dcm, "SharedFunctionalGroupsSequence", []):
        derivation_sequence = getattr(shared_group, "DerivationImageSequence", [])
        for derivation in derivation_sequence:
            for source in getattr(derivation, "SourceImageSequence", []):
                uid = getattr(source, "SeriesInstanceUID", None)
                if uid is not None:
                    uids.add(str(uid))

    return uids


def _to_jsonable(value: Any) -> Any:
    """Convert pydicom/highdicom metadata to simple serializable values."""

    if value is None:
        return None

    if isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")

    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]

    return str(value)


def _short_uid(uid: str | None) -> str:
    if uid is None:
        return "unknown"
    return str(uid).split(".")[-1]