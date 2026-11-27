from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pydicom
import SimpleITK as sitk

from okapy.core.models import (
    ImageVolume,
    MaskIdentity,
    MaskVolume,
    VolumeStage,
)
from okapy.dicom.conversion.utils import (
    ensure_inside,
    image_metadata,
    make_filename,
    referenced_series_uids,
    short_uid,
    sitk_mask_like_reference,
    to_jsonable,
    write_image_unique,
)
from okapy.dicom.models import DicomSeries

from .seg_highdicom_adapter import (
    FrameBetweenSlicesError,
    parse_seg,
    read_seg_segment_infos,
    read_seg_with_highdicom,
    segment_to_sitk,
    segments_on_reference_grid,
)

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

        seg = parse_seg(dcm)
        segment_numbers, segment_infos = read_seg_segment_infos(seg)

        # Frames on the reference grid are placed slice by slice; SEGs on
        # another in-plane grid, or with frames between the reference slices,
        # are rebuilt as a volume and resampled.
        try:
            masks_xyz = segments_on_reference_grid(dcm, reference_image.image)
        except FrameBetweenSlicesError as exc:
            logger.warning(
                "SEG series %s is not on the slices of reference series %s (%s). "
                "Resampling it onto the reference grid.",
                series.series_instance_uid,
                reference_image.series_instance_uid,
                exc,
            )
            masks_xyz = None
        raw_volume = read_seg_with_highdicom(seg) if masks_xyz is None else None

        requested_labels = set(labels) if labels is not None else None

        masks: list[MaskVolume] = []

        for segment_number in segment_numbers:
            segment_info = dict(segment_infos[segment_number])
            label = str(segment_info.get("label", f"segment_{segment_number}"))

            if requested_labels is not None and label not in requested_labels:
                continue

            if masks_xyz is not None:
                mask_xyz = masks_xyz.get(
                    segment_number,
                    np.zeros(reference_image.image.GetSize(), dtype=np.uint8),
                )
                mask_image = sitk_mask_like_reference(mask_xyz, reference_image.image)
            else:
                native_mask_image = segment_to_sitk(raw_volume, segment_number)
                mask_image = sitk.Resample(
                    native_mask_image,
                    reference_image.image,
                    sitk.Transform(),
                    sitk.sitkNearestNeighbor,
                    0,
                    sitk.sitkUInt8,
                )

            filename = self._make_filename(
                reference_image=reference_image,
                label=label,
                segment_number=segment_number,
            )
            path = write_image_unique(
                mask_image, ensure_inside(output_dir / filename, output_dir)
            )

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
                            "segment_info": to_jsonable(segment_info),
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
        return make_filename(
            reference_image.patient_id,
            label,
            f"seg{int(segment_number)}",
            "SEG",
            reference_image.modality,
            short_uid(reference_image.series_instance_uid),
            extension=self.extension,
        )


def _check_reference_uid(
    *,
    dcm,
    reference_image: ImageVolume,
    series: DicomSeries,
) -> None:
    referenced_uids = set(series.referenced_series_uids)
    referenced_uids.update(referenced_series_uids(dcm, rtstruct=False, derivation=True))

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
