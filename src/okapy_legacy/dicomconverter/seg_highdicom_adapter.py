from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import highdicom as hd
from pydicom.dataset import Dataset


@dataclass
class HighdicomSegmentReadResult:
    """Small compatibility wrapper replacing the pydicom_seg read result.

    It only implements what SegFile currently uses:
      - direction
      - spacing
      - origin
      - available_segments
      - segment_infos
      - segment_data(segment_number)
    """

    seg: hd.seg.Segmentation
    segments: dict[int, np.ndarray]
    segment_infos: dict[int, dict]
    direction: np.ndarray
    spacing: np.ndarray
    origin: np.ndarray

    @property
    def available_segments(self) -> list[int]:
        return list(self.segments.keys())

    def segment_data(self, segment_number: int) -> np.ndarray:
        return self.segments[segment_number]


def read_seg_with_highdicom(dcm: Dataset) -> HighdicomSegmentReadResult:
    """Read a DICOM SEG object using highdicom.

    Returns segment arrays in z, y, x order, matching pydicom_seg's convention
    expected by the old SegFile code.
    """

    # highdicom can wrap an already-read pydicom Dataset
    seg = hd.seg.Segmentation.from_dataset(dcm)

    segment_numbers = [int(n) for n in seg.get_segment_numbers()]

    segment_infos = {}
    for segment_number in segment_numbers:
        desc = seg.get_segment_description(segment_number)
        label = getattr(desc, "SegmentLabel", str(segment_number))
        segment_infos[segment_number] = {
            "label": str(label),
            "dataset": desc,
        }

    # Try to get a full volume per segment.
    # highdicom returns a highdicom.Volume-like object for volumetric SEG.
    segments = {}
    volume_geometry = None

    for segment_number in segment_numbers:
        volume = seg.get_volume(segment_number=segment_number)
        volume_geometry = volume

        # highdicom Volume array is generally spatial, but we normalize to z, y, x
        # for compatibility with old pydicom_seg SegmentReadResult.
        arr = np.asarray(volume.array)

        # Common case should already be 3D. If a singleton channel dimension appears,
        # squeeze it.
        arr = np.squeeze(arr)

        if arr.ndim != 3:
            raise ValueError(
                f"Expected 3D array for SEG segment {segment_number}, "
                f"got shape {arr.shape}."
            )

        segments[segment_number] = arr.astype(np.uint8)

    if volume_geometry is None:
        raise ValueError("SEG contains no readable segments.")

    # highdicom Volume exposes geometry. These attributes may differ slightly
    # across versions, so keep this isolated here.
    spacing = np.asarray(
        volume_geometry.get_pixel_measures().SpacingBetweenSlices, dtype=float
    )

    # Fallback geometry extraction from DICOM tags is more robust for legacy use.
    direction, voxel_spacing, origin = _geometry_from_dicom_seg(dcm)

    return HighdicomSegmentReadResult(
        seg=seg,
        segments=segments,
        segment_infos=segment_infos,
        direction=direction,
        spacing=voxel_spacing,
        origin=origin,
    )


def _geometry_from_dicom_seg(dcm: Dataset) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract direction, spacing, origin from a common patient-coordinate SEG.

    Returns:
      direction: 3x3 matrix with columns [row_cosines, column_cosines, normal]
      spacing: x, y, z spacing
      origin: x, y, z origin
    """

    shared = dcm.SharedFunctionalGroupsSequence[0]

    orientation = shared.PlaneOrientationSequence[0].ImageOrientationPatient
    row = np.asarray(orientation[:3], dtype=float)
    col = np.asarray(orientation[3:], dtype=float)
    normal = np.cross(row, col)
    direction = np.stack([row, col, normal], axis=1)

    measures = shared.PixelMeasuresSequence[0]

    # DICOM stores PixelSpacing as [row_spacing, column_spacing].
    row_spacing, col_spacing = [float(x) for x in measures.PixelSpacing]

    if hasattr(measures, "SpacingBetweenSlices"):
        z_spacing = float(measures.SpacingBetweenSlices)
    elif hasattr(measures, "SliceThickness"):
        z_spacing = float(measures.SliceThickness)
    else:
        z_spacing = 1.0

    spacing = np.asarray([col_spacing, row_spacing, z_spacing], dtype=float)

    first_frame = dcm.PerFrameFunctionalGroupsSequence[0]
    origin = np.asarray(
        first_frame.PlanePositionSequence[0].ImagePositionPatient,
        dtype=float,
    )

    return direction, spacing, origin
