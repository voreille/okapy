from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import highdicom as hd
import numpy as np
import SimpleITK as sitk
from pydicom.dataset import Dataset

from okapy.dicom.conversion.utils import SLICE_MATCH_TOLERANCE


@dataclass(frozen=True)
class HighdicomSegmentReadResult:
    """Decoded DICOM SEG volume and segment metadata.

    The highdicom volume contains spatial dimensions ordered as:

        slice, row, column

    and one final channel dimension containing the requested segments.
    """

    seg: hd.seg.Segmentation
    volume: hd.Volume
    segment_numbers: tuple[int, ...]
    segment_infos: dict[int, dict[str, Any]]

    @property
    def available_segments(self) -> list[int]:
        return list(self.segment_numbers)

    def segment_data(self, segment_number: int) -> np.ndarray:
        """Return one segment in z, y, x array order."""

        try:
            channel_index = self.segment_numbers.index(segment_number)
        except ValueError as exc:
            raise KeyError(
                f"Segment {segment_number} is not available. "
                f"Available segments: {self.segment_numbers}."
            ) from exc

        array = np.asarray(self.volume.array)

        if array.ndim == 3:
            # Defensive handling for a possible singleton-channel representation.
            if len(self.segment_numbers) != 1:
                raise RuntimeError(
                    "SEG volume has no segment channel dimension, but contains "
                    f"{len(self.segment_numbers)} segments."
                )
            return array

        if array.ndim != 4:
            raise RuntimeError(
                "Expected highdicom SEG volume with shape (z, y, x, segments), "
                f"got {array.shape}."
            )

        if array.shape[-1] != len(self.segment_numbers):
            raise RuntimeError(
                "SEG channel count does not match segment count: "
                f"array channels={array.shape[-1]}, "
                f"segments={len(self.segment_numbers)}."
            )

        return array[..., channel_index]


def read_seg_segment_infos(
    dcm: Dataset,
) -> tuple[tuple[int, ...], dict[int, dict[str, Any]]]:
    """Return the segment numbers and their labels / descriptions."""

    seg = hd.seg.Segmentation.from_dataset(dcm)
    return _segment_infos(seg)


def _segment_infos(
    seg: hd.seg.Segmentation,
) -> tuple[tuple[int, ...], dict[int, dict[str, Any]]]:
    segment_numbers = tuple(int(n) for n in seg.get_segment_numbers())

    if not segment_numbers:
        raise ValueError("SEG contains no segments.")

    segment_infos: dict[int, dict[str, Any]] = {}

    for segment_number in segment_numbers:
        description = seg.get_segment_description(segment_number)

        segment_infos[segment_number] = {
            "label": str(
                getattr(
                    description,
                    "SegmentLabel",
                    f"segment_{segment_number}",
                )
            ),
            "dataset": description,
        }

    return segment_numbers, segment_infos


def read_seg_with_highdicom(
    dcm: Dataset,
) -> HighdicomSegmentReadResult:
    """Read a regularly spaced patient-coordinate DICOM SEG."""

    seg = hd.seg.Segmentation.from_dataset(dcm)
    segment_numbers, segment_infos = _segment_infos(seg)

    volume = seg.get_volume(
        segment_numbers=segment_numbers,
        combine_segments=False,
        allow_missing_positions=True,
        rescale_fractional=True,
    )

    array = np.asarray(volume.array)

    if array.ndim not in {3, 4}:
        raise RuntimeError(
            "Expected a 3D SEG volume with an optional segment channel, "
            f"got shape {array.shape}."
        )

    return HighdicomSegmentReadResult(
        seg=seg,
        volume=volume,
        segment_numbers=segment_numbers,
        segment_infos=segment_infos,
    )


def segment_to_sitk(
    result: HighdicomSegmentReadResult,
    segment_number: int,
    *,
    fractional_threshold: float = 0.5,
) -> sitk.Image:
    """Convert one highdicom SEG segment into a binary SimpleITK image."""

    array = np.asarray(result.segment_data(segment_number))

    if array.ndim != 3:
        raise RuntimeError(f"Expected a 3D segment array, got shape {array.shape}.")

    segmentation_type = str(getattr(result.seg, "SegmentationType", "BINARY")).upper()

    if segmentation_type == "FRACTIONAL":
        binary_array = array >= fractional_threshold
    else:
        binary_array = array != 0

    # highdicom spatial order -> SimpleITK NumPy order
    array_for_sitk = binary_array.transpose(2, 1, 0).astype(np.uint8)

    image = sitk.GetImageFromArray(array_for_sitk)

    image.SetOrigin(tuple(float(x) for x in result.volume.position))
    image.SetSpacing(tuple(float(x) for x in result.volume.spacing))
    image.SetDirection(
        tuple(float(x) for x in np.asarray(result.volume.direction).ravel())
    )

    return image


def segments_on_reference_grid(
    dcm: Dataset,
    reference: sitk.Image,
    *,
    fractional_threshold: float = 0.5,
) -> dict[int, np.ndarray] | None:
    """Place the SEG frames on the slices of ``reference``.

    A SEG stores one 2D frame per segment and plane, each with its own
    position, so it can describe slices that are not evenly spaced (e.g.
    multi-bed PET). Instead of rebuilding a regular volume from those positions,
    each frame is put on the reference slice it lies on, the way RTSTRUCT
    contours are. This keeps the mask voxel-aligned with the reference image
    whatever the slice spacing.

    Returns binary masks in x, y, z order per segment number, or None when the
    frames do not share the in-plane grid of ``reference`` (other size,
    orientation, pixel spacing or in-plane origin); the caller then has to
    resample. Raises ValueError when a frame lies between reference slices or
    outside the reference volume.
    """

    size_xyz = reference.GetSize()
    n_frames = int(dcm.get("NumberOfFrames", 1))

    if (int(dcm.Columns), int(dcm.Rows)) != size_xyz[:2]:
        return None

    direction = np.asarray(reference.GetDirection(), dtype=float).reshape(3, 3)
    spacing = reference.GetSpacing()

    frames: list[tuple[int, int]] = []

    for frame in range(n_frames):
        orientation = _frame_attribute(
            dcm, frame, "PlaneOrientationSequence", "ImageOrientationPatient"
        )
        pixel_spacing = _frame_attribute(
            dcm, frame, "PixelMeasuresSequence", "PixelSpacing"
        )
        position = _frame_attribute(
            dcm, frame, "PlanePositionSequence", "ImagePositionPatient"
        )
        segment_number = _frame_attribute(
            dcm, frame, "SegmentIdentificationSequence", "ReferencedSegmentNumber"
        )

        if orientation is None or pixel_spacing is None or position is None:
            return None

        orientation = np.asarray(orientation, dtype=float)
        if not (
            np.allclose(orientation[:3], direction[:, 0], atol=1e-3)
            and np.allclose(orientation[3:], direction[:, 1], atol=1e-3)
        ):
            return None

        # PixelSpacing is (row spacing, column spacing) = (y, x).
        if not np.allclose(
            (float(pixel_spacing[1]), float(pixel_spacing[0])), spacing[:2], rtol=1e-3
        ):
            return None

        index = np.asarray(
            reference.TransformPhysicalPointToContinuousIndex(
                tuple(float(x) for x in position)
            )
        )

        if np.any(np.abs(index[:2]) > SLICE_MATCH_TOLERANCE):
            return None

        slice_index = round(index[2])
        offset = abs(index[2] - slice_index)

        if offset > SLICE_MATCH_TOLERANCE:
            raise ValueError(
                f"SEG frame {frame + 1} at {tuple(float(x) for x in position)} lies "
                f"{offset:.2f} slices from the nearest slice of the reference image "
                f"(tolerance {SLICE_MATCH_TOLERANCE})."
            )

        if not 0 <= slice_index < size_xyz[2]:
            raise ValueError(
                f"SEG frame {frame + 1} at {tuple(float(x) for x in position)} is "
                f"outside the reference image (slice index {slice_index}, "
                f"{size_xyz[2]} slices)."
            )

        frames.append((int(segment_number or 1), slice_index))

    pixels = np.asarray(dcm.pixel_array)
    if n_frames == 1:
        pixels = pixels[np.newaxis]

    if str(dcm.get("SegmentationType", "BINARY")).upper() == "FRACTIONAL":
        maximum = float(dcm.get("MaximumFractionalValue", 255))
        pixels = pixels / maximum >= fractional_threshold
    else:
        pixels = pixels != 0

    masks: dict[int, np.ndarray] = {}

    for frame, (segment_number, slice_index) in enumerate(frames):
        mask = masks.setdefault(segment_number, np.zeros(size_xyz, dtype=np.uint8))
        # Frames are stored as (row, column) = (y, x).
        mask[:, :, slice_index] |= pixels[frame].T.astype(np.uint8)

    return masks


def _frame_attribute(dcm: Dataset, frame: int, sequence: str, keyword: str):
    """Return ``keyword`` from the per-frame or shared functional groups."""

    groups = []
    if "PerFrameFunctionalGroupsSequence" in dcm:
        groups.append(dcm.PerFrameFunctionalGroupsSequence[frame])
    if "SharedFunctionalGroupsSequence" in dcm:
        groups.append(dcm.SharedFunctionalGroupsSequence[0])

    for group in groups:
        if sequence in group and len(group[sequence].value) > 0:
            item = group[sequence].value[0]
            if keyword in item:
                return item[keyword].value

    return None
