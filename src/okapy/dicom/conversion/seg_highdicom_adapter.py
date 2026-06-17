from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import highdicom as hd
import numpy as np
import SimpleITK as sitk
from pydicom.dataset import Dataset


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


def read_seg_with_highdicom(
    dcm: Dataset,
) -> HighdicomSegmentReadResult:
    """Read a regularly spaced patient-coordinate DICOM SEG."""

    seg = hd.seg.Segmentation.from_dataset(dcm)

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
