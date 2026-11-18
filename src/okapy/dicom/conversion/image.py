from __future__ import annotations

import logging
from pathlib import Path
from statistics import mode

import numpy as np
import pydicom

from okapy.core.models import ImageVolume, VolumeStage
from okapy.dicom.conversion.utils import (
    safe_name,
    short_uid,
    sitk_image_from_array_xyz,
    write_image_unique,
)
from okapy.dicom.identity import SeriesIdentityConfig, build_series_identity
from okapy.dicom.models import DicomSeries

logger = logging.getLogger(__name__)


class SimpleITKImageSeriesConverter:
    """Convert CT/MR-like DICOM image series to NIfTI.

    This intentionally keeps logic close to the old implementation:
    - sort slices by ImagePositionPatient projected on slice normal
    - remove inconsistent Rows/Columns
    - remove duplicate slice positions
    - handle one missing slice by interpolation
    - handle larger one-gap discontinuity by filling with modality-specific value
    """

    def __init__(
        self,
        *,
        extension: str = "nii.gz",
        dtype: np.dtype = np.float32,
        identity_config: SeriesIdentityConfig | None = None,
    ) -> None:
        self.extension = extension
        self.dtype = dtype
        self.identity_config = identity_config

    def convert(self, series: DicomSeries, output_dir: Path) -> ImageVolume:
        if not series.is_image:
            raise ValueError(f"Expected image series, got {series.modality}.")

        slices, paths, orthogonal_positions = self._read_and_sort_slices(series)
        slices, paths, orthogonal_positions = self._clean_slices(
            slices,
            paths,
            orthogonal_positions,
        )

        image_xyz = self._get_physical_values(slices, paths, series.modality)
        image_xyz = np.transpose(image_xyz, (1, 0, 2))

        d_slices = np.diff(np.asarray(orthogonal_positions, dtype=float))
        slice_spacing = _mode_rounded(d_slices)
        n_missing_slices, slice_discontinuities = _check_missing_slices(
            d_slices=d_slices,
            slice_spacing=slice_spacing,
        )

        if np.sum(slice_discontinuities) > 1:
            raise ValueError(
                f"Too many slice discontinuities in series "
                f"{series.series_instance_uid}."
            )

        if n_missing_slices == 1:
            logger.warning(
                "One slice is missing in series %s. Replacing it by interpolation.",
                series.series_instance_uid,
            )
            image_xyz = _interp_missing_slice(image_xyz, d_slices)

        if np.sum(slice_discontinuities) == 1 and n_missing_slices > 1:
            logger.warning(
                "Multiple slices are missing in series %s. Filling gap.",
                series.series_instance_uid,
            )
            image_xyz = _fill_discontinuity(
                image_xyz,
                modality=series.modality,
                n_missing_slices=n_missing_slices,
                slice_discontinuities=slice_discontinuities,
            )

        image_xyz = image_xyz.astype(self.dtype, copy=False)

        origin, spacing, direction = _geometry_from_slices(
            slices=slices,
            expected_n_slices=image_xyz.shape[2],
        )

        image = sitk_image_from_array_xyz(
            image_xyz,
            origin=origin,
            spacing=spacing,
            direction=direction,
        )

        filename = self._make_filename(series)
        path = write_image_unique(image, output_dir / filename)

        identity = build_series_identity(
            series,
            config=self.identity_config,
        )

        return ImageVolume.from_sitk(
            path=path,
            image=image,
            identity=identity,
            stage=VolumeStage.CONVERTED,
            metadata={
                "source": "dicom",
                "converter": self.__class__.__name__,
            },
        )

    def _read_and_sort_slices(self, series: DicomSeries):
        slices = [pydicom.dcmread(str(path)) for path in series.paths]
        paths = list(series.paths)

        orientation = np.asarray(slices[0].ImageOrientationPatient, dtype=float)
        normal = np.cross(orientation[:3], orientation[3:])

        orthogonal_positions = [
            float(np.dot(normal, np.asarray(s.ImagePositionPatient, dtype=float)))
            for s in slices
        ]

        zipped = list(zip(slices, paths, orthogonal_positions))
        zipped.sort(key=lambda x: x[2])

        slices, paths, orthogonal_positions = zip(*zipped)
        return list(slices), list(paths), list(orthogonal_positions)

    def _clean_slices(self, slices, paths, orthogonal_positions):
        slices, paths, orthogonal_positions = _keep_most_common_shape(
            slices,
            paths,
            orthogonal_positions,
            dimension="Rows",
        )

        slices, paths, orthogonal_positions = _keep_most_common_shape(
            slices,
            paths,
            orthogonal_positions,
            dimension="Columns",
        )

        slices, paths, orthogonal_positions = _drop_duplicate_positions(
            slices,
            paths,
            orthogonal_positions,
        )

        return slices, paths, orthogonal_positions

    def _get_physical_values(self, slices, paths, modality: str) -> np.ndarray:
        if modality == "CT":
            arrays = [
                float(s.RescaleSlope) * s.pixel_array + float(s.RescaleIntercept)
                for s in slices
            ]
        elif modality in {"MR", "NM", "US", "XA", "CR", "DX", "MG", "IO"}:
            arrays = [s.pixel_array for s in slices]
        elif modality == "PT":
            arrays = [
                float(getattr(s, "RescaleSlope", 1.0)) * s.pixel_array
                + float(getattr(s, "RescaleIntercept", 0.0))
                for s in slices
            ]
        else:
            arrays = [s.pixel_array for s in slices]

        return np.stack(arrays, axis=-1)

    def _make_filename(self, series: DicomSeries) -> str:
        patient = safe_name(series.patient_id)
        modality = safe_name(series.modality)
        uid = short_uid(series.series_instance_uid)
        return f"{patient}__{modality}__{uid}.{self.extension}"


def _keep_most_common_shape(slices, paths, orthogonal_positions, *, dimension: str):
    values = [getattr(s, dimension) for s in slices]
    unique, counts = np.unique(values, return_counts=True)
    keep_value = unique[np.argmax(counts)]

    keep = [i for i, value in enumerate(values) if value == keep_value]

    if len(keep) != len(slices):
        logger.warning(
            "Dropping %d slices with inconsistent %s.",
            len(slices) - len(keep),
            dimension,
        )

    return (
        [slices[i] for i in keep],
        [paths[i] for i in keep],
        [orthogonal_positions[i] for i in keep],
    )


def _drop_duplicate_positions(slices, paths, orthogonal_positions):
    if not slices:
        return slices, paths, orthogonal_positions

    keep = [0]

    for i in range(1, len(orthogonal_positions)):
        if orthogonal_positions[i] != orthogonal_positions[i - 1]:
            keep.append(i)

    if len(keep) != len(slices):
        logger.warning(
            "Dropping %d duplicated slice positions.", len(slices) - len(keep)
        )

    return (
        [slices[i] for i in keep],
        [paths[i] for i in keep],
        [orthogonal_positions[i] for i in keep],
    )


def _mode_rounded(values: np.ndarray) -> float:
    if len(values) == 0:
        return 1.0

    return float(mode(np.round(values, decimals=5)))


def _check_missing_slices(
    *,
    d_slices: np.ndarray,
    slice_spacing: float,
) -> tuple[int, np.ndarray]:
    if slice_spacing == 0:
        raise RuntimeError(
            "The most frequent slice spacing is 0, probably due to "
            "multi-channel or duplicate slice data."
        )

    if np.min(np.abs(d_slices)) == 0:
        raise RuntimeError("Some slices have the same position.")

    slice_discontinuities = np.abs(d_slices - slice_spacing) > 0.9 * abs(slice_spacing)

    n_missing_slices = np.round(
        np.sum(d_slices[slice_discontinuities] / slice_spacing - 1)
    ).astype(int)

    return int(n_missing_slices), slice_discontinuities


def _interp_missing_slice(image_xyz: np.ndarray, d_slices: np.ndarray) -> np.ndarray:
    mean_slice_spacing = np.mean(d_slices)
    errors = np.abs(d_slices - mean_slice_spacing)
    idx = int(np.where(errors > 0.5 * mean_slice_spacing)[0][0])

    new_slice = (image_xyz[:, :, idx] + image_xyz[:, :, idx + 1]) * 0.5
    new_slice = new_slice[..., np.newaxis]

    return np.concatenate(
        (
            image_xyz[..., :idx],
            new_slice,
            image_xyz[..., idx:],
        ),
        axis=2,
    )


def _fill_discontinuity(
    image_xyz: np.ndarray,
    *,
    modality: str,
    n_missing_slices: int,
    slice_discontinuities: np.ndarray,
) -> np.ndarray:
    fill_constant = {
        "PT": 0,
        "CT": -1000,
        "MR": 0,
    }.get(modality, 0)

    idx = int(np.where(slice_discontinuities)[0][0])
    fill_shape = image_xyz.shape[:2] + (n_missing_slices,)

    return np.concatenate(
        (
            image_xyz[..., : idx + 1],
            fill_constant * np.ones(fill_shape, dtype=image_xyz.dtype),
            image_xyz[..., idx + 1 :],
        ),
        axis=2,
    )


def _geometry_from_slices(slices, expected_n_slices: int):
    first = slices[0]
    last = slices[-1]

    origin = tuple(float(x) for x in first.ImagePositionPatient)

    orientation = np.asarray(first.ImageOrientationPatient, dtype=float)
    row = orientation[:3]
    col = orientation[3:]
    normal = np.cross(row, col)

    direction_matrix = np.stack([row, col, normal], axis=1)
    direction = tuple(float(x) for x in direction_matrix.ravel())

    row_spacing, col_spacing = [float(x) for x in first.PixelSpacing]

    if expected_n_slices > 1:
        first_pos = np.asarray(first.ImagePositionPatient, dtype=float)
        last_pos = np.asarray(last.ImagePositionPatient, dtype=float)
        z_spacing = float(
            np.linalg.norm(last_pos - first_pos) / (expected_n_slices - 1)
        )
    else:
        z_spacing = float(getattr(first, "SliceThickness", 1.0))

    spacing = (col_spacing, row_spacing, z_spacing)

    return origin, spacing, direction
