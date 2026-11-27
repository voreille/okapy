from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pydicom

from okapy.core.models import ImageVolume, VolumeStage
from okapy.dicom.conversion.utils import (
    SLICE_SPACING_TOLERANCE,
    check_no_duplicate_positions,
    check_stacking_along_normal,
    ensure_inside,
    geometry_from_positions,
    make_filename,
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
    - reject series without volumetric geometry (IPP/IOP/PixelSpacing)
    - sort slices by ImagePositionPatient projected on slice normal
    - remove inconsistent Rows/Columns
    - reject several slices at one position (4D / dynamic / multi-echo)
    - reject slices that are not stacked along the slice normal (gantry tilt)
    - handle one missing slice by interpolation
    - handle larger one-gap discontinuity by filling with modality-specific value
    - reject slice spacing that is still non-uniform after gap handling
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

        series_uid = series.series_instance_uid
        slices, paths, orthogonal_positions = self._read_and_sort_slices(series)
        slices, paths, orthogonal_positions = self._clean_slices(
            slices,
            paths,
            orthogonal_positions,
            series_uid=series_uid,
        )

        image_xyz = self._get_physical_values(slices, paths, series.modality)
        image_xyz = np.transpose(image_xyz, (1, 0, 2))

        # A single slice has no gaps to check; its z spacing is SliceThickness.
        if len(slices) > 1:
            orientation = np.asarray(slices[0].ImageOrientationPatient, dtype=float)
            check_stacking_along_normal(
                np.asarray([s.ImagePositionPatient for s in slices], dtype=float),
                np.cross(orientation[:3], orientation[3:]),
                series_uid=series_uid,
            )
            image_xyz = _handle_slice_gaps(
                image_xyz,
                orthogonal_positions=orthogonal_positions,
                modality=series.modality,
                series_uid=series_uid,
            )

        image_xyz = image_xyz.astype(self.dtype, copy=False)

        origin, spacing, direction = geometry_from_positions(
            first_position=slices[0].ImagePositionPatient,
            last_position=slices[-1].ImagePositionPatient,
            n_slices=image_xyz.shape[2],
            orientation=slices[0].ImageOrientationPatient,
            pixel_spacing=slices[0].PixelSpacing,
            single_slice_spacing=lambda: (
                _float_or_default(slices[0], "SliceThickness", 1.0) or 1.0
            ),
        )

        image = sitk_image_from_array_xyz(
            image_xyz,
            origin=origin,
            spacing=spacing,
            direction=direction,
        )

        filename = self._make_filename(series)
        path = write_image_unique(
            image, ensure_inside(output_dir / filename, output_dir)
        )

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

        _check_volumetric_geometry(slices, series=series)

        orientation = np.asarray(slices[0].ImageOrientationPatient, dtype=float)
        normal = np.cross(orientation[:3], orientation[3:])

        orthogonal_positions = [
            float(np.dot(normal, np.asarray(s.ImagePositionPatient, dtype=float)))
            for s in slices
        ]

        zipped = list(zip(slices, paths, orthogonal_positions, strict=True))
        zipped.sort(key=lambda x: x[2])

        slices, paths, orthogonal_positions = zip(*zipped, strict=True)
        return list(slices), list(paths), list(orthogonal_positions)

    def _clean_slices(self, slices, paths, orthogonal_positions, *, series_uid: str):
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

        check_no_duplicate_positions(orthogonal_positions, series_uid=series_uid)

        return slices, paths, orthogonal_positions

    def _get_physical_values(self, slices, paths, modality: str) -> np.ndarray:
        # Modality LUT: RescaleSlope / RescaleIntercept when present (1 / 0
        # otherwise), for every modality.
        arrays = [
            _float_or_default(s, "RescaleSlope", 1.0) * s.pixel_array
            + _float_or_default(s, "RescaleIntercept", 0.0)
            for s in slices
        ]

        return np.stack(arrays, axis=-1)

    def _make_filename(self, series: DicomSeries) -> str:
        return make_filename(
            series.patient_id,
            series.modality,
            short_uid(series.series_instance_uid),
            extension=self.extension,
        )


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


def _float_or_default(ds, keyword: str, default: float) -> float:
    value = getattr(ds, keyword, None)
    if value is None or value == "":
        return default
    return float(value)


def _check_volumetric_geometry(slices, *, series: DicomSeries) -> None:
    """Reject series without the geometry needed to build a volume."""

    required = ("ImagePositionPatient", "ImageOrientationPatient", "PixelSpacing")

    for ds in slices:
        missing = [
            keyword for keyword in required if getattr(ds, keyword, None) is None
        ]
        if missing:
            raise ValueError(
                f"Unsupported image series {series.series_instance_uid} "
                f"(modality {series.modality}, SOP class "
                f"{getattr(ds, 'SOPClassUID', None)}): missing {', '.join(missing)}. "
                "Only volumetric (cross-sectional) image series can be converted."
            )


def _handle_slice_gaps(
    image_xyz: np.ndarray,
    *,
    orthogonal_positions,
    modality: str,
    series_uid: str,
) -> np.ndarray:
    """Fill missing slices and check the resulting slice spacing is uniform."""

    d_slices = np.diff(np.asarray(orthogonal_positions, dtype=float))
    slice_spacing = _reference_gap(d_slices)

    if slice_spacing <= 0:
        raise ValueError(
            f"The median slice spacing of series {series_uid} is {slice_spacing}."
        )

    # Number of slice spacings covered by each gap.
    n_steps = np.round(d_slices / slice_spacing).astype(int)

    if np.any(n_steps < 1):
        raise ValueError(
            f"Series {series_uid} has near-duplicate slices: gap(s) of "
            f"{d_slices[n_steps < 1].tolist()} mm for a slice spacing of "
            f"{slice_spacing} mm."
        )

    residual = np.abs(d_slices / n_steps - slice_spacing) / slice_spacing
    if np.any(residual > SLICE_SPACING_TOLERANCE):
        raise ValueError(
            f"Series {series_uid} has non-uniform slice spacing: gaps of "
            f"{d_slices[residual > SLICE_SPACING_TOLERANCE].tolist()} mm deviate "
            f"more than {100 * SLICE_SPACING_TOLERANCE:g} % from the slice "
            f"spacing of {slice_spacing} mm."
        )

    discontinuities = np.flatnonzero(n_steps > 1)

    if len(discontinuities) > 1:
        raise ValueError(f"Too many slice discontinuities in series {series_uid}.")

    if len(discontinuities) == 0:
        return image_xyz

    idx = int(discontinuities[0])
    n_missing_slices = int(n_steps[idx]) - 1

    if n_missing_slices == 1:
        logger.warning(
            "One slice is missing in series %s. Replacing it by interpolation.",
            series_uid,
        )
        return _interp_missing_slice(image_xyz, idx)

    logger.warning(
        "Multiple slices are missing in series %s. Filling gap.",
        series_uid,
    )
    return _fill_discontinuity(
        image_xyz,
        modality=modality,
        n_missing_slices=n_missing_slices,
        idx=idx,
    )


def _reference_gap(values: np.ndarray) -> float:
    """Reference slice spacing: the median gap.

    Unlike the most frequent gap it does not depend on the slice order when
    every gap is jittered, and a few missing-slice gaps do not move it.
    """

    if len(values) == 0:
        return 1.0

    return float(np.median(np.round(values, decimals=5)))


def _interp_missing_slice(image_xyz: np.ndarray, idx: int) -> np.ndarray:
    """Insert the mean of slices ``idx`` and ``idx + 1`` between them."""

    new_slice = (image_xyz[:, :, idx] + image_xyz[:, :, idx + 1]) * 0.5
    new_slice = new_slice[..., np.newaxis]

    return np.concatenate(
        (
            image_xyz[..., : idx + 1],
            new_slice,
            image_xyz[..., idx + 1 :],
        ),
        axis=2,
    )


def _fill_discontinuity(
    image_xyz: np.ndarray,
    *,
    modality: str,
    n_missing_slices: int,
    idx: int,
) -> np.ndarray:
    fill_constant = {
        "PT": 0,
        "CT": -1000,
        "MR": 0,
    }.get(modality, 0)

    fill_shape = image_xyz.shape[:2] + (n_missing_slices,)

    return np.concatenate(
        (
            image_xyz[..., : idx + 1],
            fill_constant * np.ones(fill_shape, dtype=image_xyz.dtype),
            image_xyz[..., idx + 1 :],
        ),
        axis=2,
    )
