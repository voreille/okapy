from __future__ import annotations

import logging
import warnings
from collections.abc import Iterable
from pathlib import Path

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
    SLICE_MATCH_TOLERANCE,
    image_metadata,
    join_inside,
    make_filename,
    referenced_series_uids,
    sitk_mask_like_reference,
    write_image_unique,
)
from okapy.dicom.models import DicomSeries

logger = logging.getLogger(__name__)

_CHECK_REFERENCE_DEPRECATED = (
    "RTStructMaskConverter.check_reference is deprecated; use "
    "check_reference_uid instead."
)


def _warn_check_reference_deprecated(
    message: str = _CHECK_REFERENCE_DEPRECATED,
) -> None:
    # stacklevel=3: past this helper and the converter method, at the caller.
    warnings.warn(message, DeprecationWarning, stacklevel=3)


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
        check_reference_uid: bool = True,
        check_reference: bool | None = None,
    ) -> None:
        if check_reference is not None:
            _warn_check_reference_deprecated(
                "RTStructMaskConverter(check_reference=...) is deprecated; use "
                "check_reference_uid=... instead."
            )
            check_reference_uid = check_reference

        self.extension = extension
        self.skip_empty = skip_empty
        self.check_reference_uid = check_reference_uid

    @property
    def check_reference(self) -> bool:
        """Deprecated alias of :attr:`check_reference_uid`."""

        _warn_check_reference_deprecated()
        return self.check_reference_uid

    @check_reference.setter
    def check_reference(self, value: bool) -> None:
        _warn_check_reference_deprecated()
        self.check_reference_uid = value

    def convert(
        self,
        series: DicomSeries,
        reference_image: ImageVolume,
        output_dir: Path,
        labels: Iterable[str] | None = None,
    ) -> list[MaskVolume]:
        if not series.is_rtstruct:
            raise ValueError(f"Expected RTSTRUCT series, got {series.modality}.")

        paths = series.paths

        if not paths:
            raise ValueError(f"RTSTRUCT series {series.series_instance_uid} is empty.")

        # Only the first file is used: the others are not read.
        dcm = pydicom.dcmread(str(paths[0]))

        if len(paths) > 1:
            logger.warning(
                "Multiple RTSTRUCT files found in series %s. Using the first one.",
                series.series_instance_uid,
            )

        if self.check_reference_uid:
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
            path = write_image_unique(mask_image, join_inside(output_dir, filename))

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
        # Contours per slice as (in-plane vertices, rasterized pixel rows and
        # columns). They are combined by nesting depth once every contour has
        # been read.
        slice_contours: dict[int, list[_SliceContour]] = {}
        clipped = 0

        for contour in contour_sequence:
            contour_data = getattr(contour, "ContourData", None)

            if contour_data is None:
                continue

            nodes = np.asarray(contour_data, dtype=float).reshape(-1, 3)

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

            # Every point must lie on the reference slice the contour is drawn on.
            offset = float(np.max(np.abs(z_values - z_index)))
            if offset > SLICE_MATCH_TOLERANCE:
                raise RuntimeError(
                    f"RTSTRUCT contour for label {label!r} lies {offset:.2f} slices "
                    f"from reference slice {z_index} (tolerance "
                    f"{SLICE_MATCH_TOLERANCE}): it is not on a slice of the "
                    "reference image."
                )

            if z_index < 0 or z_index >= mask_xyz.shape[2]:
                logger.warning(
                    "Skipping RTSTRUCT contour for label %r: it is outside the "
                    "reference image grid (z index %d, %d slices).",
                    label,
                    z_index,
                    mask_xyz.shape[2],
                )
                continue

            # A pixel centre is on the grid when its index is in [0, size - 1];
            # vertices beyond half a voxel outside it mean the polygon is clipped.
            in_plane = voxel_indices[:, :2]
            if np.any(in_plane < -0.5) or np.any(
                in_plane > np.asarray(mask_xyz.shape[:2]) - 0.5
            ):
                clipped += 1

            # mask_xyz is indexed as [x, y, z]. skimage.draw.polygon returns
            # coordinates for the first and second array dimensions, so we pass
            # x and y coordinates in that order; ``shape`` clips to the grid.
            rr, cc = polygon(
                voxel_indices[:, 0], voxel_indices[:, 1], shape=mask_xyz.shape[:2]
            )

            if len(rr) == 0:
                continue

            slice_contours.setdefault(z_index, []).append((in_plane, rr, cc))

        if clipped:
            logger.warning(
                "%d RTSTRUCT contour(s) for label %r extend outside the reference "
                "image grid; they were clipped to it.",
                clipped,
                label,
            )

        for z_index, contours in slice_contours.items():
            mask_xyz[:, :, z_index] = _combine_slice_contours(
                contours, shape=mask_xyz.shape[:2]
            )

        return mask_xyz

    def _check_reference_series(
        self,
        *,
        dcm,
        reference_image: ImageVolume,
        rtstruct_series: DicomSeries,
    ) -> None:
        referenced_uids = set(referenced_series_uids(dcm))

        if not referenced_uids:
            logger.warning(
                "Could not find referenced series UID in RTSTRUCT series %s.",
                rtstruct_series.series_instance_uid,
            )
            return

        if reference_image.series_instance_uid not in referenced_uids:
            raise ValueError(
                f"RTSTRUCT series {rtstruct_series.series_instance_uid} references "
                f"image series {sorted(referenced_uids)}, not the reference image "
                f"series {reference_image.series_instance_uid}."
            )

    def _make_filename(
        self,
        *,
        reference_image: ImageVolume,
        label: str,
    ) -> str:
        return make_filename(
            reference_image.patient_id,
            label,
            "RTSTRUCT",
            reference_image.modality,
            reference_image.series_instance_uid,
            extension=self.extension,
        )


#: One contour of a slice: in-plane vertices, and the rows and columns of the
#: pixels it rasterizes to.
_SliceContour = tuple[np.ndarray, np.ndarray, np.ndarray]

#: Distance (in pixels) within which a vertex is on a polygon boundary.
_BOUNDARY_TOLERANCE = 1e-6


def _combine_slice_contours(
    contours: list[_SliceContour],
    *,
    shape: tuple[int, int],
) -> np.ndarray:
    """Combine the contours of one ROI on one slice by nesting depth.

    A contour's depth is the number of other contours of the slice that contain
    it. Even depths add area, odd depths cut a hole. Contours are painted from
    the shallowest to the deepest, so each pixel takes the value of the deepest
    contour covering it; contours that only touch (same depth) fill their union.
    """

    plane = np.zeros(shape, dtype=bool)

    if len(contours) == 1:
        _, rr, cc = contours[0]
        plane[rr, cc] = True
        return plane

    # A contour can only be contained in another whose bounding box, widened by
    # the boundary tolerance, holds its own: skip the full test otherwise. The
    # margin also covers the rounding of the containment test on far vertices.
    boxes = []
    for vertices, _, _ in contours:
        low, high = vertices.min(axis=0), vertices.max(axis=0)
        margin = _BOUNDARY_TOLERANCE + 1e-12 * float(
            np.max(np.abs(np.concatenate((low, high))))
        )
        boxes.append((low, high, margin))

    def contains(j: int, i: int) -> bool:
        other_low, other_high, margin = boxes[j]
        low, high, _ = boxes[i]
        if np.any(low < other_low - margin) or np.any(high > other_high + margin):
            return False
        return _polygon_contains(contours[j][0], contours[i][0])

    depths = [
        sum(contains(j, i) for j in range(len(contours)) if j != i)
        for i in range(len(contours))
    ]

    for depth, (_, rr, cc) in sorted(
        zip(depths, contours, strict=True), key=lambda item: item[0]
    ):
        plane[rr, cc] = depth % 2 == 0
    return plane


def _polygon_contains(polygon_xy: np.ndarray, points_xy: np.ndarray) -> bool:
    """Whether ``polygon_xy`` contains the contour with vertices ``points_xy``.

    Contained means every vertex is inside or on the boundary of the polygon
    and at least one is strictly inside, so contours that only share an edge or
    a corner are not nested.
    """

    on_boundary = _on_polygon_boundary(polygon_xy, points_xy)
    inside = _points_strictly_inside(polygon_xy, points_xy) & ~on_boundary
    return bool(np.all(inside | on_boundary) and np.any(inside))


def _on_polygon_boundary(
    polygon_xy: np.ndarray,
    points_xy: np.ndarray,
    tolerance: float = _BOUNDARY_TOLERANCE,
) -> np.ndarray:
    start = polygon_xy[None, :, :]
    end = np.roll(polygon_xy, -1, axis=0)[None, :, :]
    points = points_xy[:, None, :]
    edge = end - start
    length_sq = np.sum(edge**2, axis=2)
    projection = np.sum((points - start) * edge, axis=2)
    t = np.divide(
        projection,
        length_sq,
        out=np.zeros_like(projection),
        where=length_sq > 0,
    )
    closest = start + np.clip(t, 0.0, 1.0)[:, :, None] * edge
    distance = np.sqrt(np.sum((points - closest) ** 2, axis=2))
    return np.any(distance <= tolerance, axis=1)


def _points_strictly_inside(
    polygon_xy: np.ndarray, points_xy: np.ndarray
) -> np.ndarray:
    """Even-odd ray casting; points on the boundary give an arbitrary result."""

    x, y = points_xy[:, 0:1], points_xy[:, 1:2]
    x0, y0 = polygon_xy[None, :, 0], polygon_xy[None, :, 1]
    x1 = np.roll(polygon_xy[:, 0], -1)[None, :]
    y1 = np.roll(polygon_xy[:, 1], -1)[None, :]
    crosses = (y0 > y) != (y1 > y)
    with np.errstate(divide="ignore", invalid="ignore"):
        x_cross = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
    return np.sum(crosses & (x < x_cross), axis=1) % 2 == 1


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
