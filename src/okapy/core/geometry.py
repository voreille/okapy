from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
import SimpleITK as sitk

# Binary masks are always resampled the same way: linear (trilinear in 3D)
# interpolation of the 0/1 volume, then a threshold at MASK_THRESHOLD. This is
# the IBSI fractional route and is deliberately not configurable.
#
# Linear interpolation places the boundary with sub-voxel precision, which
# nearest neighbour cannot do. B-spline is not an option because its kernel has
# negative lobes: interpolating a 0/1 step over- and undershoots, and
# thresholding the result leaves detached speckle outside the boundary and
# pinholes inside thin structures. Clamping to [0, 1] does not help, since the
# ringing values that cross the threshold already lie inside that range.
# B-spline remains available for images through ``image_interpolator``.
MASK_INTERPOLATOR = sitk.sitkLinear
MASK_INTERPOLATOR_NAME = "linear"
MASK_THRESHOLD = 0.5


@dataclass(frozen=True)
class PhysicalBox:
    """Axis-aligned physical bounding box in world coordinates, in mm."""

    min_xyz: tuple[float, float, float]
    max_xyz: tuple[float, float, float]

    def __post_init__(self) -> None:
        if len(self.min_xyz) != 3:
            raise ValueError(f"min_xyz must have length 3, got {self.min_xyz}.")
        if len(self.max_xyz) != 3:
            raise ValueError(f"max_xyz must have length 3, got {self.max_xyz}.")

        for min_value, max_value in zip(self.min_xyz, self.max_xyz, strict=True):
            if min_value >= max_value:
                raise ValueError(
                    "PhysicalBox min values must be strictly smaller than max "
                    f"values. Got min={self.min_xyz}, max={self.max_xyz}."
                )

    @property
    def size_xyz(self) -> tuple[float, float, float]:
        return tuple(
            max_value - min_value
            for min_value, max_value in zip(self.min_xyz, self.max_xyz, strict=True)
        )


def image_physical_box(image: sitk.Image) -> PhysicalBox:
    """Return the physical FOV box of a SimpleITK image.

    This computes the physical coordinates of all voxel-domain corners and
    returns an axis-aligned physical bounding box.
    """

    if image.GetDimension() != 3:
        raise ValueError(
            f"Only 3D images are supported, got dimension={image.GetDimension()}."
        )

    size = image.GetSize()

    # Use continuous index bounds so the box covers voxel extents, not only
    # voxel centers.
    corners = [
        image.TransformContinuousIndexToPhysicalPoint(index)
        for index in product(
            (-0.5, size[0] - 0.5),
            (-0.5, size[1] - 0.5),
            (-0.5, size[2] - 0.5),
        )
    ]

    corners_array = np.asarray(corners, dtype=float)

    return PhysicalBox(
        min_xyz=tuple(float(x) for x in np.min(corners_array, axis=0)),
        max_xyz=tuple(float(x) for x in np.max(corners_array, axis=0)),
    )


def intersect_boxes(boxes: list[PhysicalBox]) -> PhysicalBox:
    """Return the intersection of axis-aligned physical boxes.

    The intersection is computed per world axis, so it is itself an
    axis-aligned box. For obliquely oriented images, whose FOV boxes (see
    :func:`image_physical_box`) are the axis-aligned bounding boxes of rotated
    volumes, the result can include corners that lie outside some of the
    images: it is a bounding box of the common FOV, not the exact overlap.
    """

    if not boxes:
        raise ValueError("Cannot compute intersection of an empty list of boxes.")

    mins = np.asarray([box.min_xyz for box in boxes], dtype=float)
    maxs = np.asarray([box.max_xyz for box in boxes], dtype=float)

    min_xyz = np.max(mins, axis=0)
    max_xyz = np.min(maxs, axis=0)

    if np.any(min_xyz >= max_xyz):
        raise ValueError(
            "Boxes do not overlap. "
            f"intersection min={tuple(min_xyz)}, max={tuple(max_xyz)}."
        )

    return PhysicalBox(
        min_xyz=tuple(float(x) for x in min_xyz),
        max_xyz=tuple(float(x) for x in max_xyz),
    )


IndexExtent = tuple[np.ndarray, np.ndarray]
"""Continuous (x, y, z) index range ``(lo, hi)`` in the voxel space of an image."""

#: Tolerance, in target voxels, used when deciding whether a target voxel
#: centre lies inside an extent. A centre exactly on the boundary (a tie) is
#: kept even when floating-point error puts it a hair outside.
_TIE_TOLERANCE = 1e-6


def _per_axis_padding(padding_voxels: int | tuple[int, int, int]) -> np.ndarray:
    if isinstance(padding_voxels, list | tuple):
        if len(padding_voxels) != 3:
            raise ValueError(
                f"padding_voxels must have 3 values, got {padding_voxels}."
            )
        values = padding_voxels
    else:
        values = (padding_voxels,) * 3

    if any(
        isinstance(v, bool) or not isinstance(v, int | np.integer) or v < 0
        for v in values
    ):
        raise ValueError(
            f"padding_voxels must be non-negative integers, got {padding_voxels}."
        )
    return np.asarray(values, dtype=int)


def _centres_inside(
    extents: list[IndexExtent], start: np.ndarray, step: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the target voxel range whose centres lie inside all ``extents``.

    Also returns the intersected extent ``(extent_lo, extent_hi)``.
    """

    extent_lo = np.max([extent[0] for extent in extents], axis=0)
    extent_hi = np.min([extent[1] for extent in extents], axis=0)

    lo = np.ceil((extent_lo - start) / step - _TIE_TOLERANCE).astype(int)
    hi = np.floor((extent_hi - start) / step + _TIE_TOLERANCE).astype(int)
    return lo, hi, extent_lo, extent_hi


def mask_index_extent(mask: sitk.Image, image: sitk.Image) -> IndexExtent:
    """Return the extent of the non-zero mask voxels in the index space of ``image``.

    The extent covers whole mask voxels (centres +/- half a voxel), so when the
    mask is on the image grid it is ``(first - 0.5, last + 0.5)``. Working in
    index space rather than physical space keeps the result correct for any
    image orientation, including flipped axes.
    """

    if mask.GetDimension() != 3:
        raise ValueError(
            f"Only 3D masks are supported, got dimension={mask.GetDimension()}."
        )

    array_zyx = sitk.GetArrayViewFromImage(mask)

    # Per-axis projections give the bounding box of the non-zero voxels.
    nonzero_x, nonzero_y, nonzero_z = (
        np.flatnonzero(np.any(array_zyx, axis=axes))
        for axes in ((0, 1), (0, 2), (1, 2))
    )

    if nonzero_z.size == 0:
        raise ValueError("Cannot compute the extent of an empty mask.")

    min_xyz = np.array([nonzero_x[0], nonzero_y[0], nonzero_z[0]])
    max_xyz = np.array([nonzero_x[-1], nonzero_y[-1], nonzero_z[-1]])

    corners = [
        image.TransformPhysicalPointToContinuousIndex(
            mask.TransformContinuousIndexToPhysicalPoint(tuple(float(x) for x in index))
        )
        for index in product(*zip(min_xyz - 0.5, max_xyz + 0.5, strict=True))
    ]

    corners_array = np.asarray(corners, dtype=float)
    return corners_array.min(axis=0), corners_array.max(axis=0)


def box_index_extent(box: PhysicalBox, image: sitk.Image) -> IndexExtent:
    """Return the extent of a physical box in the index space of ``image``."""

    corners = [
        image.TransformPhysicalPointToContinuousIndex(point)
        for point in product(*zip(box.min_xyz, box.max_xyz, strict=True))
    ]

    corners_array = np.asarray(corners, dtype=float)
    return corners_array.min(axis=0), corners_array.max(axis=0)


def make_reference_image_on_aligned_grid(
    *,
    image: sitk.Image,
    spacing: tuple[float, float, float],
    extents: list[IndexExtent] | None = None,
    padding_voxels: int | tuple[int, int, int] = 0,
    clip_extents: list[IndexExtent] | None = None,
    pixel_id: int = sitk.sitkFloat32,
) -> sitk.Image:
    """Create an empty SimpleITK image defining the target grid for ``image``.

    The target grid has the orientation of ``image`` and the requested spacing.
    Along each axis it is aligned with the image grid by centring (IBSI grid
    centre alignment): the centres of the full target grid and of the image
    grid coincide. When the spacing equals the native spacing this is the
    image grid itself, so cropping never shifts or interpolates voxels.

    The grid is then cropped to the target voxels whose centres fall inside
    the intersection of ``extents`` (in the index space of ``image``, see
    :func:`mask_index_extent`), padded by ``padding_voxels`` target voxels on
    each side (one value, or one per axis) and clipped to the image. Without
    extents the full grid is used. Finally the grid is clipped to the target
    voxels whose centres fall inside ``clip_extents`` (e.g. the common FOV),
    so the padding never extends past them.

    Voxel centres exactly on an extent boundary are kept.
    """

    if len(spacing) != 3:
        raise ValueError(f"spacing must have length 3, got {spacing}.")
    if any(s <= 0 for s in spacing):
        raise ValueError(f"spacing values must be > 0, got {spacing}.")
    padding = _per_axis_padding(padding_voxels)

    native_spacing = np.asarray(image.GetSpacing(), dtype=float)
    native_size = np.asarray(image.GetSize(), dtype=float)
    target_spacing = np.asarray(spacing, dtype=float)

    # Target spacing in units of image voxels; exactly 1 on native axes.
    step = target_spacing / native_spacing
    step[np.isclose(step, 1.0)] = 1.0

    full_size = np.ceil(native_size / step - 1e-9).astype(int)

    # Continuous image index of the first target voxel (grid centres coincide).
    start = (native_size - 1) / 2 - (full_size - 1) / 2 * step

    lo = np.zeros(3, dtype=int)
    hi = full_size - 1

    if extents:
        lo, hi, extent_lo, extent_hi = _centres_inside(extents, start, step)

        # An extent narrower than one target voxel keeps the nearest voxel.
        narrow = lo > hi
        centre = np.round(((extent_lo + extent_hi) / 2 - start) / step).astype(int)
        lo[narrow] = centre[narrow]
        hi[narrow] = centre[narrow]

        lo = np.maximum(lo - padding, 0)
        hi = np.minimum(hi + padding, full_size - 1)

        if np.any(lo > hi):
            raise ValueError(
                "The region to crop does not overlap the image: "
                f"index extent lo={tuple(extent_lo)}, hi={tuple(extent_hi)}, "
                f"image size={image.GetSize()}."
            )

    if clip_extents:
        clip_lo, clip_hi, extent_lo, extent_hi = _centres_inside(
            clip_extents, start, step
        )
        lo = np.maximum(lo, clip_lo)
        hi = np.minimum(hi, clip_hi)

        if np.any(lo > hi):
            raise ValueError(
                "The region to crop does not overlap the clipping region: "
                f"clip index extent lo={tuple(extent_lo)}, hi={tuple(extent_hi)}, "
                f"image size={image.GetSize()}."
            )

    origin_index = start + lo * step

    reference = sitk.Image(tuple(int(x) for x in hi - lo + 1), pixel_id)
    reference.SetOrigin(
        image.TransformContinuousIndexToPhysicalPoint(
            tuple(float(x) for x in origin_index)
        )
    )
    reference.SetSpacing(tuple(float(x) for x in native_spacing * step))
    reference.SetDirection(image.GetDirection())

    return reference


def resample_to_reference(
    image: sitk.Image,
    reference: sitk.Image,
    *,
    interpolator: int,
    default_value: float = 0.0,
    output_pixel_type: int | None = None,
) -> sitk.Image:
    return sitk.Resample(
        image,
        reference,
        sitk.Transform(),
        interpolator,
        default_value,
        image.GetPixelID() if output_pixel_type is None else output_pixel_type,
    )


def same_geometry(a: sitk.Image, b: sitk.Image, *, atol: float = 1e-6) -> bool:
    return (
        a.GetSize() == b.GetSize()
        and np.allclose(a.GetSpacing(), b.GetSpacing(), atol=atol)
        and np.allclose(a.GetOrigin(), b.GetOrigin(), atol=atol)
        and np.allclose(a.GetDirection(), b.GetDirection(), atol=atol)
    )


def assert_same_geometry(a: sitk.Image, b: sitk.Image, *, context: str = "") -> None:
    if same_geometry(a, b):
        return

    prefix = f"{context}: " if context else ""

    raise ValueError(
        f"{prefix}Images do not have the same geometry.\n"
        f"A size={a.GetSize()}, spacing={a.GetSpacing()}, "
        f"origin={a.GetOrigin()}, direction={a.GetDirection()}\n"
        f"B size={b.GetSize()}, spacing={b.GetSpacing()}, "
        f"origin={b.GetOrigin()}, direction={b.GetDirection()}"
    )


def interpolator_from_name(name: str | int) -> int:
    if isinstance(name, int):
        return _interpolator_from_order(name)

    key = str(name).lower()
    if key in {"nearest", "nearest_neighbor", "nn", "0"}:
        return sitk.sitkNearestNeighbor
    if key in {"linear", "bilinear", "trilinear", "1"}:
        return sitk.sitkLinear
    if key in {"bspline", "spline", "b_spline", "3"}:
        return sitk.sitkBSpline

    raise ValueError(
        f"Unsupported interpolator {name!r}. Use nearest, linear, or bspline."
    )


def _interpolator_from_order(order: int) -> int:
    if order == 0:
        return sitk.sitkNearestNeighbor
    if order == 1:
        return sitk.sitkLinear
    if order == 3:
        return sitk.sitkBSpline
    raise ValueError("Unsupported interpolation order. Use 0, 1, or 3.")
