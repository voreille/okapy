from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
import SimpleITK as sitk

from okapy.core.models import ImageGeometry


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

        for min_value, max_value in zip(self.min_xyz, self.max_xyz):
            if min_value >= max_value:
                raise ValueError(
                    "PhysicalBox min values must be strictly smaller than max "
                    f"values. Got min={self.min_xyz}, max={self.max_xyz}."
                )

    @property
    def size_xyz(self) -> tuple[float, float, float]:
        return tuple(
            max_value - min_value
            for min_value, max_value in zip(self.min_xyz, self.max_xyz)
        )

    def pad(self, padding_mm: float) -> PhysicalBox:
        padding_mm = float(padding_mm)

        if padding_mm < 0:
            raise ValueError(f"padding_mm must be >= 0, got {padding_mm}.")

        return PhysicalBox(
            min_xyz=tuple(x - padding_mm for x in self.min_xyz),
            max_xyz=tuple(x + padding_mm for x in self.max_xyz),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "min_xyz": self.min_xyz,
            "max_xyz": self.max_xyz,
            "size_xyz": self.size_xyz,
        }


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


def mask_physical_bounding_box(mask: sitk.Image) -> PhysicalBox:
    """Return the physical bounding box of non-zero mask voxels."""

    if mask.GetDimension() != 3:
        raise ValueError(
            f"Only 3D masks are supported, got dimension={mask.GetDimension()}."
        )

    array = sitk.GetArrayFromImage(mask)
    nonzero_zyx = np.argwhere(array != 0)

    if nonzero_zyx.size == 0:
        raise ValueError("Cannot compute bounding box of an empty mask.")

    min_zyx = nonzero_zyx.min(axis=0)
    max_zyx = nonzero_zyx.max(axis=0)

    # Convert z,y,x array indices to x,y,z image indices.
    min_xyz_index = (int(min_zyx[2]), int(min_zyx[1]), int(min_zyx[0]))
    max_xyz_index = (int(max_zyx[2]), int(max_zyx[1]), int(max_zyx[0]))

    # Use continuous half-voxel bounds around the non-zero voxel centers.
    corners = [
        mask.TransformContinuousIndexToPhysicalPoint(index)
        for index in product(
            (min_xyz_index[0] - 0.5, max_xyz_index[0] + 0.5),
            (min_xyz_index[1] - 0.5, max_xyz_index[1] + 0.5),
            (min_xyz_index[2] - 0.5, max_xyz_index[2] + 0.5),
        )
    ]

    corners_array = np.asarray(corners, dtype=float)

    return PhysicalBox(
        min_xyz=tuple(float(x) for x in np.min(corners_array, axis=0)),
        max_xyz=tuple(float(x) for x in np.max(corners_array, axis=0)),
    )


def union_boxes(boxes: list[PhysicalBox]) -> PhysicalBox:
    if not boxes:
        raise ValueError("Cannot compute union of an empty list of boxes.")

    mins = np.asarray([box.min_xyz for box in boxes], dtype=float)
    maxs = np.asarray([box.max_xyz for box in boxes], dtype=float)

    return PhysicalBox(
        min_xyz=tuple(float(x) for x in np.min(mins, axis=0)),
        max_xyz=tuple(float(x) for x in np.max(maxs, axis=0)),
    )


def intersect_boxes(boxes: list[PhysicalBox]) -> PhysicalBox:
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


def make_reference_image_from_physical_box(
    *,
    box: PhysicalBox,
    spacing: tuple[float, float, float],
    direction_source: sitk.Image,
    pixel_id: int = sitk.sitkFloat32,
) -> sitk.Image:
    """Create an empty SimpleITK image defining a target grid.

    The grid is axis-aligned in physical coordinates through its origin/spacing,
    and uses the direction of direction_source.
    """

    if len(spacing) != 3:
        raise ValueError(f"spacing must have length 3, got {spacing}.")
    if any(s <= 0 for s in spacing):
        raise ValueError(f"spacing values must be > 0, got {spacing}.")

    size = tuple(
        max(1, int(np.ceil((box.max_xyz[i] - box.min_xyz[i]) / spacing[i])))
        for i in range(3)
    )

    reference = sitk.Image(size, pixel_id)
    reference.SetOrigin(tuple(float(x) for x in box.min_xyz))
    reference.SetSpacing(tuple(float(x) for x in spacing))
    reference.SetDirection(direction_source.GetDirection())

    return reference


def make_reference_image_from_geometry(
    geometry: ImageGeometry,
    *,
    pixel_id: int = sitk.sitkFloat32,
) -> sitk.Image:
    reference = sitk.Image(geometry.size, pixel_id)
    reference.SetSpacing(geometry.spacing)
    reference.SetOrigin(geometry.origin)
    reference.SetDirection(geometry.direction)
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
        output_pixel_type or image.GetPixelID(),
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


#: Interpolators that may be applied to a binary mask.
MASK_INTERPOLATORS = (sitk.sitkNearestNeighbor, sitk.sitkLinear)


def mask_interpolator_from_name(name: str | int) -> int:
    """Resolve an interpolator for binary masks.

    B-spline is rejected here even though it stays valid for images. Its kernel
    has negative lobes, so interpolating a 0/1 step over- and undershoots, and
    thresholding the result leaves detached speckle outside the boundary and
    pinholes inside thin structures. Clamping to [0, 1] does not help: the
    ringing values that cross the threshold already lie inside that range.
    """

    if str(name).lower() in {"bspline", "spline", "b_spline", "3"}:
        raise ValueError(
            "B-spline interpolation is not supported for masks because it rings "
            "on the 0/1 boundary and produces speckle and pinholes after "
            "thresholding. Use 'nearest' or 'linear' instead. B-spline remains "
            "available for 'image_interpolator'."
        )

    interpolator = interpolator_from_name(name)

    if interpolator not in MASK_INTERPOLATORS:
        raise ValueError(
            f"Unsupported mask interpolator {name!r}. Use nearest or linear."
        )

    return interpolator
