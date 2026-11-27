from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pydicom
import SimpleITK as sitk

#: Maximum distance, in voxels, between a SEG frame or RTSTRUCT contour and the
#: reference slice it is assigned to.
SLICE_MATCH_TOLERANCE = 0.25


#: Relative tolerance on slice spacing: slice gaps and the in-plane drift of
#: consecutive slice positions must stay within 1 % of the slice spacing.
SLICE_SPACING_TOLERANCE = 0.01


#: Slice positions closer than this (mm) along the normal are the same position.
SAME_POSITION_TOLERANCE = 1e-5


_UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9._-]")
_MAX_NAME_LENGTH = 128


def safe_name(value: object | None) -> str:
    """Return a filename component derived from ``value`` (e.g. a DICOM header).

    Every character outside ``[A-Za-z0-9._-]`` becomes ``_``; missing/empty
    values become ``"unknown"``; ``.``/``..`` become ``_`` so the result can
    never name the current or parent directory. At most 128 characters.
    """

    if value is None:
        return "unknown"

    name = _UNSAFE_CHARS.sub("_", str(value))[:_MAX_NAME_LENGTH]

    if not name:
        return "unknown"
    if name in {".", ".."}:
        return "_"
    return name


def short_uid(uid: str | None) -> str:
    if uid is None:
        return "unknown"
    return safe_name(str(uid).split(".")[-1])


def ensure_inside(path: Path, root: Path) -> Path:
    """Return ``path`` if it resolves inside ``root``, else raise ``ValueError``."""

    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError(f"Output path {path} escapes the output directory {root}.")
    return path


def check_no_duplicate_positions(
    orthogonal_positions,
    *,
    series_uid: str,
) -> None:
    """Raise ``ValueError`` if two sorted slice positions are the same."""

    d_slices = np.diff(np.asarray(orthogonal_positions, dtype=float))
    n_duplicates = int(np.sum(np.abs(d_slices) <= SAME_POSITION_TOLERANCE))

    if n_duplicates:
        raise ValueError(
            f"Series {series_uid} has several slices at the same position "
            f"({n_duplicates} duplicate position(s)): 4D, dynamic or multi-echo "
            "series are not supported."
        )


def check_stacking_along_normal(
    positions: np.ndarray,
    normal: np.ndarray,
    *,
    series_uid: str,
) -> None:
    """Raise ``ValueError`` if slice ``positions`` do not follow the normal.

    ``positions`` are the ImagePositionPatient of the slices, sorted along the
    slice normal and without duplicates. The in-plane component of each step
    between consecutive positions must stay within
    :data:`SLICE_SPACING_TOLERANCE` of the step along the normal (gantry tilt).
    """

    steps = np.diff(np.asarray(positions, dtype=float), axis=0)
    if len(steps) == 0:
        return

    along = steps @ normal
    in_plane = np.linalg.norm(steps - along[:, np.newaxis] * normal, axis=1)

    # Relative to the step along the normal, so that a gap of several slices
    # is judged per slice spacing.
    drift = float(np.max(in_plane / np.abs(along)))
    if drift > SLICE_SPACING_TOLERANCE:
        raise ValueError(
            f"Series {series_uid} is not stacked along the slice normal: "
            f"consecutive slice positions drift {100 * drift:.1f} % in-plane "
            "(gantry tilt or oblique stacking). Such series are not supported."
        )


def read_dicom(path_or_dataset):
    if hasattr(path_or_dataset, "SOPInstanceUID"):
        return path_or_dataset
    return pydicom.dcmread(str(path_or_dataset))


def read_dicom_header(path_or_dataset):
    if hasattr(path_or_dataset, "SOPInstanceUID"):
        return path_or_dataset
    return pydicom.dcmread(str(path_or_dataset), stop_before_pixels=True)


def image_metadata(image: sitk.Image) -> dict:
    return {
        "size": tuple(image.GetSize()),
        "spacing": tuple(image.GetSpacing()),
        "origin": tuple(image.GetOrigin()),
        "direction": tuple(image.GetDirection()),
        "pixel_type": image.GetPixelIDTypeAsString(),
    }


def write_image_unique(image: sitk.Image, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not path.exists():
        sitk.WriteImage(image, str(path))
        return path

    stem = path.name
    suffix = ""

    if stem.endswith(".nii.gz"):
        base = stem.removesuffix(".nii.gz")
        suffix = ".nii.gz"
    else:
        base = path.stem
        suffix = path.suffix

    counter = 1
    while True:
        candidate = path.with_name(f"{base}({counter}){suffix}")
        if not candidate.exists():
            sitk.WriteImage(image, str(candidate))
            return candidate
        counter += 1


def sitk_image_from_array_xyz(
    array_xyz: np.ndarray,
    *,
    origin: tuple[float, float, float],
    spacing: tuple[float, float, float],
    direction: tuple[float, ...],
) -> sitk.Image:
    """Create SITK image from array in x, y, z order.

    SimpleITK expects z, y, x array order.
    """

    array_zyx = np.transpose(array_xyz, (2, 1, 0))
    image = sitk.GetImageFromArray(array_zyx)
    image.SetOrigin(tuple(float(x) for x in origin))
    image.SetSpacing(tuple(float(x) for x in spacing))
    image.SetDirection(tuple(float(x) for x in direction))
    return image


def sitk_mask_like_reference(mask_xyz: np.ndarray, reference: sitk.Image) -> sitk.Image:
    mask_zyx = np.transpose(mask_xyz.astype(np.uint8), (2, 1, 0))
    image = sitk.GetImageFromArray(mask_zyx)
    image.CopyInformation(reference)
    return image
