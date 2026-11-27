from __future__ import annotations

import hashlib
import re
from collections.abc import Callable
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from okapy.dicom.tags import get_optional_str as get_optional_str
from okapy.dicom.tags import referenced_series_uids as referenced_series_uids
from okapy.dicom.tags import to_jsonable as to_jsonable

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

#: NAME_MAX of common filesystems (ext4, APFS, NTFS), in bytes.
_MAX_FILENAME_BYTES = 255
#: Length a too-long file name is truncated to, leaving room for the ``(n)``
#: counter :func:`write_image_unique` adds on a name collision.
_TRUNCATED_FILENAME_BYTES = _MAX_FILENAME_BYTES - 16
_FILENAME_HASH_LENGTH = 10


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


def make_filename(*parts: object, extension: str) -> str:
    """Return ``part1__part2__...extension`` with every part :func:`safe_name`-d.

    :func:`safe_name` and :func:`short_uid` results pass through unchanged.
    """

    return fit_filename("__".join(safe_name(part) for part in parts), extension)


def fit_filename(stem: str, extension: str) -> str:
    """Return ``stem.extension``, shortened if it exceeds 255 bytes.

    A name that fits is returned unchanged. A longer one is truncated and ends
    with a short hash of the full name, so different long names stay distinct.
    """

    name = f"{stem}.{extension}"
    if len(name.encode()) <= _MAX_FILENAME_BYTES:
        return name

    digest = hashlib.sha1(name.encode()).hexdigest()[:_FILENAME_HASH_LENGTH]
    tail = f"__{digest}.{extension}"
    keep = _TRUNCATED_FILENAME_BYTES - len(tail.encode())
    head = stem.encode()[:keep].decode(errors="ignore")
    return head + tail


def ensure_inside(path: Path, root: Path) -> Path:
    """Return ``path`` if it resolves inside ``root``, else raise ``ValueError``."""

    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError(f"Output path {path} escapes the output directory {root}.")
    return path


def join_inside(root: Path, *parts: str | Path) -> Path:
    """Return ``root`` joined with ``parts``, checked by :func:`ensure_inside`."""

    return ensure_inside(root.joinpath(*parts), root)


def study_output_dir(output_dir: Path, study_uid: str | None, index: int) -> Path:
    """Output directory of the ``index``-th study, named after its UID."""

    study_name = safe_name(study_uid) if study_uid else f"study_{index:04d}"
    return join_inside(output_dir, study_name)


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


def geometry_from_positions(
    *,
    first_position,
    last_position,
    n_slices: int,
    orientation,
    pixel_spacing,
    single_slice_spacing: Callable[[], float],
) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, ...]]:
    """Return the (origin, spacing, direction) of a stack of slices.

    ``first_position`` / ``last_position`` are the ImagePositionPatient of the
    first and last of ``n_slices`` slices sorted along the normal, and
    ``pixel_spacing`` is PixelSpacing (row, column). The z spacing of a single
    slice is ``single_slice_spacing()``, called only in that case.
    """

    origin = tuple(float(x) for x in first_position)

    orientation = np.asarray(orientation, dtype=float)
    row = orientation[:3]
    col = orientation[3:]
    normal = np.cross(row, col)

    direction = tuple(float(x) for x in np.stack([row, col, normal], axis=1).ravel())

    row_spacing, col_spacing = (float(x) for x in pixel_spacing)

    if n_slices > 1:
        first = np.asarray(first_position, dtype=float)
        last = np.asarray(last_position, dtype=float)
        z_spacing = float(np.linalg.norm(last - first) / (n_slices - 1))
    else:
        z_spacing = single_slice_spacing()

    return origin, (col_spacing, row_spacing, z_spacing), direction


def functional_group_sequences(ds, frame_index: int, name: str) -> list:
    """Non-empty ``name`` sequences of one frame, per-frame group first.

    Shared Functional Groups define default values for all frames, while
    Per-frame Functional Groups override them where specified; the first
    sequence returned is therefore the one that applies.
    """

    groups = []

    per_frame = getattr(ds, "PerFrameFunctionalGroupsSequence", None)
    if per_frame is not None:
        groups.append(per_frame[frame_index])

    shared = getattr(ds, "SharedFunctionalGroupsSequence", None)
    if shared is not None and len(shared) > 0:
        groups.append(shared[0])

    sequences = []
    for group in groups:
        sequence = getattr(group, name, None)
        if sequence is not None and len(sequence) > 0:
            sequences.append(sequence)

    return sequences


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
