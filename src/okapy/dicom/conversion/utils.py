from __future__ import annotations

from pathlib import Path

import numpy as np
import pydicom
import SimpleITK as sitk


def safe_name(value: str | None) -> str:
    if value is None:
        return "unknown"

    return (
        str(value)
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "-")
    )


def short_uid(uid: str) -> str:
    return uid.split(".")[-1]


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
