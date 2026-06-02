from __future__ import annotations

from pathlib import Path

import numpy as np
import SimpleITK as sitk


def find_roi_mask(case_dir: Path) -> Path:
    """Find the NIfTI ROI mask provided with a DRO case."""

    candidates = (
        list(case_dir.rglob("*mask*.nii"))
        + list(case_dir.rglob("*mask*.nii.gz"))
        + list(case_dir.rglob("*roi*.nii"))
        + list(case_dir.rglob("*roi*.nii.gz"))
        + list(case_dir.rglob("*ROI*.nii"))
        + list(case_dir.rglob("*ROI*.nii.gz"))
    )

    if len(candidates) == 0:
        raise FileNotFoundError(f"No ROI/mask NIfTI found in {case_dir}")

    if len(candidates) > 1:
        # Keep this strict at first, so you notice ambiguous cases.
        raise RuntimeError(
            f"Found multiple ROI/mask candidates in {case_dir}: {candidates}"
        )

    return candidates[0]


def compute_roi_suv_stats(
    image_path: Path,
    mask_path: Path,
) -> dict[str, float]:
    image = sitk.ReadImage(str(image_path))
    mask = sitk.ReadImage(str(mask_path))

    mask = _resample_mask_to_image(mask, image)

    image_array = sitk.GetArrayFromImage(image).astype(float)
    mask_array = sitk.GetArrayFromImage(mask) > 0

    values = image_array[mask_array]

    if values.size == 0:
        raise RuntimeError(f"ROI mask is empty: {mask_path}")

    return {
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "median": float(np.median(values)),
    }


def assert_dro_suv_stats(
    stats: dict[str, float],
    *,
    atol: float = 0.015,
):
    """Assert expected DRO SUVbw values.

    Values are expected to match to two decimal digits.
    """

    np.testing.assert_allclose(stats["max"], 4.00, atol=atol)
    np.testing.assert_allclose(stats["min"], 0.20, atol=atol)
    np.testing.assert_allclose(stats["median"], 1.00, atol=atol)


def _resample_mask_to_image(mask: sitk.Image, image: sitk.Image) -> sitk.Image:
    """Resample mask to image geometry if needed."""

    same_geometry = (
        mask.GetSize() == image.GetSize()
        and np.allclose(mask.GetSpacing(), image.GetSpacing())
        and np.allclose(mask.GetOrigin(), image.GetOrigin())
        and np.allclose(mask.GetDirection(), image.GetDirection())
    )

    if same_geometry:
        return mask

    return sitk.Resample(
        mask,
        image,
        sitk.Transform(),
        sitk.sitkNearestNeighbor,
        0,
        mask.GetPixelID(),
    )
