from __future__ import annotations

import csv
import functools
from pathlib import Path

import numpy as np
import SimpleITK as sitk


@functools.lru_cache(maxsize=None)
def _expected_outcomes(dro_root: Path) -> dict[str, str]:
    """Expected SUVbw outcome per DRO, from ``docs/DRO_list.csv``.

    The DRO directory lives next to the ``docs`` folder in the
    ``oncoray/suv_computation`` checkout. Returns an empty mapping when that
    file is not reachable, in which case callers fall back to the naming
    convention.
    """

    csv_path = dro_root.parent / "docs" / "DRO_list.csv"

    if not csv_path.is_file():
        return {}

    with csv_path.open(newline="", encoding="utf-8") as fh:
        return {row["ID"]: row["SUVmax_expected"] for row in csv.DictReader(fh)}


def dro_expects_error(case_dir: Path) -> bool:
    """Whether a DRO must fail to convert rather than produce SUVbw values.

    The manual marks these as "should yield an error in SUVbw computation":
    required attributes are missing, or the metadata is inconsistent in a way
    that makes any computed value untrustworthy.
    """

    expected = _expected_outcomes(case_dir.parent).get(case_dir.name)

    if expected is not None:
        return expected.strip().upper() == "ERROR"

    return case_dir.name.startswith("DRO_error")


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
