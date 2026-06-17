from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import SimpleITK as sitk

from okapy.core.geometry import assert_same_geometry
from okapy.core.models import ImageVolume, MaskVolume
from okapy.features.backends.base import FeatureBackend
from okapy.features.models import FeatureSet

@dataclass(frozen=True)
class PETFeatureConfig:
    threshold: float = 0.0
    threshold_type: str = "absolute"
    suvpeak_diameter_mm: float = 12.0
    restrict_suvpeak_to_mask: bool = False
    preserve_legacy_names: bool = True

    def __post_init__(self) -> None:
        if self.threshold_type not in {"absolute", "relative"}:
            raise ValueError("threshold_type must be 'absolute' or 'relative'.")
        if self.suvpeak_diameter_mm <= 0:
            raise ValueError("suvpeak_diameter_mm must be positive.")

class PETFeatureBackend(FeatureBackend):
    def __init__(self, *, config: PETFeatureConfig | None = None, name: str = "pet") -> None:
        self.name = name
        self.config = config or PETFeatureConfig()

    def extract(self, image: ImageVolume, mask: MaskVolume, *, work_dir: Path) -> FeatureSet:
        del work_dir
        if image.modality != "PT":
            raise ValueError(f"PETFeatureBackend requires PT, got {image.modality!r}.")
        assert_same_geometry(image.image, mask.image, context=f"PET features for {mask.label!r}")
        image_array = sitk.GetArrayFromImage(image.image).astype(np.float64)
        mask_array = sitk.GetArrayFromImage(mask.image) != 0
        roi_values = image_array[mask_array]
        if roi_values.size == 0:
            raise ValueError(f"Mask {mask.label!r} is empty.")
        threshold = float(self.config.threshold if self.config.threshold_type == "absolute" else self.config.threshold * np.max(roi_values))
        metabolic_mask = mask_array & (image_array > threshold)
        metabolic_values = image_array[metabolic_mask]
        voxel_volume_ml = float(np.prod(image.image.GetSpacing())) / 1000.0
        mtv_ml = float(np.count_nonzero(metabolic_mask) * voxel_volume_ml)
        if metabolic_values.size == 0:
            suv_mean = suv_max = tlg = suv_peak = float("nan")
        else:
            suv_mean = float(np.mean(metabolic_values))
            suv_max = float(np.max(metabolic_values))
            tlg = float(mtv_ml * suv_mean)
            suv_peak = compute_suvpeak(
                image=image.image,
                candidate_mask=metabolic_mask,
                diameter_mm=self.config.suvpeak_diameter_mm,
                restrict_to_mask=self.config.restrict_suvpeak_to_mask,
            )
        suffix = self._feature_suffix()
        if self.config.preserve_legacy_names:
            features = {
                f"original_firstorder_MTV{suffix}": mtv_ml,
                f"original_firstorder_TLG{suffix}": tlg,
                f"original_firstorder_SUVpeak{suffix}": suv_peak,
            }
        else:
            features = {
                f"PET_MTV_ml{suffix}": mtv_ml,
                f"PET_TLG{suffix}": tlg,
                f"PET_SUVmean{suffix}": suv_mean,
                f"PET_SUVmax{suffix}": suv_max,
                f"PET_SUVpeak{suffix}": suv_peak,
            }
        return FeatureSet(
            backend_name=self.name,
            features=features,
            metadata={
                "threshold": threshold,
                "threshold_type": self.config.threshold_type,
                "suvpeak_diameter_mm": self.config.suvpeak_diameter_mm,
            },
        )

    def _feature_suffix(self) -> str:
        if self.config.threshold == 0:
            return ""
        if self.config.threshold_type == "relative":
            return f"_T_{self.config.threshold * 100:g}rel"
        return f"_T_{self.config.threshold:g}abs"

def compute_suvpeak(*, image: sitk.Image, candidate_mask: np.ndarray, diameter_mm: float, restrict_to_mask: bool) -> float:
    image_array = sitk.GetArrayFromImage(image).astype(np.float64)
    candidate_indices = np.argwhere(candidate_mask)
    if candidate_indices.size == 0:
        return float("nan")
    hottest_zyx = candidate_indices[int(np.argmax(image_array[candidate_mask]))]
    spacing_zyx = np.asarray(image.GetSpacing(), dtype=float)[::-1]
    radius_mm = float(diameter_mm) / 2.0
    radius_zyx = np.ceil(radius_mm / spacing_zyx).astype(int)
    lower = np.maximum(hottest_zyx - radius_zyx, 0)
    upper = np.minimum(hottest_zyx + radius_zyx + 1, np.asarray(image_array.shape))
    slices = tuple(slice(int(lo), int(hi)) for lo, hi in zip(lower, upper))
    neighbourhood = image_array[slices]
    coordinates = np.indices(neighbourhood.shape, dtype=float)
    center = hottest_zyx - lower
    distance_squared_mm = np.zeros(neighbourhood.shape, dtype=float)
    for axis in range(3):
        distance_squared_mm += ((coordinates[axis] - center[axis]) * spacing_zyx[axis]) ** 2
    sphere = distance_squared_mm <= radius_mm ** 2
    if restrict_to_mask:
        sphere &= candidate_mask[slices]
    values = neighbourhood[sphere]
    return float(np.mean(values)) if values.size else float("nan")
