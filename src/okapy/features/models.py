from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from okapy.core.models import ImageVolume, MaskVolume

FeatureValue = int | float | str | bool | None

@dataclass(frozen=True)
class FeatureSet:
    backend_name: str
    features: dict[str, FeatureValue]
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class FeatureRecord:
    image: ImageVolume
    mask: MaskVolume
    feature_sets: tuple[FeatureSet, ...]

    @property
    def features(self) -> dict[str, FeatureValue]:
        merged: dict[str, FeatureValue] = {}
        for feature_set in self.feature_sets:
            duplicate_names = merged.keys() & feature_set.features.keys()
            if duplicate_names:
                raise ValueError(
                    "Duplicate feature names were produced for "
                    f"image={self.image.path}, mask={self.mask.path}: "
                    f"{sorted(duplicate_names)}."
                )
            merged.update(feature_set.features)
        return merged
