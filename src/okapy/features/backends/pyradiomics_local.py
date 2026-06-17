from __future__ import annotations

from pathlib import Path
from typing import Any
import numpy as np

from okapy.core.models import ImageVolume, MaskVolume
from okapy.features.backends.base import FeatureBackend
from okapy.features.models import FeatureSet

class LocalPyradiomicsBackend(FeatureBackend):
    def __init__(
        self,
        *,
        params: Path | str | dict[str, Any],
        name: str = "pyradiomics",
        include_diagnostics: bool = False,
        execute_kwargs: dict[str, Any] | None = None,
    ) -> None:
        try:
            from radiomics.featureextractor import RadiomicsFeatureExtractor
        except ImportError as exc:
            raise ImportError(
                "PyRadiomics is required by the local backend. Install it or use a command backend."
            ) from exc
        self.name = name
        self.include_diagnostics = include_diagnostics
        self.execute_kwargs = dict(execute_kwargs or {})
        self._extractor = RadiomicsFeatureExtractor(params if isinstance(params, dict) else str(Path(params)))

    def extract(self, image: ImageVolume, mask: MaskVolume, *, work_dir: Path) -> FeatureSet:
        del work_dir
        result = self._extractor.execute(image.image, mask.image, **self.execute_kwargs)
        features = {
            str(name): _to_feature_value(value)
            for name, value in result.items()
            if self.include_diagnostics or not str(name).startswith("diagnostics")
        }
        return FeatureSet(backend_name=self.name, features=features)

def _to_feature_value(value: Any):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        if value.ndim == 0 or value.size == 1:
            return value.reshape(-1)[0].item()
        return value.tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return str(value)
