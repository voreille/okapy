from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from okapy.core.models import ImageVolume, MaskVolume
from okapy.features.models import FeatureSet

class FeatureBackend(ABC):
    name: str

    @abstractmethod
    def extract(self, image: ImageVolume, mask: MaskVolume, *, work_dir: Path) -> FeatureSet:
        raise NotImplementedError
