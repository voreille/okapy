from __future__ import annotations

import logging
from pathlib import Path

from okapy.core.models import ImageMaskSet, VolumeCollection
from okapy.features.models import FeatureRecord
from okapy.features.selector import select_backends

logger = logging.getLogger(__name__)

class FeatureExtractionStep:
    def __init__(
        self,
        *,
        backends_by_selector: dict[str, list[object]] | None = None,
        common_backends: list[object] | None = None,
        continue_on_error: bool = False,
    ) -> None:
        self.backends_by_selector = backends_by_selector or {}
        self.common_backends = common_backends or []
        self.continue_on_error = continue_on_error

    def run(self, volumes: VolumeCollection, *, work_dir: Path) -> list[FeatureRecord]:
        work_dir.mkdir(parents=True, exist_ok=True)
        return [
            record
            for study in volumes.studies
            for item in study.image_mask_sets
            for record in self._extract_image_mask_set(item, work_dir=work_dir)
        ]

    def _extract_image_mask_set(self, item: ImageMaskSet, *, work_dir: Path) -> list[FeatureRecord]:
        backends = [
            *select_backends(self.backends_by_selector, item.image.modality_key),
            *self.common_backends,
        ]
        records: list[FeatureRecord] = []
        for mask in item.masks:
            feature_sets = []
            for backend in backends:
                try:
                    feature_sets.append(backend.extract(item.image, mask, work_dir=work_dir))
                except Exception:
                    if not self.continue_on_error:
                        raise
                    logger.exception(
                        "Feature extraction failed: backend=%s, image=%s, mask=%s",
                        getattr(backend, "name", type(backend).__name__),
                        item.image.path,
                        mask.path,
                    )
            records.append(FeatureRecord(image=item.image, mask=mask, feature_sets=tuple(feature_sets)))
        return records
