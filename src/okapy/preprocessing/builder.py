from __future__ import annotations

from typing import Any

from okapy.preprocessing.models import StudyPreprocessingConfig
from okapy.preprocessing.processor_pipeline import ProcessorPipeline
from okapy.preprocessing.step import StudyPreprocessingStep


def build_preprocessing_step(
    config: dict[str, Any],
) -> StudyPreprocessingStep:
    return StudyPreprocessingStep(
        study_config=StudyPreprocessingConfig.from_dict(config),
        geometry_section=(config.get("geometry_preprocessing") or {}),
        image_processors=ProcessorPipeline.from_config(
            config.get("local_preprocessing") or config.get("intensity_preprocessing")
        ),
        mask_processors=ProcessorPipeline.from_config(config.get("mask_preprocessing")),
    )
