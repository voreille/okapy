from __future__ import annotations

from okapy.config import ConfigSource, load_config
from okapy.dicom.conversion.builder import build_dicom_conversion_step
from okapy.features.builder import build_feature_extraction_step
from okapy.pipeline.extraction import ExtractionPipeline
from okapy.pipeline.preprocessing import PreprocessingPipeline
from okapy.preprocessing.builder import build_preprocessing_step


def build_extraction_pipeline(
    config: ConfigSource,
) -> ExtractionPipeline:
    """Build the full pipeline from a config path or mapping (legacy migrated)."""

    config = load_config(config)
    feature_config = config.get("feature_extraction") or {}

    return ExtractionPipeline(
        conversion=build_dicom_conversion_step(config),
        preprocessing=build_preprocessing_step(config),
        feature_extraction=build_feature_extraction_step(config),
        result_format=str(feature_config.get("result_format", "long")),
        include_feature_backend=bool(feature_config.get("include_backend", False)),
    )


def build_preprocessing_pipeline(
    config: ConfigSource,
) -> PreprocessingPipeline:
    """Build conversion + preprocessing from a config path or mapping."""

    config = load_config(config)

    return PreprocessingPipeline(
        conversion=build_dicom_conversion_step(config),
        preprocessing=build_preprocessing_step(config),
    )
