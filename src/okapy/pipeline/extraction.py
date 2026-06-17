from __future__ import annotations

from pathlib import Path

import pandas as pd

from okapy.dicom.conversion.step import DicomConversionStep
from okapy.features.formatters import feature_records_to_dataframe
from okapy.features.step import FeatureExtractionStep
from okapy.preprocessing.step import StudyPreprocessingStep


class ExtractionPipeline:
    def __init__(
        self,
        *,
        conversion: DicomConversionStep,
        preprocessing: StudyPreprocessingStep,
        feature_extraction: FeatureExtractionStep,
        result_format: str = "long",
        include_feature_backend: bool = False,
    ) -> None:
        self.conversion = conversion
        self.preprocessing = preprocessing
        self.feature_extraction = feature_extraction
        self.result_format = result_format
        self.include_feature_backend = include_feature_backend

    def run(
        self,
        input_dir: Path,
        *,
        work_dir: Path,
        labels: list[str] | None = None,
    ) -> pd.DataFrame:
        work_dir.mkdir(parents=True, exist_ok=True)

        converted = self.conversion.run(
            input_dir=input_dir,
            output_dir=work_dir / "converted",
            labels=labels,
        )

        preprocessed = self.preprocessing.run(
            converted,
            output_dir=work_dir / "preprocessed",
        )

        records = self.feature_extraction.run(
            preprocessed,
            work_dir=work_dir / "features",
        )

        return feature_records_to_dataframe(
            records,
            result_format=self.result_format,
            include_backend=self.include_feature_backend,
        )
