from __future__ import annotations

from pathlib import Path


from okapy.dicom.conversion.step import DicomConversionStep
from okapy.preprocessing.step import StudyPreprocessingStep
from okapy.core.models import VolumeCollection


class PreprocessingPipeline:
    def __init__(
        self,
        *,
        conversion: DicomConversionStep,
        preprocessing: StudyPreprocessingStep,
        result_format: str = "long",
        include_feature_backend: bool = False,
    ) -> None:
        self.conversion = conversion
        self.preprocessing = preprocessing
        self.result_format = result_format
        self.include_feature_backend = include_feature_backend

    def run(
        self,
        input_dir: Path,
        *,
        work_dir: Path,
        labels: list[str] | None = None,
    ) -> VolumeCollection:
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

        return preprocessed
