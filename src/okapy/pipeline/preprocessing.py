from __future__ import annotations

from pathlib import Path

from okapy.core.models import VolumeCollection
from okapy.dicom.conversion.step import DicomConversionStep
from okapy.preprocessing.step import StudyPreprocessingStep


class PreprocessingPipeline:
    """DICOM conversion followed by study preprocessing (no features)."""

    def __init__(
        self,
        *,
        conversion: DicomConversionStep,
        preprocessing: StudyPreprocessingStep,
    ) -> None:
        self.conversion = conversion
        self.preprocessing = preprocessing

    def run(
        self,
        input_dir: Path | str,
        *,
        work_dir: Path | str,
        labels: list[str] | None = None,
    ) -> VolumeCollection:
        input_dir = Path(input_dir)
        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)

        converted = self.conversion.run(
            input_dir=input_dir,
            output_dir=work_dir / "converted",
            labels=labels,
        )

        return self.preprocessing.run(
            converted,
            output_dir=work_dir / "preprocessed",
        )
