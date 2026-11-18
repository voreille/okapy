from __future__ import annotations

from typing import Any

from okapy.dicom.collector import DicomStudyCollector
from okapy.dicom.conversion.image import SimpleITKImageSeriesConverter
from okapy.dicom.conversion.rtstruct import RTStructMaskConverter
from okapy.dicom.conversion.seg import SegMaskConverter
from okapy.dicom.conversion.step import DicomConversionStep
from okapy.dicom.conversion.suv import PETSUVConverter
from okapy.dicom.identity import SeriesIdentityConfig
from okapy.dicom.walker import DicomWalker


def build_dicom_conversion_step(
    config: dict[str, Any],
) -> DicomConversionStep:
    """Build the DICOM-to-NIfTI conversion step from current Okapy config."""

    general = config.get("general") or {}
    conversion = config.get("conversion") or {}

    identity_config = _build_identity_config(general)
    walker = _build_walker(general)

    collector = DicomStudyCollector(
        walker=walker,
    )

    extension = str(conversion.get("extension", "nii.gz"))

    return DicomConversionStep(
        collector=collector,
        image_converter=SimpleITKImageSeriesConverter(
            extension=extension,
            identity_config=identity_config,
        ),
        pet_converter=PETSUVConverter(
            extension=extension,
            identity_config=identity_config,
        ),
        rtstruct_converter=RTStructMaskConverter(
            extension=extension,
            check_reference=bool(
                conversion.get("check_rtstruct_reference_uid", True)
            ),
        ),
        seg_converter=SegMaskConverter(
            extension=extension,
            check_reference_uid=bool(conversion.get("check_seg_reference_uid", True)),
        ),
        continue_on_error=bool(conversion.get("continue_on_error", False)),
    )


def _build_identity_config(
    general: dict[str, Any],
) -> SeriesIdentityConfig:
    return SeriesIdentityConfig(
        use_submodalities=bool(general.get("submodalities", False)),
        submodality_separator=str(general.get("submodality_separator", " --- ")),
        submodality_modalities=tuple(
            str(modality)
            for modality in general.get(
                "submodality_modalities",
                ["MR"],
            )
        ),
    )


def _build_walker(
    general: dict[str, Any],
) -> DicomWalker:
    additional_tags = general.get("additional_dicom_tags") or []

    return DicomWalker(
        additional_dicom_tags=[str(tag) for tag in additional_tags],
    )
