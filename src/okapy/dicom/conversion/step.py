from __future__ import annotations

import logging
from pathlib import Path

from okapy.core.models import (
    ImageMaskSet,
    ImageVolume,
    MaskVolume,
    StudyVolumes,
    VolumeCollection,
)
from okapy.dicom.collector import DicomStudyCollector
from okapy.dicom.conversion.image import SimpleITKImageSeriesConverter
from okapy.dicom.conversion.rtstruct import RTStructMaskConverter
from okapy.dicom.conversion.seg import SegMaskConverter
from okapy.dicom.conversion.suv import PETSUVConverter
from okapy.dicom.models import DicomSeries, DicomStudy


logger = logging.getLogger(__name__)


class DicomConversionStep:
    """Convert grouped DICOM studies into native-grid image and mask volumes.

    Responsibilities:
    - discover supported DICOM files;
    - group files into studies and series;
    - convert CT/MR/other image series;
    - convert PT series to SUVbw;
    - resolve RTSTRUCT/SEG references;
    - convert segmentation objects onto their referenced image grid;
    - return converted volumes grouped by study.

    This step does not perform preprocessing, cropping, or resampling between
    modalities. Converted masks remain aligned with their original reference
    image series.
    """

    def __init__(
        self,
        *,
        collector: DicomStudyCollector,
        image_converter: SimpleITKImageSeriesConverter,
        pet_converter: PETSUVConverter,
        rtstruct_converter: RTStructMaskConverter,
        seg_converter: SegMaskConverter,
        continue_on_error: bool = False,
    ) -> None:
        self.collector = collector
        self.image_converter = image_converter
        self.pet_converter = pet_converter
        self.rtstruct_converter = rtstruct_converter
        self.seg_converter = seg_converter
        self.continue_on_error = continue_on_error

    def run(
        self,
        input_dir: Path,
        *,
        output_dir: Path,
        labels: list[str] | None = None,
    ) -> VolumeCollection:
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)

        if not input_dir.is_dir():
            raise NotADirectoryError(
                f"DICOM input directory does not exist: {input_dir}"
            )

        output_dir.mkdir(parents=True, exist_ok=True)

        collection = self.collector.collect(input_dir)

        converted_studies: list[StudyVolumes] = []

        for index, study in enumerate(collection.studies):
            study_uid = study.study_instance_uid
            study_name = study_uid or f"study_{index:04d}"
            study_output_dir = output_dir / study_name

            try:
                converted_study = self.convert_study(
                    study,
                    output_dir=study_output_dir,
                    labels=labels,
                )
            except Exception:
                if not self.continue_on_error:
                    raise

                logger.exception(
                    "Failed to convert DICOM study %s.",
                    study_uid,
                )
                continue

            converted_studies.append(converted_study)

        return VolumeCollection(studies=converted_studies)

    def convert_study(
        self,
        study: DicomStudy,
        *,
        output_dir: Path,
        labels: list[str] | None = None,
    ) -> StudyVolumes:
        """Convert one grouped DICOM study."""

        output_dir = Path(output_dir)
        image_output_dir = output_dir / "images"
        mask_output_dir = output_dir / "masks"

        image_output_dir.mkdir(parents=True, exist_ok=True)
        mask_output_dir.mkdir(parents=True, exist_ok=True)

        images = self._convert_images(
            study.image_series,
            output_dir=image_output_dir,
        )

        images_by_uid = {image.series_instance_uid: image for image in images}

        masks = self._convert_masks(
            study.mask_series,
            images_by_uid=images_by_uid,
            output_dir=mask_output_dir,
            labels=labels,
        )

        image_mask_sets = self._group_images_and_native_masks(
            images=images,
            masks=masks,
        )

        return StudyVolumes(
            study_instance_uid=study.study_instance_uid,
            image_mask_sets=image_mask_sets,
        )

    def _convert_images(
        self,
        series_list: list[DicomSeries],
        *,
        output_dir: Path,
    ) -> list[ImageVolume]:
        images: list[ImageVolume] = []

        for series in series_list:
            try:
                converter = (
                    self.pet_converter
                    if series.modality == "PT"
                    else self.image_converter
                )

                image = converter.convert(
                    series=series,
                    output_dir=output_dir,
                )

            except Exception:
                if not self.continue_on_error:
                    raise

                logger.exception(
                    "Failed to convert image series %s with modality %s.",
                    series.series_instance_uid,
                    series.modality,
                )
                continue

            images.append(image)

        return images

    def _convert_masks(
        self,
        series_list: list[DicomSeries],
        *,
        images_by_uid: dict[str, ImageVolume],
        output_dir: Path,
        labels: list[str] | None,
    ) -> list[MaskVolume]:
        masks: list[MaskVolume] = []

        for series in series_list:
            try:
                reference_image = self._resolve_reference_image(
                    series=series,
                    images_by_uid=images_by_uid,
                )

                if series.is_rtstruct:
                    series_masks = self.rtstruct_converter.convert(
                        series=series,
                        reference_image=reference_image,
                        output_dir=output_dir,
                        labels=labels,
                    )

                elif series.is_seg:
                    series_masks = self.seg_converter.convert(
                        series=series,
                        reference_image=reference_image,
                        output_dir=output_dir,
                        labels=labels,
                    )

                else:
                    logger.warning(
                        "Skipping unsupported mask series %s with modality %s.",
                        series.series_instance_uid,
                        series.modality,
                    )
                    continue

            except Exception:
                if not self.continue_on_error:
                    raise

                logger.exception(
                    "Failed to convert mask series %s.",
                    series.series_instance_uid,
                )
                continue

            masks.extend(series_masks)

        return masks

    @staticmethod
    def _resolve_reference_image(
        *,
        series: DicomSeries,
        images_by_uid: dict[str, ImageVolume],
    ) -> ImageVolume:
        """Resolve the image referenced by an RTSTRUCT or SEG series."""

        referenced_uids = series.referenced_series_uids

        for referenced_uid in referenced_uids:
            image = images_by_uid.get(referenced_uid)

            if image is not None:
                return image

        if len(images_by_uid) == 1:
            image = next(iter(images_by_uid.values()))

            logger.warning(
                "No referenced image series was resolved for mask series %s. "
                "Using the only available image series %s.",
                series.series_instance_uid,
                image.series_instance_uid,
            )

            return image

        available_uids = sorted(images_by_uid)

        raise RuntimeError(
            "Could not resolve reference image for mask series "
            f"{series.series_instance_uid}. "
            f"Referenced series: {list(referenced_uids)}. "
            f"Available image series: {available_uids}."
        )

    @staticmethod
    def _group_images_and_native_masks(
        *,
        images: list[ImageVolume],
        masks: list[MaskVolume],
    ) -> list[ImageMaskSet]:
        """Associate converted masks only with their native reference image.

        Cross-modality mask reuse is handled later by the preprocessing step
        when combine_segmentation=True.
        """

        masks_by_reference_uid: dict[str, list[MaskVolume]] = {}

        for mask in masks:
            masks_by_reference_uid.setdefault(
                mask.reference_series_instance_uid,
                [],
            ).append(mask)

        return [
            ImageMaskSet(
                image=image,
                masks=list(
                    masks_by_reference_uid.get(
                        image.series_instance_uid,
                        [],
                    )
                ),
            )
            for image in images
        ]
