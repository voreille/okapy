from __future__ import annotations

from pathlib import Path
from typing import Any

import SimpleITK as sitk

# Import processors for side-effect registration of built-ins.
import okapy.preprocessing.processors  # noqa: F401
from okapy.core.models import (
    ImageMaskSet,
    ImageVolume,
    MaskVolume,
    StudyVolumes,
    VolumeCollection,
)
from okapy.preprocessing.geometry import (
    compute_common_fov,
    compute_processing_roi,
    make_reference_grid,
    masks_for_image,
    resample_image_volume_to_reference,
    resample_mask_volume_to_reference,
)
from okapy.preprocessing.io import (
    image_output_name,
    mask_output_name,
    write_image_unique,
)
from okapy.preprocessing.models import (
    GeometryConfig,
    PreprocessingDiagnostics,
    PreprocessingPlan,
    StudyPreprocessingConfig,
)
from okapy.preprocessing.processor_pipeline import ProcessorPipeline
from okapy.preprocessing.selector import select_section_config


class StudyPreprocessingStep:
    """Preprocess converted image and mask volumes at study level.

    Responsibilities:
    - decide which masks apply to each image;
    - compute crop ROIs in physical coordinates;
    - define target grids;
    - resample images and masks onto matching grids;
    - apply local mask and image processors;
    - optionally write outputs.
    """

    def __init__(
        self,
        *,
        study_config: StudyPreprocessingConfig,
        geometry_section: dict[str, Any] | None = None,
        image_processors: ProcessorPipeline | None = None,
        mask_processors: ProcessorPipeline | None = None,
    ) -> None:
        self.study_config = study_config
        self.geometry_section = geometry_section or {}
        self.image_processors = image_processors or ProcessorPipeline.empty()
        self.mask_processors = mask_processors or ProcessorPipeline.empty()

    def run(
        self,
        volumes: VolumeCollection,
        *,
        output_dir: Path,
    ) -> VolumeCollection:
        """Preprocess all studies in a volume collection."""

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        processed_studies: list[StudyVolumes] = []

        for index, study in enumerate(volumes.studies):
            study_name = study.study_instance_uid or f"study_{index:04d}"
            study_output_dir = output_dir / study_name

            processed_study = self.preprocess_study(
                images=study.images,
                masks=study.masks,
                output_dir=study_output_dir,
                study_instance_uid=study.study_instance_uid,
            )

            processed_studies.append(processed_study)

        return VolumeCollection(studies=processed_studies)

    def preprocess_study(
        self,
        *,
        images: list[ImageVolume],
        masks: list[MaskVolume],
        output_dir: Path,
        study_instance_uid: str | None = None,
        return_diagnostics: bool = False,
    ) -> StudyVolumes | tuple[StudyVolumes, PreprocessingDiagnostics]:
        """Preprocess one study."""

        output_dir = Path(output_dir)
        image_output_dir = output_dir / self.study_config.image_output_subdir
        mask_output_dir = output_dir / self.study_config.mask_output_subdir

        if study_instance_uid is None and images:
            study_instance_uid = images[0].study_instance_uid

        image_mask_sets: list[ImageMaskSet] = []
        plans: list[PreprocessingPlan] = []
        skipped_images: list[ImageVolume] = []

        common_fov = None

        if images:
            try:
                common_fov = compute_common_fov(images)
            except ValueError:
                # Fail later only if a selected geometry config explicitly
                # requests crop_to_common_fov.
                common_fov = None

        for image in images:
            image_masks = masks_for_image(
                image,
                masks,
                combine_segmentation=(self.study_config.combine_segmentation),
            )

            if not image_masks and self.study_config.skip_images_without_masks:
                skipped_images.append(image)
                continue

            geometry_config = self._geometry_config_for_image(image)

            if geometry_config.crop_to_common_fov and common_fov is None:
                raise ValueError(
                    "crop_to_common_fov=True, but the image FOVs do not overlap."
                )

            physical_roi = compute_processing_roi(
                image=image,
                masks=image_masks,
                geometry_config=geometry_config,
                common_fov=common_fov,
            )

            reference_grid = make_reference_grid(
                image=image,
                roi=physical_roi,
                geometry_config=geometry_config,
                pixel_id=sitk.sitkFloat32,
            )

            output_image_path = image_output_dir / image_output_name(
                patient_id=image.patient_id,
                modality_key=image.modality_key,
                series_instance_uid=image.series_instance_uid,
            )

            plan = PreprocessingPlan(
                image=image,
                masks=image_masks,
                modality_key=image.modality_key,
                geometry_config=geometry_config,
                physical_roi=physical_roi,
                reference_grid=reference_grid,
                output_image_path=output_image_path,
                output_mask_dir=mask_output_dir,
            )
            plans.append(plan)

            geometry_image = resample_image_volume_to_reference(
                image,
                reference_grid,
                geometry_config=geometry_config,
                output_path=output_image_path,
            )

            # Masks are resampled first so image processors such as
            # MaskedStandardizer can use masks on the target grid.
            processed_masks: list[MaskVolume] = []

            for mask in image_masks:
                output_mask_path = mask_output_dir / mask_output_name(
                    patient_id=mask.patient_id,
                    label=mask.label,
                    target_modality_key=(geometry_image.modality_key),
                    target_series_instance_uid=(geometry_image.series_instance_uid),
                    reference_series_instance_uid=(mask.reference_series_instance_uid),
                )

                geometry_mask = resample_mask_volume_to_reference(
                    mask,
                    reference_grid,
                    target_image=geometry_image,
                    geometry_config=geometry_config,
                    output_path=output_mask_path,
                )

                local_mask = self.mask_processors.apply(
                    geometry_mask,
                    key=geometry_image.modality_key,
                    reference=geometry_image,
                )

                processed_masks.append(local_mask)

            processed_image = self.image_processors.apply(
                geometry_image,
                key=geometry_image.modality_key,
                masks=processed_masks,
            )

            if self.study_config.write_outputs:
                processed_image = self._write_image(processed_image)
                processed_masks = [self._write_mask(mask) for mask in processed_masks]

            image_mask_sets.append(
                ImageMaskSet(
                    image=processed_image,
                    masks=processed_masks,
                )
            )

        result = StudyVolumes(
            study_instance_uid=study_instance_uid,
            image_mask_sets=image_mask_sets,
        )

        if return_diagnostics:
            diagnostics = PreprocessingDiagnostics(
                plans=plans,
                skipped_images=skipped_images,
            )
            return result, diagnostics

        return result

    def _geometry_config_for_image(
        self,
        image: ImageVolume,
    ) -> GeometryConfig:
        config = select_section_config(
            self.geometry_section,
            key=image.modality_key,
            include_common=True,
        )

        return GeometryConfig.from_dict(config)

    @staticmethod
    def _write_image(image: ImageVolume) -> ImageVolume:
        final_path = write_image_unique(
            image.image,
            image.path,
        )

        if final_path == image.path:
            return image

        return image.with_image(
            image.image,
            path=final_path,
        )

    @staticmethod
    def _write_mask(mask: MaskVolume) -> MaskVolume:
        final_path = write_image_unique(
            mask.image,
            mask.path,
        )

        if final_path == mask.path:
            return mask

        return mask.with_image(
            mask.image,
            path=final_path,
        )
