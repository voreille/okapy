from __future__ import annotations

from pathlib import Path
from typing import Any

import SimpleITK as sitk

# Import processors for side-effect registration of built-ins.
import okapy.preprocessing.processors  # noqa: F401
from okapy.core.models import ImageMaskSet, ImageVolume, MaskVolume, StudyVolumes
from okapy.preprocessing.geometry import (
    compute_common_fov,
    compute_processing_roi,
    make_reference_grid,
    masks_for_image,
    resample_image_volume_to_reference,
    resample_mask_volume_to_reference,
)
from okapy.preprocessing.io import image_output_name, mask_output_name, write_image_unique
from okapy.preprocessing.models import (
    GeometryConfig,
    PreprocessingDiagnostics,
    PreprocessingPlan,
    StudyPreprocessingConfig,
)
from okapy.preprocessing.processor_pipeline import ProcessorPipeline
from okapy.preprocessing.selector import select_section_config


class StudyPreprocessingPipeline:
    """Preprocess image/mask volumes at the study level.

    Responsibilities:
    - decide which masks apply to each image;
    - compute crop ROI in physical coordinates;
    - resample image and masks to the same target grid;
    - apply local image/mask processors;
    - optionally write outputs.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self.study_config = StudyPreprocessingConfig.from_dict(self.config)
        self.geometry_section = self.config.get("geometry_preprocessing") or {}

        # ``local_preprocessing`` is the preferred new name. Keep
        # ``intensity_preprocessing`` as a temporary alias.
        local_config = self.config.get("local_preprocessing")
        if local_config is None:
            local_config = self.config.get("intensity_preprocessing")

        self.image_processors = ProcessorPipeline.from_config(local_config)
        self.mask_processors = ProcessorPipeline.from_config(
            self.config.get("mask_preprocessing")
        )

    def preprocess_study(
        self,
        *,
        images: list[ImageVolume],
        masks: list[MaskVolume],
        output_dir: Path,
        study_instance_uid: str | None = None,
        return_diagnostics: bool = False,
    ):
        output_dir = Path(output_dir)
        image_output_dir = output_dir / self.study_config.image_output_subdir
        mask_output_dir = output_dir / self.study_config.mask_output_subdir

        if study_instance_uid is None and images:
            study_instance_uid = images[0].study_instance_uid

        image_mask_sets: list[ImageMaskSet] = []
        diagnostics = PreprocessingDiagnostics()
        plans: list[PreprocessingPlan] = []
        skipped_images: list[ImageVolume] = []

        # Compute once. It is used only by images whose geometry config asks for it.
        common_fov = None
        if images:
            try:
                common_fov = compute_common_fov(images)
            except ValueError:
                # Do not fail unless a specific image asks for crop_to_common_fov.
                common_fov = None

        for image in images:
            image_masks = masks_for_image(
                image,
                masks,
                combine_segmentation=self.study_config.combine_segmentation,
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

            reference_grid = make_reference_grid( # TODO: change this for the -1 spacing case
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

            # First create masks on the image grid, then apply mask processors.
            # This makes masks available to local image processors such as
            # MaskedStandardizer.
            processed_masks: list[MaskVolume] = []
            for mask in image_masks:
                output_mask_path = mask_output_dir / mask_output_name(
                    patient_id=mask.patient_id,
                    label=mask.label,
                    target_modality_key=geometry_image.modality_key,
                    target_series_instance_uid=geometry_image.series_instance_uid,
                    reference_series_instance_uid=mask.reference_series_instance_uid,
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
                final_image_path = write_image_unique(
                    processed_image.image,
                    processed_image.path,
                )
                if final_image_path != processed_image.path:
                    processed_image = processed_image.with_image(
                        processed_image.image,
                        path=final_image_path,
                    )

                final_masks = []
                for mask in processed_masks:
                    final_mask_path = write_image_unique(mask.image, mask.path)
                    if final_mask_path != mask.path:
                        mask = mask.with_image(mask.image, path=final_mask_path)
                    final_masks.append(mask)
                processed_masks = final_masks

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

    def _geometry_config_for_image(self, image: ImageVolume) -> GeometryConfig:
        config = select_section_config(
            self.geometry_section,
            key=image.modality_key,
            include_common=True,
        )
        return GeometryConfig.from_dict(config)
