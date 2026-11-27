from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import SimpleITK as sitk

# Import processors for side-effect registration of built-ins.
import okapy.preprocessing.processors  # noqa: F401
from okapy.core.geometry import image_physical_box
from okapy.core.models import (
    ImageMaskSet,
    ImageVolume,
    MaskVolume,
    StudyVolumes,
    VolumeCollection,
)
from okapy.dicom.conversion.utils import ensure_inside, safe_name
from okapy.preprocessing.geometry import (
    compute_common_fov,
    drop_empty_masks,
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

logger = logging.getLogger(__name__)


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
            study_name = safe_name(study.study_instance_uid or f"study_{index:04d}")
            study_output_dir = ensure_inside(output_dir / study_name, output_dir)

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
        write_outputs = self.study_config.write_outputs
        image_output_dir = ensure_inside(
            output_dir / self.study_config.image_output_subdir, output_dir
        )
        mask_output_dir = ensure_inside(
            output_dir / self.study_config.mask_output_subdir, output_dir
        )

        if study_instance_uid is None and images:
            study_instance_uid = images[0].study_instance_uid

        image_mask_sets: list[ImageMaskSet] = []
        plans: list[PreprocessingPlan] = []
        skipped_images: list[ImageVolume] = []

        # Empty masks have no extent: drop them before any geometry is computed.
        masks = drop_empty_masks(masks)

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

            geometry_config = self._geometry_config_for_image(image)

            if not image_masks and self._skip_image_without_masks(
                image, geometry_config
            ):
                skipped_images.append(image)
                continue

            if geometry_config.crop_to_common_fov and common_fov is None:
                raise ValueError(
                    "crop_to_common_fov=True, but the image FOVs do not overlap."
                )

            reference_grid = make_reference_grid(
                image=image,
                masks=image_masks,
                geometry_config=geometry_config,
                common_fov=common_fov,
                pixel_id=sitk.sitkFloat32,
            )
            physical_roi = image_physical_box(reference_grid)

            output_image_path = None
            if write_outputs:
                output_image_path = ensure_inside(
                    image_output_dir
                    / image_output_name(
                        patient_id=image.patient_id,
                        modality_key=image.modality_key,
                        series_instance_uid=image.series_instance_uid,
                    ),
                    image_output_dir,
                )

            plan = PreprocessingPlan(
                image=image,
                masks=image_masks,
                modality_key=image.modality_key,
                geometry_config=geometry_config,
                physical_roi=physical_roi,
                reference_grid=reference_grid,
                output_image_path=output_image_path,
                output_mask_dir=mask_output_dir if write_outputs else None,
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
                output_mask_path = None
                if write_outputs:
                    output_mask_path = ensure_inside(
                        mask_output_dir
                        / mask_output_name(
                            patient_id=mask.patient_id,
                            label=mask.label,
                            target_modality_key=geometry_image.modality_key,
                            target_series_instance_uid=(
                                geometry_image.series_instance_uid
                            ),
                            reference_series_instance_uid=(
                                mask.reference_series_instance_uid
                            ),
                        ),
                        mask_output_dir,
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

            if write_outputs:
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

    def _skip_image_without_masks(
        self,
        image: ImageVolume,
        geometry_config: GeometryConfig,
    ) -> bool:
        """Decide whether an image without (non-empty) masks is skipped."""

        skip = self.study_config.skip_images_without_masks

        if skip is False and geometry_config.crop_to_masks:
            raise ValueError(
                f"Image {image.series_instance_uid} has no masks, but its "
                "geometry config has crop_to_masks=True and "
                "skip_images_without_masks=False. Set crop_to_masks=False or "
                "allow skipping images without masks."
            )

        if skip is None:
            skip = geometry_config.crop_to_masks

        if skip:
            logger.warning(
                "Skipping image %s (%s): no masks apply to it.",
                image.series_instance_uid,
                image.modality_key,
            )

        return skip

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
