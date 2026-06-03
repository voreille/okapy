from __future__ import annotations

from pathlib import Path

import numpy as np
import SimpleITK as sitk

from okapy.core.models import ImageVolume


def sitk_image_summary(image: sitk.Image) -> dict:
    array = sitk.GetArrayFromImage(image)

    return {
        "size": list(image.GetSize()),
        "spacing": [float(x) for x in image.GetSpacing()],
        "origin": [float(x) for x in image.GetOrigin()],
        "direction": [float(x) for x in image.GetDirection()],
        "pixel_type": image.GetPixelIDTypeAsString(),
        "array_shape": list(array.shape),
        "dtype": str(array.dtype),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
        "sum": float(np.sum(array)),
        "nonzero_voxels": int(np.count_nonzero(array)),
    }


def converted_images_summary(images: list[ImageVolume]) -> dict:
    summaries = []

    for image in sorted(
        images,
        key=lambda x: (
            x.patient_id or "",
            x.modality,
            x.series_instance_uid,
        ),
    ):
        summary = sitk_image_summary(image.image)

        summary.update(
            {
                "path_name": Path(image.path).name,
                "modality": image.modality,
                "patient_id": image.patient_id,
                "study_instance_uid": image.study_instance_uid,
                "series_instance_uid": image.series_instance_uid,
            }
        )

        summaries.append(summary)

    return {
        "num_images": len(summaries),
        "images": summaries,
    }