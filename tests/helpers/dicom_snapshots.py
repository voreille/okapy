from pathlib import Path

import numpy as np
import SimpleITK as sitk


def image_summary(path: Path) -> dict:
    image = sitk.ReadImage(str(path))
    array = sitk.GetArrayFromImage(image)

    return {
        "shape": list(array.shape),
        "spacing": list(image.GetSpacing()),
        "origin": list(image.GetOrigin()),
        "direction": list(image.GetDirection()),
        "dtype": str(array.dtype),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def mask_summary(path: Path) -> dict:
    image = sitk.ReadImage(str(path))
    array = sitk.GetArrayFromImage(image)

    labels, counts = np.unique(array, return_counts=True)

    return {
        "shape": list(array.shape),
        "spacing": list(image.GetSpacing()),
        "origin": list(image.GetOrigin()),
        "direction": list(image.GetDirection()),
        "dtype": str(array.dtype),
        "labels": [int(x) for x in labels],
        "voxel_count_by_label": {
            str(int(label)): int(count)
            for label, count in zip(labels, counts)
        },
    }