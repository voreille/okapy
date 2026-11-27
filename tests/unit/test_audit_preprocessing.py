"""Audit: preprocessing with empty masks, oblique common FOV and Int8 resampling."""

from __future__ import annotations

import logging

import numpy as np
import pytest
import SimpleITK as sitk

from okapy.core.geometry import image_physical_box, resample_to_reference
from okapy.preprocessing.models import StudyPreprocessingConfig
from okapy.preprocessing.step import StudyPreprocessingStep
from tests.helpers import factories

CT_UID = "1.2.840.1.101"
PT_UID = "1.2.840.1.202"


def _image(size=(20, 20, 20), spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0)):
    array = np.arange(np.prod(size), dtype=np.float32).reshape(size[::-1])
    image = sitk.GetImageFromArray(array)
    image.SetSpacing(spacing)
    image.SetOrigin(origin)
    return image


def _box_mask(image, lo, hi):
    array = np.zeros(image.GetSize()[::-1], dtype=np.uint8)
    array[lo[2] : hi[2] + 1, lo[1] : hi[1] + 1, lo[0] : hi[0] + 1] = 1
    mask = sitk.GetImageFromArray(array)
    mask.CopyInformation(image)
    return mask


def _empty_mask(image):
    mask = sitk.Image(image.GetSize(), sitk.sitkUInt8)
    mask.CopyInformation(image)
    return mask


def _step(geometry, **general):
    general.setdefault("write_outputs", False)
    return StudyPreprocessingStep(
        study_config=StudyPreprocessingConfig.from_dict({"general": general}),
        geometry_section={"common": geometry},
    )


def _ct(image, uid=CT_UID, modality="CT"):
    return factories.image_volume(
        image, modality=modality, series_instance_uid=uid, path=f"{uid}.nii.gz"
    )


def _mask(mask, label, ref_uid=CT_UID):
    return factories.mask_volume(
        mask, label=label, reference_series_instance_uid=ref_uid, path=f"{label}.nii.gz"
    )


# --------------------------------------------------------------------------
# Empty masks together with non-empty ones
# --------------------------------------------------------------------------


def test_empty_mask_of_one_image_does_not_affect_other_image(tmp_path, caplog):
    ct_image = _image()
    pt_image = _image(spacing=(2.0, 2.0, 2.0))
    ct = _ct(ct_image)
    pt = _ct(pt_image, uid=PT_UID, modality="PT")
    masks = [
        _mask(_empty_mask(ct_image), "empty_on_ct"),
        _mask(_box_mask(ct_image, (2, 2, 2), (5, 5, 5)), "gtv_on_ct"),
        _mask(_box_mask(pt_image, (1, 1, 1), (3, 3, 3)), "gtv_on_pt", ref_uid=PT_UID),
    ]

    with caplog.at_level(logging.WARNING):
        study = _step({"crop_to_masks": True}).preprocess_study(
            images=[ct, pt], masks=masks, output_dir=tmp_path
        )

    by_uid = {s.image.series_instance_uid: s for s in study.image_mask_sets}
    assert set(by_uid) == {CT_UID, PT_UID}
    assert [m.label for m in by_uid[CT_UID].masks] == ["gtv_on_ct"]
    assert by_uid[CT_UID].image.image.GetSize() == (4, 4, 4)
    assert [m.label for m in by_uid[PT_UID].masks] == ["gtv_on_pt"]
    assert any("empty_on_ct" in r.getMessage() for r in caplog.records)


def test_empty_mask_with_combine_segmentation_is_dropped_for_every_image(tmp_path):
    ct_image = _image()
    ct = _ct(ct_image)
    pt = _ct(_image(), uid=PT_UID, modality="PT")
    masks = [
        _mask(_empty_mask(ct_image), "empty"),
        _mask(_box_mask(ct_image, (2, 2, 2), (5, 5, 5)), "gtv"),
    ]

    study = _step({"crop_to_masks": True}, combine_segmentation=True).preprocess_study(
        images=[ct, pt], masks=masks, output_dir=tmp_path
    )

    assert len(study.image_mask_sets) == 2
    for image_mask_set in study.image_mask_sets:
        assert [m.label for m in image_mask_set.masks] == ["gtv"]


def test_mask_that_vanishes_on_the_target_grid_is_not_returned_silently(
    tmp_path, caplog
):
    """A 1-voxel ROI resampled 1 mm -> 4 mm thresholds to an all-zero mask.

    An all-zero mask reaching feature extraction is the same "empty ROI"
    failure the empty-mask criterion is about: it must be dropped or at least
    warned about, never passed on silently.
    """

    image = _image()
    masks = [
        _mask(_box_mask(image, (9, 9, 9), (9, 9, 9)), "tiny"),
        _mask(_box_mask(image, (2, 2, 2), (17, 17, 17)), "big"),
    ]

    with caplog.at_level(logging.WARNING):
        study = _step({"crop_to_masks": True, "spacing": [4, 4, 4]}).preprocess_study(
            images=[_ct(image)], masks=masks, output_dir=tmp_path
        )

    (image_mask_set,) = study.image_mask_sets
    for mask in image_mask_set.masks:
        if not sitk.GetArrayViewFromImage(mask.image).any():
            assert any("tiny" in r.getMessage() for r in caplog.records), (
                "all-zero mask returned without a warning"
            )


# --------------------------------------------------------------------------
# Common FOV with oblique images
# --------------------------------------------------------------------------


def _rotated_about_z(image, degrees):
    angle = np.deg2rad(degrees)
    c, s = np.cos(angle), np.sin(angle)
    image.SetDirection((c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0))
    return image


def test_common_fov_with_an_oblique_image_stays_inside_each_image(tmp_path):
    axial = _image(origin=(-10.0, -10.0, 0.0))
    oblique = _rotated_about_z(_image(origin=(0.0, -14.0, 2.0)), 45.0)

    study = _step(
        {"crop_to_masks": False, "crop_to_common_fov": True}
    ).preprocess_study(
        images=[_ct(axial), _ct(oblique, uid=PT_UID, modality="PT")],
        masks=[],
        output_dir=tmp_path,
    )

    assert len(study.image_mask_sets) == 2
    for image_mask_set, source in zip(
        study.image_mask_sets, (axial, oblique), strict=True
    ):
        grid = image_mask_set.image.image
        assert grid.GetDirection() == pytest.approx(source.GetDirection())
        box = image_physical_box(grid)
        source_box = image_physical_box(source)
        assert np.all(np.asarray(box.min_xyz) >= np.asarray(source_box.min_xyz) - 1e-6)
        assert np.all(np.asarray(box.max_xyz) <= np.asarray(source_box.max_xyz) + 1e-6)


def test_common_fov_with_padding_and_oblique_image_is_clipped(tmp_path):
    axial = _image()
    oblique = _rotated_about_z(_image(origin=(10.0, -4.0, 0.0)), 30.0)
    mask = _mask(_box_mask(axial, (0, 0, 0), (2, 2, 2)), "gtv")

    study = _step(
        {"crop_to_masks": True, "crop_to_common_fov": True, "padding_voxels": 5},
        skip_images_without_masks=True,
    ).preprocess_study(
        images=[_ct(axial), _ct(oblique, uid=PT_UID, modality="PT")],
        masks=[mask],
        output_dir=tmp_path,
    )

    (image_mask_set,) = study.image_mask_sets
    grid = image_mask_set.image.image
    # Padding never goes below index 0 of the axial image.
    assert np.all(np.asarray(grid.GetOrigin()) >= -1e-6)


# --------------------------------------------------------------------------
# Int8 resampling
# --------------------------------------------------------------------------


def test_int8_resampling_keeps_negative_values_and_type():
    array = np.array([[[-128, -1, 0, 127]]], dtype=np.int8)
    image = sitk.GetImageFromArray(array)

    out = resample_to_reference(
        image,
        image,
        interpolator=sitk.sitkNearestNeighbor,
        output_pixel_type=sitk.sitkInt8,
    )

    assert out.GetPixelID() == sitk.sitkInt8
    np.testing.assert_array_equal(sitk.GetArrayFromImage(out), array)


def test_float_to_int8_resampling_returns_int8():
    image = sitk.GetImageFromArray(np.full((2, 2, 2), 5.0, dtype=np.float32))

    out = resample_to_reference(
        image, image, interpolator=sitk.sitkLinear, output_pixel_type=sitk.sitkInt8
    )

    assert out.GetPixelID() == sitk.sitkInt8
    assert sitk.GetArrayFromImage(out).dtype == np.int8


def test_uint8_pixel_type_zero_like_constant_is_honoured():
    # sitkUInt8 == 1, sitkInt8 == 0: both falsy-adjacent ids must be honoured.
    image = sitk.GetImageFromArray(np.full((2, 2, 2), 5.0, dtype=np.float32))
    for pixel_type in (sitk.sitkInt8, sitk.sitkUInt8):
        out = resample_to_reference(
            image,
            image,
            interpolator=sitk.sitkLinear,
            output_pixel_type=pixel_type,
        )
        assert out.GetPixelID() == pixel_type
