"""Study-level preprocessing (``StudyPreprocessingStep``) behavior."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

from okapy.preprocessing.builder import build_preprocessing_step
from okapy.preprocessing.models import StudyPreprocessingConfig
from okapy.preprocessing.step import StudyPreprocessingStep
from tests.helpers import factories

CT_UID = "1.2.840.1.101"
PT_UID = "1.2.840.1.202"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _image(size=(20, 20, 20), spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0)):
    rng = np.random.default_rng(42)
    array = rng.random(size[::-1]).astype(np.float32) * 100.0
    image = sitk.GetImageFromArray(array)
    image.SetSpacing(spacing)
    image.SetOrigin(origin)
    return image


def _box_mask(image: sitk.Image, lo, hi) -> sitk.Image:
    """Binary box on inclusive index ranges [lo, hi] (x, y, z) on ``image``'s grid."""

    array = np.zeros(image.GetSize()[::-1], dtype=np.uint8)
    array[lo[2] : hi[2] + 1, lo[1] : hi[1] + 1, lo[0] : hi[0] + 1] = 1
    mask = sitk.GetImageFromArray(array)
    mask.CopyInformation(image)
    return mask


def _empty_mask(image: sitk.Image) -> sitk.Image:
    mask = sitk.Image(image.GetSize(), sitk.sitkUInt8)
    mask.CopyInformation(image)
    return mask


def _write(image: sitk.Image, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(image, str(path))
    return path


def _ct(tmp_path: Path, image: sitk.Image | None = None, uid: str = CT_UID):
    image = image if image is not None else _image()
    path = _write(image, tmp_path / "input" / f"ct_{uid}.nii.gz")
    return factories.image_volume(
        image, modality="CT", path=path, series_instance_uid=uid
    )


def _pt(tmp_path: Path, image: sitk.Image, uid: str = PT_UID):
    path = _write(image, tmp_path / "input" / f"pt_{uid}.nii.gz")
    return factories.image_volume(
        image, modality="PT", path=path, series_instance_uid=uid
    )


def _mask(tmp_path: Path, mask: sitk.Image, *, label: str, ref_uid: str = CT_UID):
    path = _write(mask, tmp_path / "input" / f"mask_{label}_{ref_uid}.nii.gz")
    return factories.mask_volume(
        mask,
        label=label,
        modality="RTSTRUCT",
        reference_modality="CT",
        path=path,
        reference_series_instance_uid=ref_uid,
    )


def _step(geometry_section: dict, **study_kwargs) -> StudyPreprocessingStep:
    return StudyPreprocessingStep(
        study_config=StudyPreprocessingConfig(**study_kwargs),
        geometry_section=geometry_section,
    )


def _warning_messages(caplog, recwarn) -> list[str]:
    messages = [
        record.getMessage()
        for record in caplog.records
        if record.levelno >= logging.WARNING
    ]
    messages += [str(warning.message) for warning in recwarn.list]
    return messages


def _assert_warned(caplog, recwarn, *substrings: str) -> None:
    """At least one warning (logging or ``warnings``) contains one substring."""

    messages = _warning_messages(caplog, recwarn)
    lowered = [m.lower() for m in messages]
    assert any(s.lower() in m for m in lowered for s in substrings), (
        f"Expected a warning mentioning one of {substrings}, got {messages}"
    )


def _sets_by_uid(study):
    return {item.image.series_instance_uid: item for item in study.image_mask_sets}


# ---------------------------------------------------------------------------
# images without masks
# ---------------------------------------------------------------------------


def test_default_skip_option_is_none():
    assert StudyPreprocessingConfig().skip_images_without_masks is None
    assert StudyPreprocessingConfig.from_dict({}).skip_images_without_masks is None
    assert (
        StudyPreprocessingConfig.from_dict({"general": {}}).skip_images_without_masks
        is None
    )


def test_default_config_skips_image_without_masks_and_processes_others(
    tmp_path, caplog, recwarn
):
    caplog.set_level(logging.WARNING)
    ct = _ct(tmp_path)
    pt = _pt(tmp_path, _image(), uid=PT_UID)
    gtv = _mask(tmp_path, _box_mask(ct.image, (5, 6, 7), (9, 10, 11)), label="GTV")

    step = build_preprocessing_step(
        {"geometry_preprocessing": {"common": {"spacing": [-1, -1, -1]}}}
    )
    study, diagnostics = step.preprocess_study(
        images=[ct, pt],
        masks=[gtv],
        output_dir=tmp_path / "out",
        return_diagnostics=True,
    )

    sets = _sets_by_uid(study)
    assert set(sets) == {CT_UID}
    assert [img.series_instance_uid for img in diagnostics.skipped_images] == [PT_UID]
    _assert_warned(caplog, recwarn, PT_UID, "mask")

    ct_set = sets[CT_UID]
    assert ct_set.image.path.exists()
    assert ct_set.image.image.GetSize() == (5, 5, 5)
    assert [m.label for m in ct_set.masks] == ["GTV"]
    assert ct_set.masks[0].path.exists()
    written = sitk.ReadImage(str(ct_set.masks[0].path))
    assert sitk.GetArrayFromImage(written).sum() == 125


def test_default_config_processes_image_without_masks_when_not_cropping(tmp_path):
    ct = _ct(tmp_path)
    pt_image = _image(size=(10, 12, 14))
    pt = _pt(tmp_path, pt_image, uid=PT_UID)
    gtv = _mask(tmp_path, _box_mask(ct.image, (5, 6, 7), (9, 10, 11)), label="GTV")

    step = _step({"common": {"spacing": [-1, -1, -1], "crop_to_masks": False}})
    study = step.preprocess_study(
        images=[ct, pt], masks=[gtv], output_dir=tmp_path / "out"
    )

    sets = _sets_by_uid(study)
    assert set(sets) == {CT_UID, PT_UID}
    assert sets[PT_UID].masks == []
    assert sets[PT_UID].image.image.GetSize() == (10, 12, 14)
    np.testing.assert_allclose(
        sitk.GetArrayFromImage(sets[PT_UID].image.image),
        sitk.GetArrayFromImage(pt_image),
        rtol=1e-6,
    )


def test_skip_true_always_skips_images_without_masks(tmp_path):
    ct = _ct(tmp_path)
    pt = _pt(tmp_path, _image(), uid=PT_UID)
    gtv = _mask(tmp_path, _box_mask(ct.image, (5, 6, 7), (9, 10, 11)), label="GTV")

    step = _step(
        {"common": {"spacing": [-1, -1, -1], "crop_to_masks": False}},
        skip_images_without_masks=True,
    )
    study, diagnostics = step.preprocess_study(
        images=[ct, pt],
        masks=[gtv],
        output_dir=tmp_path / "out",
        return_diagnostics=True,
    )

    assert set(_sets_by_uid(study)) == {CT_UID}
    assert [img.series_instance_uid for img in diagnostics.skipped_images] == [PT_UID]


def test_explicit_false_with_crop_to_masks_raises_clear_error(tmp_path):
    ct = _ct(tmp_path)
    pt = _pt(tmp_path, _image(), uid=PT_UID)
    gtv = _mask(tmp_path, _box_mask(ct.image, (5, 6, 7), (9, 10, 11)), label="GTV")

    step = _step(
        {"common": {"spacing": [-1, -1, -1], "crop_to_masks": True}},
        skip_images_without_masks=False,
    )
    with pytest.raises(ValueError, match="mask"):
        step.preprocess_study(images=[ct, pt], masks=[gtv], output_dir=tmp_path / "o")


# ---------------------------------------------------------------------------
# empty masks
# ---------------------------------------------------------------------------


def test_empty_mask_is_dropped_and_other_masks_are_cropped(tmp_path, caplog, recwarn):
    caplog.set_level(logging.WARNING)
    ct = _ct(tmp_path)
    gtv = _mask(tmp_path, _box_mask(ct.image, (5, 6, 7), (9, 10, 11)), label="GTV")
    empty = _mask(tmp_path, _empty_mask(ct.image), label="EMPTYROI")

    step = _step({"common": {"spacing": [-1, -1, -1], "padding_voxels": 1}})
    study = step.preprocess_study(
        images=[ct], masks=[empty, gtv], output_dir=tmp_path / "out"
    )

    _assert_warned(caplog, recwarn, "EMPTYROI", "empty")
    (item,) = study.image_mask_sets
    assert [m.label for m in item.masks] == ["GTV"]
    # The crop is driven by GTV alone: [5-1, 9+1] x [6-1, 10+1] x [7-1, 11+1].
    assert item.image.image.GetSize() == (7, 7, 7)
    np.testing.assert_allclose(item.image.image.GetOrigin(), (4.0, 5.0, 6.0), atol=1e-9)
    assert item.masks[0].path.exists()
    assert sitk.GetArrayFromImage(item.masks[0].image).sum() == 125


def test_image_with_only_empty_masks_is_skipped_by_default(tmp_path, caplog, recwarn):
    caplog.set_level(logging.WARNING)
    ct = _ct(tmp_path)
    pt_image = _image(origin=(0.0, 0.0, 0.0))
    pt = _pt(tmp_path, pt_image, uid=PT_UID)
    gtv = _mask(tmp_path, _box_mask(ct.image, (5, 6, 7), (9, 10, 11)), label="GTV")
    pt_empty = _mask(tmp_path, _empty_mask(pt_image), label="PTEMPTY", ref_uid=PT_UID)

    step = _step({"common": {"spacing": [-1, -1, -1]}})
    study, diagnostics = step.preprocess_study(
        images=[ct, pt],
        masks=[gtv, pt_empty],
        output_dir=tmp_path / "out",
        return_diagnostics=True,
    )

    sets = _sets_by_uid(study)
    assert set(sets) == {CT_UID}
    assert [m.label for m in sets[CT_UID].masks] == ["GTV"]
    assert [img.series_instance_uid for img in diagnostics.skipped_images] == [PT_UID]
    _assert_warned(caplog, recwarn, "PTEMPTY", "empty")


# ---------------------------------------------------------------------------
# native spacing (no ``spacing`` key)
# ---------------------------------------------------------------------------


def test_geometry_without_spacing_keeps_native_spacing_and_values(tmp_path):
    image = _image(size=(16, 18, 12), spacing=(0.8, 0.9, 2.5), origin=(-5, 3, 7))
    ct = _ct(tmp_path, image)
    gtv = _mask(tmp_path, _box_mask(image, (2, 3, 4), (6, 8, 7)), label="GTV")

    step = _step({"common": {"crop_to_masks": False}})
    study = step.preprocess_study(images=[ct], masks=[gtv], output_dir=tmp_path / "o")

    (item,) = study.image_mask_sets
    out = item.image.image
    np.testing.assert_allclose(out.GetSpacing(), image.GetSpacing())
    np.testing.assert_allclose(out.GetOrigin(), image.GetOrigin())
    assert out.GetSize() == image.GetSize()
    np.testing.assert_allclose(
        sitk.GetArrayFromImage(out), sitk.GetArrayFromImage(image), rtol=1e-6
    )
    np.testing.assert_array_equal(
        sitk.GetArrayFromImage(item.masks[0].image),
        sitk.GetArrayFromImage(gtv.image),
    )


def test_per_axis_minus_one_keeps_native_spacing_on_that_axis(tmp_path):
    image = _image(size=(20, 20, 8), spacing=(1.0, 1.0, 2.5))
    ct = _ct(tmp_path, image)
    gtv = _mask(tmp_path, _box_mask(image, (4, 4, 2), (12, 12, 5)), label="GTV")

    step = _step({"common": {"spacing": [2.0, 2.0, -1]}})
    study = step.preprocess_study(images=[ct], masks=[gtv], output_dir=tmp_path / "o")

    (item,) = study.image_mask_sets
    np.testing.assert_allclose(item.image.image.GetSpacing(), (2.0, 2.0, 2.5))


# ---------------------------------------------------------------------------
# common FOV + padding
# ---------------------------------------------------------------------------


def _voxel_centre_bounds(image: sitk.Image):
    size = image.GetSize()
    first = np.asarray(image.TransformIndexToPhysicalPoint((0, 0, 0)))
    last = np.asarray(
        image.TransformIndexToPhysicalPoint(tuple(int(n - 1) for n in size))
    )
    return np.minimum(first, last), np.maximum(first, last)


def test_padding_never_extends_grid_beyond_common_fov(tmp_path):
    # CT covers [-0.5, 39.5] mm per axis, PT covers [9.5, 29.5] mm: the
    # common FOV is the PT box.
    ct_image = _image(size=(40, 40, 40))
    pt_image = _image(size=(20, 20, 20), origin=(10.0, 10.0, 10.0))
    ct = _ct(tmp_path, ct_image)
    pt = _pt(tmp_path, pt_image)
    # Mask right at the low edge of the common FOV (CT indices 10..12).
    gtv = _mask(tmp_path, _box_mask(ct_image, (10, 10, 10), (12, 12, 12)), label="GTV")

    step = _step(
        {
            "common": {
                "spacing": [-1, -1, -1],
                "crop_to_masks": True,
                "crop_to_common_fov": True,
                "padding_voxels": 5,
            }
        },
        combine_segmentation=True,
    )
    study = step.preprocess_study(
        images=[ct, pt], masks=[gtv], output_dir=tmp_path / "out"
    )

    fov_lo, fov_hi = np.full(3, 9.5), np.full(3, 29.5)
    sets = _sets_by_uid(study)
    assert set(sets) == {CT_UID, PT_UID}
    for item in sets.values():
        lo, hi = _voxel_centre_bounds(item.image.image)
        assert np.all(lo >= fov_lo - 1e-6), (item.image.modality, lo)
        assert np.all(hi <= fov_hi + 1e-6), (item.image.modality, hi)

    # On the CT the low padding is clipped at the FOV edge, the high side
    # still gets its 5 voxels: centres 10..17.
    ct_out = sets[CT_UID].image.image
    assert ct_out.GetSize() == (8, 8, 8)
    np.testing.assert_allclose(ct_out.GetOrigin(), (10.0, 10.0, 10.0), atol=1e-9)
    assert sitk.GetArrayFromImage(sets[CT_UID].masks[0].image).sum() == 27


# ---------------------------------------------------------------------------
# write_outputs=False
# ---------------------------------------------------------------------------


def test_write_outputs_false_paths_never_point_to_missing_files(tmp_path):
    ct = _ct(tmp_path)
    gtv = _mask(tmp_path, _box_mask(ct.image, (5, 6, 7), (9, 10, 11)), label="GTV")
    out_dir = tmp_path / "out"

    step = _step({"common": {"spacing": [2.0, 2.0, 2.0]}}, write_outputs=False)
    study = step.preprocess_study(images=[ct], masks=[gtv], output_dir=out_dir)

    (item,) = study.image_mask_sets
    volumes = [item.image, *item.masks]
    assert len(volumes) == 2
    for volume in volumes:
        assert volume.path is None or Path(volume.path).exists(), volume.path


# ---------------------------------------------------------------------------
# regression: a normal crop + resample through the step
# ---------------------------------------------------------------------------


def test_native_crop_through_step_writes_expected_values(tmp_path):
    image = _image()
    ct = _ct(tmp_path, image)
    gtv = _mask(tmp_path, _box_mask(image, (5, 6, 7), (9, 10, 11)), label="GTV")

    step = _step({"common": {"spacing": [-1, -1, -1], "padding_voxels": 2}})
    study = step.preprocess_study(images=[ct], masks=[gtv], output_dir=tmp_path / "o")

    (item,) = study.image_mask_sets
    assert item.image.path.exists()
    written = sitk.ReadImage(str(item.image.path))
    assert written.GetSize() == (9, 9, 9)
    np.testing.assert_allclose(written.GetOrigin(), (3.0, 4.0, 5.0), atol=1e-6)
    expected = sitk.GetArrayFromImage(image)[5:14, 4:13, 3:12]
    np.testing.assert_allclose(sitk.GetArrayFromImage(written), expected, rtol=1e-6)
    mask_written = sitk.ReadImage(str(item.masks[0].path))
    assert sitk.GetArrayFromImage(mask_written).sum() == 125


def test_resampled_crop_through_step_has_expected_grid(tmp_path):
    image = _image()
    ct = _ct(tmp_path, image)
    gtv = _mask(tmp_path, _box_mask(image, (5, 5, 5), (9, 9, 9)), label="GTV")

    step = _step({"common": {"spacing": [2.0, 2.0, 2.0]}})
    study = step.preprocess_study(images=[ct], masks=[gtv], output_dir=tmp_path / "o")

    (item,) = study.image_mask_sets
    out = item.image.image
    # Full 2 mm grid is centred on the 1 mm grid: first centre at 0.5 mm.
    # Mask extent [4.5, 9.5] mm keeps centres 4.5, 6.5, 8.5.
    assert out.GetSize() == (3, 3, 3)
    np.testing.assert_allclose(out.GetSpacing(), (2.0, 2.0, 2.0))
    np.testing.assert_allclose(out.GetOrigin(), (4.5, 4.5, 4.5), atol=1e-6)
    assert item.masks[0].image.GetSize() == (3, 3, 3)
    assert item.masks[0].image.GetPixelID() == sitk.sitkUInt8
