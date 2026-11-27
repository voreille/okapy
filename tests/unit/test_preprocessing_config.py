"""Parsing of the preprocessing configuration."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk
import yaml

from okapy.preprocessing.builder import build_preprocessing_step
from okapy.preprocessing.models import GeometryConfig, StudyPreprocessingConfig
from tests.helpers import factories

FALSE_STRINGS = ["false", "False", "no", "0"]
TRUE_STRINGS = ["true", "yes", "1"]


# ---------------------------------------------------------------------------
# GeometryConfig: spacing
# ---------------------------------------------------------------------------


def test_geometry_spacing_defaults_to_none():
    assert GeometryConfig().spacing is None
    assert GeometryConfig.from_dict({}).spacing is None
    assert GeometryConfig.from_dict({"crop_to_masks": False}).spacing is None


def test_geometry_spacing_per_axis_minus_one_is_kept():
    config = GeometryConfig.from_dict({"spacing": [-1, 2.0, -1]})

    assert config.spacing == (-1.0, 2.0, -1.0)


@pytest.mark.parametrize("spacing", [[0, 1, 1], [1, -2, 1], [1, 1]])
def test_geometry_rejects_invalid_spacing(spacing):
    with pytest.raises(ValueError, match="spacing"):
        GeometryConfig.from_dict({"spacing": spacing})


# ---------------------------------------------------------------------------
# GeometryConfig: padding_voxels
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("padding", [[1, 2, 3], (1, 2, 3)])
def test_geometry_accepts_per_axis_padding(padding):
    config = GeometryConfig.from_dict({"spacing": [1, 1, 1], "padding_voxels": padding})

    assert tuple(config.padding_voxels) == (1, 2, 3)


@pytest.mark.parametrize(
    "padding",
    [[1, -1, 0], [1, 2], [1, 2, 3, 4], [1, 2.5, 0], [True, 1, 1], -1, 2.5],
)
def test_geometry_rejects_invalid_padding(padding):
    with pytest.raises(ValueError, match="padding_voxels"):
        GeometryConfig.from_dict({"spacing": [1, 1, 1], "padding_voxels": padding})


# ---------------------------------------------------------------------------
# GeometryConfig: unknown keys and booleans
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", ["crop_to_mask", "spacings", "interpolator"])
def test_geometry_rejects_unknown_keys(key):
    with pytest.raises(ValueError, match=key):
        GeometryConfig.from_dict({"spacing": [1, 1, 1], key: True})


@pytest.mark.parametrize("key", ["crop_to_masks", "crop_to_common_fov"])
@pytest.mark.parametrize("value", FALSE_STRINGS)
def test_geometry_bool_false_strings(key, value):
    config = GeometryConfig.from_dict({"spacing": [1, 1, 1], key: value})

    assert getattr(config, key) is False


@pytest.mark.parametrize("key", ["crop_to_masks", "crop_to_common_fov"])
@pytest.mark.parametrize("value", TRUE_STRINGS)
def test_geometry_bool_true_strings(key, value):
    config = GeometryConfig.from_dict({"spacing": [1, 1, 1], key: value})

    assert getattr(config, key) is True


@pytest.mark.parametrize("key", ["crop_to_masks", "crop_to_common_fov"])
@pytest.mark.parametrize("value", [True, False])
def test_geometry_real_bools_pass_through(key, value):
    assert (
        getattr(GeometryConfig.from_dict({"spacing": [1, 1, 1], key: value}), key)
        is value
    )


@pytest.mark.parametrize("key", ["crop_to_masks", "crop_to_common_fov"])
def test_geometry_rejects_ambiguous_bool_string(key):
    with pytest.raises(ValueError, match=f"{key}|maybe"):
        GeometryConfig.from_dict({"spacing": [1, 1, 1], key: "maybe"})


# ---------------------------------------------------------------------------
# StudyPreprocessingConfig
# ---------------------------------------------------------------------------


def test_study_config_rejects_unknown_general_key():
    with pytest.raises(ValueError, match="write_output"):
        StudyPreprocessingConfig.from_dict({"general": {"write_output": False}})


def test_study_config_accepts_known_general_keys():
    config = StudyPreprocessingConfig.from_dict(
        {
            "general": {
                "combine_segmentation": True,
                "skip_images_without_masks": True,
                "write_outputs": False,
                "image_output_subdir": "img",
                "mask_output_subdir": "msk",
            }
        }
    )

    assert config.combine_segmentation is True
    assert config.skip_images_without_masks is True
    assert config.write_outputs is False
    assert config.image_output_subdir == "img"
    assert config.mask_output_subdir == "msk"


@pytest.mark.parametrize(
    "key", ["combine_segmentation", "write_outputs", "skip_images_without_masks"]
)
@pytest.mark.parametrize("value", FALSE_STRINGS)
def test_study_bool_false_strings(key, value):
    config = StudyPreprocessingConfig.from_dict({"general": {key: value}})

    assert getattr(config, key) is False


@pytest.mark.parametrize(
    "key", ["combine_segmentation", "write_outputs", "skip_images_without_masks"]
)
@pytest.mark.parametrize("value", TRUE_STRINGS)
def test_study_bool_true_strings(key, value):
    config = StudyPreprocessingConfig.from_dict({"general": {key: value}})

    assert getattr(config, key) is True


@pytest.mark.parametrize(
    "key", ["combine_segmentation", "write_outputs", "skip_images_without_masks"]
)
def test_study_rejects_ambiguous_bool_string(key):
    with pytest.raises(ValueError, match=f"{key}|maybe"):
        StudyPreprocessingConfig.from_dict({"general": {key: "maybe"}})


def test_study_real_bools_pass_through():
    config = StudyPreprocessingConfig.from_dict(
        {"general": {"combine_segmentation": True, "write_outputs": False}}
    )

    assert config.combine_segmentation is True
    assert config.write_outputs is False


def test_study_skip_option_explicit_false_is_kept():
    config = StudyPreprocessingConfig.from_dict(
        {"general": {"skip_images_without_masks": False}}
    )

    assert config.skip_images_without_masks is False


# ---------------------------------------------------------------------------
# Empty YAML sections
# ---------------------------------------------------------------------------


def test_empty_general_section_gives_defaults():
    config = yaml.safe_load("general:\n")
    assert config == {"general": None}

    study = StudyPreprocessingConfig.from_dict(config)

    assert study.combine_segmentation is False
    assert study.write_outputs is True
    assert study.skip_images_without_masks is None


def _write(image: sitk.Image, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(image, str(path))
    return path


def test_yaml_with_empty_sections_builds_and_runs(tmp_path):
    config = yaml.safe_load(
        """
general:
geometry_preprocessing:
  common:
  MR:
local_preprocessing:
  common:
mask_preprocessing:
"""
    )
    assert config["general"] is None
    assert config["geometry_preprocessing"] == {"common": None, "MR": None}

    step = build_preprocessing_step(config)

    rng = np.random.default_rng(3)
    image = sitk.GetImageFromArray(rng.random((10, 12, 14)).astype(np.float32))
    image.SetSpacing((0.7, 0.8, 3.0))
    mask_array = np.zeros((10, 12, 14), dtype=np.uint8)
    mask_array[3:6, 4:8, 5:9] = 1
    mask = sitk.GetImageFromArray(mask_array)
    mask.CopyInformation(image)

    mr = factories.image_volume(
        image,
        modality="MR",
        path=_write(image, tmp_path / "in" / "mr.nii.gz"),
        series_instance_uid="1.2.3.4",
    )
    roi = factories.mask_volume(
        mask,
        label="GTV",
        reference_modality="MR",
        path=_write(mask, tmp_path / "in" / "gtv.nii.gz"),
        reference_series_instance_uid="1.2.3.4",
    )

    study = step.preprocess_study(images=[mr], masks=[roi], output_dir=tmp_path / "o")

    (item,) = study.image_mask_sets
    # No spacing configured: native spacing, cropped to the mask by default.
    np.testing.assert_allclose(item.image.image.GetSpacing(), (0.7, 0.8, 3.0))
    assert item.image.image.GetSize() == (4, 4, 3)
    assert sitk.GetArrayFromImage(item.masks[0].image).sum() == 48


def test_unknown_key_in_geometry_section_raises_through_step(tmp_path):
    config = yaml.safe_load(
        """
geometry_preprocessing:
  common:
    spacing: [1, 1, 1]
    crop_to_mask: false
"""
    )
    step = build_preprocessing_step(config)

    image = sitk.Image((6, 6, 6), sitk.sitkFloat32)
    ct = factories.image_volume(image, path=_write(image, tmp_path / "ct.nii.gz"))
    mask = sitk.Image((6, 6, 6), sitk.sitkUInt8)
    mask[2, 2, 2] = 1
    roi = factories.mask_volume(mask, path=_write(mask, tmp_path / "m.nii.gz"))

    with pytest.raises(ValueError, match="crop_to_mask"):
        step.preprocess_study(images=[ct], masks=[roi], output_dir=tmp_path / "o")
