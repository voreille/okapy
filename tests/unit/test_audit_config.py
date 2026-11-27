"""Audit: config strictness and legacy-config detection / migration edge cases."""

from __future__ import annotations

import math

import pytest
import yaml

from okapy.config import load_config
from okapy.config.legacy import is_legacy_config
from okapy.features.builder import build_feature_extraction_step
from okapy.preprocessing.models import (
    GeometryConfig,
    StudyPreprocessingConfig,
    parse_bool,
)
from okapy.preprocessing.selector import select_section_config

PYRADIOMICS_PARAMS = {"setting": {"binWidth": 25}}


def _legacy(*, padding=..., spacing=(1.0, 1.0, 1.0)):
    general = {"submodalities": False}
    if padding is not ...:
        general["padding"] = padding
    volume_preprocessing = {"common": {"bspline_resampler": {"order": 3}}}
    if spacing is not None:
        volume_preprocessing["common"]["bspline_resampler"]["resampling_spacing"] = (
            list(spacing)
        )
    return {
        "general": general,
        "volume_preprocessing": volume_preprocessing,
        "feature_extraction": {"CT": {"pyradiomics": {"original": PYRADIOMICS_PARAMS}}},
    }


def _ct_geometry(config):
    return select_section_config(
        config["geometry_preprocessing"], key="CT", include_common=True
    )


# --------------------------------------------------------------------------
# GeometryConfig: non-finite numbers
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "spacing",
    [
        [math.nan, 1.0, 1.0],
        [1.0, math.inf, 1.0],
        [1.0, 1.0, -math.inf],
    ],
    ids=["nan", "inf", "-inf"],
)
def test_geometry_rejects_non_finite_spacing(spacing):
    with pytest.raises(ValueError):
        GeometryConfig.from_dict({"spacing": spacing})


@pytest.mark.parametrize("padding", [math.inf, math.nan, [1, math.inf, 1]])
def test_geometry_rejects_non_finite_padding_with_value_error(padding):
    with pytest.raises(ValueError):
        GeometryConfig.from_dict({"padding_voxels": padding})


@pytest.mark.parametrize("spacing", ["1,1,1", [1.0, 1.0], [1, 1, 1, 1]])
def test_geometry_rejects_malformed_spacing(spacing):
    with pytest.raises(ValueError):
        GeometryConfig.from_dict({"spacing": spacing})


def test_geometry_spacing_from_yaml_nan_literal_is_rejected():
    config = yaml.safe_load("spacing: [.nan, 1, 1]")
    with pytest.raises(ValueError):
        GeometryConfig.from_dict(config)


# --------------------------------------------------------------------------
# Strict booleans
# --------------------------------------------------------------------------


@pytest.mark.parametrize("value", [" TRUE ", "Yes", "ON", "1"])
def test_parse_bool_true_strings_are_case_and_space_insensitive(value):
    assert parse_bool(value, key="k") is True


@pytest.mark.parametrize("value", ["", "none", "null", "t", "y", "2", "truee", None])
def test_parse_bool_rejects_ambiguous_values(value):
    with pytest.raises(ValueError, match="'k'"):
        parse_bool(value, key="k")


def test_null_geometry_boolean_is_rejected_not_defaulted():
    # ``crop_to_masks:`` with no value is None in YAML: ambiguous.
    with pytest.raises(ValueError):
        GeometryConfig.from_dict(yaml.safe_load("crop_to_masks:"))


@pytest.mark.parametrize("value", ["false", "no", "off", "0"])
def test_feature_continue_on_error_string_false_is_false(value):
    """``bool("false")`` is True: strings must be parsed strictly here too."""

    step = build_feature_extraction_step(
        {"feature_extraction": {"continue_on_error": value}}
    )
    assert step.continue_on_error is False


# --------------------------------------------------------------------------
# Empty sections and unknown keys
# --------------------------------------------------------------------------


def test_null_selector_sections_give_defaults():
    section = yaml.safe_load("common:\nCT:\ndefault:\n")
    config = select_section_config(section, key="CT", include_common=True)
    assert GeometryConfig.from_dict(config) == GeometryConfig()


def test_study_config_with_null_general_and_null_values():
    config = StudyPreprocessingConfig.from_dict(yaml.safe_load("general:\n"))
    assert config == StudyPreprocessingConfig()


def test_unknown_key_in_a_selector_is_reported():
    section = {"common": {}, "CT": {"spacnig": [1, 1, 1]}}
    config = select_section_config(section, key="CT", include_common=True)
    with pytest.raises(ValueError, match="spacnig"):
        GeometryConfig.from_dict(config)


def test_unknown_study_key_is_reported_with_its_name():
    with pytest.raises(ValueError, match="write_output"):
        StudyPreprocessingConfig.from_dict({"general": {"write_output": False}})


# --------------------------------------------------------------------------
# Legacy detection: no false positives on new-shape configs
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "config",
    [
        pytest.param(
            {
                "general": {"result_format": "wide", "combine_segmentation": True},
                "geometry_preprocessing": {
                    "common": {"spacing": [1, 1, 1], "padding_voxels": [1, 2, 3]},
                    "PT": {"spacing": [2, 2, 2]},
                },
                "local_preprocessing": {"CT": {"clipper": {"min": -1000}}},
                "mask_preprocessing": {"CT": {"some_processor": {}}},
                "feature_extraction": {
                    "continue_on_error": True,
                    "result_format": "long",
                    "include_backend": True,
                    "common": [{"type": "pet", "name": "pet"}],
                    "CT": [],
                    "PT": None,
                },
            },
            id="full-new-shape",
        ),
        pytest.param({"general": None, "feature_extraction": None}, id="nulls"),
        pytest.param({"general": {}}, id="empty-general"),
        pytest.param(
            yaml.safe_load("general:\ngeometry_preprocessing:\nfeature_extraction:\n"),
            id="yaml-empty-sections",
        ),
    ],
)
def test_new_shape_configs_are_not_detected_as_legacy(config):
    assert is_legacy_config(config) is False
    assert load_config(config) == config


def test_migrated_config_is_not_legacy_any_more():
    migrated = load_config(_legacy(padding=10))

    assert is_legacy_config(migrated) is False
    assert load_config(migrated) == migrated


def test_migrated_config_general_is_accepted_by_study_config():
    migrated = load_config(_legacy(padding=10))
    StudyPreprocessingConfig.from_dict(migrated)


# --------------------------------------------------------------------------
# Legacy padding migration edge cases
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("padding", "spacing", "expected"),
    [
        (10, (0.977, 0.977, 3.0), (10, 10, 3)),
        (10.0, (2.0, 2.0, 2.0), (5, 5, 5)),
        (1, (3.0, 3.0, 3.0), (0, 0, 0)),
        (0.4, (1.0, 1.0, 1.0), (0, 0, 0)),
    ],
)
def test_padding_migration_per_axis(padding, spacing, expected):
    geometry = _ct_geometry(load_config(_legacy(padding=padding, spacing=spacing)))
    voxels = geometry["padding_voxels"]
    if isinstance(voxels, int):
        voxels = (voxels,) * 3
    assert tuple(voxels) == expected
    GeometryConfig.from_dict(geometry)


def test_zero_padding_with_native_spacing_migrates_to_zero_voxels():
    """0 mm is 0 voxels on any grid: no spacing is needed to convert it."""

    geometry = _ct_geometry(load_config(_legacy(padding=0, spacing=None)))
    assert geometry.get("padding_voxels", 0) in (0, [0, 0, 0], (0, 0, 0))
    assert geometry.get("crop_to_masks", True) is True


@pytest.mark.parametrize("padding", [-1, "10", True, None, [10, 10, 10]])
def test_invalid_legacy_padding_is_rejected(padding):
    with pytest.raises(ValueError):
        load_config(_legacy(padding=padding))


def test_whole_image_without_any_spacing_migrates():
    geometry = _ct_geometry(load_config(_legacy(padding="whole_image", spacing=None)))
    assert geometry["crop_to_masks"] is False
