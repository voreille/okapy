"""Config loading: legacy auto-detection/migration and legacy padding migration."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from okapy.config import load_config
from okapy.preprocessing.models import GeometryConfig
from okapy.preprocessing.selector import select_section_config

PARAMETERS_DIR = Path(__file__).parents[2] / "parameters"
DEFAULTS_PYRADIOMICS_ONLY = (
    PARAMETERS_DIR / "defaults" / "defaults_pyradiomics_only.yaml"
)

PYRADIOMICS_PARAMS = {
    "imageType": {"Original": {}},
    "featureClass": {"firstorder": None},
    "setting": {"binWidth": 20, "label": 1},
}


def _is_legacy(config):
    from okapy.config import is_legacy_config

    return is_legacy_config(config)


def _legacy(*, padding=..., spacing=(1.0, 1.0, 1.0), extra_volume=None):
    general = {"submodalities": False}
    if padding is not ...:
        general["padding"] = padding
    volume_preprocessing = {
        "common": {
            "bspline_resampler": {"resampling_spacing": list(spacing), "order": 3}
        },
        "default": None,
    }
    volume_preprocessing.update(extra_volume or {})
    return {
        "general": general,
        "volume_preprocessing": volume_preprocessing,
        "feature_extraction": {"CT": {"pyradiomics": {"original": PYRADIOMICS_PARAMS}}},
    }


def _new_shape():
    return {
        "general": {"submodalities": False},
        "geometry_preprocessing": {
            "common": {"spacing": [1.0, 1.0, 1.0], "padding_voxels": 4}
        },
        "feature_extraction": {
            "result_format": "long",
            "CT": [{"type": "pet", "name": "pet"}],
        },
    }


def _effective_geometry(config, key):
    """The geometry block a ``key`` image actually gets (selector + common)."""

    return select_section_config(
        config["geometry_preprocessing"], key=key, include_common=True
    )


def _per_axis(value):
    if isinstance(value, (list, tuple)):
        return tuple(int(v) for v in value)
    return (int(value),) * 3


# ---------------------------------------------------------------------------
# is_legacy_config
# ---------------------------------------------------------------------------


def test_is_legacy_config_detects_volume_preprocessing():
    assert _is_legacy({"volume_preprocessing": {"common": {}}}) is True


def test_is_legacy_config_detects_general_padding():
    assert _is_legacy({"general": {"padding": 10}}) is True


@pytest.mark.parametrize("extractor", ["pyradiomics", "okapy"])
def test_is_legacy_config_detects_legacy_feature_extractor_blocks(extractor):
    config = {"feature_extraction": {"CT": {extractor: {"original": {}}}}}

    assert _is_legacy(config) is True


def test_is_legacy_config_detects_shipped_legacy_defaults():
    raw = yaml.safe_load(DEFAULTS_PYRADIOMICS_ONLY.read_text())

    assert _is_legacy(raw) is True


def test_is_legacy_config_is_false_for_new_shape():
    assert _is_legacy(_new_shape()) is False


def test_is_legacy_config_is_false_for_new_feature_backend_lists():
    config = {
        "general": {},
        "feature_extraction": {
            "common": [{"type": "pet"}],
            "CT": [{"type": "pyradiomics_local", "params": {}}],
        },
    }

    assert _is_legacy(config) is False


# ---------------------------------------------------------------------------
# load_config auto-migration
# ---------------------------------------------------------------------------


def _assert_migrated_shape(config):
    assert "volume_preprocessing" not in config
    assert "padding" not in (config.get("general") or {})
    assert "geometry_preprocessing" in config
    ct_backends = config["feature_extraction"]["CT"]
    assert isinstance(ct_backends, list)
    assert [b["type"] for b in ct_backends] == ["pyradiomics_local"]
    assert ct_backends[0]["params"] == PYRADIOMICS_PARAMS


def test_load_config_migrates_legacy_mapping():
    config = load_config(_legacy(padding=10))

    _assert_migrated_shape(config)


def test_load_config_migrates_legacy_yaml_file(tmp_path):
    path = tmp_path / "legacy.yaml"
    path.write_text(yaml.safe_dump(_legacy(padding=10)))

    config = load_config(path)

    _assert_migrated_shape(config)


def test_load_config_migrates_legacy_yaml_given_as_str(tmp_path):
    path = tmp_path / "legacy.yaml"
    path.write_text(yaml.safe_dump(_legacy(padding=10)))

    config = load_config(str(path))

    _assert_migrated_shape(config)


def test_load_config_migrates_shipped_pyradiomics_only_defaults():
    config = load_config(DEFAULTS_PYRADIOMICS_ONLY)

    assert "volume_preprocessing" not in config
    for modality in ("CT", "PT", "MR"):
        types = [b["type"] for b in config["feature_extraction"][modality]]
        assert types.count("pyradiomics_local") == 2
    # padding: 100 mm on a 1 mm grid -> 100 voxels per axis.
    ct = _effective_geometry(config, "CT")
    assert _per_axis(ct["padding_voxels"]) == (100, 100, 100)
    assert ct["spacing"] == [1.0, 1.0, 1.0]


def test_load_config_leaves_new_shape_unchanged():
    original = _new_shape()

    config = load_config(original)

    assert config == _new_shape()
    assert config is not original


def test_load_config_does_not_mutate_legacy_input():
    original = _legacy(padding=10)

    load_config(original)

    assert original == _legacy(padding=10)


def test_load_config_still_rejects_empty_yaml(tmp_path):
    path = tmp_path / "empty.yaml"
    path.write_text("")

    with pytest.raises(ValueError, match="empty"):
        load_config(path)


def test_load_config_still_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "missing.yaml")


# ---------------------------------------------------------------------------
# Legacy padding (mm) -> padding_voxels (per-axis voxels of the target grid)
# ---------------------------------------------------------------------------


def test_padding_on_unit_grid_is_the_same_number_of_voxels():
    config = load_config(_legacy(padding=10, spacing=(1.0, 1.0, 1.0)))

    geometry = _effective_geometry(config, "CT")
    assert _per_axis(geometry["padding_voxels"]) == (10, 10, 10)
    assert geometry.get("crop_to_masks", True) is True


def test_padding_mm_is_divided_by_target_spacing_per_axis():
    config = load_config(_legacy(padding=12, spacing=(1.0, 2.0, 4.0)))

    geometry = _effective_geometry(config, "CT")
    assert _per_axis(geometry["padding_voxels"]) == (12, 6, 3)


def test_padding_mm_is_rounded_to_nearest_voxel():
    config = load_config(_legacy(padding=10, spacing=(3.0, 0.75, 4.0)))

    geometry = _effective_geometry(config, "CT")
    # 10/3 = 3.33 -> 3; 10/0.75 = 13.33 -> 13; 10/4 = 2.5 -> round() -> 2
    assert _per_axis(geometry["padding_voxels"]) == (3, 13, round(2.5))


def test_padding_uses_the_spacing_of_each_selector():
    config = load_config(
        _legacy(
            padding=10,
            spacing=(1.0, 1.0, 1.0),
            extra_volume={
                "CT": {
                    "bspline_resampler": {
                        "resampling_spacing": [2.0, 2.0, 2.0],
                        "order": 3,
                    }
                }
            },
        )
    )

    ct = _effective_geometry(config, "CT")
    pt = _effective_geometry(config, "PT")
    assert ct["spacing"] == [2.0, 2.0, 2.0]
    assert _per_axis(ct["padding_voxels"]) == (5, 5, 5)
    assert pt["spacing"] == [1.0, 1.0, 1.0]
    assert _per_axis(pt["padding_voxels"]) == (10, 10, 10)


def test_missing_padding_defaults_to_ten_mm():
    config = load_config(_legacy(spacing=(2.0, 2.0, 2.0)))

    geometry = _effective_geometry(config, "CT")
    assert _per_axis(geometry["padding_voxels"]) == (5, 5, 5)
    assert geometry.get("crop_to_masks", True) is True


def test_whole_image_padding_disables_cropping():
    config = load_config(_legacy(padding="whole_image", spacing=(2.0, 2.0, 2.0)))

    geometry = _effective_geometry(config, "CT")
    assert geometry["crop_to_masks"] is False
    assert _per_axis(geometry.get("padding_voxels", 0)) == (0, 0, 0)


def test_whole_image_padding_with_native_spacing_is_accepted():
    config = load_config(_legacy(padding="whole_image", spacing=(1.0, 1.0, -1)))

    geometry = _effective_geometry(config, "CT")
    assert geometry["crop_to_masks"] is False


def test_padding_with_native_spacing_axis_is_rejected():
    with pytest.raises(ValueError, match="padding_voxels"):
        load_config(_legacy(padding=10, spacing=(1.0, 1.0, -1)))


def test_padding_without_any_target_spacing_is_rejected():
    legacy = {
        "general": {"padding": 10},
        "feature_extraction": {"CT": {"pyradiomics": {"original": PYRADIOMICS_PARAMS}}},
    }

    with pytest.raises(ValueError, match="padding_voxels"):
        load_config(legacy)


def test_default_padding_with_native_spacing_axis_is_rejected():
    with pytest.raises(ValueError, match="padding_voxels"):
        load_config(_legacy(spacing=(-1, 1.0, 1.0)))


def test_migrated_padding_is_accepted_by_geometry_config():
    config = load_config(_legacy(padding=12, spacing=(1.0, 2.0, 4.0)))

    geometry = GeometryConfig.from_dict(_effective_geometry(config, "CT"))

    assert geometry.spacing == (1.0, 2.0, 4.0)
    assert _per_axis(geometry.padding_voxels) == (12, 6, 3)
    assert geometry.crop_to_masks is True
