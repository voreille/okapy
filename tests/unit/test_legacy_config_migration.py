import pytest

from okapy.config.legacy import migrate_legacy_preprocessing_config


def _legacy_config(mask_resampler):
    return {
        "general": {"padding": 10},
        "volume_preprocessing": {
            "common": {
                "bspline_resampler": {
                    "resampling_spacing": [1, 1, 1],
                    "order": 3,
                }
            }
        },
        "mask_preprocessing": {
            "default": {"binary_bspline_resampler": mask_resampler}
        },
    }


def test_linear_mask_resampler_migrates_spacing_and_cval_only():
    migrated = migrate_legacy_preprocessing_config(
        _legacy_config(
            {
                "order": 1,
                "threshold": 0.5,
                "resampling_spacing": [2, 2, 2],
                "cval": 0,
            }
        )
    )

    assert migrated["geometry_preprocessing"]["default"] == {
        "spacing": [2, 2, 2],
        "default_mask_value": 0,
    }
    assert migrated["mask_preprocessing"] == {}

    # The image path still migrates order 3 to B-spline.
    common = migrated["geometry_preprocessing"]["common"]
    assert common["image_interpolator"] == "bspline"


def test_order_one_without_threshold_is_accepted_and_dropped():
    migrated = migrate_legacy_preprocessing_config(_legacy_config({"order": 1}))

    assert "default" not in migrated["geometry_preprocessing"]
    assert migrated["mask_preprocessing"] == {}


@pytest.mark.parametrize(
    "mask_resampler",
    [
        {"order": 0},
        {"order": 3},
        {"order": 1, "threshold": 0.7},
    ],
)
def test_other_mask_resampler_settings_are_rejected(mask_resampler):
    with pytest.raises(ValueError, match="binary_bspline_resampler"):
        migrate_legacy_preprocessing_config(_legacy_config(mask_resampler))


def test_config_without_mask_preprocessing_migrates():
    config = _legacy_config({"order": 1})
    del config["mask_preprocessing"]

    migrated = migrate_legacy_preprocessing_config(config)

    assert migrated["mask_preprocessing"] == {}


def test_other_mask_processors_pass_through():
    config = _legacy_config({"order": 1})
    config["mask_preprocessing"]["default"]["cast_mask"] = {"pixel_type": "uint8"}

    migrated = migrate_legacy_preprocessing_config(config)

    assert migrated["mask_preprocessing"]["default"] == {
        "cast_mask": {"pixel_type": "uint8"}
    }
