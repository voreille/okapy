import pytest

from okapy.preprocessing.models import GeometryConfig


def test_from_dict_builds_without_mask_options():
    config = GeometryConfig.from_dict({"spacing": [1.0, 1.0, 1.0]})

    assert config.spacing == (1.0, 1.0, 1.0)
    assert not hasattr(config, "mask_interpolator")
    assert not hasattr(config, "mask_threshold")


@pytest.mark.parametrize(
    "key, value",
    [
        ("mask_interpolator", "nearest"),
        ("mask_interpolator", "linear"),
        ("mask_threshold", 0.5),
    ],
)
def test_from_dict_rejects_removed_mask_keys(key, value):
    with pytest.raises(ValueError, match=key):
        GeometryConfig.from_dict({"spacing": [1.0, 1.0, 1.0], key: value})
