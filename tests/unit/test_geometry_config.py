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


def test_from_dict_reads_padding_voxels():
    config = GeometryConfig.from_dict({"spacing": [1.0, 1.0, 1.0], "padding_voxels": 5})

    assert config.padding_voxels == 5


def test_from_dict_rejects_padding_mm():
    with pytest.raises(ValueError, match="padding_voxels"):
        GeometryConfig.from_dict({"spacing": [1.0, 1.0, 1.0], "padding_mm": 10})


@pytest.mark.parametrize("value", [-1, 2.5, True])
def test_from_dict_rejects_invalid_padding_voxels(value):
    with pytest.raises(ValueError, match="padding_voxels"):
        GeometryConfig.from_dict({"spacing": [1.0, 1.0, 1.0], "padding_voxels": value})
