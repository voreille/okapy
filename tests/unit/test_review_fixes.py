"""Behaviours approved after the code review of the quality pass (#37).

One section per numbered point of the review-fixes contract.
"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path

import numpy as np
import pydicom
import pytest
import SimpleITK as sitk
import yaml
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence as DicomSequence

from okapy.config import load_config
from okapy.core.config_values import parse_bool
from okapy.core.models import ImageVolume, MaskIdentity, MaskVolume, SeriesIdentity
from okapy.dicom.conversion.builder import build_dicom_conversion_step
from okapy.dicom.conversion.enhanced_pet import EnhancedPETSUVConverter
from okapy.dicom.conversion.image import SimpleITKImageSeriesConverter
from okapy.dicom.conversion.rtstruct import RTStructMaskConverter
from okapy.dicom.conversion.suv import PETSUVConverter
from okapy.dicom.conversion.suv_common import parse_dicom_datetime
from okapy.features.backends.command import CommandFeatureBackend
from okapy.features.builder import build_feature_extraction_step
from okapy.pipeline import build_extraction_pipeline
from okapy.pipeline.builder import build_preprocessing_pipeline
from okapy.preprocessing.models import GeometryConfig, StudyPreprocessingConfig
from tests.helpers.dicom_factories import (
    enhanced_pet_dataset,
    load_series,
    rtstruct_dataset,
    seg_dataset,
    square_contour,
    write_dataset,
    write_image_series,
)

ROOT = Path(__file__).parents[2]
PYPROJECT = ROOT / "pyproject.toml"

MISSING_UID = "1.2.826.0.1.3680043.8.498.999999"


def _warnings(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]


# --------------------------------------------------------------------------
# 1. tqdm in the dev extra
# --------------------------------------------------------------------------


def _toml():
    try:
        import tomllib
    except ModuleNotFoundError:  # Python < 3.11
        tomllib = pytest.importorskip("tomli")
    return tomllib


def _requirement_name(requirement: str) -> str:
    return re.split(r"[\s<>=!~;\[]", requirement.strip(), maxsplit=1)[0].lower()


def test_tqdm_is_in_dev_extra():
    pyproject = _toml().loads(PYPROJECT.read_text())
    dev = pyproject["project"]["optional-dependencies"]["dev"]

    assert "tqdm" in {_requirement_name(r) for r in dev}


# --------------------------------------------------------------------------
# 2. Mixed legacy + new-shape configs are rejected
# --------------------------------------------------------------------------

_LEGACY_VOLUME_PREPROCESSING = {
    "common": {"bspline_resampler": {"resampling_spacing": [1.0, 1.0, 1.0]}}
}


@pytest.mark.parametrize(
    ("legacy_part", "legacy_key"),
    [
        # 0 mm migrates on any grid, so only the mixing itself can raise.
        pytest.param({"general": {"padding": 0}}, "padding", id="general-padding"),
        pytest.param(
            {"volume_preprocessing": _LEGACY_VOLUME_PREPROCESSING},
            "volume_preprocessing",
            id="volume-preprocessing",
        ),
    ],
)
@pytest.mark.parametrize(
    ("new_part", "new_section"),
    [
        pytest.param(
            {"geometry_preprocessing": {"common": {"spacing": [1.0, 1.0, 1.0]}}},
            "geometry_preprocessing",
            id="geometry",
        ),
        pytest.param(
            {"local_preprocessing": {"CT": {"standardizer": {}}}},
            "local_preprocessing",
            id="local",
        ),
    ],
)
def test_mixed_legacy_and_new_shape_config_is_rejected(
    legacy_part, legacy_key, new_part, new_section
):
    config = {**legacy_part, **new_part}

    with pytest.raises(ValueError) as excinfo:
        load_config(config)

    message = str(excinfo.value)
    assert legacy_key in message
    assert new_section in message


def test_legacy_config_with_new_shape_feature_extraction_is_rejected():
    config = {
        "volume_preprocessing": _LEGACY_VOLUME_PREPROCESSING,
        "feature_extraction": {"PT": [{"type": "pet", "name": "pet"}]},
    }

    with pytest.raises(ValueError) as excinfo:
        load_config(config)

    message = str(excinfo.value)
    assert "volume_preprocessing" in message
    assert "feature_extraction" in message


def test_pure_legacy_config_still_migrates():
    config = {
        "general": {"padding": 10},
        "volume_preprocessing": _LEGACY_VOLUME_PREPROCESSING,
        "feature_extraction": {"CT": {"pyradiomics": {"original": {"a": 1}}}},
    }

    migrated = load_config(config)

    assert "volume_preprocessing" not in migrated
    assert "padding" not in migrated["general"]
    assert migrated["geometry_preprocessing"]["common"]["spacing"] == [1.0, 1.0, 1.0]
    assert migrated["geometry_preprocessing"]["common"]["padding_voxels"] == 10
    (backend,) = migrated["feature_extraction"]["CT"]
    assert backend["type"] == "pyradiomics_local"
    assert backend["params"] == {"a": 1}


def test_pure_new_shape_config_is_returned_unchanged():
    config = {
        "general": {"write_outputs": False},
        "geometry_preprocessing": {"common": {"spacing": [1.0, 1.0, 1.0]}},
        "local_preprocessing": {"CT": {"standardizer": {}}},
        "feature_extraction": {"PT": [{"type": "pet", "name": "pet"}]},
    }

    assert load_config(config) == config


# --------------------------------------------------------------------------
# 3. Reference resolution with the reference UID check disabled
# --------------------------------------------------------------------------


def _ct(directory, series_instance_uid=None):
    return write_image_series(
        directory,
        positions=[0.0, 1.0, 2.0, 3.0],
        values=[1, 2, 3, 4],
        rows=4,
        columns=5,
        rescale=(1.0, 0.0),
        series_instance_uid=series_instance_uid,
    )


def _rtstruct_on_missing_series(directory):
    ds = rtstruct_dataset(
        rois=[(1, "GTV", [square_contour(0.5, 0.5, 3.5, 2.5, 1.0)])],
        referenced_series_uid=MISSING_UID,
    )
    write_dataset(ds, directory / "rtstruct.dcm")
    return ds.SeriesInstanceUID


def _seg_on_unconverted_series(tmp_path, input_dir):
    """SEG built on a CT that is NOT part of the converted input."""

    _, source_paths = _ct(tmp_path / "not_input" / "ct")
    sources = [pydicom.dcmread(str(p)) for p in source_paths]
    mask_zyx = np.zeros((len(sources), 4, 5), dtype=np.uint8)
    mask_zyx[1, 1:3, 2:4] = 1
    seg = seg_dataset(source_images=sources, mask_zyx=mask_zyx, label="Tumor")
    write_dataset(seg, input_dir / "seg" / "seg.dcm")
    return seg.SeriesInstanceUID


def _make_mask(kind, tmp_path, input_dir):
    if kind == "rtstruct":
        return _rtstruct_on_missing_series(input_dir / "rt")
    pytest.importorskip("highdicom")
    return _seg_on_unconverted_series(tmp_path, input_dir)


_CHECK_KEY = {
    "rtstruct": "check_rtstruct_reference_uid",
    "seg": "check_seg_reference_uid",
}


@pytest.mark.parametrize("kind", ["rtstruct", "seg"])
def test_unchecked_mask_with_unconverted_reference_uses_the_only_image(
    tmp_path, caplog, kind
):
    input_dir = tmp_path / "in"
    image_uid, _ = _ct(input_dir / "ct")
    mask_uid = _make_mask(kind, tmp_path, input_dir)
    step = build_dicom_conversion_step({"conversion": {_CHECK_KEY[kind]: False}})

    with caplog.at_level(logging.WARNING):
        collection = step.run(input_dir, output_dir=tmp_path / "out")

    masks = collection.masks
    assert len(masks) == 1
    assert masks[0].reference_series_instance_uid == image_uid
    assert sitk.GetArrayFromImage(masks[0].image).sum() > 0
    assert any(mask_uid in m or image_uid in m for m in _warnings(caplog))


@pytest.mark.parametrize("kind", ["rtstruct", "seg"])
def test_unchecked_mask_with_unconverted_reference_and_two_images_raises(
    tmp_path, kind
):
    input_dir = tmp_path / "in"
    _ct(input_dir / "ct1")
    _ct(input_dir / "ct2")
    mask_uid = _make_mask(kind, tmp_path, input_dir)
    step = build_dicom_conversion_step({"conversion": {_CHECK_KEY[kind]: False}})

    with pytest.raises(ValueError, match=re.escape(mask_uid)):
        step.run(input_dir, output_dir=tmp_path / "out")


@pytest.mark.parametrize("kind", ["rtstruct", "seg"])
def test_checked_mask_with_unconverted_reference_still_raises(tmp_path, kind):
    input_dir = tmp_path / "in"
    _ct(input_dir / "ct")
    mask_uid = _make_mask(kind, tmp_path, input_dir)
    step = build_dicom_conversion_step({})

    with pytest.raises(ValueError, match=re.escape(mask_uid)):
        step.run(input_dir, output_dir=tmp_path / "out")


# --------------------------------------------------------------------------
# 4. Command backend with in-memory volumes (path is None)
# --------------------------------------------------------------------------

PATH_REPORTER = """
import argparse
import json
import os

import SimpleITK as sitk

parser = argparse.ArgumentParser()
parser.add_argument("--image")
parser.add_argument("--mask")
parser.add_argument("--params")
parser.add_argument("--output")
args = parser.parse_args()

features = {}
for role in ("image", "mask"):
    value = getattr(args, role)
    exists = os.path.isfile(value)
    features[role] = os.path.abspath(value)
    features[role + "_exists"] = exists
    if exists:
        array = sitk.GetArrayFromImage(sitk.ReadImage(value))
        features[role + "_sum"] = float(array.sum())
        features[role + "_shape"] = list(array.shape)

with open(args.output, "w") as f:
    json.dump({"features": features}, f)
"""


def _array_image(values):
    image = sitk.GetImageFromArray(np.asarray(values, dtype=np.float32))
    image.SetSpacing((1.0, 1.0, 2.0))
    return image


def _mask_array():
    mask = np.zeros((3, 4, 5), dtype=np.uint8)
    mask[1, 1:3, 1:4] = 1
    return mask


def _image_volume(path, image):
    return ImageVolume.from_sitk(
        path=path,
        image=image,
        identity=SeriesIdentity(
            modality="CT",
            patient_id="P1",
            study_instance_uid="1.2.3",
            series_instance_uid="1.2.3.4",
        ),
    )


def _mask_volume(path, image):
    return MaskVolume.from_sitk(
        path=path,
        image=image,
        identity=MaskIdentity(
            label="GTV",
            modality="RTSTRUCT",
            reference_modality="CT",
            patient_id="P1",
            study_instance_uid="1.2.3",
            reference_series_instance_uid="1.2.3.4",
        ),
    )


@pytest.mark.parametrize(
    ("image_in_memory", "mask_in_memory"),
    [(True, True), (True, False), (False, True)],
    ids=["both-in-memory", "image-in-memory", "mask-in-memory"],
)
def test_command_backend_writes_in_memory_volumes_for_the_command(
    tmp_path, image_in_memory, mask_in_memory
):
    script = tmp_path / "reporter.py"
    script.write_text(PATH_REPORTER)
    params = tmp_path / "params.yaml"
    params.write_text("setting: {}\n")

    image_values = np.arange(60, dtype=np.float32).reshape(3, 4, 5)
    image_sitk = _array_image(image_values)
    mask_sitk = sitk.GetImageFromArray(_mask_array())
    mask_sitk.CopyInformation(image_sitk)

    image_path = None
    if not image_in_memory:
        image_path = tmp_path / "on_disk_image.nii.gz"
        sitk.WriteImage(image_sitk, str(image_path))
    mask_path = None
    if not mask_in_memory:
        mask_path = tmp_path / "on_disk_mask.nii.gz"
        sitk.WriteImage(mask_sitk, str(mask_path))

    work_dir = tmp_path / "work"
    backend = CommandFeatureBackend(
        command=[sys.executable, str(script)],
        params_path=params,
        name="ext",
        timeout_seconds=60,
    )
    result = backend.extract(
        _image_volume(image_path, image_sitk),
        _mask_volume(mask_path, mask_sitk),
        work_dir=work_dir,
    )
    features = result.features

    for role in ("image", "mask"):
        assert Path(features[role]).name != "None"
        assert features[f"{role}_exists"] is True
        assert features[f"{role}_shape"] == [3, 4, 5]
    assert features["image_sum"] == pytest.approx(float(image_values.sum()))
    assert features["mask_sum"] == pytest.approx(float(_mask_array().sum()))

    for role, in_memory in (("image", image_in_memory), ("mask", mask_in_memory)):
        passed = Path(features[role])
        if in_memory:
            assert passed.resolve().is_relative_to(work_dir.resolve())
            assert passed.name.endswith((".nii", ".nii.gz"))
    if not image_in_memory:
        assert Path(features["image"]) == image_path.resolve()
    if not mask_in_memory:
        assert Path(features["mask"]) == mask_path.resolve()


# --------------------------------------------------------------------------
# 5. general.result_format in a new-shape config
# --------------------------------------------------------------------------


def test_new_shape_general_result_format_is_rejected():
    config = {
        "general": {"result_format": "wide"},
        "geometry_preprocessing": {"common": {"spacing": [1.0, 1.0, 1.0]}},
    }

    with pytest.raises(ValueError, match=re.escape("feature_extraction.result_format")):
        build_extraction_pipeline(config)


def test_legacy_general_result_format_still_migrates():
    config = {
        "general": {"padding": 10, "result_format": "wide"},
        "volume_preprocessing": _LEGACY_VOLUME_PREPROCESSING,
    }

    migrated = load_config(config)

    assert migrated["feature_extraction"]["result_format"] == "wide"
    assert "result_format" not in migrated["general"]
    pipeline = build_extraction_pipeline(config)
    assert pipeline is not None


# --------------------------------------------------------------------------
# 6. Legacy general.patch_size / general.study_processor
# --------------------------------------------------------------------------


def _legacy_with_general(**general):
    return {
        "general": {"padding": 10, **general},
        "volume_preprocessing": _LEGACY_VOLUME_PREPROCESSING,
    }


def test_legacy_patch_size_is_dropped_with_a_warning(caplog):
    config = _legacy_with_general(patch_size=[64, 64, 64])

    with caplog.at_level(logging.WARNING):
        migrated = load_config(config)

    assert "patch_size" not in migrated["general"]
    assert any("patch_size" in m for m in _warnings(caplog))
    build_preprocessing_pipeline(config)


def test_legacy_studyprocessor_deep_is_rejected():
    config = _legacy_with_general(study_processor="studyprocessor_deep")

    with pytest.raises(ValueError, match="studyprocessor_deep") as excinfo:
        load_config(config)

    assert "not supported" in str(excinfo.value).lower()


def test_legacy_standard_studyprocessor_is_dropped_and_config_builds():
    config = _legacy_with_general(study_processor="studyprocessor")

    migrated = load_config(config)

    assert "study_processor" not in migrated["general"]
    build_preprocessing_pipeline(config)


# --------------------------------------------------------------------------
# 7. SUV: acquisition vs series datetime compared at whole seconds
# --------------------------------------------------------------------------

F18_HALF_LIFE_S = 6588.0
DOSE_BQ = 370e6
WEIGHT_KG = 70.0
STORED_VALUES = [100, 200]


def _suv_with_acquisition_datetime(tmp_path, acquisition_datetime):
    rph = Dataset()
    rph.RadionuclideTotalDose = DOSE_BQ
    rph.RadionuclideHalfLife = F18_HALF_LIFE_S
    rph.RadiopharmaceuticalStartDateTime = "20200101110000"
    extra = {
        "Units": "BQML",
        "SUVType": "BW",
        "DecayCorrection": "START",
        "PatientWeight": WEIGHT_KG,
        "Manufacturer": "SIEMENS",
        "SeriesDate": "20200101",
        "SeriesTime": "120000",
        "AcquisitionDate": "20200101",
        "AcquisitionTime": "120000",
        "AcquisitionDateTime": acquisition_datetime,
        "ActualFrameDuration": 120000,
        # Far from 0, so the fallback reference time would differ by minutes.
        "FrameReferenceTime": 600000,
        "RadiopharmaceuticalInformationSequence": DicomSequence([rph]),
    }
    _, paths = write_image_series(
        tmp_path / "dicom",
        modality="PT",
        positions=[0, 3],
        values=STORED_VALUES,
        rescale=(1.0, 0.0),
        extra=extra,
    )
    volume = PETSUVConverter().convert(load_series(paths), tmp_path / "out")
    return sitk.GetArrayFromImage(volume.image)[:, 0, 0]


def _expected_suvbw(decay_s):
    corrected = DOSE_BQ * 2 ** (-decay_s / F18_HALF_LIFE_S)
    return np.asarray(STORED_VALUES, dtype=float) * WEIGHT_KG * 1000.0 / corrected


def test_fractional_acquisition_datetime_equals_series_datetime_at_whole_seconds(
    tmp_path,
):
    whole = _suv_with_acquisition_datetime(tmp_path / "whole", "20200101120000")
    fractional = _suv_with_acquisition_datetime(
        tmp_path / "fractional", "20200101120000.40"
    )

    # The whole-second series uses the AcquisitionDateTime itself (1 h uptake).
    np.testing.assert_allclose(whole, _expected_suvbw(3600.0), rtol=1e-5)
    # 0.4 s changes the SUV by ~4e-5; the frame-reference fallback by ~6 %.
    np.testing.assert_allclose(fractional, whole, rtol=1e-3)


def test_dicom_datetime_parser_keeps_fractional_seconds():
    assert parse_dicom_datetime("20200101120000.40").microsecond == 400000


# --------------------------------------------------------------------------
# 8. Slice spacing reference with one missing slice among three
# --------------------------------------------------------------------------


def _convert(tmp_path, positions, values):
    _, paths = write_image_series(
        tmp_path / "dicom", positions=positions, values=values, rescale=(1.0, 0.0)
    )
    return SimpleITKImageSeriesConverter().convert(load_series(paths), tmp_path / "out")


def _slice_values(volume):
    return sitk.GetArrayFromImage(volume.image)[:, 0, 0].tolist()


def test_three_slices_with_one_missing_are_interpolated_to_four(tmp_path):
    volume = _convert(tmp_path, [0, 1, 3], [10, 20, 40])

    assert volume.image.GetSize()[2] == 4
    np.testing.assert_allclose(volume.image.GetSpacing()[2], 1.0)
    np.testing.assert_allclose(volume.image.GetOrigin()[2], 0.0)
    assert _slice_values(volume) == [10.0, 20.0, 30.0, 40.0]


def test_three_slices_with_first_gap_missing_are_interpolated_to_four(tmp_path):
    volume = _convert(tmp_path, [0, 2, 3], [10, 30, 40])

    assert volume.image.GetSize()[2] == 4
    np.testing.assert_allclose(volume.image.GetSpacing()[2], 1.0)
    assert _slice_values(volume) == [10.0, 20.0, 30.0, 40.0]


def test_six_slices_with_one_missing_still_interpolate(tmp_path):
    volume = _convert(tmp_path, [0, 1, 2, 3, 5, 6], [10, 20, 30, 40, 60, 70])

    assert volume.image.GetSize()[2] == 7
    np.testing.assert_allclose(volume.image.GetSpacing()[2], 1.0)
    assert _slice_values(volume) == [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0]


@pytest.mark.parametrize("reverse", [False, True], ids=["ordered", "reversed"])
def test_unique_jittered_gaps_still_convert_regardless_of_order(tmp_path, reverse):
    gaps = [2.515, 2.485, 2.505, 2.495, 2.51, 2.49]
    if reverse:
        gaps = gaps[::-1]
    positions = np.concatenate([[0.0], np.cumsum(gaps)]).tolist()

    volume = _convert(tmp_path, positions, list(range(7)))

    assert volume.image.GetSize()[2] == 7
    np.testing.assert_allclose(volume.image.GetSpacing()[2], 2.5, atol=1e-2)


# --------------------------------------------------------------------------
# 9. Command backend: command given as a string
# --------------------------------------------------------------------------


@pytest.mark.parametrize("section", ["CT", "common"])
def test_command_backend_string_command_is_rejected(tmp_path, section):
    config = {
        "feature_extraction": {
            section: [
                {
                    "type": "command",
                    "name": "ext",
                    "command": "python x.py",
                    "params": str(tmp_path / "params.yaml"),
                }
            ]
        }
    }

    with pytest.raises(ValueError) as excinfo:
        build_feature_extraction_step(config)

    message = str(excinfo.value)
    assert "command" in message
    assert "list" in message.lower()


def test_command_backend_list_command_still_builds(tmp_path):
    config = {
        "feature_extraction": {
            "CT": [
                {
                    "type": "command",
                    "name": "ext",
                    "command": ["python", "x.py"],
                    "params": str(tmp_path / "params.yaml"),
                }
            ]
        }
    }

    step = build_feature_extraction_step(config)

    (backend,) = step.backends_by_selector["CT"]
    assert backend.command == ["python", "x.py"]


# --------------------------------------------------------------------------
# 10. parse_bool with integers
# --------------------------------------------------------------------------


@pytest.mark.parametrize(("value", "expected"), [(1, True), (0, False)])
def test_parse_bool_accepts_integer_one_and_zero(value, expected):
    assert parse_bool(value, key="flag") is expected


@pytest.mark.parametrize("value", [2, -1, 1.0, 0.0, None])
def test_parse_bool_rejects_other_numbers_and_none(value):
    with pytest.raises(ValueError, match="flag"):
        parse_bool(value, key="flag")


@pytest.mark.parametrize("value", [True, False])
def test_parse_bool_passes_real_booleans(value):
    assert parse_bool(value, key="flag") is value


def test_yaml_crop_to_masks_zero_parses_as_false():
    config = yaml.safe_load("crop_to_masks: 0\n")

    assert GeometryConfig.from_dict(config).crop_to_masks is False


def test_yaml_write_outputs_one_parses_as_true():
    config = yaml.safe_load("general:\n  write_outputs: 1\n")

    assert StudyPreprocessingConfig.from_dict(config).write_outputs is True


# --------------------------------------------------------------------------
# 11. Study options at the top level instead of under general
# --------------------------------------------------------------------------

_STUDY_OPTIONS = {
    "combine_segmentation": True,
    "skip_images_without_masks": True,
    "write_outputs": False,
    "image_output_subdir": "imgs",
    "mask_output_subdir": "msks",
}


@pytest.mark.parametrize(("key", "value"), list(_STUDY_OPTIONS.items()))
def test_top_level_study_option_is_rejected(key, value):
    with pytest.raises(ValueError, match="general") as excinfo:
        StudyPreprocessingConfig.from_dict({key: value})

    assert key in str(excinfo.value)


def test_study_options_under_general_still_parse():
    config = StudyPreprocessingConfig.from_dict({"general": dict(_STUDY_OPTIONS)})

    assert config.combine_segmentation is True
    assert config.skip_images_without_masks is True
    assert config.write_outputs is False
    assert config.image_output_subdir == "imgs"
    assert config.mask_output_subdir == "msks"


# --------------------------------------------------------------------------
# 12. Legacy migration with the implicit 10 mm padding
# --------------------------------------------------------------------------

_CT_ONLY_SPACING = {
    "CT": {"bspline_resampler": {"resampling_spacing": [2.0, 2.0, 5.0]}}
}


def test_implicit_padding_without_common_spacing_migrates_with_a_warning(caplog):
    config = {"volume_preprocessing": _CT_ONLY_SPACING}

    with caplog.at_level(logging.WARNING):
        migrated = load_config(config)

    geometry = migrated["geometry_preprocessing"]
    assert geometry["common"].get("padding_voxels", 0) == 0
    assert geometry["CT"]["padding_voxels"] == [5, 5, 2]
    assert any("padding" in m for m in _warnings(caplog))
    build_preprocessing_pipeline(config)


def test_implicit_padding_with_common_spacing_still_converts_to_voxels():
    config = {
        "volume_preprocessing": {
            "common": {"bspline_resampler": {"resampling_spacing": [1.0, 1.0, 2.0]}}
        }
    }

    migrated = load_config(config)

    assert migrated["geometry_preprocessing"]["common"]["padding_voxels"] == [
        10,
        10,
        5,
    ]


def test_explicit_padding_without_common_spacing_still_raises():
    config = {"general": {"padding": 10}, "volume_preprocessing": _CT_ONLY_SPACING}

    with pytest.raises(ValueError, match="padding"):
        load_config(config)


def test_explicit_padding_error_says_the_spacing_is_missing():
    config = {"general": {"padding": 10}, "volume_preprocessing": _CT_ONLY_SPACING}

    with pytest.raises(ValueError) as excinfo:
        load_config(config)

    message = str(excinfo.value)
    assert "missing" in message.lower()
    assert "native" not in message.lower()


# --------------------------------------------------------------------------
# 13. Enhanced PET: PatientSize-in-cm warning once per image
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "unit_code", ["g/ml{SUVlbm}", "cm2/ml{SUVbsa}"], ids=["gml-lbm", "cm2ml-bsa"]
)
def test_enhanced_pet_patient_size_warning_is_logged_once_per_image(
    tmp_path, caplog, unit_code
):
    ds = enhanced_pet_dataset(
        frame_positions=[(0.0, 0.0, 0.0), (0.0, 0.0, 3.0), (0.0, 0.0, 6.0)],
        frame_values=[5, 6, 7],
    )
    mapping = ds.SharedFunctionalGroupsSequence[0].RealWorldValueMappingSequence[0]
    mapping.MeasurementUnitsCodeSequence[0].CodeValue = unit_code
    ds.PatientSize = 175.0  # centimetres, should be metres
    ds.PatientSex = "M"
    path = write_dataset(ds, tmp_path / "dicom" / "enhanced_pet.dcm")

    with caplog.at_level(logging.WARNING):
        EnhancedPETSUVConverter().convert(load_series([path]), tmp_path / "out")

    size_warnings = [m for m in _warnings(caplog) if "PatientSize" in m]
    assert len(size_warnings) == 1


# --------------------------------------------------------------------------
# 14. Backwards-compatible aliases
# --------------------------------------------------------------------------


def test_compute_suvpeak_is_importable_from_native_pet():
    from okapy.features.backends.pet import compute_suvpeak as backend_suvpeak
    from okapy.features.native.pet import compute_suvpeak

    assert compute_suvpeak is backend_suvpeak


def test_rtstruct_check_reference_setter_is_deprecated_alias():
    converter = RTStructMaskConverter()

    with pytest.warns(DeprecationWarning):
        converter.check_reference = False

    assert converter.check_reference_uid is False
