"""Hardening of the frozen ``okapy_legacy`` package: YAML loading, cleanup, naming."""

from __future__ import annotations

import shlex
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from okapy_legacy.dicomconverter.converter import ExtractorConverter
from okapy_legacy.dicomconverter.volume import ReferenceFrame, Volume
from okapy_legacy.yaml import yaml as legacy_yaml

MINIMAL_PARAMS = {"general": {}, "volume_preprocessing": {"default": None}}


def _marker_command(marker: Path) -> str:
    code = f"open({str(marker)!r}, 'w').close()"
    return f"{shlex.quote(sys.executable)} -c {shlex.quote(code)}"


def _malicious_yaml(marker: Path) -> str:
    command = _marker_command(marker).replace("'", "''")
    return (
        "general: {}\n"
        "volume_preprocessing: {default: null}\n"
        f"extra: !!python/object/apply:os.system ['{command}']\n"
    )


def _load(path: Path):
    with path.open("r") as f:
        return legacy_yaml.load(f)


# ---------------------------------------------------------------------------
# Safe YAML loading
# ---------------------------------------------------------------------------


def test_marker_command_really_creates_marker(tmp_path):
    """Sanity check: the payload would have an effect if it were executed."""

    import subprocess

    marker = tmp_path / "sanity"
    subprocess.run(_marker_command(marker), shell=True, check=True)

    assert marker.exists()


def test_legacy_yaml_loader_rejects_python_object_tags(tmp_path):
    marker = tmp_path / "pwned"
    config = tmp_path / "params.yaml"
    config.write_text(_malicious_yaml(marker))

    with pytest.raises(yaml.YAMLError):
        _load(config)

    assert not marker.exists()


def test_legacy_yaml_loader_rejects_python_name_tags(tmp_path):
    config = tmp_path / "params.yaml"
    config.write_text("value: !!python/name:os.system\n")

    with pytest.raises(yaml.YAMLError):
        _load(config)


def test_extractor_converter_from_params_rejects_python_object_tags(tmp_path):
    marker = tmp_path / "pwned"
    config = tmp_path / "params.yaml"
    config.write_text(_malicious_yaml(marker))

    with pytest.raises((yaml.YAMLError, ValueError)):
        ExtractorConverter.from_params(str(config))

    assert not marker.exists()


def test_extractor_converter_from_params_rejects_python_tags_in_includes(tmp_path):
    marker = tmp_path / "pwned"
    (tmp_path / "evil.yaml").write_text(_malicious_yaml(marker))
    config = tmp_path / "params.yaml"
    config.write_text(
        "general: {}\nvolume_preprocessing: {default: null}\n"
        "extra: !include evil.yaml\n"
    )

    with pytest.raises((yaml.YAMLError, ValueError)):
        ExtractorConverter.from_params(str(config))

    assert not marker.exists()


def test_legacy_yaml_loader_still_loads_plain_yaml(tmp_path):
    config = tmp_path / "params.yaml"
    config.write_text("general:\n  padding: 10\n  labels: [a, b]\n")

    assert _load(config) == {"general": {"padding": 10, "labels": ["a", "b"]}}


def test_include_of_file_in_config_directory_works(tmp_path):
    (tmp_path / "settings.yaml").write_text("binWidth: 20\nlabel: 1\n")
    (tmp_path / "notes.txt").write_text("hello\n")
    config = tmp_path / "params.yaml"
    config.write_text("setting: !include settings.yaml\nnotes: !include notes.txt\n")

    assert _load(config) == {
        "setting": {"binWidth": 20, "label": 1},
        "notes": "hello\n",
    }


def test_include_of_file_in_config_subdirectory_works(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "settings.yaml").write_text("binWidth: 5\n")
    config = tmp_path / "params.yaml"
    config.write_text("setting: !include sub/settings.yaml\n")

    assert _load(config) == {"setting": {"binWidth": 5}}


def test_include_outside_config_directory_is_rejected(tmp_path):
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (tmp_path / "outside.yaml").write_text("secret: 1\n")
    config = config_dir / "params.yaml"
    config.write_text("setting: !include ../outside.yaml\n")

    with pytest.raises((yaml.YAMLError, ValueError)) as excinfo:
        _load(config)

    assert not isinstance(excinfo.value, FileNotFoundError)


def test_include_of_absolute_path_outside_config_directory_is_rejected(tmp_path):
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    outside = tmp_path / "outside.yaml"
    outside.write_text("secret: 1\n")
    config = config_dir / "params.yaml"
    config.write_text(f"setting: !include {outside}\n")

    with pytest.raises((yaml.YAMLError, ValueError)):
        _load(config)


def test_from_params_with_include_in_config_directory_works(tmp_path):
    (tmp_path / "general.yaml").write_text("additional_dicom_tags: []\n")
    config = tmp_path / "params.yaml"
    config.write_text(
        "general: !include general.yaml\nvolume_preprocessing:\n  default:\n"
    )

    converter = ExtractorConverter.from_params(str(config))

    assert converter.additional_dicom_tags == []


# ---------------------------------------------------------------------------
# Error-path cleanup only removes a temporary dir the converter created
# ---------------------------------------------------------------------------


def test_error_with_user_output_folder_keeps_folder_and_contents(tmp_path):
    converter = ExtractorConverter.from_params(dict(MINIMAL_PARAMS))
    empty_input = tmp_path / "empty_input"
    empty_input.mkdir()
    output = tmp_path / "user_output"
    output.mkdir()
    precious = output / "precious.txt"
    precious.write_text("keep me")

    with pytest.raises(RuntimeError, match="No valid DICOM"):
        converter(empty_input, output_folder=output)

    assert output.is_dir()
    assert precious.read_text() == "keep me"


def test_error_with_new_user_output_folder_does_not_delete_parent(tmp_path):
    converter = ExtractorConverter.from_params(dict(MINIMAL_PARAMS))
    empty_input = tmp_path / "empty_input"
    empty_input.mkdir()
    sibling = tmp_path / "sibling.txt"
    sibling.write_text("keep me")

    with pytest.raises(RuntimeError, match="No valid DICOM"):
        converter(empty_input, output_folder=tmp_path / "new_output")

    assert sibling.read_text() == "keep me"
    assert empty_input.is_dir()


def test_error_without_output_folder_removes_its_temporary_dir(tmp_path, monkeypatch):
    import tempfile

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "tmp"))
    (tmp_path / "tmp").mkdir()
    converter = ExtractorConverter.from_params(dict(MINIMAL_PARAMS))
    empty_input = tmp_path / "empty_input"
    empty_input.mkdir()

    with pytest.raises(RuntimeError, match="No valid DICOM"):
        converter(empty_input)

    assert list((tmp_path / "tmp").iterdir()) == []


# ---------------------------------------------------------------------------
# Output naming of header-derived values
# ---------------------------------------------------------------------------


def _legacy_volume(patient_id: str, modality: str = "CT") -> Volume:
    header = SimpleNamespace(
        PatientID=patient_id,
        SeriesNumber=1,
        series_datetime="20200101 101010",
    )
    frame = ReferenceFrame(
        origin=[0.0, 0.0, 0.0],
        orientation_matrix=np.eye(3),
        voxel_spacing=[1.0, 1.0, 1.0],
        shape=(3, 3, 3),
    )
    return Volume(
        array=np.zeros((3, 3, 3), dtype=np.float32),
        reference_frame=frame,
        modality=modality,
        dicom_header=header,
    )


@pytest.mark.parametrize("naming", [0, 1, 2])
@pytest.mark.parametrize("patient_id", ["../../x", "..", "/abs/x", "..\\..\\x"])
def test_legacy_volume_path_stays_inside_output_folder(tmp_path, naming, patient_id):
    converter = ExtractorConverter.from_params(dict(MINIMAL_PARAMS))
    converter.naming = naming
    output = tmp_path / "a" / "b" / "out"
    output.mkdir(parents=True)

    path = converter.get_path(_legacy_volume(patient_id), ouput_folder=output)

    assert path.resolve().parent == output.resolve()


def test_legacy_write_with_hostile_patient_id_writes_inside_output_folder(tmp_path):
    converter = ExtractorConverter.from_params(dict(MINIMAL_PARAMS))
    output = tmp_path / "a" / "b" / "out"
    output.mkdir(parents=True)

    info = converter.write(_legacy_volume("../../x"), output_folder=output)

    written = [p.resolve() for p in tmp_path.rglob("*.nii.gz")]
    assert written == [Path(info["path"]).resolve()]
    assert written[0].parent == output.resolve()


def test_legacy_volume_name_uses_shared_safe_name(tmp_path):
    from okapy.dicom.conversion.utils import safe_name

    converter = ExtractorConverter.from_params(dict(MINIMAL_PARAMS))
    converter.naming = 0

    path = converter.get_path(_legacy_volume("a b/c:d"), ouput_folder=tmp_path)

    assert path.name.startswith(safe_name("a b/c:d") + "__")
    assert path.name == "a_b_c_d__CT.nii.gz"
