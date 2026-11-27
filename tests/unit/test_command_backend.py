"""Command feature backend.

The external command is the real system boundary, so it is replaced by a tiny
Python script run with ``sys.executable``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import SimpleITK as sitk

from okapy.core.models import (
    ImageMaskSet,
    MaskIdentity,
    MaskVolume,
    StudyVolumes,
    VolumeCollection,
)
from okapy.features import FeatureExtractionStep
from okapy.features.backends.command import CommandFeatureBackend
from tests.helpers.factories import image_volume

FAKE_EXTRACTOR = """
import argparse
import json
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--mode", default="ok")
parser.add_argument("--image")
parser.add_argument("--mask")
parser.add_argument("--params")
parser.add_argument("--output")
args = parser.parse_args()

preexisting = os.path.exists(args.output)

if args.mode == "fail":
    sys.exit(3)
if args.mode == "skip":
    sys.exit(0)

payload = {
    "features": {
        "value": 1.5,
        "preexisting": preexisting,
        "output": os.path.abspath(args.output),
        "mask": args.mask,
    },
    "metadata": {"mode": args.mode},
}
with open(args.output, "w") as f:
    json.dump(payload, f)
"""


@pytest.fixture
def script(tmp_path):
    path = tmp_path / "fake_extractor.py"
    path.write_text(FAKE_EXTRACTOR)
    return path


@pytest.fixture
def params(tmp_path):
    path = tmp_path / "params.yaml"
    path.write_text("setting: {}\n")
    return path


def _backend(script, params, *, mode="ok", name="ext"):
    return CommandFeatureBackend(
        command=[sys.executable, str(script), "--mode", mode],
        params_path=params,
        name=name,
        timeout_seconds=60,
    )


def _sitk():
    image = sitk.Image(4, 4, 4, sitk.sitkUInt8)
    image[1, 1, 1] = 1
    return image


def _image(series_instance_uid="1.2.3.4"):
    return image_volume(
        _sitk(), path="image.nii.gz", series_instance_uid=series_instance_uid
    )


def _mask(
    *,
    label="GTV",
    path="mask.nii.gz",
    reference_series_instance_uid="1.2.3.4",
    metadata=None,
):
    return MaskVolume.from_sitk(
        path=Path(path),
        image=_sitk(),
        identity=MaskIdentity(
            label=label,
            modality="RTSTRUCT",
            reference_modality="CT",
            patient_id="P1",
            study_instance_uid="1.2.3",
            reference_series_instance_uid=reference_series_instance_uid,
            metadata=metadata or {},
        ),
    )


def _output_dir(feature_set) -> Path:
    return Path(feature_set.features["output"]).parent


def _assert_inside(path: Path, root: Path):
    resolved = Path(path).resolve()
    assert resolved.is_relative_to(root.resolve()), f"{resolved} escapes {root}"


# ---------------------------------------------------------------------------
# Regular behavior
# ---------------------------------------------------------------------------


def test_command_backend_returns_features_written_by_command(script, params, tmp_path):
    result = _backend(script, params).extract(
        _image(), _mask(), work_dir=tmp_path / "work"
    )

    assert result.backend_name == "ext"
    assert result.features["value"] == 1.5
    assert result.features["preexisting"] is False
    assert result.features["mask"] == "mask.nii.gz"
    assert result.metadata == {"mode": "ok"}
    _assert_inside(_output_dir(result), tmp_path / "work")


def test_command_backend_nonzero_exit_raises(script, params, tmp_path):
    with pytest.raises(RuntimeError, match="exit code 3"):
        _backend(script, params, mode="fail").extract(
            _image(), _mask(), work_dir=tmp_path / "work"
        )


def test_command_backend_missing_output_raises(script, params, tmp_path):
    with pytest.raises(RuntimeError, match="did not create"):
        _backend(script, params, mode="skip").extract(
            _image(), _mask(), work_dir=tmp_path / "work"
        )


def test_command_backend_rejects_empty_command(params):
    with pytest.raises(ValueError, match="non-empty"):
        CommandFeatureBackend(command=[], params_path=params, name="ext")


# ---------------------------------------------------------------------------
# Stale output files
# ---------------------------------------------------------------------------


def test_stale_output_is_not_read_when_command_does_not_write(script, params, tmp_path):
    work_dir = tmp_path / "work"
    first = _backend(script, params).extract(_image(), _mask(), work_dir=work_dir)
    stale_output = Path(first.features["output"])
    assert stale_output.is_file()

    with pytest.raises(RuntimeError, match="did not create"):
        _backend(script, params, mode="skip").extract(
            _image(), _mask(), work_dir=work_dir
        )

    assert not stale_output.exists()


def test_stale_output_is_removed_before_command_runs(script, params, tmp_path):
    work_dir = tmp_path / "work"
    backend = _backend(script, params)
    backend.extract(_image(), _mask(), work_dir=work_dir)

    second = backend.extract(_image(), _mask(), work_dir=work_dir)

    assert second.features["preexisting"] is False


def test_stale_output_is_removed_when_command_fails(script, params, tmp_path):
    work_dir = tmp_path / "work"
    first = _backend(script, params).extract(_image(), _mask(), work_dir=work_dir)
    stale_output = Path(first.features["output"])

    with pytest.raises(RuntimeError):
        _backend(script, params, mode="fail").extract(
            _image(), _mask(), work_dir=work_dir
        )

    assert not stale_output.exists()


# ---------------------------------------------------------------------------
# Per-pair work dirs
# ---------------------------------------------------------------------------


def test_two_masks_with_same_label_and_different_uids_get_different_dirs(
    script, params, tmp_path
):
    work_dir = tmp_path / "work"
    backend = _backend(script, params)
    mask_a = _mask(
        path="a.nii.gz", metadata={"rtstruct_series_instance_uid": "1.2.3.100"}
    )
    mask_b = _mask(
        path="b.nii.gz", metadata={"rtstruct_series_instance_uid": "1.2.3.200"}
    )

    result_a = backend.extract(_image(), mask_a, work_dir=work_dir)
    result_b = backend.extract(_image(), mask_b, work_dir=work_dir)

    assert _output_dir(result_a) != _output_dir(result_b)
    assert result_a.features["mask"] == "a.nii.gz"
    assert result_b.features["mask"] == "b.nii.gz"


def test_labels_that_sanitize_alike_get_different_dirs(script, params, tmp_path):
    work_dir = tmp_path / "work"
    backend = _backend(script, params)

    result_a = backend.extract(
        _image(), _mask(label="GTV 1", path="a.nii.gz"), work_dir=work_dir
    )
    result_b = backend.extract(
        _image(), _mask(label="GTV_1", path="b.nii.gz"), work_dir=work_dir
    )

    assert _output_dir(result_a) != _output_dir(result_b)


def test_same_mask_on_two_images_gets_different_dirs(script, params, tmp_path):
    work_dir = tmp_path / "work"
    backend = _backend(script, params)
    mask = _mask()

    result_a = backend.extract(_image("1.2.3.4"), mask, work_dir=work_dir)
    result_b = backend.extract(_image("1.2.3.5"), mask, work_dir=work_dir)

    assert _output_dir(result_a) != _output_dir(result_b)


def test_step_with_two_same_label_masks_reads_each_own_output(script, params, tmp_path):
    image = _image()
    masks = [
        _mask(path="a.nii.gz", metadata={"seg_series_instance_uid": "9.1"}),
        _mask(path="b.nii.gz", metadata={"seg_series_instance_uid": "9.2"}),
    ]
    volumes = VolumeCollection(
        studies=[
            StudyVolumes(
                study_instance_uid="1.2.3",
                image_mask_sets=[ImageMaskSet(image=image, masks=masks)],
            )
        ]
    )
    step = FeatureExtractionStep(common_backends=[_backend(script, params)])

    records = step.run(volumes, work_dir=tmp_path / "work")

    assert [r.features["mask"] for r in records] == ["a.nii.gz", "b.nii.gz"]
    assert records[0].features["output"] != records[1].features["output"]


@pytest.mark.parametrize(
    "label, series_uid, name",
    [
        ("../../x", "1.2.3.4", "ext"),
        ("GTV", "../../x", "ext"),
        ("GTV", "1.2.3.4", "../../x"),
        ("..", "..", ".."),
        ("/etc/passwd", "/tmp/abs", "/abs"),
        ("..\\..\\x", "1.2.3.4", "ext"),
    ],
)
def test_hostile_identifiers_cannot_escape_work_dir(
    script, params, tmp_path, label, series_uid, name
):
    work_dir = tmp_path / "deep" / "nested" / "work"
    backend = _backend(script, params, name=name)

    result = backend.extract(
        _image(series_uid),
        _mask(label=label, reference_series_instance_uid=series_uid),
        work_dir=work_dir,
    )

    output = Path(result.features["output"])
    _assert_inside(output, work_dir)
    assert output.resolve() != work_dir.resolve()
    assert output.parent.resolve() != work_dir.resolve()
    # Nothing was created outside the work dir.
    created = {p.name for p in (tmp_path / "deep").iterdir()}
    assert created == {"nested"}
    assert {p.name for p in (tmp_path / "deep" / "nested").iterdir()} == {"work"}


def test_extract_accepts_str_work_dir(script, params, tmp_path):
    work_dir = tmp_path / "work"

    result = _backend(script, params).extract(_image(), _mask(), work_dir=str(work_dir))

    _assert_inside(Path(result.features["output"]), work_dir)


def test_output_file_is_valid_json_features(script, params, tmp_path):
    result = _backend(script, params).extract(
        _image(), _mask(), work_dir=tmp_path / "work"
    )

    payload = json.loads(Path(result.features["output"]).read_text())
    assert payload["features"]["value"] == 1.5
