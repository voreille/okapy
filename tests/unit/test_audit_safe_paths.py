"""Audit: safe_name, output path containment, symlinks and the command work dir.

Contract: ``safe_name`` maps every char outside ``[A-Za-z0-9._-]`` to ``_``,
``.``/``..`` to ``_``, empty/None to ``unknown`` and caps the result at 128
chars; the resolved output path is always inside the output directory.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

from okapy.core.models import MaskIdentity, MaskVolume
from okapy.dicom.conversion.image import SimpleITKImageSeriesConverter
from okapy.dicom.conversion.utils import ensure_inside, make_filename, safe_name
from okapy.features.backends.command import CommandFeatureBackend
from okapy.preprocessing.models import StudyPreprocessingConfig
from okapy.preprocessing.step import StudyPreprocessingStep
from tests.helpers import factories
from tests.helpers.dicom_factories import load_series, write_image_series

#: NAME_MAX of common filesystems (ext4, APFS, NTFS).
NAME_MAX = 255

CT_UID = "1.2.840.1.101"


# --------------------------------------------------------------------------
# safe_name
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("a\x00b", "a_b"),
        ("..\\..\\windows", ".._.._windows"),
        ("C:\\Users\\x", "C__Users_x"),
        ("a/b", "a_b"),
        ("\n", "_"),
        ("é", "_"),
        ("١٢٣", "___"),  # Arabic-Indic digits are not ASCII digits
        ("...", "..."),
        (". .", "._."),
        (b"x", "b_x_"),
        (0, "0"),
    ],
)
def test_safe_name_edge_cases(value, expected):
    assert safe_name(value) == expected


@pytest.mark.parametrize("value", ["." * 300, "a" * 127 + "/", "/" * 129])
def test_safe_name_long_values_are_capped_and_never_a_dot_dir(value):
    name = safe_name(value)
    assert 0 < len(name) <= 128
    assert name not in {".", ".."}
    assert Path(name).name == name


def test_names_colliding_after_sanitizing_are_detectable():
    assert safe_name("a/b") == safe_name("a?b")


def test_make_filename_of_maximal_parts_is_a_valid_file_name():
    """Every part is capped at 128 chars; the joined file name must still fit.

    UIDs, labels and patient IDs over 64 chars are non-conformant but are
    found in the wild; the writer then fails with ENAMETOOLONG.
    """

    name = make_filename("p" * 200, "l" * 200, "CT", "1." * 40, extension="nii.gz")
    assert len(name.encode()) <= NAME_MAX


# --------------------------------------------------------------------------
# ensure_inside and symlinks
# --------------------------------------------------------------------------


def test_ensure_inside_rejects_a_symlinked_subdir_pointing_outside(tmp_path):
    root = tmp_path / "out"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    os.symlink(outside, root / "images")

    with pytest.raises(ValueError):
        ensure_inside(root / "images" / "x.nii.gz", root)


def test_ensure_inside_rejects_a_dangling_symlink_pointing_outside(tmp_path):
    root = tmp_path / "out"
    root.mkdir()
    os.symlink(tmp_path / "nowhere" / "x.nii.gz", root / "x.nii.gz")

    with pytest.raises(ValueError):
        ensure_inside(root / "x.nii.gz", root)


def test_ensure_inside_accepts_a_symlinked_root(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    os.symlink(real, link)

    assert ensure_inside(link / "x.nii.gz", link) == link / "x.nii.gz"


def test_ensure_inside_accepts_a_relative_root(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path("out").mkdir()

    assert ensure_inside(Path("out") / "x", Path("out")) == Path("out") / "x"


def test_conversion_does_not_write_through_a_symlink_to_outside(tmp_path):
    uid, paths = write_image_series(
        tmp_path / "dicom", positions=[0, 1, 2], rescale=(1.0, 0.0)
    )
    series = load_series(paths)
    out = tmp_path / "out"
    out.mkdir()
    victim = tmp_path / "victim.nii.gz"
    victim.write_text("keep me")
    filename = SimpleITKImageSeriesConverter()._make_filename(series)
    os.symlink(victim, out / filename)

    try:
        volume = SimpleITKImageSeriesConverter().convert(series, out)
    except ValueError:
        pass
    else:
        assert volume.path.resolve().is_relative_to(out.resolve())

    assert victim.read_text() == "keep me"


# --------------------------------------------------------------------------
# Preprocessing writers
# --------------------------------------------------------------------------


def _image(size=(8, 8, 8)):
    image = sitk.GetImageFromArray(np.ones(size[::-1], dtype=np.float32))
    return image


def _box_mask(image, lo, hi):
    array = np.zeros(image.GetSize()[::-1], dtype=np.uint8)
    array[lo[2] : hi[2] + 1, lo[1] : hi[1] + 1, lo[0] : hi[0] + 1] = 1
    mask = sitk.GetImageFromArray(array)
    mask.CopyInformation(image)
    return mask


def _step(**general):
    return StudyPreprocessingStep(
        study_config=StudyPreprocessingConfig.from_dict({"general": general}),
        geometry_section={"common": {"crop_to_masks": True}},
    )


def _ct_and_masks(tmp_path, labels, *, patient_id="P1"):
    image = _image()
    ct = factories.image_volume(
        image,
        path=tmp_path / "input" / "ct.nii.gz",
        series_instance_uid=CT_UID,
        patient_id=patient_id,
    )
    masks = [
        factories.mask_volume(
            _box_mask(image, (1, 1, 1), (3 + i, 3, 3)),
            label=label,
            path=tmp_path / "input" / f"m{i}.nii.gz",
            reference_series_instance_uid=CT_UID,
            patient_id=patient_id,
        )
        for i, label in enumerate(labels)
    ]
    return ct, masks


def test_preprocessing_symlinked_image_subdir_does_not_write_outside(tmp_path):
    ct, masks = _ct_and_masks(tmp_path, ["GTV"])
    out = tmp_path / "out"
    out.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    os.symlink(outside, out / "images")

    with pytest.raises(ValueError):
        _step().preprocess_study(images=[ct], masks=masks, output_dir=out)

    assert list(outside.iterdir()) == []


def test_labels_colliding_after_sanitizing_both_get_written(tmp_path):
    ct, masks = _ct_and_masks(tmp_path, ["GTV/1", "GTV?1"])

    study = _step().preprocess_study(
        images=[ct], masks=masks, output_dir=tmp_path / "out"
    )

    (image_mask_set,) = study.image_mask_sets
    paths = [mask.path for mask in image_mask_set.masks]
    assert len(set(paths)) == 2
    for mask, path in zip(image_mask_set.masks, paths, strict=True):
        on_disk = sitk.GetArrayFromImage(sitk.ReadImage(str(path)))
        np.testing.assert_array_equal(on_disk, sitk.GetArrayFromImage(mask.image))


def test_very_long_header_values_do_not_break_writing(tmp_path):
    """Non-conformant 200-char PatientID / ROI label must still be writable."""

    ct, masks = _ct_and_masks(tmp_path, ["L" * 200], patient_id="P" * 200)

    study = _step().preprocess_study(
        images=[ct], masks=masks, output_dir=tmp_path / "out"
    )

    (image_mask_set,) = study.image_mask_sets
    assert image_mask_set.image.path.exists()
    assert all(mask.path.exists() for mask in image_mask_set.masks)


# --------------------------------------------------------------------------
# Command backend work dir
# --------------------------------------------------------------------------

SCRIPT = """
import argparse, json
p = argparse.ArgumentParser()
for a in ("--image", "--mask", "--params", "--output"):
    p.add_argument(a)
a = p.parse_args()
with open(a.output, "w") as f:
    json.dump({"features": {"out": a.output}}, f)
"""


@pytest.fixture
def backend_factory(tmp_path):
    script = tmp_path / "ext.py"
    script.write_text(SCRIPT)
    params = tmp_path / "params.yaml"
    params.write_text("{}\n")

    def make(name="ext"):
        return CommandFeatureBackend(
            command=[sys.executable, str(script)], params_path=params, name=name
        )

    return make


def _pair(uid="1.2.3.4", label="GTV"):
    image = factories.image_volume(_image(), series_instance_uid=uid)
    mask = MaskVolume.from_sitk(
        path=Path("mask.nii.gz"),
        image=_box_mask(image.image, (1, 1, 1), (2, 2, 2)),
        identity=MaskIdentity(
            label=label,
            modality="RTSTRUCT",
            reference_modality="CT",
            patient_id="P1",
            study_instance_uid="1.2.3",
            reference_series_instance_uid=uid,
        ),
    )
    return image, mask


def test_symlinked_pair_dir_pointing_outside_is_refused(tmp_path, backend_factory):
    work = tmp_path / "work"
    work.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    os.symlink(outside, work / safe_name("1.2.3.4"))

    with pytest.raises(ValueError):
        backend_factory().extract(*_pair(), work_dir=work)

    assert list(outside.iterdir()) == []


@pytest.mark.parametrize(
    "names",
    [("a/b", "a?b"), ("..", "."), ("", "unknown"), ("x" * 200, "x" * 201)],
)
def test_backend_names_that_sanitize_alike_get_different_dirs(
    tmp_path, backend_factory, names
):
    work = tmp_path / "work"
    outputs = [
        Path(backend_factory(name).extract(*_pair(), work_dir=work).features["out"])
        for name in names
    ]

    assert outputs[0] != outputs[1]
    for output in outputs:
        assert output.resolve().is_relative_to(work.resolve())


@pytest.mark.parametrize(
    "label", ["\x00", "../" * 50, "a\\..\\..\\b", "-rf", "$(touch pwn)"]
)
def test_hostile_labels_stay_inside_and_are_not_interpreted(
    tmp_path, backend_factory, label, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    work = tmp_path / "work"

    result = backend_factory().extract(*_pair(label=label), work_dir=work)

    assert Path(result.features["out"]).resolve().is_relative_to(work.resolve())
    assert not (tmp_path / "pwn").exists()
