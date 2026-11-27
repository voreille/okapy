"""Shared ``safe_name`` and output path containment for header-derived names."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

from okapy.dicom.conversion import utils as conversion_utils
from okapy.preprocessing import io as preprocessing_io

SAFE_NAMES = [conversion_utils.safe_name, preprocessing_io.safe_name]
SAFE_IDS = ["conversion.utils", "preprocessing.io"]
ALLOWED = re.compile(r"[A-Za-z0-9._-]*")


@pytest.fixture(params=SAFE_NAMES, ids=SAFE_IDS)
def safe_name(request):
    return request.param


def test_preprocessing_io_uses_the_shared_safe_name():
    shared = conversion_utils.safe_name
    local = preprocessing_io.safe_name
    if local is shared:
        return
    for value in [None, "", ".", "..", "../etc/passwd", "a b/c:d", "é", "x" * 500]:
        assert local(value) == shared(value)


@pytest.mark.parametrize("value", [None, ""])
def test_missing_values_become_unknown(safe_name, value):
    assert safe_name(value) == "unknown"


@pytest.mark.parametrize("value", [".", ".."])
def test_dot_names_are_replaced(safe_name, value):
    assert safe_name(value) == "_"


@pytest.mark.parametrize(
    "value, expected",
    [
        ("../etc/passwd", ".._etc_passwd"),
        ("a b/c:d", "a_b_c_d"),
        ("é", "_"),
        ("a\\b", "a_b"),
        ("/abs", "_abs"),
        ("a\x00b", "a_b"),
        ("tab\there", "tab_here"),
        ("P1*?<>|\"'", "P1_______"),
    ],
)
def test_disallowed_characters_become_underscores(safe_name, value, expected):
    assert safe_name(value) == expected


@pytest.mark.parametrize(
    "value",
    ["P1", "1.2.840.10008.5.1", "GTV-1_left", "A.b-C_9"],
)
def test_allowed_names_are_unchanged(safe_name, value):
    assert safe_name(value) == value


def test_non_string_values_are_stringified(safe_name):
    assert safe_name(12) == "12"
    assert safe_name(1.5) == "1.5"


def test_result_is_at_most_128_characters(safe_name):
    result = safe_name("x" * 500)

    assert 0 < len(result) <= 128
    assert set(result) == {"x"}


@pytest.mark.parametrize(
    "value",
    ["../../x", "..", ".", "a/../../b", "é/..", "\\..\\..", "x" * 300 + "/../"],
)
def test_result_is_a_single_safe_path_component(safe_name, value, tmp_path):
    result = safe_name(value)

    assert ALLOWED.fullmatch(result)
    assert result not in {"", ".", ".."}
    assert len(result) <= 128
    assert Path(result).name == result
    assert (tmp_path / result).resolve().parent == tmp_path.resolve()


# ---------------------------------------------------------------------------
# okapy.preprocessing.io output names
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("patient_id", ["..", "../x", "../../x", "/etc/passwd", "."])
def test_image_output_name_stays_in_output_dir(patient_id, tmp_path):
    name = preprocessing_io.image_output_name(
        patient_id=patient_id, modality_key="CT", series_instance_uid="1.2.3.4"
    )

    assert Path(name).name == name
    assert (tmp_path / name).resolve().parent == tmp_path.resolve()


@pytest.mark.parametrize(
    "patient_id, label",
    [("..", "GTV"), ("../x", "GTV"), ("P1", "../../x"), ("P1", ".."), (".", "/a")],
)
def test_mask_output_name_stays_in_output_dir(patient_id, label, tmp_path):
    name = preprocessing_io.mask_output_name(
        patient_id=patient_id,
        label=label,
        target_modality_key="CT",
        target_series_instance_uid="1.2.3.4",
        reference_series_instance_uid="1.2.3.5",
    )

    assert Path(name).name == name
    assert (tmp_path / name).resolve().parent == tmp_path.resolve()


def test_hostile_modality_key_stays_in_output_dir(tmp_path):
    name = preprocessing_io.image_output_name(
        patient_id="P1", modality_key="../../CT", series_instance_uid="1.2.3.4"
    )

    assert Path(name).name == name


# ---------------------------------------------------------------------------
# End-to-end: conversion + preprocessing writers with hostile headers
# ---------------------------------------------------------------------------


def _write_ct_series(directory: Path, *, patient_id: str, study_uid: str):
    directory.mkdir(parents=True, exist_ok=True)
    series_uid = generate_uid()
    frame_uid = generate_uid()
    for index in range(3):
        sop_uid = generate_uid()
        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = CTImageStorage
        meta.MediaStorageSOPInstanceUID = sop_uid
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        path = directory / f"ct{index}.dcm"
        ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
        ds.SOPClassUID = CTImageStorage
        ds.SOPInstanceUID = sop_uid
        ds.Modality = "CT"
        ds.PatientID = patient_id
        ds.StudyInstanceUID = study_uid
        ds.SeriesInstanceUID = series_uid
        ds.FrameOfReferenceUID = frame_uid
        ds.ImagePositionPatient = [0.0, 0.0, float(index)]
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        ds.PixelSpacing = [1.0, 1.0]
        ds.SliceThickness = 1.0
        ds.Rows = 4
        ds.Columns = 4
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 1
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.RescaleSlope = 1
        ds.RescaleIntercept = 0
        ds.InstanceNumber = index + 1
        ds.PixelData = np.zeros((4, 4), dtype=np.int16).tobytes()
        ds.save_as(str(path), enforce_file_format=True)


def _all_files(root: Path) -> set[Path]:
    return {p.resolve() for p in root.rglob("*") if p.is_file()}


@pytest.mark.parametrize(
    "patient_id, study_uid",
    [
        ("..", "1.2.826.0.1.3680043.8.498.1"),
        ("../x", "1.2.826.0.1.3680043.8.498.2"),
        ("../../../x", "1.2.826.0.1.3680043.8.498.3"),
        ("P1", ".."),
        ("P1", "../../escaped"),
    ],
)
def test_written_files_stay_inside_work_dir(tmp_path, patient_id, study_uid):
    try:
        from okapy.pipeline import build_preprocessing_pipeline
    except ImportError:  # not exported yet; exercise the writers regardless
        from okapy.pipeline.builder import build_preprocessing_pipeline

    sandbox = tmp_path / "a" / "b" / "c"
    input_dir = sandbox / "dicom"
    work_dir = sandbox / "work"
    _write_ct_series(input_dir, patient_id=patient_id, study_uid=study_uid)
    before = _all_files(tmp_path)
    pipeline = build_preprocessing_pipeline(
        {
            "general": {},
            "geometry_preprocessing": {
                "common": {"spacing": [1.0, 1.0, 1.0], "crop_to_masks": False}
            },
        }
    )

    result = pipeline.run(input_dir, work_dir=work_dir)

    new_files = _all_files(tmp_path) - before
    assert new_files, "pipeline wrote nothing"
    for path in new_files:
        assert path.is_relative_to(work_dir.resolve()), path
    for image in result.images:
        assert image.path.resolve().is_relative_to(work_dir.resolve())
    for source_image in result.images:
        source = source_image.source_path
        if source is not None:
            assert Path(source).resolve().is_relative_to(work_dir.resolve())
