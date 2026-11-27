"""DICOM discovery (DicomWalker) and grouping (DicomStudyCollector)."""

from __future__ import annotations

import logging

import pydicom

from okapy.dicom.collector import DicomStudyCollector
from okapy.dicom.walker import DicomWalker
from tests.helpers.dicom_factories import (
    DEFAULT_PATIENT_ID,
    DEFAULT_STUDY_UID,
    rtstruct_dataset,
    square_contour,
    write_dataset,
    write_image_series,
)

# A DICOM file whose first data element has an unknown VR: pydicom raises
# NotImplementedError while reading it.
MALFORMED_DICOM = b"\0" * 128 + b"DICM" + b"\x08\x00\x60\x00ZZ\x04\x00CT  "


def _write_study(root):
    ct_uid, ct_paths = write_image_series(
        root / "ct", positions=[0.0, 2.0, 4.0], values=[1, 2, 3]
    )
    rtstruct = rtstruct_dataset(
        rois=[(1, "GTV", [square_contour(1, 1, 3, 3, 2.0)])],
        referenced_series_uid=ct_uid,
    )
    rt_path = write_dataset(rtstruct, root / "rt" / "rtstruct.dcm")
    return ct_uid, ct_paths, rtstruct.SeriesInstanceUID, rt_path


# --------------------------------------------------------------------------
# Smoke: walker + collector
# --------------------------------------------------------------------------


def test_collector_groups_ct_and_rtstruct_into_one_study(tmp_path):
    ct_uid, ct_paths, rt_uid, rt_path = _write_study(tmp_path)

    collection = DicomStudyCollector().collect(tmp_path)

    assert len(collection.studies) == 1
    study = collection.studies[0]
    assert study.study_instance_uid == DEFAULT_STUDY_UID
    assert study.patient_id == DEFAULT_PATIENT_ID

    assert [s.series_instance_uid for s in study.image_series] == [ct_uid]
    ct = study.image_series[0]
    assert ct.modality == "CT"
    # Records are sorted along z.
    assert [r.image_position_patient[2] for r in ct.records] == [0.0, 2.0, 4.0]
    assert set(ct.paths) == {p.resolve() for p in ct_paths}

    assert [s.series_instance_uid for s in study.rtstruct_series] == [rt_uid]
    rt = study.rtstruct_series[0]
    assert rt.paths == (rt_path.resolve(),)
    assert rt.referenced_series_uids == (ct_uid,)
    assert study.seg_series == ()


# --------------------------------------------------------------------------
# Hidden paths
# --------------------------------------------------------------------------


def test_data_under_a_hidden_ancestor_directory_is_found(tmp_path):
    root = tmp_path / ".cache" / "x"
    ct_uid, ct_paths = write_image_series(
        root / "ct", positions=[0.0, 1.0, 2.0], values=[1, 2, 3]
    )

    records = DicomWalker().collect_records(root)

    assert {r.path for r in records} == {p.resolve() for p in ct_paths}

    collection = DicomStudyCollector().collect(root)
    assert [s.series_instance_uid for s in collection.image_series] == [ct_uid]


def test_hidden_subdirectories_inside_the_root_are_still_skipped(tmp_path):
    root = tmp_path / ".cache" / "x"
    _, visible = write_image_series(root / "ct", positions=[0.0, 1.0], values=[1, 2])
    _, hidden = write_image_series(
        root / ".hidden", positions=[0.0, 1.0], values=[1, 2]
    )
    hidden_file_uid, hidden_file = write_image_series(
        root / "ct2", positions=[0.0], values=[1], prefix=".slice"
    )

    paths = {r.path for r in DicomWalker().collect_records(root)}

    assert paths == {p.resolve() for p in visible}
    assert not paths & {p.resolve() for p in hidden + hidden_file}


def test_hidden_subdirectories_are_skipped_for_a_plain_root(tmp_path):
    _, visible = write_image_series(
        tmp_path / "ct", positions=[0.0, 1.0], values=[1, 2]
    )
    write_image_series(tmp_path / ".git" / "ct", positions=[0.0, 1.0], values=[1, 2])

    paths = {r.path for r in DicomWalker().collect_records(tmp_path)}

    assert paths == {p.resolve() for p in visible}


# --------------------------------------------------------------------------
# Malformed files
# --------------------------------------------------------------------------


def test_a_malformed_dicom_file_is_logged_and_skipped(tmp_path, caplog):
    _, ct_paths = write_image_series(
        tmp_path / "ct", positions=[0.0, 1.0, 2.0], values=[1, 2, 3]
    )
    bad = tmp_path / "ct" / "broken.dcm"
    bad.write_bytes(MALFORMED_DICOM)

    with caplog.at_level(logging.DEBUG, logger="okapy"):
        records = DicomWalker().collect_records(tmp_path)

    assert {r.path for r in records} == {p.resolve() for p in ct_paths}
    assert "broken.dcm" in caplog.text


def test_collect_survives_a_malformed_dicom_file(tmp_path):
    ct_uid, _ = write_image_series(
        tmp_path / "ct", positions=[0.0, 1.0, 2.0], values=[1, 2, 3]
    )
    (tmp_path / "broken.dcm").write_bytes(MALFORMED_DICOM)

    collection = DicomStudyCollector().collect(tmp_path)

    assert [s.series_instance_uid for s in collection.image_series] == [ct_uid]


def test_non_dicom_files_are_skipped(tmp_path):
    _, ct_paths = write_image_series(tmp_path, positions=[0.0, 1.0], values=[1, 2])
    (tmp_path / "notes.txt").write_text("not a dicom file")

    records = DicomWalker().collect_records(tmp_path)

    assert {r.path for r in records} == {p.resolve() for p in ct_paths}


# --------------------------------------------------------------------------
# Series with mixed modalities
# --------------------------------------------------------------------------


def test_mixed_modality_series_does_not_abort_collect(tmp_path):
    good_uid, _ = write_image_series(
        tmp_path / "good", positions=[0.0, 1.0, 2.0], values=[1, 2, 3]
    )
    bad_uid, bad_paths = write_image_series(
        tmp_path / "bad", positions=[0.0, 1.0, 2.0], values=[1, 2, 3]
    )
    ds = pydicom.dcmread(str(bad_paths[-1]))
    ds.Modality = "MR"
    ds.save_as(str(bad_paths[-1]), enforce_file_format=True)

    collection = DicomStudyCollector().collect(tmp_path)

    image_uids = {s.series_instance_uid for s in collection.image_series}
    assert good_uid in image_uids
    good = next(s for s in collection.image_series if s.series_instance_uid == good_uid)
    assert len(good.records) == 3
