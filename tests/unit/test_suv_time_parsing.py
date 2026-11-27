"""DICOM TM/DT parsing used by the SUV computation, and its effect on SUVbw.

DICOM PS3.5 6.2 allows reduced precision in TM (``HH``, ``HHMM``, ``HHMMSS``,
``HHMMSS.F{1-6}``) and DT (``YYYYMMDD[HH[MM[SS[.F{1-6}]]]][&ZZXX]``). The old
parser forced ``%H%M%S`` through ``strptime``, which silently misread ``"1030"``
as 10:03:00 - a ~27 minute error in the decay time of the injected dose.
"""

from __future__ import annotations

from datetime import datetime, time, timedelta
from pathlib import Path

import numpy as np
import pydicom
import pytest
import SimpleITK as sitk
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.sequence import Sequence
from pydicom.uid import ExplicitVRLittleEndian

from okapy.dicom.conversion.suv import PETSUVConverter
from okapy.dicom.conversion.suv_common import (
    SUVComputationError,
    parse_dicom_datetime,
    parse_dicom_time,
)
from okapy.dicom.models import DicomFileRecord, DicomSeries

PET_IMAGE_STORAGE = "1.2.840.10008.5.1.4.1.1.128"
SERIES_UID = "1.2.826.0.1.3680043.8.498.4242"
STUDY_UID = "1.2.826.0.1.3680043.8.498.42"

HALF_LIFE_S = 6588.0  # 18F
DOSE_BQ = 370e6
WEIGHT_KG = 70.0
STORED = np.array([[1000, 2000], [3000, 4000]], dtype=np.uint16)


# --------------------------------------------------------------------------
# parse_dicom_time
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("12", time(12, 0, 0)),
        ("1230", time(12, 30, 0)),
        ("123045", time(12, 30, 45)),
        ("123045.5", time(12, 30, 45, 500000)),
        ("123045.12", time(12, 30, 45, 120000)),
        ("123045.123", time(12, 30, 45, 123000)),
        ("123045.1234", time(12, 30, 45, 123400)),
        ("123045.12345", time(12, 30, 45, 123450)),
        ("123045.123456", time(12, 30, 45, 123456)),
        ("000000", time(0, 0, 0)),
        ("235959.999999", time(23, 59, 59, 999999)),
        ("0930", time(9, 30, 0)),
        (" 1230 ", time(12, 30, 0)),
    ],
)
def test_parse_dicom_time_accepts_reduced_precision(value, expected):
    assert parse_dicom_time(value) == expected


def test_parse_dicom_time_hhmm_is_not_misread_as_hhmss():
    """Regression: strptime('%H%M%S') read '1030' as 10:03:00."""
    assert parse_dicom_time("1030") == time(10, 30, 0)
    assert parse_dicom_time("1030") != time(10, 3, 0)


@pytest.mark.parametrize(
    "value",
    [
        "930",  # odd length
        "12304",  # odd length
        "1",  # odd length
        "1230451",  # odd length, no dot
        "2500",  # hour out of range
        "2400",  # hour out of range
        "1260",  # minute out of range
        "126000",  # minute out of range
        "123060",  # second out of range (leap second not representable)
        "",
        "   ",
        "ab",
        "12ab",
        "12h30",
        "123045.",  # dot without fraction
        "123045.1234567",  # more than 6 fractional digits
        "1230.5",  # fraction only allowed after seconds
        "-1230",
    ],
)
def test_parse_dicom_time_rejects_invalid_values(value):
    with pytest.raises(SUVComputationError):
        parse_dicom_time(value)


# --------------------------------------------------------------------------
# parse_dicom_datetime
# --------------------------------------------------------------------------


def _wall_clock(dt: datetime) -> datetime:
    return dt.replace(tzinfo=None)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("20240315123045", datetime(2024, 3, 15, 12, 30, 45)),
        ("2024031512", datetime(2024, 3, 15, 12, 0, 0)),
        ("202403151230", datetime(2024, 3, 15, 12, 30, 0)),
        ("20240315", datetime(2024, 3, 15, 0, 0, 0)),
        ("20240315123045.5", datetime(2024, 3, 15, 12, 30, 45, 500000)),
        ("20240315123045.123456", datetime(2024, 3, 15, 12, 30, 45, 123456)),
        ("20240229080000", datetime(2024, 2, 29, 8, 0, 0)),
    ],
)
def test_parse_dicom_datetime_accepts_reduced_precision(value, expected):
    result = parse_dicom_datetime(value)

    assert result == expected
    assert result.tzinfo is None


@pytest.mark.parametrize(
    ("value", "wall_clock", "offset"),
    [
        (
            "20240315123045+0100",
            datetime(2024, 3, 15, 12, 30, 45),
            timedelta(hours=1),
        ),
        (
            "20240315123045-0500",
            datetime(2024, 3, 15, 12, 30, 45),
            timedelta(hours=-5),
        ),
        (
            "20240315123045.25-0500",
            datetime(2024, 3, 15, 12, 30, 45, 250000),
            timedelta(hours=-5),
        ),
        # Reduced precision combined with a negative offset: splitting on "-"
        # must not eat into the date/time digits.
        ("2024031512-0500", datetime(2024, 3, 15, 12, 0, 0), timedelta(hours=-5)),
        ("202403151230+0530", datetime(2024, 3, 15, 12, 30, 0), timedelta(hours=5.5)),
        ("20240315-0500", datetime(2024, 3, 15, 0, 0, 0), timedelta(hours=-5)),
    ],
)
def test_parse_dicom_datetime_keeps_wall_clock_fields_with_utc_offset(
    value, wall_clock, offset
):
    """Whether the result is tz-aware is left to the implementation, but the
    fields written in the string must survive, and if an offset is attached
    it must be the one written."""
    result = parse_dicom_datetime(value)

    assert _wall_clock(result) == wall_clock

    if result.tzinfo is not None:
        assert result.utcoffset() == offset


@pytest.mark.parametrize(
    "value",
    [
        "2024133",  # odd length
        "20241301120000",  # month 13
        "20240015120000",  # month 0
        "20240230120000",  # Feb 30
        "20230229120000",  # not a leap year
        "20240315250000",  # hour 25
        "20240315126000",  # minute 60
        "20240315123060",  # second 60
        "202403151",  # odd length
        "2024031512304",  # odd length
        "",
        "garbage",
        "2024-03-15T12:30:45",  # ISO, not DICOM DT
        "20240315123045.",  # dot without fraction
        "20240315123045+01",  # truncated offset
        "20240315123045+0100x",
    ],
)
def test_parse_dicom_datetime_rejects_invalid_values(value):
    with pytest.raises(SUVComputationError):
        parse_dicom_datetime(value)


# --------------------------------------------------------------------------
# Effect on SUVbw through PETSUVConverter
# --------------------------------------------------------------------------


def _write_pet_series(
    directory: Path,
    *,
    acquisition_time: str,
    series_time: str | None = None,
    start_time: str | None = None,
    start_datetime: str | None = None,
    acquisition_date: str = "20240315",
    timezone_offset: str | None = None,
    manufacturer: str = "SIEMENS",
    n_slices: int = 2,
) -> DicomSeries:
    directory.mkdir(parents=True, exist_ok=True)
    records = []

    for index in range(n_slices):
        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = PET_IMAGE_STORAGE
        meta.MediaStorageSOPInstanceUID = f"{SERIES_UID}.{index + 1}"
        meta.TransferSyntaxUID = ExplicitVRLittleEndian

        path = directory / f"slice_{index}.dcm"
        ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
        ds.SOPClassUID = PET_IMAGE_STORAGE
        ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
        ds.StudyInstanceUID = STUDY_UID
        ds.SeriesInstanceUID = SERIES_UID
        ds.Modality = "PT"
        ds.PatientID = "P1"
        ds.Manufacturer = manufacturer
        ds.InstanceNumber = index + 1

        ds.Units = "BQML"
        ds.DecayCorrection = "START"
        ds.PatientWeight = WEIGHT_KG
        ds.PatientSize = 1.75
        ds.PatientSex = "M"
        ds.SeriesDate = acquisition_date
        ds.SeriesTime = series_time if series_time is not None else acquisition_time
        ds.AcquisitionDate = acquisition_date
        ds.AcquisitionTime = acquisition_time
        ds.ActualFrameDuration = 120000
        ds.FrameReferenceTime = 0
        if timezone_offset is not None:
            ds.TimezoneOffsetFromUTC = timezone_offset

        rph = Dataset()
        rph.RadionuclideTotalDose = DOSE_BQ
        rph.RadionuclideHalfLife = HALF_LIFE_S
        if start_time is not None:
            rph.RadiopharmaceuticalStartTime = start_time
        if start_datetime is not None:
            rph.RadiopharmaceuticalStartDateTime = start_datetime
        ds.RadiopharmaceuticalInformationSequence = Sequence([rph])

        ds.ImagePositionPatient = [0.0, 0.0, float(index) * 3.0]
        ds.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        ds.PixelSpacing = [2.0, 2.0]
        ds.SliceThickness = 3.0
        ds.Rows, ds.Columns = STORED.shape
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 0
        ds.RescaleSlope = 1.0
        ds.RescaleIntercept = 0.0
        ds.PixelData = (STORED * (index + 1)).astype(np.uint16).tobytes()

        ds.save_as(str(path), enforce_file_format=True)

        records.append(
            DicomFileRecord(
                path=path,
                sop_class_uid=PET_IMAGE_STORAGE,
                modality="PT",
                patient_id="P1",
                study_instance_uid=STUDY_UID,
                series_instance_uid=SERIES_UID,
                sop_instance_uid=ds.SOPInstanceUID,
                instance_number=index + 1,
                series_description=None,
                image_position_patient=tuple(ds.ImagePositionPatient),
                image_orientation_patient=tuple(ds.ImageOrientationPatient),
            )
        )

    return DicomSeries(
        series_instance_uid=SERIES_UID, modality="PT", records=tuple(records)
    )


def _convert(series: DicomSeries, tmp_path: Path) -> np.ndarray:
    out = tmp_path / "out"
    out.mkdir(exist_ok=True)
    volume = PETSUVConverter().convert(series, out)
    # sitk array is (z, y, x).
    return sitk.GetArrayFromImage(sitk.ReadImage(str(volume.path)))


def _expected_suvbw(decay_s: float) -> np.ndarray:
    corrected_dose = DOSE_BQ * 2 ** (-decay_s / HALF_LIFE_S)
    stored = np.stack([STORED * (i + 1) for i in range(2)]).astype(np.float64)
    return stored * WEIGHT_KG * 1000.0 / corrected_dose


def test_synthetic_pet_series_is_readable(tmp_path):
    """Sanity check for the fixture itself (pydicom round trip)."""
    series = _write_pet_series(tmp_path / "dcm", acquisition_time="110000")
    ds = pydicom.dcmread(str(series.paths[0]))
    assert ds.pixel_array.tolist() == STORED.tolist()


@pytest.mark.parametrize(
    ("start_time", "scan_time", "decay_s"),
    [
        ("1000", "1100", 3600.0),
        # Old parser: 10:03 -> 11:00 == 3420 s, i.e. 27 min too short.
        ("1030", "1100", 1800.0),
        # Old parser: 10:00 -> 11:03 == 3780 s.
        ("1000", "1130", 5400.0),
        # Mixed precision between start and scan time.
        ("103000", "1100", 1800.0),
        ("1030", "110000.00", 1800.0),
        # HH only.
        ("10", "11", 3600.0),
    ],
)
def test_hhmm_times_give_correct_decay_in_suvbw(
    tmp_path, start_time, scan_time, decay_s
):
    series = _write_pet_series(
        tmp_path / "dcm", acquisition_time=scan_time, start_time=start_time
    )

    suvbw = _convert(series, tmp_path)

    np.testing.assert_allclose(suvbw, _expected_suvbw(decay_s), rtol=1e-5)


def test_hhmm_series_time_matches_hhmmss_acquisition_time(tmp_path):
    """SeriesTime '1100' equals AcquisitionTime '110000': the acquisition
    datetime is the scan start (manual, second choice), not a frame-shifted
    fallback."""
    series = _write_pet_series(
        tmp_path / "dcm",
        acquisition_time="110000",
        series_time="1100",
        start_time="1030",
    )

    suvbw = _convert(series, tmp_path)

    np.testing.assert_allclose(suvbw, _expected_suvbw(1800.0), rtol=1e-5)


def test_reduced_precision_start_datetime_gives_correct_decay(tmp_path):
    series = _write_pet_series(
        tmp_path / "dcm",
        acquisition_time="1100",
        start_datetime="202403151030",
    )

    suvbw = _convert(series, tmp_path)

    np.testing.assert_allclose(suvbw, _expected_suvbw(1800.0), rtol=1e-5)


def test_start_datetime_with_utc_offset_matching_dataset_timezone(tmp_path):
    """Offset equals TimezoneOffsetFromUTC: both naive-local and tz-aware
    readings give a 1 h uptake, and mixing them must not crash."""
    series = _write_pet_series(
        tmp_path / "dcm",
        acquisition_time="110000",
        start_datetime="20240315100000+0100",
        timezone_offset="+0100",
    )

    suvbw = _convert(series, tmp_path)

    np.testing.assert_allclose(suvbw, _expected_suvbw(3600.0), rtol=1e-5)


def test_invalid_acquisition_time_raises_suv_error_naming_series(tmp_path):
    series = _write_pet_series(
        tmp_path / "dcm",
        acquisition_time="1",
        series_time="110000",
        start_time="1000",
    )

    with pytest.raises(SUVComputationError, match=SERIES_UID.replace(".", r"\.")):
        _convert(series, tmp_path)


def test_invalid_start_time_raises_suv_error_naming_series(tmp_path):
    series = _write_pet_series(
        tmp_path / "dcm", acquisition_time="110000", start_time="930"
    )

    with pytest.raises(SUVComputationError, match=SERIES_UID.replace(".", r"\.")):
        _convert(series, tmp_path)
