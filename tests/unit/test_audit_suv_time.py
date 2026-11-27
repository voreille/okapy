"""Audit: DICOM TM/DT boundaries and decay across midnight / leap days."""

from __future__ import annotations

import re
from datetime import datetime, time

import numpy as np
import pytest
import SimpleITK as sitk
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence as DicomSequence

from okapy.dicom.conversion.suv import PETSUVConverter
from okapy.dicom.conversion.suv_common import (
    SUVComputationError,
    administration_datetime,
    decay_corrected_dose_bq,
    parse_dicom_date,
    parse_dicom_datetime,
    parse_dicom_time,
)
from tests.helpers.dicom_factories import load_series, write_image_series

F18_HALF_LIFE_S = 6588.0


# --------------------------------------------------------------------------
# DT: UTC offsets and calendar boundaries
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("20240101000000+1400", datetime(2024, 1, 1)),
        ("20240101000000-1200", datetime(2024, 1, 1)),
        ("20240101+0100", datetime(2024, 1, 1)),
        ("2024010123+0000", datetime(2024, 1, 1, 23)),
        ("20240229235959.999999", datetime(2024, 2, 29, 23, 59, 59, 999999)),
        ("20000229", datetime(2000, 2, 29)),
        ("20241231235959", datetime(2024, 12, 31, 23, 59, 59)),
        ("20240101 ", datetime(2024, 1, 1)),  # DICOM space padding
    ],
)
def test_dt_boundaries_parse(value, expected):
    assert parse_dicom_datetime(value) == expected


@pytest.mark.parametrize(
    "value",
    [
        "20240101000000+1401",
        "20240101000000-1201",
        "20240101000000+0060",
        "20240101000000+14",
        "20240101000000+",
        "20240101000000 +0100",
        "20230229",
        "19000229",
        "20241301",
        "20240100",
        "20240132",
        "202401011260",
        "20240101240000",
        "２０２４０１０１",  # full-width digits
        "20240101120000.",
        "202401011200.5",
    ],
)
def test_dt_invalid_values_raise_suv_error(value):
    with pytest.raises(SUVComputationError):
        parse_dicom_datetime(value)


@pytest.mark.parametrize("value", ["١٢٣٠", "１２３０", "12:30", "12:3045", "1230 45"])
def test_tm_non_ascii_or_malformed_is_rejected(value):
    with pytest.raises(SUVComputationError):
        parse_dicom_time(value)


@pytest.mark.parametrize(
    ("value", "expected"),
    [("12:30:45", time(12, 30, 45)), ("23:59:59.5", time(23, 59, 59, 500000))],
)
def test_tm_legacy_colon_form(value, expected):
    assert parse_dicom_time(value) == expected


@pytest.mark.parametrize("value", ["20230229", "2024022", "20240229x", ""])
def test_da_invalid_raises_suv_error(value):
    with pytest.raises(SUVComputationError):
        parse_dicom_date(value)


def test_parse_rejects_none_with_suv_error():
    with pytest.raises(SUVComputationError):
        parse_dicom_time(None)
    with pytest.raises(SUVComputationError):
        parse_dicom_datetime(None)


# --------------------------------------------------------------------------
# Administration time across midnight / leap day / year end
# --------------------------------------------------------------------------


def _rph(*, start_time=None, start_datetime=None):
    rph = Dataset()
    if start_time is not None:
        rph.RadiopharmaceuticalStartTime = start_time
    if start_datetime is not None:
        rph.RadiopharmaceuticalStartDateTime = start_datetime
    return rph


@pytest.mark.parametrize(
    ("reference", "start_time", "expected"),
    [
        (datetime(2024, 3, 1, 0, 20), "2350", datetime(2024, 2, 29, 23, 50)),
        (datetime(2023, 3, 1, 0, 20), "2350", datetime(2023, 2, 28, 23, 50)),
        (datetime(2025, 1, 1, 0, 10), "235500", datetime(2024, 12, 31, 23, 55)),
        (datetime(2024, 2, 29, 10, 0), "08", datetime(2024, 2, 29, 8, 0)),
    ],
)
def test_time_only_administration_across_day_boundaries(
    reference, start_time, expected
):
    admin = administration_datetime(
        _rph(start_time=start_time),
        reference_dt=reference,
        half_life_s=F18_HALF_LIFE_S,
        emitted_warnings=set(),
    )
    assert admin == expected


def test_reduced_precision_start_datetime_on_leap_day_is_used_as_is():
    admin = administration_datetime(
        _rph(start_datetime="202402292330"),
        reference_dt=datetime(2024, 3, 1, 0, 30),
        half_life_s=F18_HALF_LIFE_S,
        emitted_warnings=set(),
    )
    assert admin == datetime(2024, 2, 29, 23, 30)


def test_injection_shortly_after_reference_time_increases_dose():
    """Injection 30 min after the decay reference: the dose grows back."""

    admin = administration_datetime(
        _rph(start_datetime="20240101103000"),
        reference_dt=datetime(2024, 1, 1, 10, 0),
        half_life_s=F18_HALF_LIFE_S,
        emitted_warnings=set(),
    )
    assert admin == datetime(2024, 1, 1, 10, 30)

    dose = decay_corrected_dose_bq(
        dose_bq=100.0,
        half_life_s=F18_HALF_LIFE_S,
        reference_dt=datetime(2024, 1, 1, 10, 0),
        administration_dt=admin,
    )
    assert dose == pytest.approx(100.0 * 2 ** (1800.0 / F18_HALF_LIFE_S))


def test_fractional_seconds_are_used_in_decay():
    admin = administration_datetime(
        _rph(start_datetime="20240101090000.5"),
        reference_dt=datetime(2024, 1, 1, 10, 0),
        half_life_s=F18_HALF_LIFE_S,
        emitted_warnings=set(),
    )
    assert admin == datetime(2024, 1, 1, 9, 0, 0, 500000)


def test_invalid_start_datetime_raises_suv_error_not_value_error():
    with pytest.raises(SUVComputationError):
        administration_datetime(
            _rph(start_datetime="20240230100000"),
            reference_dt=datetime(2024, 3, 1, 10, 0),
            half_life_s=F18_HALF_LIFE_S,
            emitted_warnings=set(),
        )


# --------------------------------------------------------------------------
# UTC offsets on both sides (user decision): differing offsets raise
# --------------------------------------------------------------------------

DOSE_BQ = 370e6
WEIGHT_KG = 70.0
STORED_VALUES = [100, 200]


def _bqml_pet(tmp_path, *, start_datetime: str, acquisition_datetime: str):
    """Two-slice BQML PET, DecayCorrection START, scan start = AcquisitionDateTime.

    SeriesDate/Time equal the wall clock of ``acquisition_datetime`` so the
    Acquisition DateTime is the decay reference (manual, second choice).
    """

    rph = Dataset()
    rph.RadionuclideTotalDose = DOSE_BQ
    rph.RadionuclideHalfLife = F18_HALF_LIFE_S
    rph.RadiopharmaceuticalStartDateTime = start_datetime

    wall_clock = acquisition_datetime[:14]
    extra = {
        "Units": "BQML",
        "SUVType": "BW",
        "DecayCorrection": "START",
        "PatientWeight": WEIGHT_KG,
        "Manufacturer": "SIEMENS",
        "SeriesDate": wall_clock[:8],
        "SeriesTime": wall_clock[8:],
        "AcquisitionDate": wall_clock[:8],
        "AcquisitionTime": wall_clock[8:],
        "AcquisitionDateTime": acquisition_datetime,
        "ActualFrameDuration": 120000,
        "FrameReferenceTime": 0,
        "RadiopharmaceuticalInformationSequence": DicomSequence([rph]),
    }
    uid, paths = write_image_series(
        tmp_path / "dicom",
        modality="PT",
        positions=[0, 3],
        values=STORED_VALUES,
        rescale=(1.0, 0.0),
        extra=extra,
    )
    return uid, load_series(paths)


def _suvbw(tmp_path, **kwargs) -> np.ndarray:
    _, series = _bqml_pet(tmp_path, **kwargs)
    volume = PETSUVConverter().convert(series, tmp_path / "out")
    return sitk.GetArrayFromImage(volume.image)[:, 0, 0]


def _expected_suvbw(decay_s: float) -> np.ndarray:
    corrected = DOSE_BQ * 2 ** (-decay_s / F18_HALF_LIFE_S)
    return np.asarray(STORED_VALUES, dtype=float) * WEIGHT_KG * 1000.0 / corrected


@pytest.mark.parametrize(
    ("start", "scan"),
    [
        ("20240315100000+0100", "20240315110000+0000"),
        ("20240315100000+0000", "20240315110000+0100"),
        ("20240315100000-0500", "20240315110000+0530"),
        # Across midnight: 23:50+0100 injection, 00:10+0000 scan.
        ("20240314235000+0100", "20240315001000+0000"),
    ],
)
def test_differing_utc_offsets_raise_suv_error_naming_series(tmp_path, start, scan):
    uid, series = _bqml_pet(tmp_path, start_datetime=start, acquisition_datetime=scan)

    with pytest.raises(SUVComputationError, match=re.escape(uid)):
        PETSUVConverter().convert(series, tmp_path / "out")


@pytest.mark.parametrize("offset", ["+0100", "+0000", "-0500"])
def test_equal_utc_offsets_give_the_same_suv_as_no_offsets(tmp_path, offset):
    with_offsets = _suvbw(
        tmp_path / "with",
        start_datetime=f"20240315100000{offset}",
        acquisition_datetime=f"20240315110000{offset}",
    )
    without = _suvbw(
        tmp_path / "without",
        start_datetime="20240315100000",
        acquisition_datetime="20240315110000",
    )

    np.testing.assert_allclose(with_offsets, without, rtol=1e-9)
    np.testing.assert_allclose(with_offsets, _expected_suvbw(3600.0), rtol=1e-5)


@pytest.mark.parametrize(
    ("start", "scan"),
    [
        ("20240315100000+0100", "20240315110000"),
        ("20240315100000", "20240315110000+0000"),
    ],
)
def test_offset_on_one_side_only_uses_the_wall_clock_difference(tmp_path, start, scan):
    suvbw = _suvbw(tmp_path, start_datetime=start, acquisition_datetime=scan)

    np.testing.assert_allclose(suvbw, _expected_suvbw(3600.0), rtol=1e-5)
