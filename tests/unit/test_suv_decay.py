"""Unit tests for decay correction of the radionuclide dose.

These focus on ``_dose_at_image_reference_time`` and, in particular, on the
"Administration time" recommendations of the SUV computation manual (v3.0.0):

* Radiopharmaceutical Start DateTime (0018,1078) is trusted only when the
  resulting uptake time lands in ``[-3600 s, 2 x half-life)``;
* outside that window - and whenever only the deprecated Radiopharmaceutical
  Start Time (0018,1072) is available - the administration *date* is taken from
  the decay-correction reference datetime, which is only defensible for
  short-lived radionuclides (half-life < 41,400 s);
* for longer-lived radionuclides the uptake time may span whole days, so a
  missing or untrustworthy administration date must abort the computation
  rather than silently under-correct the dose.

The date substitution matters in practice because DICOM anonymizers rewrite the
public ``AcquisitionDate`` while leaving the private GE/Siemens datetime tags
(which carry the real date) untouched, producing multi-year offsets.
"""

from __future__ import annotations

import math

import pytest
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence

from okapy.dicom.conversion.suv import (
    PETMetadata,
    SUVComputationError,
    _dose_at_image_reference_time,
)
from okapy.dicom.conversion.suv_common import average_count_rate_time_s

# 18F half-life in seconds, matching the test data.
HALF_LIFE_S = 6588.0

# 89Zr: long-lived enough that the uptake time can span several days, so the
# administration date can never be inferred from the reference datetime.
ZR89_HALF_LIFE_S = 282276.0

TOTAL_DOSE_BQ = 636354944.0


def _make_metadata(
    manufacturer: str = "GE MEDICAL SYSTEMS",
    decay_correction: str = "START",
) -> PETMetadata:
    return PETMetadata(
        units="BQML",
        suv_type=None,
        manufacturer=manufacturer,
        decay_correction=decay_correction,
        patient_weight_kg=70.0,
        patient_size_m=1.75,
        patient_sex="M",
    )


def _make_ge_slice(
    *,
    acquisition_date: str,
    acquisition_time: str = "132630.00",
    series_time: str | None = None,
    start_time: str | None = "113500.00",
    start_datetime: str | None = None,
    half_life_s: float = HALF_LIFE_S,
    private_reference_dt: str | None = "19960223132630.00",
) -> Dataset:
    """Build a minimal GE PET slice for the START decay-correction path.

    ``private_reference_dt`` is the GE private image datetime tag (0009,100D)
    used as the reference time; set it to ``None`` to omit it.
    """
    rph = Dataset()
    rph.RadionuclideTotalDose = TOTAL_DOSE_BQ
    rph.RadionuclideHalfLife = half_life_s

    if start_time is not None:
        rph.RadiopharmaceuticalStartTime = start_time

    if start_datetime is not None:
        rph.RadiopharmaceuticalStartDateTime = start_datetime

    ds = Dataset()
    ds.RadiopharmaceuticalInformationSequence = Sequence([rph])
    ds.AcquisitionDate = acquisition_date
    ds.AcquisitionTime = acquisition_time
    ds.SeriesDate = acquisition_date
    # Single-bed acquisition: Series Date/Time equals Acquisition Date/Time,
    # which is the manual's second-choice source for the scan start datetime.
    ds.SeriesTime = series_time if series_time is not None else acquisition_time

    if private_reference_dt is not None:
        ds.add_new(0x0009100D, "DT", private_reference_dt)

    return ds


def _expected_corrected_dose(
    delta_s: float, half_life_s: float = HALF_LIFE_S
) -> float:
    return TOTAL_DOSE_BQ * 2 ** (-delta_s / half_life_s)


def test_consistent_dates_use_full_datetime():
    """Start time only, dates agree: the reference date is the injection date."""
    s = _make_ge_slice(acquisition_date="19960223")

    dose = _dose_at_image_reference_time(s, _make_metadata())

    # Injection 11:35:00 -> reference 13:26:30 == 6690 s.
    assert dose == pytest.approx(_expected_corrected_dose(6690.0))


def test_anonymized_year_falls_back_to_time_of_day():
    """A 1885 AcquisitionDate vs. a real 1996 private tag must not zero the dose."""
    s = _make_ge_slice(acquisition_date="18850827")

    dose = _dose_at_image_reference_time(s, _make_metadata())

    # The date comes from the private reference tag, not from AcquisitionDate.
    assert dose > 0
    assert math.isfinite(dose)
    assert dose == pytest.approx(_expected_corrected_dose(6690.0))


def test_anonymized_year_previously_raised_zero_dose():
    """Guarantee we no longer raise the historical 'Invalid decay-corrected dose'."""
    s = _make_ge_slice(acquisition_date="18850827")

    # Should not raise.
    _dose_at_image_reference_time(s, _make_metadata())


def test_time_of_day_fallback_handles_midnight_crossing():
    """Injection late in the day, scan just after midnight, with garbage dates."""
    # Reference (private tag) at 00:30 on the real date; injection time 23:50.
    s = _make_ge_slice(
        acquisition_date="18850827",
        acquisition_time="003000.00",
        start_time="235000.00",
        private_reference_dt="19960224003000.00",
    )

    dose = _dose_at_image_reference_time(s, _make_metadata())

    # 00:30 - 23:50 = -84000 -> administration on the previous day -> 2400 s.
    assert dose == pytest.approx(_expected_corrected_dose(2400.0))


def test_plausible_start_datetime_is_used_as_is():
    """A Start DateTime inside the plausible window keeps its date component."""
    s = _make_ge_slice(
        acquisition_date="19960223",
        start_datetime="19960223113500.00",
        start_time=None,
    )

    dose = _dose_at_image_reference_time(s, _make_metadata())

    assert dose == pytest.approx(_expected_corrected_dose(6690.0))


def test_long_uptake_time_keeps_the_administration_date():
    """A multi-day uptake with a long-lived radionuclide is legitimate.

    This is DRO_4_5: 89Zr administered three days before the scan. The offset
    is below two half-lifes, so the Start DateTime must be used verbatim rather
    than being collapsed onto the reference date.
    """
    three_days_s = 3 * 86400.0

    s = _make_ge_slice(
        acquisition_date="20250101",
        acquisition_time="110000.00",
        start_datetime="20241229110000.00",
        start_time="110000.00",
        half_life_s=ZR89_HALF_LIFE_S,
        private_reference_dt="20250101110000.00",
    )

    dose = _dose_at_image_reference_time(s, _make_metadata())

    assert dose == pytest.approx(
        _expected_corrected_dose(three_days_s, ZR89_HALF_LIFE_S)
    )


def test_long_lived_radionuclide_without_start_datetime_raises():
    """DRO_error_4_1: only a start *time*, so the uptake day count is unknown."""
    s = _make_ge_slice(
        acquisition_date="20250101",
        acquisition_time="110000.00",
        start_time="110000.00",
        half_life_s=ZR89_HALF_LIFE_S,
        private_reference_dt="20250101110000.00",
    )

    with pytest.raises(SUVComputationError, match="administration date"):
        _dose_at_image_reference_time(s, _make_metadata())


def test_long_lived_radionuclide_with_anonymized_start_datetime_raises():
    """DRO_error_4_2: the Start DateTime date was rewritten to 1960."""
    s = _make_ge_slice(
        acquisition_date="20250101",
        acquisition_time="110000.00",
        start_datetime="19601229110000.00",
        start_time="110000.00",
        half_life_s=ZR89_HALF_LIFE_S,
        private_reference_dt="20250101110000.00",
    )

    with pytest.raises(SUVComputationError, match="administration date"):
        _dose_at_image_reference_time(s, _make_metadata())


def test_short_lived_radionuclide_with_anonymized_start_datetime_uses_reference_date():
    """DRO_4_4: 18F, anonymized acquisition date, uptake spanning midnight."""
    s = _make_ge_slice(
        acquisition_date="19600101",
        acquisition_time="003000.00",
        start_datetime="20250101233000.00",
        start_time="233000.00",
        private_reference_dt="19600101003000.00",
    )

    dose = _dose_at_image_reference_time(s, _make_metadata())

    # The reference date (1960-01-01) wins, and 00:30 vs. 23:30 pushes the
    # administration back one day, leaving a one-hour uptake time.
    assert dose == pytest.approx(_expected_corrected_dose(3600.0))


def test_admin_decay_correction_returns_raw_dose():
    """ADMIN correction ignores timing entirely."""
    s = _make_ge_slice(acquisition_date="18850827")
    meta = _make_metadata(decay_correction="ADMIN")

    dose = _dose_at_image_reference_time(s, meta)

    assert dose == pytest.approx(TOTAL_DOSE_BQ)


def test_invalid_decay_correction_raises():
    s = _make_ge_slice(acquisition_date="19960223")
    meta = _make_metadata(decay_correction="BOGUS")

    with pytest.raises(SUVComputationError):
        _dose_at_image_reference_time(s, meta)


def test_unrecognized_manufacturer_still_computes_a_dose():
    """Manual: an unknown vendor gets a warning, not an error."""
    s = _make_ge_slice(acquisition_date="19960223", private_reference_dt=None)
    meta = _make_metadata(manufacturer="SYNTHETIC")

    # Acquisition Date/Time equals Series Date/Time, so the reference datetime
    # is the acquisition datetime for any vendor.
    dose = _dose_at_image_reference_time(s, meta)

    assert dose == pytest.approx(_expected_corrected_dose(6690.0))


@pytest.mark.parametrize("frame_duration_s", [1.0, 60.0, 301.0, 603.0, 3600.0])
def test_average_count_rate_time_is_below_half_the_frame(frame_duration_s: float):
    """T_ave is pulled below T/2 because activity decays within the frame."""
    tave_s = average_count_rate_time_s(frame_duration_s, HALF_LIFE_S)

    assert 0 < tave_s <= frame_duration_s / 2.0


def test_average_count_rate_time_tends_to_half_the_frame_for_short_frames():
    """T/2 - the approximation Okapy used before - is the short-frame limit."""
    assert average_count_rate_time_s(1.0, HALF_LIFE_S) == pytest.approx(0.5, rel=1e-4)

    # A one-hour frame is where the approximation visibly breaks down.
    assert average_count_rate_time_s(3600.0, HALF_LIFE_S) < 0.98 * 1800.0


def test_average_count_rate_time_matches_the_closed_form():
    decay_constant = math.log(2.0) / HALF_LIFE_S
    frame_duration_s = 603.0
    x = decay_constant * frame_duration_s

    expected = math.log(x / (1.0 - math.exp(-x))) / decay_constant

    assert average_count_rate_time_s(frame_duration_s, HALF_LIFE_S) == pytest.approx(
        expected
    )
