"""Unit tests for decay correction of the radionuclide dose.

These focus on ``_dose_at_image_reference_time`` and, in particular, the guard
that protects against inconsistent reference/administration dates left behind by
DICOM anonymizers (the private GE/Siemens datetime tags keep the real date while
the public ``AcquisitionDate`` gets rewritten, e.g. to year 1885).
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

# 18F half-life in seconds, matching the test data.
HALF_LIFE_S = 6588.0
TOTAL_DOSE_BQ = 636354944.0


def _make_metadata(manufacturer: str = "GE MEDICAL SYSTEMS") -> PETMetadata:
    return PETMetadata(
        units="BQML",
        suv_type=None,
        manufacturer=manufacturer,
        decay_correction="START",
        patient_weight_kg=70.0,
        patient_size_m=1.75,
        patient_sex="M",
    )


def _make_ge_slice(
    *,
    acquisition_date: str,
    acquisition_time: str = "132630.00",
    start_time: str = "113500.00",
    private_reference_dt: str | None = "19960223132630.00",
) -> Dataset:
    """Build a minimal GE PET slice for the START decay-correction path.

    ``private_reference_dt`` is the GE private image datetime tag (0009,100D)
    used as the reference time; set it to ``None`` to omit it.
    """
    rph = Dataset()
    rph.RadionuclideTotalDose = TOTAL_DOSE_BQ
    rph.RadionuclideHalfLife = HALF_LIFE_S
    rph.RadiopharmaceuticalStartTime = start_time

    ds = Dataset()
    ds.RadiopharmaceuticalInformationSequence = Sequence([rph])
    ds.AcquisitionDate = acquisition_date
    ds.AcquisitionTime = acquisition_time
    ds.SeriesDate = acquisition_date

    if private_reference_dt is not None:
        ds.add_new(0x0009100D, "DT", private_reference_dt)

    return ds


def _expected_corrected_dose(delta_s: float) -> float:
    return TOTAL_DOSE_BQ * 2 ** (-delta_s / HALF_LIFE_S)


def test_consistent_dates_use_full_datetime():
    """When the dates agree, the guard stays out of the way."""
    s = _make_ge_slice(acquisition_date="19960223")

    dose = _dose_at_image_reference_time(s, _make_metadata())

    # Injection 11:35:00 -> reference 13:26:30 == 6690 s.
    assert dose == pytest.approx(_expected_corrected_dose(6690.0))


def test_anonymized_year_falls_back_to_time_of_day():
    """A 1885 AcquisitionDate vs. a real 1996 private tag must not zero the dose."""
    s = _make_ge_slice(acquisition_date="18850827")

    dose = _dose_at_image_reference_time(s, _make_metadata())

    # The ~110-year gap is discarded; the true 6690 s offset is recovered.
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

    # 00:30 - 23:50 = -84000 -> +86400 = 2400 s (40 minutes).
    assert dose == pytest.approx(_expected_corrected_dose(2400.0))


def test_admin_decay_correction_returns_raw_dose():
    """ADMIN correction ignores timing entirely."""
    s = _make_ge_slice(acquisition_date="18850827")
    meta = PETMetadata(
        units="BQML",
        suv_type=None,
        manufacturer="GE MEDICAL SYSTEMS",
        decay_correction="ADMIN",
        patient_weight_kg=70.0,
        patient_size_m=1.75,
        patient_sex="M",
    )

    dose = _dose_at_image_reference_time(s, meta)

    assert dose == pytest.approx(TOTAL_DOSE_BQ)


def test_invalid_decay_correction_raises():
    s = _make_ge_slice(acquisition_date="19960223")
    meta = PETMetadata(
        units="BQML",
        suv_type=None,
        manufacturer="GE MEDICAL SYSTEMS",
        decay_correction="BOGUS",
        patient_weight_kg=70.0,
        patient_size_m=1.75,
        patient_sex="M",
    )

    with pytest.raises(SUVComputationError):
        _dose_at_image_reference_time(s, meta)
