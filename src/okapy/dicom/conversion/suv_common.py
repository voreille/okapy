# src/okapy/dicom/conversion/suv_common.py

"""Shared building blocks for SUVbw computation.

These helpers are shared by the Positron Emission Tomography SOP class
converter (:mod:`okapy.dicom.conversion.suv`) and the Enhanced PET SOP class
converter (:mod:`okapy.dicom.conversion.enhanced_pet`).

Everything here follows the IBSI-SUV manual (v3.0.0), "Standardizing SUV
Computation": https://oncoray.github.io/suv_computation/suv.html
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
import logging
import math

import numpy as np

logger = logging.getLogger(__name__)


class SUVComputationError(RuntimeError):
    """Raised when SUVbw cannot be computed reliably."""


# Manual, "Administration time": a decay-correction reference datetime is
# expected between one hour before and two half-lifes after administration.
MIN_UPTAKE_OFFSET_S = -3_600.0

# Manual, "Administration time": above this half-life an uptake time of more
# than 23 hours is plausible, so a reliable administration *date* is required
# and a time-only fallback is not allowed.
MAX_HALF_LIFE_WITHOUT_DATE_S = 41_400.0

# Manual, "Radiopharmaceutical dose": doses below this are stored in MBq.
MBQ_DOSE_THRESHOLD = 1e4

# Manual, "Patient's weight": values at or above this are stored in grams.
GRAM_WEIGHT_THRESHOLD = 1000.0


def warn_once(emitted_warnings: set[str], key: str, message: str, *args) -> None:
    """Log ``message`` once per image rather than once per frame."""

    if key in emitted_warnings:
        return

    emitted_warnings.add(key)
    logger.warning(message, *args)


# --------------------------------------------------------------------------
# Attribute access
# --------------------------------------------------------------------------


def get_optional_str(ds, name: str) -> str | None:
    value = getattr(ds, name, None)

    if value is None:
        return None

    value = str(value).strip()
    return value or None


def get_optional_float(ds, name: str) -> float | None:
    value = getattr(ds, name, None)

    if value is None or str(value).strip() == "":
        return None

    return float(value)


def get_required_float(ds, name: str) -> float:
    value = get_optional_float(ds, name)

    if value is None:
        raise SUVComputationError(f"Missing required DICOM attribute: {name}.")

    return value


def positive_float_tag(ds, tag: tuple[int, int]) -> float | None:
    if tag not in ds:
        return None

    value = ds[tag].value
    if value is None or str(value).strip() == "":
        return None

    value = float(value)
    if value <= 0:
        return None

    return value


# --------------------------------------------------------------------------
# Rescaling
# --------------------------------------------------------------------------


def apply_rescale(
    stored_values: np.ndarray,
    *,
    slope: float,
    intercept: float,
    emitted_warnings: set[str],
    slope_name: str = "RescaleSlope",
    intercept_name: str = "RescaleIntercept",
) -> np.ndarray:
    """Convert stored voxel values to real-world values.

    Manual, "Rescale slope and intercept": both attributes must be present, but
    a non-positive slope or a non-zero intercept only warrants a warning - the
    values are still applied.
    """

    if slope <= 0:
        warn_once(
            emitted_warnings,
            "rescale_slope_non_positive",
            "%s is %g, but a positive value is expected. The image is rescaled "
            "with this value anyway, so the resulting SUVbw values are not "
            "trustworthy and must be checked before use (a zero slope blanks "
            "the frame, a negative slope flips its sign).",
            slope_name,
            slope,
        )

    if intercept != 0:
        warn_once(
            emitted_warnings,
            "rescale_intercept_non_zero",
            "%s is %g, but the Positron Emission Tomography SOP class requires "
            "0. A non-zero value indicates non-standard rescaling applied "
            "during post-processing. The intercept is applied as stored, so the "
            "resulting SUVbw values must be interpreted with caution.",
            intercept_name,
            intercept,
        )

    return slope * stored_values.astype(np.float64) + intercept


# --------------------------------------------------------------------------
# Patient attributes and SUV normalization factors
# --------------------------------------------------------------------------


def patient_weight_kg(weight: float | None) -> float:
    if weight is None or weight <= 0:
        raise SUVComputationError("PatientWeight is required and must be positive.")

    # Manual recommendation: values >= 1000 should be interpreted as grams.
    if weight >= GRAM_WEIGHT_THRESHOLD:
        return weight / 1000.0

    return weight


def patient_weight_g(weight: float | None) -> float:
    return patient_weight_kg(weight) * 1000.0


def patient_size_m(size: float | None) -> float:
    if size is None or size <= 0:
        raise SUVComputationError("PatientSize is required and must be positive.")

    return size


def patient_sex(sex: str | None) -> str:
    value = (sex or "").upper()

    if value not in {"M", "F", "O"}:
        raise SUVComputationError(
            f"PatientSex must be M, F, or O for this SUV conversion, got {value!r}."
        )

    return value


def normalize_suv_type(suv_type: str | None) -> str | None:
    if suv_type is None:
        return None

    return suv_type.strip().upper()


def normalize_manufacturer(manufacturer: str | None) -> str:
    m = (manufacturer or "").upper()

    if "SIEMENS" in m:
        return "SIEMENS"

    if "PHILIPS" in m:
        return "PHILIPS"

    if "GE" in m or "GEMS" in m or "GENERAL ELECTRIC" in m:
        return "GE"

    return "UNKNOWN"


def normalization_factor_kg(
    *,
    suv_type: str,
    weight_kg: float,
    height_m: float,
    sex: str,
) -> float:
    if suv_type in {"LBM", "LBMJAMES128"}:
        return _sex_specific_or_average(
            sex=sex,
            male=lambda: _lbm_james_male_kg(weight_kg, height_m),
            female=lambda: _lbm_james_female_kg(weight_kg, height_m),
        )

    if suv_type == "LBMJANMA":
        return _sex_specific_or_average(
            sex=sex,
            male=lambda: _lbm_janma_male_kg(weight_kg, height_m),
            female=lambda: _lbm_janma_female_kg(weight_kg, height_m),
        )

    if suv_type == "IBW":
        return _sex_specific_or_average(
            sex=sex,
            male=lambda: _ibw_male_kg(height_m),
            female=lambda: _ibw_female_kg(height_m),
        )

    raise SUVComputationError(f"Unsupported SUVType={suv_type!r} for Units=GML.")


def _sex_specific_or_average(sex: str, male, female) -> float:
    if sex == "M":
        return male()

    if sex == "F":
        return female()

    # Manual recommendation for PatientSex=O: use the mean of sex-specific factors.
    if sex == "O":
        return 0.5 * (male() + female())

    raise SUVComputationError(f"Unsupported PatientSex={sex!r}.")


def _lbm_james_male_kg(weight_kg: float, height_m: float) -> float:
    height_cm = height_m * 100.0
    return 1.10 * weight_kg - 128.0 * (weight_kg / height_cm) ** 2


def _lbm_james_female_kg(weight_kg: float, height_m: float) -> float:
    height_cm = height_m * 100.0
    return 1.07 * weight_kg - 148.0 * (weight_kg / height_cm) ** 2


def _lbm_janma_male_kg(weight_kg: float, height_m: float) -> float:
    bmi = weight_kg / (height_m**2)
    return 9270.0 * weight_kg / (6680.0 + 216.0 * bmi)


def _lbm_janma_female_kg(weight_kg: float, height_m: float) -> float:
    bmi = weight_kg / (height_m**2)
    return 9270.0 * weight_kg / (8780.0 + 244.0 * bmi)


def _ibw_male_kg(height_m: float) -> float:
    height_cm = height_m * 100.0
    return 48.0 + 1.06 * (height_cm - 152.0)


def _ibw_female_kg(height_m: float) -> float:
    height_cm = height_m * 100.0
    return 45.5 + 0.91 * (height_cm - 152.0)


def body_surface_area_m2(weight_kg: float, height_m: float) -> float:
    """Du Bois body surface area in m^2."""

    height_cm = height_m * 100.0
    return 0.007184 * (weight_kg**0.425) * (height_cm**0.725)


# --------------------------------------------------------------------------
# Radiopharmaceutical dose and decay
# --------------------------------------------------------------------------


def radiopharmaceutical_item(ds):
    """Return the first item of the Radiopharmaceutical Information Sequence."""

    try:
        return ds.RadiopharmaceuticalInformationSequence[0]
    except Exception as exc:  # noqa: BLE001 - absent, empty, or malformed
        raise SUVComputationError(
            "Missing RadiopharmaceuticalInformationSequence (0054,0016)."
        ) from exc


def radionuclide_total_dose_bq(rph) -> float:
    dose = get_optional_float(rph, "RadionuclideTotalDose")

    if dose is None:
        raise SUVComputationError("Missing RadionuclideTotalDose.")

    if dose <= 0:
        raise SUVComputationError(
            f"RadionuclideTotalDose must be positive, got {dose}."
        )

    # Manual recommendation: dose > 0 and < 1e4 indicates MBq.
    if dose < MBQ_DOSE_THRESHOLD:
        return dose * 1e6

    return dose


def radionuclide_half_life_s(rph) -> float:
    half_life = get_optional_float(rph, "RadionuclideHalfLife")

    if half_life is None:
        raise SUVComputationError("Missing RadionuclideHalfLife.")

    if half_life <= 0:
        raise SUVComputationError(
            f"RadionuclideHalfLife must be positive, got {half_life}."
        )

    return half_life


def average_count_rate_time_s(frame_duration_s: float, half_life_s: float) -> float:
    """Time of average activity (T_ave) within a frame, in seconds.

    Manual, "Determining the scan start datetime":

        T_ave = 1/lambda * ln( lambda*T / (1 - exp(-lambda*T)) )

    which tends to ``T / 2`` for frames that are short relative to the
    half-life.
    """

    if frame_duration_s <= 0:
        return 0.0

    decay_constant = math.log(2.0) / half_life_s
    x = decay_constant * frame_duration_s

    # For tiny x the closed form loses all precision in the logarithm; its
    # limit is exactly half the frame duration.
    if x < 1e-8:
        return frame_duration_s / 2.0

    return math.log(x / -math.expm1(-x)) / decay_constant


def administration_datetime(
    rph,
    *,
    reference_dt: datetime,
    half_life_s: float,
    emitted_warnings: set[str],
    allow_start_time: bool = True,
) -> datetime:
    """Datetime at which the radiotracer was administered.

    Implements the "Administration time" recommendations of the manual: the
    Radiopharmaceutical Start DateTime (0018,1078) is trusted only when the
    resulting uptake time is plausible, otherwise its date component is
    replaced by the date of the decay-correction reference datetime - and that
    substitution is only permitted for short-lived radionuclides.

    ``allow_start_time`` is disabled for the Enhanced PET SOP class, where the
    deprecated Radiopharmaceutical Start Time (0018,1072) is always absent.
    """

    start_datetime = get_optional_str(rph, "RadiopharmaceuticalStartDateTime")

    if start_datetime is not None:
        admin_dt = parse_dicom_datetime(start_datetime)
        offset_s = (reference_dt - admin_dt).total_seconds()

        if MIN_UPTAKE_OFFSET_S <= offset_s < 2.0 * half_life_s:
            return admin_dt

        warn_once(
            emitted_warnings,
            "admin_datetime_implausible",
            "RadiopharmaceuticalStartDateTime (0018,1078) is %s and the "
            "decay-correction reference datetime is %s, an uptake time of "
            "%.0f s. That is outside the plausible window [%.0f s, 2 x "
            "half-life = %.0f s), so at least one of the two dates is wrong "
            "(anonymization is the usual cause). Falling back to the reference "
            "date combined with the administration time; check this image "
            "before trusting its SUVbw values.",
            admin_dt.isoformat(sep=" "),
            reference_dt.isoformat(sep=" "),
            offset_s,
            MIN_UPTAKE_OFFSET_S,
            2.0 * half_life_s,
        )

        return _administration_datetime_from_reference_date(
            admin_dt.time(),
            reference_dt=reference_dt,
            half_life_s=half_life_s,
            emitted_warnings=emitted_warnings,
            source="RadiopharmaceuticalStartDateTime (0018,1078)",
        )

    if allow_start_time:
        start_time = get_optional_str(rph, "RadiopharmaceuticalStartTime")

        if start_time is not None:
            warn_once(
                emitted_warnings,
                "admin_time_only",
                "RadiopharmaceuticalStartDateTime (0018,1078) is absent, so the "
                "administration date is unknown and is assumed to be the date "
                "of the decay-correction reference datetime. The deprecated "
                "RadiopharmaceuticalStartTime (0018,1072) is used for the time "
                "of day.",
            )

            return _administration_datetime_from_reference_date(
                parse_dicom_time(start_time),
                reference_dt=reference_dt,
                half_life_s=half_life_s,
                emitted_warnings=emitted_warnings,
                source="RadiopharmaceuticalStartTime (0018,1072)",
            )

        raise SUVComputationError(
            "Missing RadiopharmaceuticalStartDateTime (0018,1078) and "
            "RadiopharmaceuticalStartTime (0018,1072); the administration "
            "datetime is required for dose correction."
        )

    raise SUVComputationError(
        "Missing RadiopharmaceuticalStartDateTime (0018,1078); it is the only "
        "source of the administration datetime in this SOP class."
    )


def _administration_datetime_from_reference_date(
    admin_time: time,
    *,
    reference_dt: datetime,
    half_life_s: float,
    emitted_warnings: set[str],
    source: str,
) -> datetime:
    if half_life_s >= MAX_HALF_LIFE_WITHOUT_DATE_S:
        raise SUVComputationError(
            f"Cannot determine the administration date: {source} provides no "
            f"usable date and RadionuclideHalfLife is {half_life_s:.0f} s "
            f"(>= {MAX_HALF_LIFE_WITHOUT_DATE_S:.0f} s), so the uptake time may "
            "span more than one day. Assuming same-day administration could "
            "under-correct the dose by whole days, therefore SUVbw is not "
            "computed."
        )

    admin_dt = datetime.combine(reference_dt.date(), admin_time)
    time_offset_s = _seconds_of_day(reference_dt) - _seconds_of_time(admin_time)

    if time_offset_s < MIN_UPTAKE_OFFSET_S:
        admin_dt -= timedelta(days=1)
        warn_once(
            emitted_warnings,
            "admin_previous_day",
            "The administration time (%s) is %.0f s after the time of day of "
            "the decay-correction reference datetime (%s). Assuming the uptake "
            "time spans midnight and the radiotracer was administered on the "
            "previous day.",
            admin_time.isoformat(),
            -time_offset_s,
            reference_dt.time().isoformat(),
        )

    return admin_dt


def decay_corrected_dose_bq(
    *,
    dose_bq: float,
    half_life_s: float,
    reference_dt: datetime,
    administration_dt: datetime,
) -> float:
    """Administered dose decayed to the decay-correction reference datetime."""

    delta_s = (reference_dt - administration_dt).total_seconds()
    corrected_dose = dose_bq * 2 ** (-delta_s / half_life_s)

    if corrected_dose <= 0 or not math.isfinite(corrected_dose):
        raise SUVComputationError(f"Invalid decay-corrected dose: {corrected_dose}.")

    return corrected_dose


def warn_unrecognized_manufacturer(
    manufacturer: str | None,
    *,
    emitted_warnings: set[str],
) -> None:
    """Warn when dose correction relies on the vendor-neutral rules.

    Manual, "Radiopharmaceutical dose": the strategy is verified for Siemens,
    GE, and Philips. It is still applied to other vendors, but with a warning.
    """

    if normalize_manufacturer(manufacturer) != "UNKNOWN":
        return

    warn_once(
        emitted_warnings,
        "unrecognized_manufacturer",
        "Manufacturer (0008,0070) is %r, which is not recognized as SIEMENS, "
        "GE, or PHILIPS. Dose correction falls back to the vendor-neutral "
        "rules of the SUV computation manual; these are unverified for this "
        "vendor, so the SUVbw values should be interpreted with caution.",
        manufacturer or "",
    )


# --------------------------------------------------------------------------
# DICOM date/time parsing
# --------------------------------------------------------------------------


def parse_dicom_datetime(value: str) -> datetime:
    value = str(value).strip()

    # Drop the fractional seconds and the UTC offset suffix.
    value = value.split(".")[0]
    value = value.split("+")[0]
    value = value.split("-")[0] if len(value) > 8 else value

    return datetime.strptime(value, "%Y%m%d%H%M%S")


def parse_dicom_date(value: str) -> date:
    return datetime.strptime(str(value).strip(), "%Y%m%d").date()


def parse_dicom_time(value: str) -> time:
    value = str(value).strip()

    if "." in value:
        value = value.split(".")[0]

    value = value.replace(":", "")

    return datetime.strptime(value, "%H%M%S").time()


def _seconds_of_day(dt: datetime) -> int:
    return dt.hour * 3600 + dt.minute * 60 + dt.second


def _seconds_of_time(t: time) -> int:
    return t.hour * 3600 + t.minute * 60 + t.second
