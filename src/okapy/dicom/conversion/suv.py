# src/okapy/dicom/conversion/suv.py

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
import logging
import math

import numpy as np

from okapy.dicom.models import DicomSeries
from okapy.dicom.conversion.image import SimpleITKImageSeriesConverter
from okapy.core.models import ImageVolume

logger = logging.getLogger(__name__)


class SUVComputationError(RuntimeError):
    """Raised when SUVbw cannot be computed reliably."""


@dataclass(frozen=True)
class PETMetadata:
    units: str
    suv_type: str | None
    manufacturer: str
    decay_correction: str | None
    patient_weight_kg: float | None
    patient_size_m: float | None
    patient_sex: str | None


class PETSUVConverter(SimpleITKImageSeriesConverter):
    """Convert PET DICOM series to SUVbw.

    Implemented according to the SUV computation manual:
      - slice-wise rescale slope
      - BQML -> SUVbw
      - GML SUVbw/LBM/IBW/LBMJANMA -> SUVbw
      - CM2ML BSA -> SUVbw
      - CNTS Philips factors or DCAL fallback
      - CPS DCAL fallback
      - dose correction ADMIN/START/NONE
    """

    def convert(self, series: DicomSeries, output_dir: Path) -> ImageVolume:
        if series.modality != "PT":
            raise ValueError(f"Expected PT series, got {series.modality}.")

        return super().convert(series, output_dir)

    def _get_physical_values(self, slices, paths, modality: str) -> np.ndarray:
        meta = _extract_pet_metadata(slices[0])
        arrays = [self._slice_to_suvbw(s, slices, meta) for s in slices]
        return np.stack(arrays, axis=-1)

    def _slice_to_suvbw(self, s, all_slices, meta: PETMetadata) -> np.ndarray:
        scaled = _scaled_pixel_array(s)

        units = meta.units.upper()

        if units == "BQML":
            return self._bqml_to_suvbw(scaled, s, meta)

        if units == "GML":
            return self._gml_to_suvbw(scaled, meta)

        if units == "CM2ML":
            return self._cm2ml_to_suvbw(scaled, meta)

        if units == "CNTS":
            return self._cnts_to_suvbw(scaled, s, meta)

        if units == "CPS":
            return self._cps_to_suvbw(scaled, s, meta)

        raise SUVComputationError(
            f"Unsupported PET Units={meta.units!r}. "
            "Supported units are BQML, GML, CM2ML, CNTS, CPS."
        )

    def _bqml_to_suvbw(
        self, activity_bqml: np.ndarray, s, meta: PETMetadata
    ) -> np.ndarray:
        weight_g = _patient_weight_g(meta)
        dose_bq = _dose_at_image_reference_time(s, meta)
        return activity_bqml * weight_g / dose_bq

    def _gml_to_suvbw(self, scaled: np.ndarray, meta: PETMetadata) -> np.ndarray:
        suv_type = _normalize_suv_type(meta.suv_type)

        if suv_type in {None, "", "BW"}:
            return scaled

        weight_kg = _patient_weight_kg(meta)
        factor_kg = _normalization_factor_kg(
            suv_type=suv_type,
            weight_kg=weight_kg,
            height_m=_patient_size_m(meta),
            sex=_patient_sex(meta),
        )

        # scaled is SUVx = activity * factor_g / dose.
        # SUVbw = SUVx / factor_g * weight_g.
        return scaled * weight_kg / factor_kg

    def _cm2ml_to_suvbw(self, scaled: np.ndarray, meta: PETMetadata) -> np.ndarray:
        suv_type = _normalize_suv_type(meta.suv_type)

        if suv_type not in {"BSA"}:
            raise SUVComputationError(
                f"Units=CM2ML requires SUVType=BSA, got {meta.suv_type!r}."
            )

        weight_g = _patient_weight_g(meta)

        bsa_m2 = _body_surface_area_m2(
            weight_kg=_patient_weight_kg(meta),
            height_m=_patient_size_m(meta),
        )
        bsa_cm2 = bsa_m2 * 10_000.0

        return scaled * weight_g / bsa_cm2

    def _cnts_to_suvbw(
        self, scaled_counts: np.ndarray, s, meta: PETMetadata
    ) -> np.ndarray:
        manufacturer = _normalize_manufacturer(meta.manufacturer)

        # Prefer Philips activity concentration factor when present.
        # It converts counts to Bq/ml, then normal SUVbw logic applies.
        activity_factor = _positive_float_tag(s, (0x7053, 0x1009))
        if manufacturer == "PHILIPS" and activity_factor is not None:
            activity_bqml = scaled_counts * activity_factor
            return self._bqml_to_suvbw(activity_bqml, s, meta)

        # Then Philips SUV factor: direct SUVbw if SUVType is BW/empty.
        suv_factor = _positive_float_tag(s, (0x7053, 0x1000))
        if manufacturer == "PHILIPS" and suv_factor is not None:
            suv_type = _normalize_suv_type(meta.suv_type)
            if suv_type not in {None, "", "BW"}:
                raise SUVComputationError(
                    f"Philips SUV scale factor only supported for SUVType=BW/empty, "
                    f"got {meta.suv_type!r}."
                )
            return scaled_counts * suv_factor

        # Fallback: CNTS -> CPS -> Bq/ml if DCAL.
        if not _has_correction(s, "DCAL"):
            raise SUVComputationError(
                "Units=CNTS without Philips scale factors requires DCAL correction."
            )

        frame_duration_s = _actual_frame_duration_s(s)
        voxel_volume_ml = _voxel_volume_ml(s)

        activity_bqml = scaled_counts / frame_duration_s / voxel_volume_ml
        return self._bqml_to_suvbw(activity_bqml, s, meta)

    def _cps_to_suvbw(self, scaled_cps: np.ndarray, s, meta: PETMetadata) -> np.ndarray:
        if not _has_correction(s, "DCAL"):
            raise SUVComputationError("Units=CPS requires DCAL correction.")

        voxel_volume_ml = _voxel_volume_ml(s)
        activity_bqml = scaled_cps / voxel_volume_ml
        return self._bqml_to_suvbw(activity_bqml, s, meta)


def _extract_pet_metadata(s) -> PETMetadata:
    units = getattr(s, "Units", None)
    if units is None or str(units).strip() == "":
        raise SUVComputationError("Missing PET Units (0054,1001).")

    return PETMetadata(
        units=str(units).strip().upper(),
        suv_type=_get_optional_str(s, "SUVType"),
        manufacturer=_get_optional_str(s, "Manufacturer") or "",
        decay_correction=_get_optional_str(s, "DecayCorrection"),
        patient_weight_kg=_get_optional_float(s, "PatientWeight"),
        patient_size_m=_get_optional_float(s, "PatientSize"),
        patient_sex=_get_optional_str(s, "PatientSex"),
    )


def _scaled_pixel_array(s) -> np.ndarray:
    slope = _get_required_positive_float(s, "RescaleSlope")
    intercept = _get_required_float(s, "RescaleIntercept")

    if intercept != 0:
        raise SUVComputationError(
            f"PET RescaleIntercept must be 0 for SUV computation, got {intercept}."
        )

    return slope * s.pixel_array.astype(np.float64)


def _get_required_float(s, name: str) -> float:
    value = getattr(s, name, None)
    if value is None or str(value).strip() == "":
        raise SUVComputationError(f"Missing required DICOM attribute: {name}.")
    return float(value)


def _get_required_positive_float(s, name: str) -> float:
    value = _get_required_float(s, name)
    if value <= 0:
        raise SUVComputationError(f"{name} must be positive, got {value}.")
    return value


def _get_optional_float(s, name: str) -> float | None:
    value = getattr(s, name, None)
    if value is None or str(value).strip() == "":
        return None
    return float(value)


def _get_optional_str(s, name: str) -> str | None:
    value = getattr(s, name, None)
    if value is None:
        return None
    value = str(value).strip()
    return value or None


def _positive_float_tag(s, tag: tuple[int, int]) -> float | None:
    if tag not in s:
        return None

    value = s[tag].value
    if value is None or str(value).strip() == "":
        return None

    value = float(value)
    if value <= 0:
        return None

    return value


def _normalize_suv_type(suv_type: str | None) -> str | None:
    if suv_type is None:
        return None
    return suv_type.strip().upper()


def _normalize_manufacturer(manufacturer: str) -> str:
    m = manufacturer.upper()

    if "SIEMENS" in m:
        return "SIEMENS"

    if "PHILIPS" in m:
        return "PHILIPS"

    if "GE" in m or "GEMS" in m or "GENERAL ELECTRIC" in m:
        return "GE"

    return "UNKNOWN"


def _patient_weight_kg(meta: PETMetadata) -> float:
    weight = meta.patient_weight_kg

    if weight is None or weight <= 0:
        raise SUVComputationError("PatientWeight is required and must be positive.")

    # Manual recommendation: values >= 1000 should be interpreted as grams.
    if weight >= 1000:
        return weight / 1000.0

    return weight


def _patient_weight_g(meta: PETMetadata) -> float:
    return _patient_weight_kg(meta) * 1000.0


def _patient_size_m(meta: PETMetadata) -> float:
    size = meta.patient_size_m

    if size is None or size <= 0:
        raise SUVComputationError("PatientSize is required and must be positive.")

    return size


def _patient_sex(meta: PETMetadata) -> str:
    sex = (meta.patient_sex or "").upper()

    if sex not in {"M", "F", "O"}:
        raise SUVComputationError(
            f"PatientSex must be M, F, or O for this SUV conversion, got {sex!r}."
        )

    return sex


def _normalization_factor_kg(
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


def _body_surface_area_m2(weight_kg: float, height_m: float) -> float:
    height_cm = height_m * 100.0
    return 0.007184 * (weight_kg**0.425) * (height_cm**0.725)


def _dose_at_image_reference_time(s, meta: PETMetadata) -> float:
    dose_bq = _radionuclide_total_dose_bq(s)

    decay_correction = (meta.decay_correction or "").upper()

    if decay_correction not in {"ADMIN", "START", "NONE"}:
        raise SUVComputationError(
            f"DecayCorrection must be ADMIN, START, or NONE, got {decay_correction!r}."
        )

    if decay_correction == "ADMIN":
        return dose_bq

    half_life_s = _radionuclide_half_life_s(s)
    administration_dt = _radiopharmaceutical_start_datetime(s)

    if decay_correction == "START":
        reference_dt = _image_reference_datetime_for_start(s, meta)
    else:
        reference_dt = _voxel_measurement_datetime_for_none(s, meta)

    delta_s = (reference_dt - administration_dt).total_seconds()
    corrected_dose = dose_bq * 2 ** (-delta_s / half_life_s)

    if corrected_dose <= 0 or not math.isfinite(corrected_dose):
        raise SUVComputationError(f"Invalid decay-corrected dose: {corrected_dose}.")

    return corrected_dose


def _radionuclide_total_dose_bq(s) -> float:
    try:
        dose = float(s.RadiopharmaceuticalInformationSequence[0].RadionuclideTotalDose)
    except Exception as exc:
        raise SUVComputationError("Missing RadionuclideTotalDose.") from exc

    if dose <= 0:
        raise SUVComputationError(
            f"RadionuclideTotalDose must be positive, got {dose}."
        )

    # Manual recommendation: dose > 0 and < 1e4 indicates MBq.
    if dose < 1e4:
        return dose * 1e6

    return dose


def _radionuclide_half_life_s(s) -> float:
    try:
        half_life = float(
            s.RadiopharmaceuticalInformationSequence[0].RadionuclideHalfLife
        )
    except Exception as exc:
        raise SUVComputationError("Missing RadionuclideHalfLife.") from exc

    if half_life <= 0:
        raise SUVComputationError(
            f"RadionuclideHalfLife must be positive, got {half_life}."
        )

    return half_life


def _image_reference_datetime_for_start(s, meta: PETMetadata) -> datetime:
    manufacturer = _normalize_manufacturer(meta.manufacturer)

    if manufacturer == "SIEMENS":
        private_dt = _private_datetime(s, (0x0071, 0x1022))
        if private_dt is not None:
            return private_dt

    if manufacturer == "GE":
        private_dt = _private_datetime(s, (0x0009, 0x100D))
        if private_dt is not None:
            return private_dt

    acquisition_dt = _acquisition_datetime(s)
    series_dt = _series_datetime(s)

    if (
        manufacturer in {"SIEMENS", "GE", "PHILIPS"}
        and series_dt is not None
        and _seconds_of_day(acquisition_dt) == _seconds_of_day(series_dt)
    ):
        return acquisition_dt

    frame_reference_s = _frame_reference_time_s(s)

    if manufacturer in {"SIEMENS", "PHILIPS"}:
        tave_s = _average_count_rate_time_s(s)
        return acquisition_dt + timedelta(seconds=tave_s - frame_reference_s)

    if manufacturer == "GE":
        return acquisition_dt - timedelta(seconds=frame_reference_s)

    raise SUVComputationError(
        f"Cannot determine START reference time for manufacturer={meta.manufacturer!r}."
    )


def _voxel_measurement_datetime_for_none(s, meta: PETMetadata) -> datetime:
    manufacturer = _normalize_manufacturer(meta.manufacturer)

    if manufacturer not in {"SIEMENS", "GE", "PHILIPS"}:
        raise SUVComputationError(
            f"DecayCorrection=NONE not supported for manufacturer={meta.manufacturer!r}."
        )

    return _acquisition_datetime(s) + timedelta(seconds=_average_count_rate_time_s(s))


def _average_count_rate_time_s(s) -> float:
    # The manual discusses Tave; for standard static frames this is commonly half
    # of ActualFrameDuration. Keep this isolated for future refinement.
    return _actual_frame_duration_s(s) / 2.0


def _actual_frame_duration_s(s) -> float:
    value = _get_optional_float(s, "ActualFrameDuration")
    if value is None or value <= 0:
        raise SUVComputationError("ActualFrameDuration must be present and positive.")

    # DICOM stores ActualFrameDuration in ms.
    return value / 1000.0


def _frame_reference_time_s(s) -> float:
    value = _get_optional_float(s, "FrameReferenceTime")
    if value is None or value < 0:
        raise SUVComputationError(
            "FrameReferenceTime must be present and non-negative."
        )

    # DICOM stores FrameReferenceTime in ms.
    return value / 1000.0


def _radiopharmaceutical_start_datetime(s) -> datetime:
    rph = s.RadiopharmaceuticalInformationSequence[0]

    start_datetime = getattr(rph, "RadiopharmaceuticalStartDateTime", None)
    if start_datetime is not None and str(start_datetime).strip() != "":
        return _parse_dicom_datetime(str(start_datetime))

    start_time = getattr(rph, "RadiopharmaceuticalStartTime", None)
    if start_time is None or str(start_time).strip() == "":
        raise SUVComputationError(
            "Missing RadiopharmaceuticalStartDateTime and RadiopharmaceuticalStartTime."
        )

    acquisition_dt = _acquisition_datetime(s)
    injection_t = _parse_dicom_time(str(start_time))

    injection_dt = datetime.combine(acquisition_dt.date(), injection_t)

    # Manual recommendation: if injection time appears >1h after acquisition time,
    # assume administration was on the preceding day.
    if (injection_dt - acquisition_dt).total_seconds() > 3600:
        injection_dt -= timedelta(days=1)

    return injection_dt


def _acquisition_datetime(s) -> datetime:
    if hasattr(s, "AcquisitionDateTime"):
        value = str(s.AcquisitionDateTime).strip()
        if value:
            return _parse_dicom_datetime(value)

    date_value = getattr(s, "AcquisitionDate", None)
    time_value = getattr(s, "AcquisitionTime", None)

    if date_value is None or time_value is None:
        raise SUVComputationError("Missing AcquisitionDate/AcquisitionTime.")

    return datetime.combine(
        _parse_dicom_date(str(date_value)),
        _parse_dicom_time(str(time_value)),
    )


def _series_datetime(s) -> datetime | None:
    date_value = getattr(s, "SeriesDate", None)
    time_value = getattr(s, "SeriesTime", None)

    if date_value is None or time_value is None:
        return None

    try:
        return datetime.combine(
            _parse_dicom_date(str(date_value)),
            _parse_dicom_time(str(time_value)),
        )
    except ValueError:
        return None


def _private_datetime(s, tag: tuple[int, int]) -> datetime | None:
    if tag not in s:
        return None

    value = s[tag].value

    if value is None:
        return None

    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="ignore")

    value = str(value).strip().split(".")[0]

    if not value:
        return None

    try:
        return _parse_dicom_datetime(value)
    except ValueError:
        logger.warning("Could not parse private datetime tag %s: %r", tag, value)
        return None


def _parse_dicom_datetime(value: str) -> datetime:
    value = value.strip()

    # Remove fractional part and timezone for first implementation.
    value = value.split(".")[0]
    value = value.split("+")[0]
    value = value.split("-")[0] if len(value) > 8 else value

    return datetime.strptime(value, "%Y%m%d%H%M%S")


def _parse_dicom_date(value: str) -> date:
    return datetime.strptime(value.strip(), "%Y%m%d").date()


def _parse_dicom_time(value: str):
    value = value.strip()

    if "." in value:
        value = value.split(".")[0]

    value = value.replace(":", "")

    return datetime.strptime(value, "%H%M%S").time()


def _seconds_of_day(dt: datetime) -> int:
    return dt.hour * 3600 + dt.minute * 60 + dt.second


def _has_correction(s, correction: str) -> bool:
    corrected_image = getattr(s, "CorrectedImage", None)

    if corrected_image is None:
        return False

    if isinstance(corrected_image, str):
        values = [corrected_image]
    else:
        values = list(corrected_image)

    return correction.upper() in {str(v).upper() for v in values}


def _voxel_volume_ml(s) -> float:
    try:
        row_spacing, col_spacing = [float(x) for x in s.PixelSpacing]
    except Exception as exc:
        raise SUVComputationError("Missing PixelSpacing.") from exc

    slice_thickness = _get_optional_float(s, "SliceThickness")

    if slice_thickness is None or slice_thickness <= 0:
        raise SUVComputationError("SliceThickness is required and must be positive.")

    # mm^3 to ml: 1000 mm^3 = 1 ml.
    volume_ml = row_spacing * col_spacing * slice_thickness / 1000.0

    if volume_ml <= 0:
        raise SUVComputationError(f"Invalid voxel volume: {volume_ml} ml.")

    return volume_ml
