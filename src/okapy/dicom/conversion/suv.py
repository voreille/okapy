# src/okapy/dicom/conversion/suv.py

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from okapy.core.models import ImageVolume
from okapy.dicom.conversion.enhanced_pet import (
    ENHANCED_PET_SOP_CLASS_UIDS,
    EnhancedPETSUVConverter,
)
from okapy.dicom.conversion.image import SimpleITKImageSeriesConverter
from okapy.dicom.conversion.suv_common import (
    SUVComputationError,
    administration_datetime,
    apply_rescale,
    average_count_rate_time_s,
    bqml_to_suvbw,
    check_administration_utc_offset,
    cm2ml_to_suvbw,
    decay_corrected_dose_bq,
    dicom_datetime_utc_offset,
    get_optional_float,
    get_optional_str,
    get_required_float,
    gml_to_suvbw,
    normalize_manufacturer,
    normalize_suv_type,
    parse_dicom_date,
    parse_dicom_datetime,
    parse_dicom_time,
    patient_weight_g,
    positive_float_tag,
    private_tag,
    radionuclide_half_life_s,
    radionuclide_total_dose_bq,
    radiopharmaceutical_item,
    warn_unrecognized_manufacturer,
)
from okapy.dicom.models import DicomSeries

logger = logging.getLogger(__name__)

#: A decay reference datetime, the UTC offset of the DT value it was read from
#: (``None`` when it has none) and the name of that source.
_ReferenceDatetime = tuple[datetime, timedelta | None, str]

__all__ = [
    "PETMetadata",
    "PETSUVConverter",
    "SUVComputationError",
]

# Private attributes as (group, Private Creator, element offset within the
# creator's block). PS3.5 7.8.1: the element is only meaningful inside the
# block reserved by its creator, e.g. (0009,0010)=GEMS_PETD_01 -> (0009,100D).

# Private scan start datetime attributes (manual, "Determining the scan start
# datetime"): the most reliable source, but vendor-specific.
SIEMENS_SCAN_START_TAG = (0x0071, "SIEMENS MED PT", 0x22)
GE_SCAN_START_TAG = (0x0009, "GEMS_PETD_01", 0x0D)

# Philips private scale factors for Units=CNTS.
PHILIPS_SUV_SCALE_FACTOR_TAG = (0x7053, "Philips PET Private Group", 0x00)
PHILIPS_ACTIVITY_SCALE_FACTOR_TAG = (0x7053, "Philips PET Private Group", 0x09)

SUPPORTED_UNITS = ("BQML", "GML", "CM2ML", "CNTS")


@dataclass(frozen=True)
class PETMetadata:
    units: str
    suv_type: str | None
    manufacturer: str
    decay_correction: str | None
    patient_weight_kg: float | None
    patient_size_m: float | None
    patient_sex: str | None
    # Warnings already emitted for this image; lets slice-level code warn once
    # per image instead of once per slice.
    emitted_warnings: set[str] = field(default_factory=set, compare=False)


class PETSUVConverter(SimpleITKImageSeriesConverter):
    """Convert PET DICOM series to SUVbw.

    Implemented according to the SUV computation manual (v3.0.0):
      - slice-wise rescale slope and intercept
      - BQML -> SUVbw
      - GML SUVbw/LBM/LBMJAMES128/LBMJANMA/IBW -> SUVbw
      - CM2ML BSA -> SUVbw
      - CNTS via the Philips activity concentration / SUV scale factors
      - dose correction for DecayCorrection ADMIN/START/NONE

    Series of the Enhanced PET SOP class are delegated to
    :class:`~okapy.dicom.conversion.enhanced_pet.EnhancedPETSUVConverter`,
    which stores the same information in functional groups instead.
    """

    def convert(self, series: DicomSeries, output_dir: Path) -> ImageVolume:
        if series.modality != "PT":
            raise ValueError(f"Expected PT series, got {series.modality}.")

        if _is_enhanced_pet(series):
            return self._enhanced_converter().convert(series, output_dir)

        try:
            return super().convert(series, output_dir)
        except SUVComputationError as exc:
            uid = series.series_instance_uid
            if uid and uid in str(exc):
                raise
            raise SUVComputationError(
                f"Cannot compute SUVbw for series {uid}: {exc}"
            ) from exc

    def _enhanced_converter(self) -> EnhancedPETSUVConverter:
        return EnhancedPETSUVConverter(
            extension=self.extension,
            dtype=self.dtype,
            identity_config=self.identity_config,
        )

    def _get_physical_values(self, slices, paths, modality: str) -> np.ndarray:
        meta = _extract_pet_metadata(slices[0])
        arrays = [self._slice_to_suvbw(s, slices, meta) for s in slices]
        return np.stack(arrays, axis=-1)

    def _slice_to_suvbw(self, s, all_slices, meta: PETMetadata) -> np.ndarray:
        scaled = _scaled_pixel_array(s, meta)

        units = meta.units.upper()

        if units == "BQML":
            return self._bqml_to_suvbw(scaled, s, meta)

        if units == "GML":
            return gml_to_suvbw(
                scaled,
                suv_type=meta.suv_type,
                patient_weight=meta.patient_weight_kg,
                patient_size=meta.patient_size_m,
                sex=meta.patient_sex,
                emitted_warnings=meta.emitted_warnings,
            )

        if units == "CM2ML":
            return cm2ml_to_suvbw(
                scaled,
                suv_type=meta.suv_type,
                patient_weight=meta.patient_weight_kg,
                patient_size=meta.patient_size_m,
                emitted_warnings=meta.emitted_warnings,
            )

        if units == "CNTS":
            return self._cnts_to_suvbw(scaled, s, meta)

        raise SUVComputationError(
            f"Unsupported PET Units={meta.units!r}. "
            f"Supported units are {', '.join(SUPPORTED_UNITS)}."
        )

    def _bqml_to_suvbw(
        self, activity_bqml: np.ndarray, s, meta: PETMetadata
    ) -> np.ndarray:
        return bqml_to_suvbw(
            activity_bqml,
            weight_g=patient_weight_g(meta.patient_weight_kg),
            dose_bq=_dose_at_image_reference_time(s, meta),
        )

    def _cnts_to_suvbw(
        self, scaled_counts: np.ndarray, s, meta: PETMetadata
    ) -> np.ndarray:
        manufacturer = normalize_manufacturer(meta.manufacturer)

        if manufacturer == "PHILIPS":
            # Prefer the activity concentration factor: it converts counts to
            # Bq/ml, after which the normal SUVbw logic applies.
            activity_factor = positive_float_tag(
                s, private_tag(s, *PHILIPS_ACTIVITY_SCALE_FACTOR_TAG)
            )
            if activity_factor is not None:
                activity_bqml = scaled_counts * activity_factor
                return self._bqml_to_suvbw(activity_bqml, s, meta)

            # Then the SUV factor: a direct SUVbw if SUVType is BW/empty.
            suv_factor = positive_float_tag(
                s, private_tag(s, *PHILIPS_SUV_SCALE_FACTOR_TAG)
            )
            if suv_factor is not None:
                suv_type = normalize_suv_type(meta.suv_type)
                if suv_type not in {None, "", "BW"}:
                    raise SUVComputationError(
                        "Philips SUV scale factor only supported for "
                        f"SUVType=BW/empty, got {meta.suv_type!r}."
                    )
                return scaled_counts * suv_factor

        raise SUVComputationError(
            "Units=CNTS requires the Philips Activity Concentration Scale Factor "
            "(7053,xx09) or SUV Scale Factor (7053,xx00) of the private creator "
            "'Philips PET Private Group' on a PHILIPS image; "
            f"Manufacturer is {meta.manufacturer!r} and neither factor is usable."
        )


def _is_enhanced_pet(series: DicomSeries) -> bool:
    return any(
        record.sop_class_uid in ENHANCED_PET_SOP_CLASS_UIDS for record in series.records
    )


def _extract_pet_metadata(s) -> PETMetadata:
    units = getattr(s, "Units", None)
    if units is None or str(units).strip() == "":
        raise SUVComputationError("Missing PET Units (0054,1001).")

    return PETMetadata(
        units=str(units).strip().upper(),
        suv_type=get_optional_str(s, "SUVType"),
        manufacturer=get_optional_str(s, "Manufacturer") or "",
        decay_correction=get_optional_str(s, "DecayCorrection"),
        patient_weight_kg=get_optional_float(s, "PatientWeight"),
        patient_size_m=get_optional_float(s, "PatientSize"),
        patient_sex=get_optional_str(s, "PatientSex"),
    )


def _scaled_pixel_array(s, meta: PETMetadata) -> np.ndarray:
    return apply_rescale(
        s.pixel_array,
        slope=get_required_float(s, "RescaleSlope"),
        intercept=get_required_float(s, "RescaleIntercept"),
        emitted_warnings=meta.emitted_warnings,
    )


def _dose_at_image_reference_time(s, meta: PETMetadata) -> float:
    rph = radiopharmaceutical_item(s)
    dose_bq = radionuclide_total_dose_bq(rph)

    decay_correction = (meta.decay_correction or "").upper()

    if decay_correction not in {"ADMIN", "START", "NONE"}:
        raise SUVComputationError(
            f"DecayCorrection must be ADMIN, START, or NONE, got {decay_correction!r}."
        )

    if decay_correction == "ADMIN":
        # Voxel values are already corrected to the administration time, so the
        # stored dose needs no correction at all.
        return dose_bq

    half_life_s = radionuclide_half_life_s(rph)

    if decay_correction == "START":
        reference_dt, reference_offset, reference_source = (
            _image_reference_datetime_for_start(s, meta, half_life_s)
        )
    else:
        reference_dt, reference_offset, reference_source = (
            _voxel_measurement_datetime_for_none(s, meta, half_life_s)
        )

    check_administration_utc_offset(
        rph,
        reference_offset=reference_offset,
        reference_source=reference_source,
        series_uid=get_optional_str(s, "SeriesInstanceUID"),
    )

    return decay_corrected_dose_bq(
        dose_bq=dose_bq,
        half_life_s=half_life_s,
        reference_dt=reference_dt,
        administration_dt=administration_datetime(
            rph,
            reference_dt=reference_dt,
            half_life_s=half_life_s,
            emitted_warnings=meta.emitted_warnings,
        ),
    )


def _image_reference_datetime_for_start(
    s, meta: PETMetadata, half_life_s: float
) -> _ReferenceDatetime:
    """Scan start datetime, i.e. the time the voxel values were corrected to.

    Manual, "Determining the scan start datetime", in order of preference:
      1. the vendor private scan start datetime (Siemens / GE only);
      2. the Acquisition Date/Time when it equals the Series Date/Time;
      3. one frame reference time before the Acquisition Date/Time (GE);
      4. one frame reference time before the measurement time (any vendor).
    """

    manufacturer = normalize_manufacturer(meta.manufacturer)
    warn_unrecognized_manufacturer(
        meta.manufacturer,
        emitted_warnings=meta.emitted_warnings,
    )

    if manufacturer == "SIEMENS":
        private_dt = _private_datetime(s, SIEMENS_SCAN_START_TAG)
        if private_dt is not None:
            return (*private_dt, "the Siemens private scan start datetime")

    if manufacturer == "GE":
        private_dt = _private_datetime(s, GE_SCAN_START_TAG)
        if private_dt is not None:
            return (*private_dt, "the GE private scan start datetime")

    acquisition_dt, acquisition_offset = _acquisition_datetime(s)
    source = "AcquisitionDateTime (0008,002A)"
    series_dt = _series_datetime(s)

    # Compared at whole seconds: fractional seconds in one of the two values
    # must not disable the shortcut. The returned datetime keeps them.
    if series_dt is not None and acquisition_dt.replace(
        microsecond=0
    ) == series_dt.replace(microsecond=0):
        return acquisition_dt, acquisition_offset, source

    frame_reference_s = _frame_reference_time_s(s)

    if manufacturer == "GE":
        return (
            acquisition_dt - timedelta(seconds=frame_reference_s),
            acquisition_offset,
            source,
        )

    tave_s = average_count_rate_time_s(_actual_frame_duration_s(s), half_life_s)
    return (
        acquisition_dt + timedelta(seconds=tave_s - frame_reference_s),
        acquisition_offset,
        source,
    )


def _voxel_measurement_datetime_for_none(
    s, meta: PETMetadata, half_life_s: float
) -> _ReferenceDatetime:
    """Measurement time, i.e. the time the uncorrected voxel values occurred."""

    warn_unrecognized_manufacturer(
        meta.manufacturer,
        emitted_warnings=meta.emitted_warnings,
    )

    tave_s = average_count_rate_time_s(_actual_frame_duration_s(s), half_life_s)
    acquisition_dt, acquisition_offset = _acquisition_datetime(s)
    return (
        acquisition_dt + timedelta(seconds=tave_s),
        acquisition_offset,
        "AcquisitionDateTime (0008,002A)",
    )


def _actual_frame_duration_s(s) -> float:
    value = get_optional_float(s, "ActualFrameDuration")
    if value is None or value <= 0:
        raise SUVComputationError("ActualFrameDuration must be present and positive.")

    # DICOM stores ActualFrameDuration in ms.
    return value / 1000.0


def _frame_reference_time_s(s) -> float:
    value = get_optional_float(s, "FrameReferenceTime")
    if value is None or value < 0:
        raise SUVComputationError(
            "FrameReferenceTime must be present and non-negative."
        )

    # DICOM stores FrameReferenceTime in ms.
    return value / 1000.0


def _acquisition_datetime(s) -> tuple[datetime, timedelta | None]:
    """Acquisition datetime and the UTC offset of its DT value, if any."""

    value = get_optional_str(s, "AcquisitionDateTime")
    if value is not None:
        return parse_dicom_datetime(value), dicom_datetime_utc_offset(value)

    date_value = getattr(s, "AcquisitionDate", None)
    time_value = getattr(s, "AcquisitionTime", None)

    if date_value is None or time_value is None:
        raise SUVComputationError("Missing AcquisitionDate/AcquisitionTime.")

    acquisition_dt = datetime.combine(
        parse_dicom_date(str(date_value)),
        parse_dicom_time(str(time_value)),
    )
    return acquisition_dt, None


def _series_datetime(s) -> datetime | None:
    date_value = getattr(s, "SeriesDate", None)
    time_value = getattr(s, "SeriesTime", None)

    if date_value is None or time_value is None:
        return None

    try:
        return datetime.combine(
            parse_dicom_date(str(date_value)),
            parse_dicom_time(str(time_value)),
        )
    except SUVComputationError:
        # Optional: an unparsable SeriesTime only disables the "acquisition
        # equals series" shortcut.
        return None


def _private_datetime(
    s, private: tuple[int, str, int]
) -> tuple[datetime, timedelta | None] | None:
    """Private datetime tag and the UTC offset of its value, if any."""

    tag = private_tag(s, *private)
    if tag is None:
        return None

    value = s[tag].value

    if value is None:
        return None

    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="ignore")

    value = str(value).strip().rstrip("\x00").strip()

    if not value:
        return None

    try:
        return parse_dicom_datetime(value), dicom_datetime_utc_offset(value)
    except SUVComputationError:
        logger.warning(
            "Could not parse private datetime tag (%04X,%04X): %r", *tag, value
        )
        return None
