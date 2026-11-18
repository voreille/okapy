# src/okapy/dicom/conversion/enhanced_pet.py

"""SUVbw conversion for the Enhanced PET Image Storage SOP class.

Unlike the classic Positron Emission Tomography SOP class, an Enhanced PET
instance is a single multi-frame object: geometry, rescaling, units, and frame
timing live in the Shared (5200,9229) and Per-Frame (5200,9230) Functional
Groups Sequences rather than as top-level attributes.

Implemented according to the "Computing SUVbw in other SOP classes" section of
the SUV computation manual (v3.0.0).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import logging

import numpy as np
import pydicom

from okapy.core.models import ImageVolume, VolumeStage
from okapy.dicom.conversion.suv_common import (
    SUVComputationError,
    administration_datetime,
    apply_rescale,
    average_count_rate_time_s,
    body_surface_area_m2,
    decay_corrected_dose_bq,
    get_optional_float,
    get_optional_str,
    normalization_factor_kg,
    normalize_suv_type,
    parse_dicom_datetime,
    patient_sex,
    patient_size_m,
    patient_weight_g,
    patient_weight_kg,
    radionuclide_half_life_s,
    radionuclide_total_dose_bq,
    radiopharmaceutical_item,
    warn_once,
)
from okapy.dicom.conversion.utils import (
    safe_name,
    short_uid,
    sitk_image_from_array_xyz,
    write_image_unique,
)
from okapy.dicom.identity import SeriesIdentityConfig, build_series_identity
from okapy.dicom.models import DicomSeries

logger = logging.getLogger(__name__)

ENHANCED_PET_SOP_CLASS_UID = "1.2.840.10008.5.1.4.1.1.130"
LEGACY_CONVERTED_ENHANCED_PET_SOP_CLASS_UID = "1.2.840.10008.5.1.4.1.1.128.1"

# The manual gives the Legacy Converted class the same treatment as the
# Enhanced PET class.
ENHANCED_PET_SOP_CLASS_UIDS = frozenset(
    {
        ENHANCED_PET_SOP_CLASS_UID,
        LEGACY_CONVERTED_ENHANCED_PET_SOP_CLASS_UID,
    }
)

# Measurement Units Code Sequence (0040,08EA) code values, CID 84 and CID 85,
# mapped onto the classic Units (0054,1001) / SUV Type (0054,1006) pair.
UNIT_CODES: dict[str, tuple[str, str | None]] = {
    "bq/ml": ("BQML", None),
    "g/ml{suvbw}": ("GML", "BW"),
    "g/ml{suvlbm}": ("GML", "LBM"),
    "g/ml{suvlbmjames128}": ("GML", "LBMJAMES128"),
    "g/ml{suvlbmjanma}": ("GML", "LBMJANMA"),
    "g/ml{suvibw}": ("GML", "IBW"),
    "cm2/ml{suvbsa}": ("CM2ML", "BSA"),
}

# Rescale Type (0028,1054) values usable as a fallback for the physical units.
FALLBACK_RESCALE_TYPES = frozenset({"BQML", "GML", "CM2ML"})


@dataclass(frozen=True)
class EnhancedPETMetadata:
    """Instance-level attributes shared by every frame of one instance."""

    decay_corrected: str | None
    decay_correction_datetime: str | None
    suv_type: str | None
    patient_weight_kg: float | None
    patient_size_m: float | None
    patient_sex: str | None
    # Warnings already emitted for this image; lets frame-level code warn once
    # per image instead of once per frame.
    emitted_warnings: set[str] = field(default_factory=set, compare=False)


@dataclass(frozen=True)
class _Frame:
    """One frame of one Enhanced PET instance, with its resolved geometry."""

    dataset: pydicom.Dataset
    index: int
    meta: EnhancedPETMetadata
    position: tuple[float, float, float]
    orthogonal_position: float


class EnhancedPETSUVConverter:
    """Convert an Enhanced PET series to an SUVbw NIfTI volume."""

    def __init__(
        self,
        *,
        extension: str = "nii.gz",
        dtype: np.dtype = np.float32,
        identity_config: SeriesIdentityConfig | None = None,
    ) -> None:
        self.extension = extension
        self.dtype = dtype
        self.identity_config = identity_config

    def convert(self, series: DicomSeries, output_dir: Path) -> ImageVolume:
        if series.modality != "PT":
            raise ValueError(f"Expected PT series, got {series.modality}.")

        frames, orientation, pixel_spacing = self._read_frames(series)

        image_xyz = np.stack(
            [self._frame_to_suvbw(frame) for frame in frames],
            axis=-1,
        )
        image_xyz = np.transpose(image_xyz, (1, 0, 2))
        image_xyz = image_xyz.astype(self.dtype, copy=False)

        origin, spacing, direction = _geometry(
            frames=frames,
            orientation=orientation,
            pixel_spacing=pixel_spacing,
        )

        image = sitk_image_from_array_xyz(
            image_xyz,
            origin=origin,
            spacing=spacing,
            direction=direction,
        )

        path = write_image_unique(image, output_dir / self._make_filename(series))

        return ImageVolume.from_sitk(
            path=path,
            image=image,
            identity=build_series_identity(series, config=self.identity_config),
            stage=VolumeStage.CONVERTED,
            metadata={
                "source": "dicom",
                "converter": self.__class__.__name__,
            },
        )

    # -- reading ----------------------------------------------------------

    def _read_frames(self, series: DicomSeries):
        frames: list[_Frame] = []
        orientation: np.ndarray | None = None
        pixel_spacing: tuple[float, float] | None = None

        for path in series.paths:
            ds = pydicom.dcmread(str(path))
            meta = _extract_metadata(ds)

            per_frame = getattr(ds, "PerFrameFunctionalGroupsSequence", None)
            if per_frame is None or len(per_frame) == 0:
                raise SUVComputationError(
                    "Missing Per-Frame Functional Groups Sequence (5200,9230) in "
                    f"{path.name}; the Enhanced PET instance cannot be read."
                )

            frame_orientation = _frame_orientation(ds, 0, path=path)
            normal = np.cross(frame_orientation[:3], frame_orientation[3:])

            if orientation is None:
                orientation = frame_orientation
                pixel_spacing = _frame_pixel_spacing(ds, 0, path=path)

            for index in range(len(per_frame)):
                position = _frame_position(ds, index, path=path)
                frames.append(
                    _Frame(
                        dataset=ds,
                        index=index,
                        meta=meta,
                        position=position,
                        orthogonal_position=float(np.dot(normal, position)),
                    )
                )

        if not frames:
            raise SUVComputationError(
                f"Enhanced PET series {series.series_instance_uid} contains no frames."
            )

        frames.sort(key=lambda frame: frame.orthogonal_position)

        return frames, orientation, pixel_spacing

    def _make_filename(self, series: DicomSeries) -> str:
        patient = safe_name(series.patient_id)
        modality = safe_name(series.modality)
        uid = short_uid(series.series_instance_uid)
        return f"{patient}__{modality}__{uid}.{self.extension}"

    # -- SUV computation --------------------------------------------------

    def _frame_to_suvbw(self, frame: _Frame) -> np.ndarray:
        units, suv_type, scaled = _real_world_values(frame)
        meta = frame.meta

        if units == "GML":
            return self._gml_to_suvbw(scaled, suv_type, meta)

        if units == "CM2ML":
            return self._cm2ml_to_suvbw(scaled, suv_type, meta)

        if units == "BQML":
            weight_g = patient_weight_g(meta.patient_weight_kg)
            return scaled * weight_g / _dose_at_reference_time(frame)

        raise SUVComputationError(
            f"Unsupported physical units {units!r} in the Enhanced PET frame."
        )

    def _gml_to_suvbw(
        self,
        scaled: np.ndarray,
        suv_type: str | None,
        meta: EnhancedPETMetadata,
    ) -> np.ndarray:
        suv_type = normalize_suv_type(suv_type) or normalize_suv_type(meta.suv_type)

        if suv_type in {None, "", "BW"}:
            return scaled

        weight_kg = patient_weight_kg(meta.patient_weight_kg)
        factor_kg = normalization_factor_kg(
            suv_type=suv_type,
            weight_kg=weight_kg,
            height_m=patient_size_m(meta.patient_size_m),
            sex=patient_sex(meta.patient_sex),
        )

        return scaled * weight_kg / factor_kg

    def _cm2ml_to_suvbw(
        self,
        scaled: np.ndarray,
        suv_type: str | None,
        meta: EnhancedPETMetadata,
    ) -> np.ndarray:
        suv_type = normalize_suv_type(suv_type) or normalize_suv_type(meta.suv_type)

        if suv_type not in {"BSA"}:
            raise SUVComputationError(
                f"Units=CM2ML requires SUVType=BSA, got {suv_type!r}."
            )

        bsa_cm2 = (
            body_surface_area_m2(
                weight_kg=patient_weight_kg(meta.patient_weight_kg),
                height_m=patient_size_m(meta.patient_size_m),
            )
            * 10_000.0
        )

        return scaled * patient_weight_g(meta.patient_weight_kg) / bsa_cm2


# --------------------------------------------------------------------------
# Metadata
# --------------------------------------------------------------------------


def _extract_metadata(ds) -> EnhancedPETMetadata:
    return EnhancedPETMetadata(
        decay_corrected=get_optional_str(ds, "DecayCorrected"),
        decay_correction_datetime=get_optional_str(ds, "DecayCorrectionDateTime"),
        suv_type=get_optional_str(ds, "SUVType"),
        patient_weight_kg=get_optional_float(ds, "PatientWeight"),
        patient_size_m=get_optional_float(ds, "PatientSize"),
        patient_sex=get_optional_str(ds, "PatientSex"),
    )


def _functional_group(ds, frame_index: int, name: str):
    """Per-frame functional group item, falling back to the shared one.

    Manual: "Shared Functional Groups define default values for all frames,
    while Per-frame Functional Groups override them where specified."
    """

    per_frame = ds.PerFrameFunctionalGroupsSequence[frame_index]
    sequence = getattr(per_frame, name, None)

    if sequence is not None and len(sequence) > 0:
        return sequence[0]

    shared = getattr(ds, "SharedFunctionalGroupsSequence", None)
    if shared is None or len(shared) == 0:
        return None

    sequence = getattr(shared[0], name, None)

    if sequence is None or len(sequence) == 0:
        return None

    return sequence[0]


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------


def _frame_orientation(ds, frame_index: int, *, path: Path) -> np.ndarray:
    item = _functional_group(ds, frame_index, "PlaneOrientationSequence")
    value = getattr(item, "ImageOrientationPatient", None) if item else None

    if value is None:
        raise SUVComputationError(
            "Missing Image Orientation (Patient) (0020,0037) in the Plane "
            f"Orientation Sequence (0020,9116) of {path.name}."
        )

    return np.asarray(value, dtype=float)


def _frame_position(ds, frame_index: int, *, path: Path) -> tuple[float, float, float]:
    item = _functional_group(ds, frame_index, "PlanePositionSequence")
    value = getattr(item, "ImagePositionPatient", None) if item else None

    if value is None:
        raise SUVComputationError(
            "Missing Image Position (Patient) (0020,0032) in the Plane Position "
            f"Sequence (0020,9113) of frame {frame_index} in {path.name}."
        )

    return tuple(float(x) for x in value)


def _frame_pixel_spacing(ds, frame_index: int, *, path: Path) -> tuple[float, float]:
    item = _functional_group(ds, frame_index, "PixelMeasuresSequence")
    value = getattr(item, "PixelSpacing", None) if item else None

    if value is None:
        raise SUVComputationError(
            "Missing Pixel Spacing (0028,0030) in the Pixel Measures Sequence "
            f"(0028,9110) of {path.name}."
        )

    row_spacing, col_spacing = (float(x) for x in value)
    return row_spacing, col_spacing


def _geometry(*, frames, orientation, pixel_spacing):
    row = orientation[:3]
    col = orientation[3:]
    normal = np.cross(row, col)

    direction = tuple(
        float(x) for x in np.stack([row, col, normal], axis=1).ravel()
    )

    origin = frames[0].position
    row_spacing, col_spacing = pixel_spacing

    if len(frames) > 1:
        first = np.asarray(frames[0].position, dtype=float)
        last = np.asarray(frames[-1].position, dtype=float)
        z_spacing = float(np.linalg.norm(last - first) / (len(frames) - 1))
    else:
        item = _functional_group(
            frames[0].dataset,
            frames[0].index,
            "PixelMeasuresSequence",
        )
        z_spacing = float(getattr(item, "SliceThickness", 1.0) if item else 1.0)

    return origin, (col_spacing, row_spacing, z_spacing), direction


# --------------------------------------------------------------------------
# Real-world values
# --------------------------------------------------------------------------


def _real_world_values(frame: _Frame) -> tuple[str, str | None, np.ndarray]:
    """Return (units, suv_type, rescaled frame values).

    Manual, "Enhanced PET Image Storage SOP Class": prefer a Real World Value
    Mapping that yields SUVbw, then one that yields another SUV type, then one
    that yields Bq/ml, and only then fall back to the Pixel Value
    Transformation Sequence combined with Rescale Type.
    """

    stored = frame.dataset.pixel_array[frame.index]
    mapping = _select_real_world_value_mapping(frame)

    if mapping is not None:
        units, suv_type, item = mapping
        slope = get_optional_float(item, "RealWorldValueSlope")
        intercept = get_optional_float(item, "RealWorldValueIntercept")

        scaled = apply_rescale(
            stored,
            slope=slope,
            intercept=intercept if intercept is not None else 0.0,
            emitted_warnings=frame.meta.emitted_warnings,
            slope_name="RealWorldValueSlope (0040,9225)",
            intercept_name="RealWorldValueIntercept (0040,9224)",
        )
        return units, suv_type, scaled

    item = _functional_group(
        frame.dataset,
        frame.index,
        "PixelValueTransformationSequence",
    )
    rescale_type = (get_optional_str(item, "RescaleType") or "").upper() if item else ""
    slope = get_optional_float(item, "RescaleSlope") if item else None

    if item is None or rescale_type not in FALLBACK_RESCALE_TYPES or slope is None:
        raise SUVComputationError(
            "No Real World Value Mapping Sequence (0040,9096) with usable units "
            "and rescaling was found, and the Pixel Value Transformation "
            "Sequence (0028,9145) fallback is unusable "
            f"(Rescale Type (0028,1054) is {rescale_type or 'absent'}, expected "
            f"one of {', '.join(sorted(FALLBACK_RESCALE_TYPES))}; Rescale Slope "
            f"(0028,1053) is {'absent' if slope is None else slope}). "
            "SUVbw cannot be computed."
        )

    warn_once(
        frame.meta.emitted_warnings,
        "rwvm_fallback",
        "No usable Real World Value Mapping Sequence (0040,9096) was found. "
        "Falling back to the Pixel Value Transformation Sequence (0028,9145) "
        "with Rescale Type (0028,1054) = %s, which is unspecified for PET and "
        "therefore only a best-effort interpretation of the physical units.",
        rescale_type,
    )

    scaled = apply_rescale(
        stored,
        slope=slope,
        intercept=get_optional_float(item, "RescaleIntercept") or 0.0,
        emitted_warnings=frame.meta.emitted_warnings,
    )

    return rescale_type, None, scaled


def _select_real_world_value_mapping(frame: _Frame):
    """Pick the best Real World Value Mapping item for one frame."""

    per_frame = frame.dataset.PerFrameFunctionalGroupsSequence[frame.index]
    sequence = getattr(per_frame, "RealWorldValueMappingSequence", None)

    if sequence is None or len(sequence) == 0:
        shared = getattr(frame.dataset, "SharedFunctionalGroupsSequence", None)
        if shared is not None and len(shared) > 0:
            sequence = getattr(shared[0], "RealWorldValueMappingSequence", None)

    if sequence is None or len(sequence) == 0:
        return None

    candidates: list[tuple[int, str, str | None, object]] = []

    for item in sequence:
        units_and_type = _units_from_measurement_code(item)

        if units_and_type is None:
            continue

        units, suv_type = units_and_type

        if get_optional_float(item, "RealWorldValueSlope") is None:
            if "RealWorldValueLUTData" in item:
                warn_once(
                    frame.meta.emitted_warnings,
                    "rwvm_lut_unsupported",
                    "A Real World Value Mapping item uses Real World Value LUT "
                    "Data (0040,9212), which Okapy does not implement. This "
                    "mapping is ignored; SUVbw is computed from another mapping "
                    "if one is available.",
                )
            continue

        # Manual priority: SUVbw, then any other SUV type, then Bq/ml.
        if units == "GML" and suv_type == "BW":
            rank = 0
        elif suv_type is not None:
            rank = 1
        else:
            rank = 2

        candidates.append((rank, units, suv_type, item))

    if not candidates:
        return None

    rank, units, suv_type, item = min(candidates, key=lambda c: c[0])
    return units, suv_type, item


def _units_from_measurement_code(item) -> tuple[str, str | None] | None:
    sequence = getattr(item, "MeasurementUnitsCodeSequence", None)

    if sequence is None or len(sequence) == 0:
        return None

    code_item = sequence[0]

    for name in ("CodeValue", "LongCodeValue", "URNCodeValue"):
        code = get_optional_str(code_item, name)

        if code is None:
            continue

        mapped = UNIT_CODES.get(code.lower())

        if mapped is not None:
            return mapped

    return None


# --------------------------------------------------------------------------
# Dose correction
# --------------------------------------------------------------------------


def _dose_at_reference_time(frame: _Frame) -> float:
    ds = frame.dataset
    meta = frame.meta

    rph = radiopharmaceutical_item(ds)
    dose_bq = radionuclide_total_dose_bq(rph)
    half_life_s = radionuclide_half_life_s(rph)

    reference_dt = _decay_correction_reference_datetime(frame, half_life_s)

    return decay_corrected_dose_bq(
        dose_bq=dose_bq,
        half_life_s=half_life_s,
        reference_dt=reference_dt,
        administration_dt=administration_datetime(
            rph,
            reference_dt=reference_dt,
            half_life_s=half_life_s,
            emitted_warnings=meta.emitted_warnings,
            # Radiopharmaceutical Start Time (0018,1072) is always absent in
            # this SOP class.
            allow_start_time=False,
        ),
    )


def _decay_correction_reference_datetime(
    frame: _Frame, half_life_s: float
) -> datetime:
    meta = frame.meta
    decay_corrected = (meta.decay_corrected or "").upper()

    if decay_corrected not in {"YES", "NO"}:
        raise SUVComputationError(
            "DecayCorrected (0018,9758) must be YES or NO, got "
            f"{decay_corrected or 'absent'!r}."
        )

    if decay_corrected == "YES":
        if meta.decay_correction_datetime is None:
            raise SUVComputationError(
                "DecayCorrected (0018,9758) is YES but Decay Correction DateTime "
                "(0018,9701) is absent; the decay-correction reference datetime "
                "is unknown and SUVbw cannot be computed."
            )

        return parse_dicom_datetime(meta.decay_correction_datetime)

    content = _functional_group(frame.dataset, frame.index, "FrameContentSequence")

    if content is None:
        raise SUVComputationError(
            "DecayCorrected (0018,9758) is NO but the Frame Content Sequence "
            "(0020,9111) is absent, so the measurement time is unknown."
        )

    # Frame Reference DateTime is, by definition, the time at which the average
    # activity occurred.
    frame_reference_dt = get_optional_str(content, "FrameReferenceDateTime")

    if frame_reference_dt is not None:
        return parse_dicom_datetime(frame_reference_dt)

    frame_acquisition_dt = get_optional_str(content, "FrameAcquisitionDateTime")
    frame_duration_ms = get_optional_float(content, "FrameAcquisitionDuration")

    if (
        frame_acquisition_dt is None
        or frame_duration_ms is None
        or frame_duration_ms < 0
    ):
        raise SUVComputationError(
            "DecayCorrected (0018,9758) is NO and Frame Reference DateTime "
            "(0018,9151) is absent; Frame Acquisition DateTime (0018,9074) and "
            "a non-negative Frame Acquisition Duration (0018,9220) are then "
            "required to determine the measurement time, but at least one is "
            "missing. SUVbw cannot be computed."
        )

    warn_once(
        meta.emitted_warnings,
        "frame_reference_datetime_absent",
        "Frame Reference DateTime (0018,9151) is absent on a non "
        "decay-corrected image. The measurement time is back-computed as Frame "
        "Acquisition DateTime (0018,9074) plus the time of average activity "
        "within the Frame Acquisition Duration (0018,9220).",
    )

    # Frame Acquisition Duration is stored in ms.
    tave_s = average_count_rate_time_s(frame_duration_ms / 1000.0, half_life_s)

    return parse_dicom_datetime(frame_acquisition_dt) + timedelta(seconds=tave_s)
