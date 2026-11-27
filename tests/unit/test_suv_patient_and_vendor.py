"""Patient size units, vendor detection and private-tag access in SUV conversion.

* PatientSize (0010,1020) is in metres, but values in cm are common; a value
  above 3 can only be cm and must be converted (with a warning) rather than
  feeding a 175 m patient into the lean-body-mass / BSA formulas.
* Vendor detection must match GE vendor strings only, not any manufacturer
  whose name happens to contain "GE".
* Vendor private scan-start / scale-factor tags are only meaningful inside the
  private block reserved by the matching Private Creator (PS3.5 7.8.1); the
  bare element number may belong to a different vendor.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.sequence import Sequence
from pydicom.uid import ExplicitVRLittleEndian

from okapy.dicom.conversion.suv import PETSUVConverter
from okapy.dicom.conversion.suv_common import (
    SUVComputationError,
    normalize_manufacturer,
    patient_size_m,
    warn_unrecognized_manufacturer,
)
from okapy.dicom.models import DicomFileRecord, DicomSeries

PET_IMAGE_STORAGE = "1.2.840.10008.5.1.4.1.1.128"
SERIES_UID = "1.2.826.0.1.3680043.8.498.777"
STUDY_UID = "1.2.826.0.1.3680043.8.498.77"

HALF_LIFE_S = 6588.0  # 18F
DOSE_BQ = 370e6
WEIGHT_KG = 70.0
FRAME_DURATION_MS = 120000
STORED = np.array([[1000, 2000], [3000, 4000]], dtype=np.uint16)

GE_CREATOR = "GEMS_PETD_01"
SIEMENS_CREATOR = "SIEMENS MED PT"
PHILIPS_CREATOR = "Philips PET Private Group"


def _write_pet_series(
    directory: Path,
    *,
    manufacturer: str = "SIEMENS",
    units: str = "BQML",
    suv_type: str | None = None,
    patient_size: float | None = 1.75,
    patient_sex: str = "M",
    acquisition_time: str = "110000",
    series_time: str | None = None,
    start_time: str = "100000",
    rescale_slope: float = 1.0,
    customize: Callable[[Dataset], None] | None = None,
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

        ds.Units = units
        if suv_type is not None:
            ds.SUVType = suv_type
        ds.DecayCorrection = "START"
        ds.PatientWeight = WEIGHT_KG
        if patient_size is not None:
            ds.PatientSize = patient_size
        ds.PatientSex = patient_sex
        ds.SeriesDate = "20240315"
        ds.SeriesTime = series_time if series_time is not None else acquisition_time
        ds.AcquisitionDate = "20240315"
        ds.AcquisitionTime = acquisition_time
        ds.ActualFrameDuration = FRAME_DURATION_MS
        ds.FrameReferenceTime = 0

        rph = Dataset()
        rph.RadionuclideTotalDose = DOSE_BQ
        rph.RadionuclideHalfLife = HALF_LIFE_S
        rph.RadiopharmaceuticalStartTime = start_time
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
        ds.RescaleSlope = rescale_slope
        ds.RescaleIntercept = 0.0
        ds.PixelData = (STORED * (index + 1)).astype(np.uint16).tobytes()

        if customize is not None:
            customize(ds)

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
    return sitk.GetArrayFromImage(sitk.ReadImage(str(volume.path)))


def _stored_volume(slope: float = 1.0) -> np.ndarray:
    return np.stack([STORED * (i + 1) for i in range(2)]).astype(np.float64) * slope


def _bqml_suvbw(decay_s: float, activity: np.ndarray | None = None) -> np.ndarray:
    activity = _stored_volume() if activity is None else activity
    corrected_dose = DOSE_BQ * 2 ** (-decay_s / HALF_LIFE_S)
    return activity * WEIGHT_KG * 1000.0 / corrected_dose


def _tave_s(frame_s: float) -> float:
    lam = math.log(2.0) / HALF_LIFE_S
    x = lam * frame_s
    return math.log(x / (1.0 - math.exp(-x))) / lam


# --------------------------------------------------------------------------
# PatientSize in cm
# --------------------------------------------------------------------------


def _lbm_james_male_kg(weight_kg: float, height_m: float) -> float:
    return 1.10 * weight_kg - 128.0 * (weight_kg / (height_m * 100.0)) ** 2


def _bsa_dubois_m2(weight_kg: float, height_m: float) -> float:
    return 0.007184 * weight_kg**0.425 * (height_m * 100.0) ** 0.725


@pytest.mark.parametrize("size", [1.75, 175.0])
def test_lbm_suv_uses_height_in_metres_even_when_patient_size_is_in_cm(tmp_path, size):
    slope = 1e-4
    series = _write_pet_series(
        tmp_path / "dcm",
        units="GML",
        suv_type="LBM",
        patient_size=size,
        rescale_slope=slope,
    )

    suvbw = _convert(series, tmp_path)

    expected = _stored_volume(slope) * WEIGHT_KG / _lbm_james_male_kg(WEIGHT_KG, 1.75)
    np.testing.assert_allclose(suvbw, expected, rtol=1e-5)


@pytest.mark.parametrize("size", [1.75, 175.0])
def test_bsa_suv_uses_height_in_metres_even_when_patient_size_is_in_cm(tmp_path, size):
    slope = 1e-3
    series = _write_pet_series(
        tmp_path / "dcm",
        units="CM2ML",
        suv_type="BSA",
        patient_size=size,
        rescale_slope=slope,
    )

    suvbw = _convert(series, tmp_path)

    bsa_cm2 = _bsa_dubois_m2(WEIGHT_KG, 1.75) * 10_000.0
    expected = _stored_volume(slope) * WEIGHT_KG * 1000.0 / bsa_cm2
    np.testing.assert_allclose(suvbw, expected, rtol=1e-5)


def test_patient_size_in_cm_logs_a_warning_during_conversion(tmp_path, caplog):
    series = _write_pet_series(
        tmp_path / "dcm",
        units="GML",
        suv_type="LBM",
        patient_size=175.0,
        rescale_slope=1e-4,
    )

    with caplog.at_level(logging.WARNING):
        _convert(series, tmp_path)

    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("PatientSize" in m for m in warnings), warnings


def test_patient_size_in_metres_does_not_warn(tmp_path, caplog):
    series = _write_pet_series(
        tmp_path / "dcm",
        units="GML",
        suv_type="LBM",
        patient_size=1.75,
        rescale_slope=1e-4,
    )

    with caplog.at_level(logging.WARNING):
        _convert(series, tmp_path)

    assert "PatientSize" not in caplog.text


@pytest.mark.parametrize(
    ("size", "expected_m"),
    [(175.0, 1.75), (160.5, 1.605), (1.75, 1.75), (3.0, 3.0)],
)
def test_patient_size_m_normalizes_cm_to_metres(size, expected_m):
    assert patient_size_m(size) == pytest.approx(expected_m)


@pytest.mark.parametrize("size", [None, 0.0, -1.7])
def test_patient_size_m_still_rejects_missing_or_non_positive(size):
    with pytest.raises(SUVComputationError, match="PatientSize"):
        patient_size_m(size)


def test_bqml_suvbw_does_not_depend_on_patient_size(tmp_path):
    """Regression: SUVbw from BQML never needs the height, cm or not."""
    series = _write_pet_series(tmp_path / "dcm", patient_size=175.0)

    suvbw = _convert(series, tmp_path)

    np.testing.assert_allclose(suvbw, _bqml_suvbw(3600.0), rtol=1e-5)


# --------------------------------------------------------------------------
# Vendor detection
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "manufacturer",
    [
        "GE MEDICAL SYSTEMS",
        "GE Medical Systems",
        "ge medical systems",
        "GEMS",
        "gems",
        "GE HEALTHCARE",
        "GE Healthcare",
        "  GE MEDICAL SYSTEMS  ",
    ],
)
def test_ge_vendor_strings_are_detected_as_ge(manufacturer):
    assert normalize_manufacturer(manufacturer) == "GE"


@pytest.mark.parametrize(
    "manufacturer",
    [
        "Imagen",
        "SEGAMI",
        "Segami Corporation",
        "Agfa-Gevaert",
        "MEGAMED",
        "Gemini Medical",
        "GENESIS IMAGING",
        "Georgetown Research Scanner",
    ],
)
def test_names_merely_containing_ge_are_not_ge(manufacturer):
    assert normalize_manufacturer(manufacturer) != "GE"


@pytest.mark.parametrize(
    ("manufacturer", "expected"),
    [
        ("SIEMENS", "SIEMENS"),
        ("Siemens Healthineers", "SIEMENS"),
        ("Philips Medical Systems", "PHILIPS"),
        ("PHILIPS HEALTHCARE GERMANY", "PHILIPS"),
        ("UNITED IMAGING", "UNKNOWN"),
        ("", "UNKNOWN"),
        (None, "UNKNOWN"),
    ],
)
def test_other_vendors_are_unaffected(manufacturer, expected):
    assert normalize_manufacturer(manufacturer) == expected


def test_non_ge_vendor_containing_ge_gets_unrecognized_warning(caplog):
    with caplog.at_level(logging.WARNING):
        warn_unrecognized_manufacturer("Imagen", emitted_warnings=set())

    assert "not recognized" in caplog.text


# --------------------------------------------------------------------------
# Private scan-start tags and Private Creators
# --------------------------------------------------------------------------
#
# Scenario shared by the scan-start tests: injection 10:00:00, the private
# scan-start tag says 11:00:00 (3600 s uptake), but Acquisition (11:10:00) and
# Series (10:55:00) differ, so without the private tag the manual's fallback
# applies:
#   GE:      acquisition - FrameReferenceTime           -> 4200 s
#   others:  acquisition + T_ave - FrameReferenceTime   -> 4200 s + T_ave

PRIVATE_SCAN_START = "20240315110000.00"
SCENARIO = {"acquisition_time": "111000", "series_time": "105500"}


def _private(group: int, creator: str | None, element: int, vr: str, value):
    """Return a customizer writing ``value`` at (group, element).

    ``element`` is the full element number, e.g. 0x100D; the creator (if any)
    is written into the block that element belongs to.
    """

    def customize(ds: Dataset) -> None:
        if creator is not None:
            ds.add_new((group, element >> 8), "LO", creator)
        ds.add_new((group, element), vr, value)

    return customize


def _ge_tag(creator: str | None, element: int = 0x100D):
    return _private(0x0009, creator, element, "DT", PRIVATE_SCAN_START)


def _siemens_tag(creator: str | None, element: int = 0x1022):
    return _private(0x0071, creator, element, "DT", PRIVATE_SCAN_START)


def test_ge_private_scan_start_is_used_with_its_private_creator(tmp_path):
    series = _write_pet_series(
        tmp_path / "dcm",
        manufacturer="GE MEDICAL SYSTEMS",
        customize=_ge_tag(GE_CREATOR),
        **SCENARIO,
    )

    np.testing.assert_allclose(
        _convert(series, tmp_path), _bqml_suvbw(3600.0), rtol=1e-5
    )


@pytest.mark.parametrize("creator", [None, "SOME_OTHER_VENDOR", "GEMS_IDEN_01"])
def test_ge_private_scan_start_ignored_without_matching_creator(tmp_path, creator):
    series = _write_pet_series(
        tmp_path / "dcm",
        manufacturer="GE MEDICAL SYSTEMS",
        customize=_ge_tag(creator),
        **SCENARIO,
    )

    np.testing.assert_allclose(
        _convert(series, tmp_path), _bqml_suvbw(4200.0), rtol=1e-5
    )


def test_ge_private_scan_start_found_in_another_private_block(tmp_path):
    """The creator reserves (0009,11xx) here; the element must be looked up
    in that block, and the foreign (0009,100D) ignored."""

    def customize(ds: Dataset) -> None:
        ds.add_new((0x0009, 0x0010), "LO", "SOME_OTHER_VENDOR")
        ds.add_new((0x0009, 0x100D), "DT", "20240315105000.00")
        ds.add_new((0x0009, 0x0011), "LO", GE_CREATOR)
        ds.add_new((0x0009, 0x110D), "DT", PRIVATE_SCAN_START)

    series = _write_pet_series(
        tmp_path / "dcm",
        manufacturer="GE MEDICAL SYSTEMS",
        customize=customize,
        **SCENARIO,
    )

    np.testing.assert_allclose(
        _convert(series, tmp_path), _bqml_suvbw(3600.0), rtol=1e-5
    )


def test_siemens_private_scan_start_is_used_with_its_private_creator(tmp_path):
    series = _write_pet_series(
        tmp_path / "dcm",
        manufacturer="SIEMENS",
        customize=_siemens_tag(SIEMENS_CREATOR),
        **SCENARIO,
    )

    np.testing.assert_allclose(
        _convert(series, tmp_path), _bqml_suvbw(3600.0), rtol=1e-5
    )


@pytest.mark.parametrize("creator", [None, "SOME_OTHER_VENDOR"])
def test_siemens_private_scan_start_ignored_without_matching_creator(tmp_path, creator):
    series = _write_pet_series(
        tmp_path / "dcm",
        manufacturer="SIEMENS",
        customize=_siemens_tag(creator),
        **SCENARIO,
    )

    decay_s = 4200.0 + _tave_s(FRAME_DURATION_MS / 1000.0)
    np.testing.assert_allclose(
        _convert(series, tmp_path), _bqml_suvbw(decay_s), rtol=1e-5
    )


def test_non_ge_vendor_containing_ge_does_not_use_ge_private_tag(tmp_path):
    """'Imagen' must not be routed to the GE private tag or GE fallback."""
    series = _write_pet_series(
        tmp_path / "dcm",
        manufacturer="Imagen",
        customize=_ge_tag(GE_CREATOR),
        **SCENARIO,
    )

    decay_s = 4200.0 + _tave_s(FRAME_DURATION_MS / 1000.0)
    np.testing.assert_allclose(
        _convert(series, tmp_path), _bqml_suvbw(decay_s), rtol=1e-5
    )


def _philips_factor(creator: str | None, value: float = 0.5):
    return _private(0x7053, creator, 0x1009, "DS", value)


def test_philips_activity_scale_factor_is_used_with_its_private_creator(tmp_path):
    series = _write_pet_series(
        tmp_path / "dcm",
        manufacturer="Philips Medical Systems",
        units="CNTS",
        customize=_philips_factor(PHILIPS_CREATOR),
    )

    expected = _bqml_suvbw(3600.0, activity=_stored_volume() * 0.5)
    np.testing.assert_allclose(_convert(series, tmp_path), expected, rtol=1e-5)


@pytest.mark.parametrize("creator", [None, "SOME_OTHER_VENDOR"])
def test_philips_scale_factor_ignored_without_matching_creator(tmp_path, creator):
    series = _write_pet_series(
        tmp_path / "dcm",
        manufacturer="Philips Medical Systems",
        units="CNTS",
        customize=_philips_factor(creator),
    )

    with pytest.raises(SUVComputationError, match="CNTS|7053"):
        _convert(series, tmp_path)
