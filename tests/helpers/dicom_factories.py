"""Synthetic DICOM builders for conversion / walker unit tests.

Everything here is built with pydicom (and highdicom for SEG) and written to a
caller-supplied directory, usually ``tmp_path``. Pixel arrays are given in
DICOM (rows, columns) = (y, x) order.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.sequence import Sequence as DicomSequence
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from okapy.dicom.models import DicomSeries
from okapy.dicom.walker import DicomWalker

CT_IMAGE_STORAGE = "1.2.840.10008.5.1.4.1.1.2"
MR_IMAGE_STORAGE = "1.2.840.10008.5.1.4.1.1.4"
PET_IMAGE_STORAGE = "1.2.840.10008.5.1.4.1.1.128"
ENHANCED_PET_IMAGE_STORAGE = "1.2.840.10008.5.1.4.1.1.130"
DX_IMAGE_STORAGE = "1.2.840.10008.5.1.4.1.1.1.1"
CR_IMAGE_STORAGE = "1.2.840.10008.5.1.4.1.1.1"
RTSTRUCT_STORAGE = "1.2.840.10008.5.1.4.1.1.481.3"

SOP_CLASS_BY_MODALITY = {
    "CT": CT_IMAGE_STORAGE,
    "MR": MR_IMAGE_STORAGE,
    "PT": PET_IMAGE_STORAGE,
    "DX": DX_IMAGE_STORAGE,
    "CR": CR_IMAGE_STORAGE,
}

IDENTITY_ORIENTATION = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)

DEFAULT_PATIENT_ID = "PAT1"
DEFAULT_STUDY_UID = "1.2.826.0.1.3680043.8.498.1"
DEFAULT_FRAME_OF_REFERENCE_UID = "1.2.826.0.1.3680043.8.498.2"


# --------------------------------------------------------------------------
# Generic helpers
# --------------------------------------------------------------------------


def _file_dataset(sop_class_uid: str, sop_instance_uid: str) -> FileDataset:
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = sop_class_uid
    meta.MediaStorageSOPInstanceUID = sop_instance_uid
    meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = FileDataset("synthetic", {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID = sop_class_uid
    ds.SOPInstanceUID = sop_instance_uid
    return ds


def write_dataset(ds: Dataset, path: Path) -> Path:
    """Write ``ds`` as a DICOM Part 10 file and return its path."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.save_as(str(path), enforce_file_format=True)
    return path


def normal_of(orientation: Sequence[float]) -> np.ndarray:
    orientation = np.asarray(orientation, dtype=float)
    return np.cross(orientation[:3], orientation[3:])


def positions_along_normal(
    offsets: Iterable[float],
    *,
    orientation: Sequence[float] = IDENTITY_ORIENTATION,
    origin: Sequence[float] = (0.0, 0.0, 0.0),
) -> list[tuple[float, float, float]]:
    """ImagePositionPatient values at ``offsets`` mm along the slice normal."""

    normal = normal_of(orientation)
    origin = np.asarray(origin, dtype=float)
    return [tuple(float(x) for x in origin + float(o) * normal) for o in offsets]


def constant_pixels(value: int, *, rows: int = 4, columns: int = 5) -> np.ndarray:
    return np.full((rows, columns), value, dtype=np.uint16)


# --------------------------------------------------------------------------
# Single-frame image slices (CT / MR / PT / DX ...)
# --------------------------------------------------------------------------


def image_slice(
    *,
    modality: str = "CT",
    pixels: np.ndarray,
    position: Sequence[float] | None = (0.0, 0.0, 0.0),
    orientation: Sequence[float] | None = IDENTITY_ORIENTATION,
    pixel_spacing: Sequence[float] | None = (1.0, 1.0),
    slice_thickness: float | None = 1.0,
    rescale: tuple[float, float] | None = (1.0, 0.0),
    series_instance_uid: str,
    study_instance_uid: str = DEFAULT_STUDY_UID,
    patient_id: str = DEFAULT_PATIENT_ID,
    sop_instance_uid: str | None = None,
    sop_class_uid: str | None = None,
    instance_number: int = 1,
    series_description: str | None = None,
    frame_of_reference_uid: str = DEFAULT_FRAME_OF_REFERENCE_UID,
    extra: dict[str, object] | None = None,
) -> FileDataset:
    """One single-frame image slice.

    ``pixels`` is an unsigned 16-bit (rows, columns) array. ``position``,
    ``orientation``, ``pixel_spacing``, ``slice_thickness`` and ``rescale``
    can be ``None`` to leave the corresponding attributes out.
    """

    sop_instance_uid = sop_instance_uid or generate_uid()
    sop_class_uid = sop_class_uid or SOP_CLASS_BY_MODALITY.get(
        modality, CT_IMAGE_STORAGE
    )
    ds = _file_dataset(sop_class_uid, sop_instance_uid)

    ds.Modality = modality
    ds.PatientID = patient_id
    ds.PatientName = "Synthetic^Patient"
    ds.PatientBirthDate = "19700101"
    ds.PatientSex = "O"
    ds.StudyDate = "20240101"
    ds.StudyTime = "100000"
    ds.StudyID = "1"
    ds.AccessionNumber = "1"
    ds.ReferringPhysicianName = ""
    ds.StudyInstanceUID = study_instance_uid
    ds.SeriesInstanceUID = series_instance_uid
    ds.FrameOfReferenceUID = frame_of_reference_uid
    ds.InstanceNumber = int(instance_number)
    ds.SeriesNumber = 1
    if series_description is not None:
        ds.SeriesDescription = series_description

    if position is not None:
        ds.ImagePositionPatient = [float(x) for x in position]
    if orientation is not None:
        ds.ImageOrientationPatient = [float(x) for x in orientation]
    if pixel_spacing is not None:
        ds.PixelSpacing = [float(x) for x in pixel_spacing]
    if slice_thickness is not None:
        ds.SliceThickness = float(slice_thickness)
    if rescale is not None:
        ds.RescaleSlope = float(rescale[0])
        ds.RescaleIntercept = float(rescale[1])

    pixels = np.asarray(pixels, dtype=np.uint16)
    ds.Rows, ds.Columns = (int(n) for n in pixels.shape)
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.PixelData = pixels.tobytes()

    for keyword, value in (extra or {}).items():
        setattr(ds, keyword, value)

    return ds


def pet_gml_attributes() -> dict[str, object]:
    """Attributes making a classic PT slice already SUVbw (Units=GML, BW)."""

    return {
        "Units": "GML",
        "SUVType": "BW",
        "DecayCorrection": "START",
        "PatientWeight": 70.0,
        "CorrectedImage": ["ATTN", "DECY"],
    }


def write_image_series(
    directory: Path,
    *,
    positions: Sequence[Sequence[float]] | Sequence[float],
    values: Sequence[int] | None = None,
    pixels: Sequence[np.ndarray] | None = None,
    modality: str = "CT",
    series_instance_uid: str | None = None,
    orientation: Sequence[float] | None = IDENTITY_ORIENTATION,
    rows: int = 4,
    columns: int = 5,
    prefix: str = "slice",
    **slice_kwargs,
) -> tuple[str, list[Path]]:
    """Write one image series and return ``(series_uid, paths)``.

    ``positions`` are either full ImagePositionPatient triples or scalar
    offsets along the slice normal of ``orientation``. Each slice gets either
    the constant stored value ``values[i]`` or the array ``pixels[i]``.
    Files are written in reverse order so the converter has to sort them.
    """

    series_instance_uid = series_instance_uid or generate_uid()
    positions = list(positions)

    if positions and np.ndim(positions[0]) == 0:
        positions = positions_along_normal(
            positions, orientation=orientation or IDENTITY_ORIENTATION
        )

    if pixels is None:
        if values is None:
            values = [100 + 10 * i for i in range(len(positions))]
        pixels = [constant_pixels(v, rows=rows, columns=columns) for v in values]

    if modality == "PT":
        extra = {**pet_gml_attributes(), **(slice_kwargs.pop("extra", None) or {})}
        slice_kwargs["extra"] = extra

    directory = Path(directory)
    paths: list[Path] = []

    for index in reversed(range(len(positions))):
        ds = image_slice(
            modality=modality,
            pixels=pixels[index],
            position=positions[index],
            orientation=orientation,
            series_instance_uid=series_instance_uid,
            instance_number=index + 1,
            **slice_kwargs,
        )
        paths.append(write_dataset(ds, directory / f"{prefix}_{index:03d}.dcm"))

    return series_instance_uid, list(reversed(paths))


def load_series(paths: Iterable[Path]) -> DicomSeries:
    """Build a :class:`DicomSeries` from written files via the real walker."""

    records = DicomWalker().collect_records([Path(p) for p in paths])
    assert records, "no readable DICOM records"
    return DicomSeries(
        series_instance_uid=records[0].series_instance_uid,
        modality=records[0].modality,
        records=tuple(records),
    )


# --------------------------------------------------------------------------
# Enhanced PET (multi-frame)
# --------------------------------------------------------------------------


def enhanced_pet_dataset(
    *,
    frame_positions: Sequence[Sequence[float]],
    frame_values: Sequence[int],
    rows: int = 4,
    columns: int = 5,
    orientation: Sequence[float] = IDENTITY_ORIENTATION,
    pixel_spacing: Sequence[float] = (2.0, 2.0),
    slice_thickness: float = 3.0,
    series_instance_uid: str | None = None,
    study_instance_uid: str = DEFAULT_STUDY_UID,
    patient_id: str = DEFAULT_PATIENT_ID,
) -> FileDataset:
    """Enhanced PET instance whose stored values are already SUVbw.

    The Real World Value Mapping uses the ``g/ml{SUVbw}`` unit with slope 1
    and intercept 0, so output voxel == stored value.
    """

    ds = _file_dataset(ENHANCED_PET_IMAGE_STORAGE, generate_uid())
    ds.Modality = "PT"
    ds.PatientID = patient_id
    ds.StudyInstanceUID = study_instance_uid
    ds.SeriesInstanceUID = series_instance_uid or generate_uid()
    ds.FrameOfReferenceUID = DEFAULT_FRAME_OF_REFERENCE_UID
    ds.InstanceNumber = 1
    ds.PatientWeight = 70.0
    ds.DecayCorrected = "YES"
    ds.DecayCorrectionDateTime = "20240101100000"

    n_frames = len(frame_positions)
    ds.NumberOfFrames = n_frames

    shared = Dataset()
    plane_orientation = Dataset()
    plane_orientation.ImageOrientationPatient = [float(x) for x in orientation]
    shared.PlaneOrientationSequence = DicomSequence([plane_orientation])
    measures = Dataset()
    measures.PixelSpacing = [float(x) for x in pixel_spacing]
    measures.SliceThickness = float(slice_thickness)
    shared.PixelMeasuresSequence = DicomSequence([measures])

    unit_code = Dataset()
    unit_code.CodeValue = "g/ml{SUVbw}"
    unit_code.CodingSchemeDesignator = "UCUM"
    unit_code.CodeMeaning = "Standardized Uptake Value body weight"
    mapping = Dataset()
    mapping.MeasurementUnitsCodeSequence = DicomSequence([unit_code])
    mapping.RealWorldValueSlope = 1.0
    mapping.RealWorldValueIntercept = 0.0
    mapping.RealWorldValueFirstValueMapped = 0
    mapping.RealWorldValueLastValueMapped = 65535
    shared.RealWorldValueMappingSequence = DicomSequence([mapping])
    ds.SharedFunctionalGroupsSequence = DicomSequence([shared])

    per_frame = []
    for position in frame_positions:
        group = Dataset()
        plane_position = Dataset()
        plane_position.ImagePositionPatient = [float(x) for x in position]
        group.PlanePositionSequence = DicomSequence([plane_position])
        per_frame.append(group)
    ds.PerFrameFunctionalGroupsSequence = DicomSequence(per_frame)

    frames = np.stack(
        [np.full((rows, columns), v, dtype=np.uint16) for v in frame_values]
    )
    ds.Rows, ds.Columns = rows, columns
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.PixelData = frames.tobytes()

    return ds


# --------------------------------------------------------------------------
# RTSTRUCT
# --------------------------------------------------------------------------


def rtstruct_dataset(
    *,
    rois: Sequence[tuple[int, str, Sequence[np.ndarray]]],
    referenced_series_uid: str | None,
    series_instance_uid: str | None = None,
    study_instance_uid: str = DEFAULT_STUDY_UID,
    patient_id: str = DEFAULT_PATIENT_ID,
) -> FileDataset:
    """RT Structure Set.

    ``rois`` is a list of ``(roi_number, roi_name, contours)``, each contour an
    (N, 3) array of physical points. With ``referenced_series_uid=None`` the
    structure set carries no referenced series at all.
    """

    ds = _file_dataset(RTSTRUCT_STORAGE, generate_uid())
    ds.Modality = "RTSTRUCT"
    ds.PatientID = patient_id
    ds.StudyInstanceUID = study_instance_uid
    ds.SeriesInstanceUID = series_instance_uid or generate_uid()
    ds.InstanceNumber = 1
    ds.StructureSetLabel = "synthetic"

    if referenced_series_uid is not None:
        series_ref = Dataset()
        series_ref.SeriesInstanceUID = referenced_series_uid
        series_ref.ContourImageSequence = DicomSequence([])
        study_ref = Dataset()
        study_ref.ReferencedSOPClassUID = "1.2.840.10008.3.1.2.3.1"
        study_ref.ReferencedSOPInstanceUID = study_instance_uid
        study_ref.RTReferencedSeriesSequence = DicomSequence([series_ref])
        frame_ref = Dataset()
        frame_ref.FrameOfReferenceUID = DEFAULT_FRAME_OF_REFERENCE_UID
        frame_ref.RTReferencedStudySequence = DicomSequence([study_ref])
        ds.ReferencedFrameOfReferenceSequence = DicomSequence([frame_ref])

    structure_set_rois = []
    roi_contours = []

    for roi_number, roi_name, contours in rois:
        roi = Dataset()
        roi.ROINumber = int(roi_number)
        roi.ROIName = roi_name
        roi.ReferencedFrameOfReferenceUID = DEFAULT_FRAME_OF_REFERENCE_UID
        structure_set_rois.append(roi)

        contour_items = []
        for points in contours:
            points = np.asarray(points, dtype=float).reshape(-1, 3)
            item = Dataset()
            item.ContourGeometricType = "CLOSED_PLANAR"
            item.NumberOfContourPoints = int(points.shape[0])
            item.ContourData = [float(x) for x in points.ravel()]
            contour_items.append(item)

        roi_contour = Dataset()
        roi_contour.ReferencedROINumber = int(roi_number)
        roi_contour.ROIDisplayColor = [255, 0, 0]
        roi_contour.ContourSequence = DicomSequence(contour_items)
        roi_contours.append(roi_contour)

    ds.StructureSetROISequence = DicomSequence(structure_set_rois)
    ds.ROIContourSequence = DicomSequence(roi_contours)

    return ds


def square_contour(x0: float, y0: float, x1: float, y1: float, z: float) -> np.ndarray:
    """Closed square contour (4 points) in physical coordinates."""

    return np.asarray(
        [(x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z)],
        dtype=float,
    )


# --------------------------------------------------------------------------
# SEG (highdicom)
# --------------------------------------------------------------------------


def seg_dataset(
    *,
    source_images: Sequence[Dataset],
    mask_zyx: np.ndarray,
    label: str = "Tumor",
    series_instance_uid: str | None = None,
) -> Dataset:
    """Binary DICOM SEG built by highdicom on ``source_images``.

    ``mask_zyx`` has shape (n_source_images, rows, columns) and follows the
    order of ``source_images``.
    """

    import highdicom as hd
    from pydicom.sr.codedict import codes

    description = hd.seg.SegmentDescription(
        segment_number=1,
        segment_label=label,
        segmented_property_category=codes.SCT.MorphologicallyAbnormalStructure,
        segmented_property_type=codes.SCT.Neoplasm,
        algorithm_type=hd.seg.SegmentAlgorithmTypeValues.MANUAL,
    )

    return hd.seg.Segmentation(
        source_images=list(source_images),
        pixel_array=np.asarray(mask_zyx, dtype=np.uint8),
        segmentation_type=hd.seg.SegmentationTypeValues.BINARY,
        segment_descriptions=[description],
        series_instance_uid=series_instance_uid or generate_uid(),
        series_number=100,
        sop_instance_uid=generate_uid(),
        instance_number=1,
        manufacturer="okapy-tests",
        manufacturer_model_name="synthetic",
        software_versions="0",
        device_serial_number="0",
        omit_empty_frames=False,
    )
