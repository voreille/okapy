from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path
import logging
from typing import Any

import pydicom
from pydicom.errors import InvalidDicomError

from okapy.dicom.models import (
    ALLOWED_MODALITIES,
    IMAGE_MODALITIES,
    DicomFileRecord,
)

logger = logging.getLogger(__name__)


class DicomWalker:
    """Find DICOM files relevant for image/mask conversion.

    The walker only discovers and lightly classifies files.
    It does not build studies, resolve references, or sort slices fully.

    Parameters
    ----------
    recursive:
        Whether to recursively search directories.
    include_hidden:
        Whether to include files or directories whose path contains hidden parts.
    extensions:
        Optional file extensions to keep. If None, every file is attempted.
    allowed_modalities:
        DICOM modalities to keep.
    stop_before_pixels:
        Passed to pydicom.dcmread. Usually True for lightweight discovery.
    force:
        Passed to pydicom.dcmread.
    additional_dicom_tags:
        Extra DICOM keywords to read and store in DicomFileRecord.extra_dicom_tags.
        These are useful for carrying metadata to the final feature table.
    """

    BASE_TAGS = [
        "SOPClassUID",
        "Modality",
        "PatientID",
        "StudyInstanceUID",
        "SeriesInstanceUID",
        "SOPInstanceUID",
        "InstanceNumber",
        "SeriesDescription",
        "ImagePositionPatient",
        "ImageOrientationPatient",
        "ReferencedSeriesSequence",
        # RTSTRUCT references are nested in this sequence.
        "ReferencedFrameOfReferenceSequence",
    ]

    def __init__(
        self,
        *,
        recursive: bool = True,
        include_hidden: bool = False,
        extensions: Iterable[str] | None = None,
        allowed_modalities: Iterable[str] = ALLOWED_MODALITIES,
        stop_before_pixels: bool = True,
        force: bool = False,
        additional_dicom_tags: Iterable[str] | None = None,
    ) -> None:
        self.recursive = recursive
        self.include_hidden = include_hidden
        self.extensions = (
            {ext.lower() for ext in extensions} if extensions is not None else None
        )
        self.allowed_modalities = {m.upper() for m in allowed_modalities}
        self.stop_before_pixels = stop_before_pixels
        self.force = force
        self.additional_dicom_tags = tuple(additional_dicom_tags or ())
        self.specific_tags = _unique_preserve_order(
            [*self.BASE_TAGS, *self.additional_dicom_tags]
        )

    def iter_paths(self, root: Path | str | Iterable[Path | str]) -> Iterator[Path]:
        """Yield candidate file paths from one root or several roots."""
        if isinstance(root, (str, Path)):
            yield from self._iter_paths_one(Path(root))
            return

        for item in root:
            yield from self._iter_paths_one(Path(item))

    def iter_records(
        self,
        root: Path | str | Iterable[Path | str],
    ) -> Iterator[DicomFileRecord]:
        """Yield DicomFileRecord objects for valid and supported DICOM files."""
        for path in self.iter_paths(root):
            record = self._read_record(path)
            if record is not None:
                yield record

    def collect_records(
        self,
        root: Path | str | Iterable[Path | str],
    ) -> list[DicomFileRecord]:
        """Collect all records into a list."""
        return list(self.iter_records(root))

    def _iter_paths_one(self, root: Path) -> Iterator[Path]:
        if root.is_file():
            if self._accept_path(root):
                yield root
            return

        if not root.exists():
            raise FileNotFoundError(f"DICOM root does not exist: {root}")

        pattern = "**/*" if self.recursive else "*"

        for path in root.glob(pattern):
            if path.is_file() and self._accept_path(path):
                yield path

    def _accept_path(self, path: Path) -> bool:
        if not self.include_hidden and _is_hidden(path):
            return False

        if self.extensions is None:
            return True

        return path.suffix.lower() in self.extensions

    def _read_record(self, path: Path) -> DicomFileRecord | None:
        try:
            ds = pydicom.dcmread(
                str(path),
                stop_before_pixels=self.stop_before_pixels,
                force=self.force,
                specific_tags=self.specific_tags,
            )
        except (InvalidDicomError, OSError, EOFError) as exc:
            logger.debug("Skipping non-DICOM file %s: %s", path, exc)
            return None

        modality = _get_str(ds, "Modality")

        if modality is None:
            logger.debug("Skipping DICOM without Modality, likely DICOMDIR: %s", path)
            return None

        modality = modality.upper()

        if modality not in self.allowed_modalities:
            logger.debug("Skipping unsupported DICOM modality %s: %s", modality, path)
            return None

        # For image modalities, require at least a series UID.
        # RTSTRUCT and SEG are allowed even without image geometry.
        if modality in IMAGE_MODALITIES and not hasattr(ds, "SeriesInstanceUID"):
            logger.debug("Skipping image without SeriesInstanceUID: %s", path)
            return None

        try:
            image_position_patient = _get_float_tuple3(ds, "ImagePositionPatient")
            image_orientation_patient = _get_float_tuple6(ds, "ImageOrientationPatient")
        except ValueError as exc:
            logger.warning(
                "Skipping DICOM file with invalid geometry %s: %s", path, exc
            )
            return None

        return DicomFileRecord(
            path=path.resolve(),
            sop_class_uid=_get_str(ds, "SOPClassUID"),
            modality=modality,
            patient_id=_get_str(ds, "PatientID"),
            study_instance_uid=_get_str(ds, "StudyInstanceUID"),
            series_instance_uid=_get_str(ds, "SeriesInstanceUID"),
            sop_instance_uid=_get_str(ds, "SOPInstanceUID"),
            instance_number=_get_int(ds, "InstanceNumber"),
            series_description=_get_str(ds, "SeriesDescription"),
            image_position_patient=image_position_patient,
            image_orientation_patient=image_orientation_patient,
            referenced_series_uids=_get_referenced_series_uids(ds),
            extra_dicom_tags=self._get_extra_dicom_tags(ds),
        )

    def _get_extra_dicom_tags(self, ds) -> dict[str, Any]:
        """Return explicitly requested tags for later reporting/export.

        Standard/base tags are still included when requested. This is intentional:
        if a tag is listed in additional_dicom_tags, the caller likely wants it in
        the final feature table even if Okapy also stores it as a first-class field.
        """
        return {
            tag: _get_dicom_value(ds, tag)
            for tag in self.additional_dicom_tags
            if hasattr(ds, tag)
        }


def _unique_preserve_order(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _get_dicom_value(ds, name: str) -> Any:
    """Extract a DICOM value in a CSV/JSON-friendly form.

    This is deliberately conservative: simple scalar values are preserved,
    bytes are decoded, multi-values become lists, and complex DICOM objects are
    converted to strings. Sequence-valued tags should usually be handled by
    dedicated code instead of being requested as extra tags.
    """
    value = getattr(ds, name, None)

    if value is None:
        return None

    if isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")

    # pydicom MultiValue behaves enough like a list for this conversion.
    if isinstance(value, (list, tuple)):
        return [_dicom_scalar_to_python(v) for v in value]

    return _dicom_scalar_to_python(value)


def _dicom_scalar_to_python(value) -> Any:
    if value is None:
        return None

    if isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")

    # pydicom DSfloat/IS can generally be converted to float/int, but keeping
    # str is safer for metadata export because it avoids surprises with VRs.
    return str(value)


def _get_str(ds, name: str) -> str | None:
    value = getattr(ds, name, None)
    if value is None:
        return None

    value = str(value).strip()
    return value or None


def _get_int(ds, name: str) -> int | None:
    value = getattr(ds, name, None)
    if value is None:
        return None

    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _get_float_tuple3(ds, name: str) -> tuple[float, float, float] | None:
    value = getattr(ds, name, None)
    if value is None:
        return None

    try:
        result = tuple(float(x) for x in value)
    except (TypeError, ValueError):
        return None

    if len(result) != 3:
        raise ValueError(
            f"Expected DICOM tag {name} to have length 3, got {len(result)}."
        )

    return result[0], result[1], result[2]


def _get_float_tuple6(
    ds,
    name: str,
) -> tuple[float, float, float, float, float, float] | None:
    value = getattr(ds, name, None)
    if value is None:
        return None

    try:
        result = tuple(float(x) for x in value)
    except (TypeError, ValueError):
        return None

    if len(result) != 6:
        raise ValueError(
            f"Expected DICOM tag {name} to have length 6, got {len(result)}."
        )

    return result[0], result[1], result[2], result[3], result[4], result[5]


def _get_referenced_series_uids(ds) -> tuple[str, ...]:
    """Extract referenced SeriesInstanceUID values from common DICOM locations.

    SEG commonly exposes ReferencedSeriesSequence directly.
    RTSTRUCT usually nests it under:
      ReferencedFrameOfReferenceSequence
        -> RTReferencedStudySequence
        -> RTReferencedSeriesSequence
    """
    refs: list[str] = []

    # Common for SEG and some other objects.
    for item in getattr(ds, "ReferencedSeriesSequence", []):
        uid = getattr(item, "SeriesInstanceUID", None)
        if uid is not None:
            refs.append(str(uid))

    # Common for RTSTRUCT.
    for frame_item in getattr(ds, "ReferencedFrameOfReferenceSequence", []):
        for study_item in getattr(frame_item, "RTReferencedStudySequence", []):
            for series_item in getattr(study_item, "RTReferencedSeriesSequence", []):
                uid = getattr(series_item, "SeriesInstanceUID", None)
                if uid is not None:
                    refs.append(str(uid))

    return tuple(dict.fromkeys(refs))


def _is_hidden(path: Path) -> bool:
    return any(part.startswith(".") for part in path.parts)
