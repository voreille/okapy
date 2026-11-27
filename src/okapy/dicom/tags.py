"""DICOM tag helpers that only need pydicom datasets (no SimpleITK).

They are used by :mod:`okapy.dicom.walker`, which must stay light, and are
re-exported by :mod:`okapy.dicom.conversion.utils`.
"""

from __future__ import annotations

from typing import Any


def get_optional_str(ds, name: str) -> str | None:
    """Attribute ``name`` of ``ds`` as a stripped string, None if absent/empty."""

    value = getattr(ds, name, None)

    if value is None:
        return None

    value = str(value).strip()
    return value or None


def to_jsonable(value: Any) -> Any:
    """Convert pydicom/highdicom metadata to simple serializable values.

    Scalars are kept, bytes are decoded, lists/tuples and dicts are converted
    item by item and anything else (including pydicom multi-values) becomes a
    string.
    """

    if value is None:
        return None

    if isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")

    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]

    return str(value)


def referenced_series_uids(
    ds,
    *,
    rtstruct: bool = True,
    derivation: bool = False,
) -> tuple[str, ...]:
    """Unique referenced SeriesInstanceUID values of ``ds``, in order found.

    Always read from ReferencedSeriesSequence (SEG and others). ``rtstruct``
    adds the RTSTRUCT nesting ReferencedFrameOfReferenceSequence ->
    RTReferencedStudySequence -> RTReferencedSeriesSequence; ``derivation``
    adds the SourceImageSequence items of the DerivationImageSequence in the
    per-frame, then shared, functional groups (best effort: they often
    reference SOP instances only).
    """

    items = list(getattr(ds, "ReferencedSeriesSequence", []))

    if rtstruct:
        for frame_item in getattr(ds, "ReferencedFrameOfReferenceSequence", []):
            for study_item in getattr(frame_item, "RTReferencedStudySequence", []):
                items.extend(getattr(study_item, "RTReferencedSeriesSequence", []))

    if derivation:
        for groups_name in (
            "PerFrameFunctionalGroupsSequence",
            "SharedFunctionalGroupsSequence",
        ):
            for group in getattr(ds, groups_name, []):
                for derivation_item in getattr(group, "DerivationImageSequence", []):
                    items.extend(getattr(derivation_item, "SourceImageSequence", []))

    uids = (getattr(item, "SeriesInstanceUID", None) for item in items)
    return tuple(dict.fromkeys(str(uid) for uid in uids if uid is not None))
