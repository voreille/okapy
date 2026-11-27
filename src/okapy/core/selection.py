from __future__ import annotations

from collections.abc import Mapping
from typing import TypeVar

T = TypeVar("T")


def modality_from_key(key: str) -> str:
    """Return the modality part of a modality key (``"PT_suv"`` -> ``"PT"``)."""

    return key.split("_", maxsplit=1)[0]


def select(mapping: Mapping[str, T], key: str, default: T) -> T:
    """Look up ``key``, then its modality (see ``modality_from_key``).

    Returns ``default`` when neither is in ``mapping``. Callers decide what the
    default is (usually ``mapping.get("default")``), so ``default`` is never
    looked up here.
    """

    if key in mapping:
        return mapping[key]

    modality = modality_from_key(key)
    if modality in mapping:
        return mapping[modality]

    return default
