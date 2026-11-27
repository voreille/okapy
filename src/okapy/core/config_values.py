"""Parsing helpers for config values shared by every okapy config section."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

#: Keys of ``feature_extraction`` that are options, not selectors (besides the
#: ``common`` backend list).
FEATURE_OPTION_KEYS = frozenset(
    {"continue_on_error", "result_format", "include_backend"}
)

_TRUE_STRINGS = frozenset({"true", "yes", "1", "on"})
_FALSE_STRINGS = frozenset({"false", "no", "0", "off"})


def parse_bool(value: Any, *, key: str) -> bool:
    """Parse a boolean config value strictly.

    Real booleans pass through; the integers ``1``/``0`` and the strings
    ``true/yes/1/on`` and ``false/no/0/off`` (case-insensitive) are accepted.
    Anything else raises ``ValueError`` naming ``key``.
    """

    if isinstance(value, bool):
        return value

    if isinstance(value, int) and value in (0, 1):
        return bool(value)

    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in _TRUE_STRINGS:
            return True
        if lowered in _FALSE_STRINGS:
            return False

    raise ValueError(f"{key!r} must be a boolean (true/false), got {value!r}.")


def get_bool(mapping: Mapping[str, Any], key: str, default: bool) -> bool:
    """:func:`parse_bool` of ``mapping[key]``, ``default`` if ``key`` is absent."""

    if key not in mapping:
        return default
    return parse_bool(mapping[key], key=key)
