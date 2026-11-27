"""Parsing helpers for config values shared by every okapy config section."""

from __future__ import annotations

from typing import Any

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
