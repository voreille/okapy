from __future__ import annotations

from copy import deepcopy
from typing import Any

from okapy.core.selection import modality_from_key, select

__all__ = ["deep_merge", "modality_from_key", "select_section_config"]


def select_section_config(
    section: dict[str, Any] | None,
    *,
    key: str,
    include_common: bool = False,
) -> dict[str, Any]:
    """Select a config block using ``key > modality > default``.

    If include_common=True, the selected block is deep-merged on top of
    ``common``.
    """

    section = section or {}
    selected = deepcopy(select(section, key, section.get("default")) or {})

    if include_common:
        return deep_merge(section.get("common") or {}, selected)

    return selected


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)

    return result
