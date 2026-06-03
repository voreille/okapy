from __future__ import annotations

from copy import deepcopy
from typing import Any


def modality_from_key(key: str) -> str:
    return key.split("_", maxsplit=1)[0]


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
    selected = _lookup_selector(section, key=key)

    if include_common:
        return deep_merge(section.get("common") or {}, selected)

    return deepcopy(selected)


def _lookup_selector(section: dict[str, Any], *, key: str) -> dict[str, Any]:
    if key in section:
        return deepcopy(section[key] or {})

    modality = modality_from_key(key)
    if modality in section:
        return deepcopy(section[modality] or {})

    return deepcopy(section.get("default") or {})


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)

    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)

    return result
