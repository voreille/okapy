from __future__ import annotations

from collections.abc import Sequence

from okapy.core.selection import select


def select_backends(
    backends_by_selector: dict[str, Sequence[object]], modality_key: str
) -> list[object]:
    default = backends_by_selector.get("default", ())
    return list(select(backends_by_selector, modality_key, default))
