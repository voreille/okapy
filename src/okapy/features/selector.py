from __future__ import annotations
from collections.abc import Sequence

def select_backends(backends_by_selector: dict[str, Sequence[object]], modality_key: str) -> list[object]:
    if modality_key in backends_by_selector:
        return list(backends_by_selector[modality_key])
    modality = modality_key.split("_", maxsplit=1)[0]
    if modality in backends_by_selector:
        return list(backends_by_selector[modality])
    return list(backends_by_selector.get("default", ()))
