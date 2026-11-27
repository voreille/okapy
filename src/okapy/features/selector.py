from __future__ import annotations

from collections.abc import Mapping, Sequence

from okapy.core.selection import select
from okapy.features.backends.base import FeatureBackend


def select_backends(
    backends_by_selector: Mapping[str, Sequence[FeatureBackend]], modality_key: str
) -> list[FeatureBackend]:
    default = backends_by_selector.get("default", ())
    return list(select(backends_by_selector, modality_key, default))
