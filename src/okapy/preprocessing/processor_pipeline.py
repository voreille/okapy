from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from okapy.preprocessing.registry import build_processor
from okapy.preprocessing.selector import modality_from_key


Processor = Callable[..., Any]


@dataclass(frozen=True)
class ProcessorPipeline:
    """Apply local processors using common + key/modality/default dispatch."""

    common: list[Processor] = field(default_factory=list)
    selectors: dict[str, list[Processor]] = field(default_factory=dict)
    default: list[Processor] = field(default_factory=list)

    @classmethod
    def from_config(cls, config: dict[str, Any] | None) -> ProcessorPipeline:
        config = config or {}

        common = _build_processor_list(config.get("common"))
        default = _build_processor_list(config.get("default"))
        selectors = {
            key: _build_processor_list(value)
            for key, value in config.items()
            if key not in {"common", "default"}
        }

        return cls(common=common, selectors=selectors, default=default)

    def apply(self, item, *, key: str | None = None, **kwargs):
        key = key or getattr(item, "modality_key", None) or getattr(item, "modality", None)
        result = item

        for processor in self.common:
            result = processor(result, **kwargs)

        for processor in self._processors_for_key(key):
            result = processor(result, **kwargs)

        return result

    def _processors_for_key(self, key: str | None) -> list[Processor]:
        if key is None:
            return self.default

        if key in self.selectors:
            return self.selectors[key]

        modality = modality_from_key(key)
        if modality in self.selectors:
            return self.selectors[modality]

        return self.default


def _build_processor_list(config: dict[str, Any] | None) -> list[Processor]:
    if not config:
        return []

    return [
        build_processor(name, params)
        for name, params in config.items()
    ]
