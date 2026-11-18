from __future__ import annotations

from typing import Any, Callable


ProcessorFactory = Callable[..., object]

PROCESSOR_REGISTRY: dict[str, ProcessorFactory] = {}


def register_processor(name: str, factory: ProcessorFactory, *, overwrite: bool = False) -> None:
    if not overwrite and name in PROCESSOR_REGISTRY:
        raise ValueError(f"Processor {name!r} is already registered.")
    PROCESSOR_REGISTRY[name] = factory


def build_processor(name: str, params: dict[str, Any] | None = None):
    try:
        factory = PROCESSOR_REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(sorted(PROCESSOR_REGISTRY))
        raise ValueError(
            f"Unknown preprocessing processor {name!r}. "
            f"Available processors: {available}."
        ) from exc

    return factory(**(params or {}))
