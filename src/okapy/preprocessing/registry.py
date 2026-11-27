from __future__ import annotations

from collections.abc import Callable
from typing import Any

ProcessorFactory = Callable[..., object]

PROCESSOR_REGISTRY: dict[str, ProcessorFactory] = {}

_builtins_registered = False


def _register_builtins() -> None:
    """Register the built-in processors on first use of the registry."""

    global _builtins_registered
    if _builtins_registered:
        return
    _builtins_registered = True

    from okapy.preprocessing import processors

    PROCESSOR_REGISTRY.update(
        {
            "identity_processor": processors.IdentityProcessor,
            "standardizer": processors.Standardizer,
            "masked_standardizer": processors.MaskedStandardizer,
            "clip_intensity": processors.ClipIntensity,
            "cast_image": processors.CastImage,
            "cast_mask": processors.CastMask,
            "binarize_mask": processors.BinarizeMask,
        }
    )


def register_processor(
    name: str, factory: ProcessorFactory, *, overwrite: bool = False
) -> None:
    _register_builtins()
    if not overwrite and name in PROCESSOR_REGISTRY:
        raise ValueError(f"Processor {name!r} is already registered.")
    PROCESSOR_REGISTRY[name] = factory


def build_processor(name: str, params: dict[str, Any] | None = None):
    _register_builtins()
    try:
        factory = PROCESSOR_REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(sorted(PROCESSOR_REGISTRY))
        raise ValueError(
            f"Unknown preprocessing processor {name!r}. "
            f"Available processors: {available}."
        ) from exc

    return factory(**(params or {}))
