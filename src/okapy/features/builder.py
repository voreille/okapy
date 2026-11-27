from __future__ import annotations

from pathlib import Path
from typing import Any

from okapy.core.config_values import parse_bool
from okapy.features.backends.command import CommandFeatureBackend
from okapy.features.backends.pet import PETFeatureBackend, PETFeatureConfig
from okapy.features.backends.pyradiomics_local import LocalPyradiomicsBackend
from okapy.features.step import FeatureExtractionStep


def build_feature_extraction_step(config: dict[str, Any]) -> FeatureExtractionStep:
    feature_config = config.get("feature_extraction") or {}
    common_backends = [
        _build_backend(item) for item in feature_config.get("common") or []
    ]
    reserved = {"common", "continue_on_error", "result_format", "include_backend"}
    backends_by_selector = {
        selector: [_build_backend(item) for item in backend_configs or []]
        for selector, backend_configs in feature_config.items()
        if selector not in reserved
    }
    return FeatureExtractionStep(
        backends_by_selector=backends_by_selector,
        common_backends=common_backends,
        continue_on_error=parse_bool(
            feature_config.get("continue_on_error", False), key="continue_on_error"
        ),
    )


def _build_backend(config: dict[str, Any]):
    backend_type = str(config["type"])
    name = str(config.get("name", backend_type))
    if backend_type == "pyradiomics_local":
        return LocalPyradiomicsBackend(
            name=name,
            params=config["params"],
            include_diagnostics=parse_bool(
                config.get("include_diagnostics", False), key="include_diagnostics"
            ),
            execute_kwargs=dict(config.get("execute_kwargs") or {}),
        )
    if backend_type == "command":
        return CommandFeatureBackend(
            name=name,
            command=[str(x) for x in config["command"]],
            params_path=Path(config["params"]),
            timeout_seconds=config.get("timeout_seconds"),
            environment=config.get("environment"),
        )
    if backend_type == "pet":
        return PETFeatureBackend(
            name=name,
            config=PETFeatureConfig(
                threshold=float(config.get("threshold", 0.0)),
                threshold_type=str(config.get("threshold_type", "absolute")),
                suvpeak_diameter_mm=float(config.get("suvpeak_diameter_mm", 12.0)),
                restrict_suvpeak_to_mask=parse_bool(
                    config.get("restrict_suvpeak_to_mask", False),
                    key="restrict_suvpeak_to_mask",
                ),
                preserve_legacy_names=parse_bool(
                    config.get("preserve_legacy_names", True),
                    key="preserve_legacy_names",
                ),
            ),
        )
    raise ValueError(f"Unknown feature backend type {backend_type!r}.")
