from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from okapy.core.models import ImageVolume, MaskVolume
from okapy.features.backends.base import FeatureBackend
from okapy.features.models import FeatureSet


class CommandFeatureBackend(FeatureBackend):
    """CLI contract: --image, --mask, --params, --output."""

    def __init__(
        self,
        *,
        command: list[str],
        params_path: Path | str,
        name: str,
        timeout_seconds: float | None = None,
        environment: dict[str, str] | None = None,
    ) -> None:
        if not command:
            raise ValueError("CommandFeatureBackend requires a non-empty command.")
        self.name = name
        self.command = [str(item) for item in command]
        self.params_path = Path(params_path)
        self.timeout_seconds = timeout_seconds
        self.environment = environment

    def extract(
        self, image: ImageVolume, mask: MaskVolume, *, work_dir: Path
    ) -> FeatureSet:
        pair_dir = (
            work_dir
            / _safe_name(image.series_instance_uid)
            / _safe_name(mask.label)
            / _safe_name(self.name)
        )
        pair_dir.mkdir(parents=True, exist_ok=True)
        output_path = pair_dir / "features.json"
        command = [
            *self.command,
            "--image",
            str(image.path),
            "--mask",
            str(mask.path),
            "--params",
            str(self.params_path),
            "--output",
            str(output_path),
        ]
        env = None if self.environment is None else {**os.environ, **self.environment}
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=self.timeout_seconds,
            env=env,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"Feature backend {self.name!r} failed with exit code {completed.returncode}.\n"
                f"Command: {command}\n"
                f"stdout:\n{completed.stdout}\n"
                f"stderr:\n{completed.stderr}"
            )
        if not output_path.is_file():
            raise RuntimeError(
                f"Feature backend {self.name!r} did not create {output_path}."
            )
        payload = json.loads(output_path.read_text())
        if "features" not in payload or not isinstance(payload["features"], dict):
            raise ValueError(
                "External feature output must contain a 'features' mapping."
            )
        return FeatureSet(
            backend_name=self.name,
            features={str(k): v for k, v in payload["features"].items()},
            metadata=dict(payload.get("metadata") or {}),
        )


def _safe_name(value: str) -> str:
    return (
        str(value)
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "-")
    )
