from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path

import SimpleITK as sitk

from okapy.core.models import ImageVolume, MaskVolume
from okapy.dicom.conversion.utils import join_inside, safe_name
from okapy.features.backends.base import FeatureBackend
from okapy.features.models import FeatureSet

#: Mask metadata keys holding the series UID of the RTSTRUCT/SEG a mask comes
#: from (set by the mask converters).
_MASK_SERIES_UID_KEYS = ("rtstruct_series_instance_uid", "seg_series_instance_uid")


class CommandFeatureBackend(FeatureBackend):
    """CLI contract: --image, --mask, --params, --output.

    The command is run as configured: configs using this backend are trusted
    input. A volume that is only in memory (``path is None``, e.g. with
    ``general.write_outputs: false``) is written as NIfTI into the per-pair
    work dir before the command runs. The same in-memory image is written
    once: later pairs get a hard link (or a copy) of that file in their own
    pair dir.
    """

    def __init__(
        self,
        *,
        command: Sequence[str],
        params_path: Path | str,
        name: str,
        timeout_seconds: float | None = None,
        environment: dict[str, str] | None = None,
    ) -> None:
        if isinstance(command, str | bytes):
            raise ValueError(
                f"Feature backend {name!r}: 'command' must be a list of "
                f"arguments (e.g. ['python', 'script.py']), got the string "
                f"{command!r}."
            )
        if not command:
            raise ValueError("CommandFeatureBackend requires a non-empty command.")
        self.name = name
        self.command = [str(item) for item in command]
        self.params_path = Path(params_path)
        self.timeout_seconds = timeout_seconds
        self.environment = environment
        # Last in-memory image written per role ("image"/"mask") and its file.
        # Holding the image keeps its identity from being reused; one entry
        # per role keeps older volumes from being held alive.
        self._written: dict[str, tuple[sitk.Image, Path]] = {}

    def extract(
        self, image: ImageVolume, mask: MaskVolume, *, work_dir: Path | str
    ) -> FeatureSet:
        work_dir = Path(work_dir)
        pair_dir = self._pair_dir(image, mask, work_dir)
        pair_dir.mkdir(parents=True, exist_ok=True)
        output_path = pair_dir / "features.json"
        # Never read a previous run's output if the command does not write one.
        output_path.unlink(missing_ok=True)
        image_path = self._volume_path("image", image.path, image.image, pair_dir)
        mask_path = self._volume_path("mask", mask.path, mask.image, pair_dir)
        command = [
            *self.command,
            "--image",
            str(image_path),
            "--mask",
            str(mask_path),
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
                f"Feature backend {self.name!r} failed with exit code "
                f"{completed.returncode}.\n"
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

    def _volume_path(
        self, role: str, path: Path | None, image: sitk.Image, pair_dir: Path
    ) -> Path:
        """``path``, or ``image`` as ``<role>.nii.gz`` in ``pair_dir`` when it
        is only in memory.

        An image already written by this backend is linked (or copied) from
        that file instead of being written again.
        """

        if path is not None:
            return path
        target = pair_dir / f"{role}.nii.gz"
        cached = self._written.get(role)
        if cached is not None and cached[0] is image and cached[1].is_file():
            source = cached[1]
            if source == target:
                return target
            # Never write through a stale file (it may be a link to another).
            target.unlink(missing_ok=True)
            try:
                os.link(source, target)
            except OSError:
                shutil.copyfile(source, target)
            return target
        target.unlink(missing_ok=True)
        sitk.WriteImage(image, str(target))
        self._written[role] = (image, target)
        return target

    def _pair_dir(self, image: ImageVolume, mask: MaskVolume, work_dir: Path) -> Path:
        """Work dir unique per (image, mask, backend).

        Sanitized names keep it readable; a hash of the raw identifiers keeps
        it unique when different values sanitize to the same name.
        """

        mask_series_uid = next(
            (
                mask.identity.metadata[key]
                for key in _MASK_SERIES_UID_KEYS
                if mask.identity.metadata.get(key)
            ),
            None,
        )
        raw = [
            image.series_instance_uid,
            str(image.path),
            mask.label,
            mask_series_uid,
            str(mask.path),
            self.name,
        ]
        digest = hashlib.sha1(json.dumps(raw).encode("utf-8")).hexdigest()[:12]
        return join_inside(
            work_dir,
            safe_name(image.series_instance_uid),
            f"{safe_name(mask.label)}-{digest}",
            safe_name(self.name),
        )
