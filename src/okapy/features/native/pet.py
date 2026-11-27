"""Backwards-compatible alias: the PET backend lives in ``features.backends``."""

from okapy.features.backends.pet import (
    PETFeatureBackend,
    PETFeatureConfig,
    compute_suvpeak,
)

__all__ = ["PETFeatureBackend", "PETFeatureConfig", "compute_suvpeak"]
