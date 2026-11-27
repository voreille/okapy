# src/okapy/config/loader.py

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from okapy.config.legacy import (
    check_not_mixed_config,
    legacy_config_keys,
    migrate_legacy_config,
)

Config = dict[str, Any]
ConfigSource = str | Path | Mapping[str, Any]


def load_config(source: ConfigSource) -> Config:
    """Load an Okapy configuration.

    Parameters
    ----------
    source:
        Either:
        - a path to a YAML configuration file;
        - an already-loaded mapping.

    Returns
    -------
    dict
        A mutable copy of the configuration. A legacy-shaped configuration
        (see :func:`okapy.config.legacy.is_legacy_config`) is migrated to the
        new shape.

    Raises
    ------
    FileNotFoundError
        If the configuration path does not exist.
    ValueError
        If the YAML is empty or its root is not a mapping, a legacy
        configuration cannot be migrated, or legacy keys are mixed with
        new-shape sections.
    """

    if isinstance(source, Mapping):
        config = deepcopy(dict(source))
    else:
        path = Path(source).expanduser()

        if not path.is_file():
            raise FileNotFoundError(f"Okapy configuration file does not exist: {path}")

        with path.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file)

        if config is None:
            raise ValueError(f"Okapy configuration file is empty: {path}")

        if not isinstance(config, dict):
            raise ValueError(
                "The root of an Okapy configuration must be a YAML mapping, "
                f"got {type(config).__name__} in {path}."
            )

    if not config:
        raise ValueError("Okapy configuration cannot be empty.")

    legacy_keys = legacy_config_keys(config)
    if legacy_keys:
        check_not_mixed_config(config, legacy_keys)
        return migrate_legacy_config(config)

    return config
