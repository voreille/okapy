from __future__ import annotations

from copy import deepcopy
from typing import Any


def migrate_legacy_preprocessing_config(config: dict[str, Any]) -> dict[str, Any]:
    """Best-effort migration from the old preprocessing YAML shape.

    Old YAML used ``volume_preprocessing`` and ``mask_preprocessing`` stacks,
    where geometry and local processors were mixed.

    New YAML separates:

    - ``geometry_preprocessing`` for crop/resample/grid definition;
    - ``local_preprocessing`` for image-only/local operations;
    - ``mask_preprocessing`` for post-geometry mask cleanup.

    Legacy:
        general.padding

    becomes:
        geometry_preprocessing.common.padding_mm
    """

    new_config = deepcopy(config)

    general = deepcopy(config.get("general") or {})
    volume_preprocessing = deepcopy(config.get("volume_preprocessing") or {})
    mask_preprocessing = deepcopy(config.get("mask_preprocessing") or {})

    geometry_preprocessing: dict[str, Any] = {}
    local_preprocessing: dict[str, Any] = {}
    new_mask_preprocessing: dict[str, Any] = {}

    # ---------------------------------------------------------------------
    # General legacy options
    # ---------------------------------------------------------------------
    common_geometry: dict[str, Any] = {}

    if "padding" in general:
        common_geometry["padding_mm"] = general["padding"]

    # Reasonable default matching the old behavior:
    # crop around masks if padding was provided.
    if "padding" in general:
        common_geometry["crop_to_masks"] = True

    if common_geometry:
        geometry_preprocessing["common"] = common_geometry

    # ---------------------------------------------------------------------
    # Volume preprocessing: split geometry processors from local processors.
    # ---------------------------------------------------------------------
    for selector, stack in volume_preprocessing.items():
        stack = stack or {}

        local_stack: dict[str, Any] = {}
        geometry_cfg: dict[str, Any] = {}

        for name, params in stack.items():
            params = params or {}

            if name == "bspline_resampler":
                if "resampling_spacing" in params:
                    geometry_cfg["spacing"] = params["resampling_spacing"]

                if "order" in params:
                    geometry_cfg["image_interpolator"] = _order_to_interpolator(
                        params["order"]
                    )

                if "cval" in params:
                    geometry_cfg["default_image_value"] = params["cval"]

            else:
                local_stack[name] = params

        if geometry_cfg:
            geometry_preprocessing[selector] = _merge_dicts(
                geometry_preprocessing.get(selector, {}),
                geometry_cfg,
            )

        if local_stack:
            local_preprocessing[selector] = local_stack

    # ---------------------------------------------------------------------
    # Mask preprocessing: geometry info goes to geometry_preprocessing,
    # thresholding/casting stays in mask_preprocessing.
    # ---------------------------------------------------------------------
    for selector, stack in mask_preprocessing.items():
        stack = stack or {}

        mask_stack: dict[str, Any] = {}
        geometry_cfg: dict[str, Any] = {}

        for name, params in stack.items():
            params = params or {}

            if name == "binary_bspline_resampler":
                if "resampling_spacing" in params:
                    geometry_cfg["spacing"] = params["resampling_spacing"]

                if "order" in params:
                    geometry_cfg["mask_interpolator"] = _order_to_interpolator(
                        params["order"]
                    )

                if "cval" in params:
                    geometry_cfg["default_mask_value"] = params["cval"]

                threshold = params.get("threshold")
                if threshold is not None:
                    mask_stack["binarize_mask"] = {"threshold": threshold}

                # In the new pipeline, masks should almost always be uint8.
                mask_stack.setdefault("cast_mask", {"pixel_type": "uint8"})

            else:
                mask_stack[name] = params

        if geometry_cfg:
            geometry_preprocessing[selector] = _merge_dicts(
                geometry_preprocessing.get(selector, {}),
                geometry_cfg,
            )

        if mask_stack:
            new_mask_preprocessing[selector] = mask_stack

    # ---------------------------------------------------------------------
    # Clean old keys and inject new structure.
    # ---------------------------------------------------------------------
    new_config.pop("volume_preprocessing", None)

    new_config["general"] = general
    new_config["geometry_preprocessing"] = geometry_preprocessing
    new_config["local_preprocessing"] = local_preprocessing
    new_config["mask_preprocessing"] = new_mask_preprocessing

    return new_config


def _order_to_interpolator(order: int) -> str:
    order = int(order)

    if order == 0:
        return "nearest"
    if order == 1:
        return "linear"
    if order == 3:
        return "bspline"

    raise ValueError(f"Cannot migrate unsupported interpolation order={order}.")


def _merge_dicts(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    merged.update(update)
    return merged
