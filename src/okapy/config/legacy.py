from __future__ import annotations

from copy import deepcopy
from typing import Any

from okapy.core.geometry import MASK_THRESHOLD


def migrate_legacy_config(
    config: dict[str, Any],
) -> dict[str, Any]:
    migrated = deepcopy(config)

    preprocessing = migrate_legacy_preprocessing_config(config)

    features = migrate_legacy_feature_extraction_config(
        config.get("feature_extraction") or {},
        result_format=str((config.get("general") or {}).get("result_format", "long")),
    )

    migrated.pop("volume_preprocessing", None)
    migrated.pop("mask_preprocessing", None)

    migrated["geometry_preprocessing"] = preprocessing["geometry_preprocessing"]
    migrated["local_preprocessing"] = preprocessing["local_preprocessing"]
    migrated["mask_preprocessing"] = preprocessing["mask_preprocessing"]
    migrated["feature_extraction"] = features

    general = deepcopy(config.get("general") or {})
    general.pop("padding", None)
    general.pop("result_format", None)

    migrated["general"] = general

    return migrated


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

    A legacy ``mask_preprocessing.<selector>.binary_bspline_resampler`` only
    migrates ``resampling_spacing`` and ``cval``. The interpolation order and
    threshold are fixed to linear + 0.5 in the new pipeline; any other value
    is rejected rather than silently changed.
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
    # Mask preprocessing: spacing/cval go to geometry_preprocessing. The
    # interpolation order and threshold are not configurable any more (masks
    # are always linear + 0.5), so they are validated and dropped. Any other
    # mask processor passes through to the new mask_preprocessing stack.
    # ---------------------------------------------------------------------
    for selector, stack in mask_preprocessing.items():
        stack = stack or {}

        mask_stack: dict[str, Any] = {}
        geometry_cfg: dict[str, Any] = {}

        for name, params in stack.items():
            params = params or {}

            if name == "binary_bspline_resampler":
                _check_legacy_mask_resampler(params)

                if "resampling_spacing" in params:
                    geometry_cfg["spacing"] = params["resampling_spacing"]

                if "cval" in params:
                    geometry_cfg["default_mask_value"] = params["cval"]

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


def migrate_legacy_feature_extraction_config(
    feature_config: dict[str, Any],
    *,
    result_format: str = "long",
) -> dict[str, Any]:
    migrated: dict[str, Any] = {
        "result_format": result_format,
        "continue_on_error": False,
        "common": [],
    }

    for selector, extractor_groups in feature_config.items():
        if not extractor_groups:
            continue

        backends: list[dict[str, Any]] = []

        for extractor_type, named_extractors in extractor_groups.items():
            if extractor_type != "pyradiomics":
                raise ValueError(
                    f"Legacy feature extractor type {extractor_type!r} "
                    "is not supported by the refactor."
                )

            for extractor_name, extractor_params in named_extractors.items():
                backends.append(
                    {
                        "type": "pyradiomics_local",
                        "name": f"pyradiomics_{extractor_name}",
                        "params": deepcopy(extractor_params or {}),
                        "include_diagnostics": False,
                    }
                )

        if selector == "PT":
            backends.append(
                {
                    "type": "pet",
                    "name": "pet",
                    "threshold": 0.0,
                    "threshold_type": "absolute",
                    "suvpeak_diameter_mm": 12.0,
                    "restrict_suvpeak_to_mask": False,
                    "preserve_legacy_names": True,
                }
            )

        migrated[selector] = backends

    return migrated


def _order_to_interpolator(order: int) -> str:
    order = int(order)

    if order == 0:
        return "nearest"
    if order == 1:
        return "linear"
    if order == 3:
        return "bspline"

    raise ValueError(f"Cannot migrate unsupported interpolation order={order}.")


#: Legacy scipy order that matches the new pipeline's linear interpolation.
_LEGACY_MASK_ORDER = 1


def _check_legacy_mask_resampler(params: dict[str, Any]) -> None:
    """Refuse legacy mask resampler settings the new pipeline cannot honour.

    Masks are always resampled with linear interpolation (order 1) and
    thresholded at 0.5. A legacy config asking for anything else is rejected
    rather than migrated, because silently switching would change feature
    values without notice.
    """

    order = params.get("order")
    if order is not None and int(order) != _LEGACY_MASK_ORDER:
        raise ValueError(
            f"Cannot migrate 'binary_bspline_resampler' with order={order}: "
            "masks are always resampled with linear interpolation (order 1) "
            "and thresholded at 0.5. Remove 'order' from the legacy config."
        )

    threshold = params.get("threshold")
    if threshold is not None and float(threshold) != MASK_THRESHOLD:
        raise ValueError(
            f"Cannot migrate 'binary_bspline_resampler' with threshold={threshold}: "
            f"masks are always thresholded at {MASK_THRESHOLD} after linear "
            "interpolation. Remove 'threshold' from the legacy config."
        )


def _merge_dicts(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    merged.update(update)
    return merged
