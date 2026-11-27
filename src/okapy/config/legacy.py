from __future__ import annotations

import logging
from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from okapy.core.config_values import FEATURE_OPTION_KEYS
from okapy.core.geometry import MASK_THRESHOLD

logger = logging.getLogger(__name__)

#: Legacy ``general.padding`` default, in millimetres.
_LEGACY_DEFAULT_PADDING_MM = 10.0

#: Legacy ``general.padding`` value meaning "do not crop".
_LEGACY_WHOLE_IMAGE = "whole_image"

#: Legacy ``general.study_processor`` values.
_LEGACY_STANDARD_STUDY_PROCESSOR = "studyprocessor"
_LEGACY_DEEP_STUDY_PROCESSOR = "studyprocessor_deep"

#: Top-level sections that only exist in the new config shape.
_NEW_SHAPE_SECTIONS = ("geometry_preprocessing", "local_preprocessing")


def is_legacy_config(config: Mapping[str, Any]) -> bool:
    """Whether ``config`` uses the legacy (pre-refactor) YAML shape.

    Only keys that cannot appear in a new-shape config are used:

    - a top-level ``volume_preprocessing`` section;
    - ``general.padding`` (millimetres; replaced by ``padding_voxels``);
    - ``feature_extraction`` selectors mapping to extractor groups, e.g.
      ``{CT: {pyradiomics: {original: ...}}}``, instead of backend lists;
    - a ``binary_bspline_resampler`` in ``mask_preprocessing``.
    """

    return bool(legacy_config_keys(config))


def legacy_config_keys(config: Mapping[str, Any]) -> list[str]:
    """The legacy-only keys found in ``config`` (see :func:`is_legacy_config`)."""

    keys: list[str] = []

    if "volume_preprocessing" in config:
        keys.append("volume_preprocessing")

    general = config.get("general") or {}
    if isinstance(general, Mapping) and "padding" in general:
        keys.append("general.padding")

    if any(isinstance(value, Mapping) for value in _feature_selector_values(config)):
        keys.append("feature_extraction")

    mask_preprocessing = config.get("mask_preprocessing") or {}
    if isinstance(mask_preprocessing, Mapping) and any(
        isinstance(stack, Mapping) and "binary_bspline_resampler" in stack
        for stack in mask_preprocessing.values()
    ):
        keys.append("mask_preprocessing.binary_bspline_resampler")

    return keys


def new_shape_config_sections(config: Mapping[str, Any]) -> list[str]:
    """The new-shape-only sections found in ``config``.

    These are ``geometry_preprocessing``, ``local_preprocessing`` and a
    ``feature_extraction`` whose selectors already map to backend lists.
    """

    sections = [name for name in _NEW_SHAPE_SECTIONS if name in config]

    if any(
        isinstance(value, list) and value for value in _feature_selector_values(config)
    ):
        sections.append("feature_extraction")

    return sections


def _feature_selector_values(config: Mapping[str, Any]) -> list[Any]:
    """Values of the ``feature_extraction`` selectors (``common`` included)."""

    features = config.get("feature_extraction") or {}
    if not isinstance(features, Mapping):
        return []
    return [
        value
        for selector, value in features.items()
        if selector not in FEATURE_OPTION_KEYS
    ]


def check_not_mixed_config(
    config: Mapping[str, Any],
    legacy_keys: list[str] | None = None,
) -> None:
    """Reject a config that mixes legacy keys with new-shape sections.

    ``legacy_keys`` is :func:`legacy_config_keys` of ``config``, computed here
    when not given.
    """

    if legacy_keys is None:
        legacy_keys = legacy_config_keys(config)
    new_sections = new_shape_config_sections(config)
    if legacy_keys and new_sections:
        raise ValueError(
            "The configuration mixes legacy keys "
            f"({', '.join(legacy_keys)}) with new-shape sections "
            f"({', '.join(new_sections)}). Do not mix them: either use the "
            "legacy shape only (it is migrated automatically) or migrate the "
            "legacy keys to the new shape."
        )


def migrate_legacy_config(
    config: dict[str, Any],
) -> dict[str, Any]:
    _check_legacy_study_processor(config.get("general") or {})

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
    if "patch_size" in general:
        logger.warning(
            "Legacy general.patch_size=%r is not used by the new pipeline; "
            "dropping it.",
            general.pop("patch_size"),
        )
    if "study_processor" in general:
        logger.warning(
            "Legacy general.study_processor=%r is the only study processor of "
            "the new pipeline; dropping the option.",
            general.pop("study_processor"),
        )

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

    Legacy ``general.padding`` is in millimetres (default 10 mm). It becomes
    ``padding_voxels`` (voxels of the target grid, per axis) for ``common`` and
    for every selector with its own image resampling spacing:
    ``round(padding / spacing)`` on each axis. A target spacing that is native
    (``-1``) or missing along any axis cannot be converted and is rejected.
    ``padding: whole_image`` becomes ``crop_to_masks: False``.

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
    explicit_padding = "padding" in general
    padding = general.get("padding", _LEGACY_DEFAULT_PADDING_MM)
    crop_to_masks = padding != _LEGACY_WHOLE_IMAGE
    geometry_preprocessing["common"] = {"crop_to_masks": crop_to_masks}
    # Selectors whose image resampling spacing defines their padding grid.
    image_spacing_selectors = {"common"}

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
                    image_spacing_selectors.add(selector)

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
        mask_geometry_cfg: dict[str, Any] = {}

        for name, params in stack.items():
            params = params or {}

            if name == "binary_bspline_resampler":
                _check_legacy_mask_resampler(params)

                if "resampling_spacing" in params:
                    mask_geometry_cfg["spacing"] = params["resampling_spacing"]

                if "cval" in params:
                    mask_geometry_cfg["default_mask_value"] = params["cval"]

            else:
                mask_stack[name] = params

        if mask_geometry_cfg:
            geometry_preprocessing[selector] = _merge_dicts(
                geometry_preprocessing.get(selector, {}),
                mask_geometry_cfg,
            )

        if mask_stack:
            new_mask_preprocessing[selector] = mask_stack

    # ---------------------------------------------------------------------
    # Padding: millimetres -> voxels of each selector's target grid.
    # ---------------------------------------------------------------------
    if crop_to_masks:
        for selector in sorted(image_spacing_selectors):
            spacing = geometry_preprocessing[selector].get("spacing")
            if selector == "common" and spacing is None and not explicit_padding:
                # The implicit 10 mm default cannot be converted without a
                # target grid: keep the new default (no padding) instead.
                logger.warning(
                    "Legacy config has no general.padding (default %s mm) and "
                    "no common target spacing: the default padding cannot be "
                    "converted to voxels, so 'common' gets no padding. Set "
                    "'padding_voxels' in 'geometry_preprocessing' if needed.",
                    _LEGACY_DEFAULT_PADDING_MM,
                )
                continue
            geometry_preprocessing[selector]["padding_voxels"] = _padding_to_voxels(
                padding,
                spacing,
                selector=selector,
            )

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
        if selector in FEATURE_OPTION_KEYS:
            migrated[selector] = deepcopy(extractor_groups)
            continue
        if not extractor_groups:
            continue
        if not isinstance(extractor_groups, Mapping):
            # Already a new-shape backend list.
            migrated[selector] = deepcopy(extractor_groups)
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


def _padding_to_voxels(padding: Any, spacing: Any, *, selector: str) -> int | list[int]:
    """Legacy padding (mm) as voxels of the ``spacing`` grid, per axis."""

    if isinstance(padding, bool) or not isinstance(padding, (int, float)):
        raise ValueError(
            f"Cannot migrate legacy general.padding={padding!r}: expected a "
            f"number of millimetres or {_LEGACY_WHOLE_IMAGE!r}."
        )
    if padding < 0:
        raise ValueError(
            f"Cannot migrate legacy general.padding={padding!r}: it must be >= 0."
        )
    if padding == 0:
        # 0 mm is 0 voxels on any grid, native spacing included.
        return 0

    if spacing is None:
        raise ValueError(
            f"Cannot migrate legacy general.padding={padding} mm for selector "
            f"{selector!r}: its target spacing is missing, so millimetres cannot "
            "be converted to voxels. Set an explicit 'padding_voxels' in "
            "'geometry_preprocessing' instead."
        )

    axes = [float(value) for value in spacing]
    if len(axes) != 3 or any(value <= 0 for value in axes):
        raise ValueError(
            f"Cannot migrate legacy general.padding={padding} mm for selector "
            f"{selector!r}: its target spacing ({spacing}) is native along at "
            "least one axis, so millimetres cannot be converted to voxels. Set "
            "an explicit 'padding_voxels' in 'geometry_preprocessing' instead."
        )

    voxels = [round(padding / value) for value in axes]
    if len(set(voxels)) == 1:
        return voxels[0]
    return voxels


def _check_legacy_study_processor(general: Mapping[str, Any]) -> None:
    """Refuse the legacy deep-learning study processor."""

    if general.get("study_processor") == _LEGACY_DEEP_STUDY_PROCESSOR:
        raise ValueError(
            f"Legacy general.study_processor={_LEGACY_DEEP_STUDY_PROCESSOR!r} is "
            "not supported by the new pipeline. Remove it (or use "
            f"{_LEGACY_STANDARD_STUDY_PROCESSOR!r})."
        )


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
