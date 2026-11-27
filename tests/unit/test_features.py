"""Feature formatting, feature builder config handling and the feature step."""

from __future__ import annotations

import pandas as pd
import pytest
import SimpleITK as sitk

from okapy.core.models import ImageMaskSet, StudyVolumes, VolumeCollection
from okapy.features import (
    FeatureExtractionStep,
    FeatureRecord,
    FeatureSet,
    build_feature_extraction_step,
    feature_records_to_dataframe,
)
from tests.helpers.factories import image_volume, mask_volume


def _image():
    image = sitk.Image(4, 4, 4, sitk.sitkFloat32)
    image.SetSpacing((1.0, 1.0, 1.0))
    return image


def _mask():
    mask = sitk.Image(4, 4, 4, sitk.sitkUInt8)
    mask[1, 1, 1] = 1
    return mask


def _record(label="GTV", feature_sets=()):
    return FeatureRecord(
        image=image_volume(_image()),
        mask=mask_volume(_mask(), label=label),
        feature_sets=tuple(feature_sets),
    )


def _collection(*labels):
    image = image_volume(_image())
    masks = [mask_volume(_mask(), label=label) for label in labels]
    return VolumeCollection(
        studies=[
            StudyVolumes(
                study_instance_uid="1.2.3",
                image_mask_sets=[ImageMaskSet(image=image, masks=masks)],
            )
        ]
    )


class _FixedBackend:
    def __init__(self, name="fixed", features=None):
        self.name = name
        self.features = features or {"f1": 1.0}
        self.work_dirs = []

    def extract(self, image, mask, *, work_dir):
        self.work_dirs.append(work_dir)
        return FeatureSet(backend_name=self.name, features=dict(self.features))


class _FailingBackend:
    name = "failing"

    def extract(self, image, mask, *, work_dir):
        raise RuntimeError("boom")


# ---------------------------------------------------------------------------
# Formatters: records without feature sets
# ---------------------------------------------------------------------------


def test_long_format_emits_a_row_for_a_record_without_feature_sets():
    df = feature_records_to_dataframe([_record()], result_format="long")

    assert len(df) == 1
    row = df.iloc[0]
    assert row["patient_id"] == "P1"
    assert row["series_instance_uid"] == "1.2.3.4"
    assert row["VOI"] == "GTV"
    if "feature_name" in df.columns:
        assert pd.isna(row["feature_name"])
    if "feature_value" in df.columns:
        assert pd.isna(row["feature_value"])


def test_long_format_with_backend_column_emits_row_for_empty_record():
    df = feature_records_to_dataframe(
        [_record()], result_format="long", include_backend=True
    )

    assert len(df) == 1
    assert df.iloc[0]["VOI"] == "GTV"


def test_long_format_keeps_empty_records_alongside_populated_ones():
    populated = _record(
        label="A",
        feature_sets=[FeatureSet(backend_name="b", features={"x": 1.0, "y": 2.0})],
    )
    empty = _record(label="B")

    df = feature_records_to_dataframe([populated, empty], result_format="long")

    assert len(df) == 3
    assert sorted(df["VOI"]) == ["A", "A", "B"]
    populated_rows = df[df["VOI"] == "A"].set_index("feature_name")["feature_value"]
    assert populated_rows.to_dict() == {"x": 1.0, "y": 2.0}
    empty_row = df[df["VOI"] == "B"].iloc[0]
    assert pd.isna(empty_row["feature_name"])
    assert pd.isna(empty_row["feature_value"])


def test_long_and_wide_formats_cover_the_same_records():
    records = [
        _record(
            label="A",
            feature_sets=[FeatureSet(backend_name="b", features={"x": 1.0})],
        ),
        _record(label="B"),
    ]

    long_df = feature_records_to_dataframe(records, result_format="long")
    wide_df = feature_records_to_dataframe(records, result_format="wide")

    assert set(long_df["VOI"]) == set(wide_df["VOI"]) == {"A", "B"}
    assert len(wide_df) == 2


def test_long_format_regular_records_unchanged():
    record = _record(
        feature_sets=[
            FeatureSet(backend_name="b1", features={"x": 1.0}),
            FeatureSet(backend_name="b2", features={"y": 2.0}),
        ]
    )

    df = feature_records_to_dataframe(
        [record], result_format="long", include_backend=True
    )

    assert list(df["feature_name"]) == ["x", "y"]
    assert list(df["feature_value"]) == [1.0, 2.0]
    assert list(df["feature_backend"]) == ["b1", "b2"]


def test_failed_extraction_with_continue_on_error_still_reported_in_long_format(
    tmp_path,
):
    step = FeatureExtractionStep(
        common_backends=[_FailingBackend()], continue_on_error=True
    )

    records = step.run(_collection("GTV"), work_dir=tmp_path / "features")
    df = feature_records_to_dataframe(records, result_format="long")

    assert len(records) == 1
    assert records[0].feature_sets == ()
    assert len(df) == 1
    assert df.iloc[0]["VOI"] == "GTV"


def test_unknown_result_format_is_rejected():
    with pytest.raises(ValueError, match="format"):
        feature_records_to_dataframe([_record()], result_format="xml")


# ---------------------------------------------------------------------------
# Builder: null YAML sections
# ---------------------------------------------------------------------------


def test_null_common_section_does_not_crash_builder():
    step = build_feature_extraction_step({"feature_extraction": {"common": None}})

    assert list(step.common_backends) == []


def test_null_modality_section_does_not_crash_builder():
    step = build_feature_extraction_step(
        {"feature_extraction": {"CT": None, "common": [{"type": "pet"}]}}
    )

    assert list(step.backends_by_selector.get("CT") or []) == []
    assert [b.name for b in step.common_backends] == ["pet"]


def test_null_sections_parsed_from_yaml_do_not_crash_builder():
    import yaml

    config = yaml.safe_load("feature_extraction:\n  common:\n  CT:\n  PT:\n")

    step = build_feature_extraction_step(config)

    assert list(step.common_backends) == []
    for selector in ("CT", "PT"):
        assert list(step.backends_by_selector.get(selector) or []) == []


def test_null_feature_extraction_section_builds_empty_step():
    step = build_feature_extraction_step({"feature_extraction": None})

    assert list(step.common_backends) == []
    assert not any(step.backends_by_selector.values())


def test_builder_still_builds_pet_backend_for_modality():
    step = build_feature_extraction_step(
        {"feature_extraction": {"PT": [{"type": "pet", "name": "pet_feats"}]}}
    )

    assert [b.name for b in step.backends_by_selector["PT"]] == ["pet_feats"]


def test_builder_rejects_unknown_backend_type():
    with pytest.raises(ValueError, match="riesz"):
        build_feature_extraction_step(
            {"feature_extraction": {"CT": [{"type": "riesz"}]}}
        )


# ---------------------------------------------------------------------------
# Feature step: work_dir as str
# ---------------------------------------------------------------------------


def test_feature_step_accepts_str_work_dir(tmp_path):
    backend = _FixedBackend()
    step = FeatureExtractionStep(common_backends=[backend])
    work_dir = tmp_path / "features"

    records = step.run(_collection("GTV"), work_dir=str(work_dir))

    assert work_dir.is_dir()
    assert len(records) == 1
    assert records[0].features == {"f1": 1.0}
    assert len(backend.work_dirs) == 1
    assert str(backend.work_dirs[0]) == str(work_dir)


def test_feature_step_returns_one_record_per_mask(tmp_path):
    step = FeatureExtractionStep(common_backends=[_FixedBackend()])

    records = step.run(_collection("A", "B"), work_dir=tmp_path)

    assert [r.mask.label for r in records] == ["A", "B"]


def test_feature_step_reraises_without_continue_on_error(tmp_path):
    step = FeatureExtractionStep(common_backends=[_FailingBackend()])

    with pytest.raises(RuntimeError, match="boom"):
        step.run(_collection("GTV"), work_dir=tmp_path)
