from __future__ import annotations
from typing import Any
import pandas as pd
from okapy.features.models import FeatureRecord

def feature_records_to_dataframe(
    records: list[FeatureRecord],
    *,
    result_format: str = "long",
    include_backend: bool = False,
) -> pd.DataFrame:
    if result_format == "long":
        return _to_long_dataframe(records, include_backend=include_backend)
    if result_format == "wide":
        return _to_wide_dataframe(records)
    raise ValueError(f"Unsupported feature result format {result_format!r}.")

def _base_row(record: FeatureRecord) -> dict[str, Any]:
    image, mask = record.image, record.mask
    return {
        "patient_id": image.patient_id,
        "study_instance_uid": image.study_instance_uid,
        "series_instance_uid": image.series_instance_uid,
        "modality": image.modality,
        "submodality": image.submodality,
        "modality_key": image.modality_key,
        "VOI": mask.label,
        "mask_modality": mask.modality,
        "mask_reference_modality": mask.reference_modality,
        "mask_reference_series_instance_uid": mask.reference_series_instance_uid,
        **image.identity.extra_dicom_tags,
    }

def _to_long_dataframe(records: list[FeatureRecord], *, include_backend: bool) -> pd.DataFrame:
    rows=[]
    for record in records:
        base=_base_row(record)
        for feature_set in record.feature_sets:
            for feature_name, feature_value in feature_set.features.items():
                row={**base,"feature_name":feature_name,"feature_value":feature_value}
                if include_backend:
                    row["feature_backend"]=feature_set.backend_name
                rows.append(row)
    return pd.DataFrame(rows)

def _to_wide_dataframe(records: list[FeatureRecord]) -> pd.DataFrame:
    return pd.DataFrame([{**_base_row(record), **record.features} for record in records])
