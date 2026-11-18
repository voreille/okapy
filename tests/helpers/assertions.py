import json
from pathlib import Path

import numpy as np
import pandas as pd


def assert_json_close(actual, expected, *, rtol: float = 1e-5, atol: float = 1e-5):
    if isinstance(actual, dict):
        assert isinstance(expected, dict)
        assert actual.keys() == expected.keys()

        for key in actual:
            assert_json_close(actual[key], expected[key], rtol=rtol, atol=atol)

    elif isinstance(actual, list):
        assert isinstance(expected, list)
        assert len(actual) == len(expected)

        for a, e in zip(actual, expected):
            assert_json_close(a, e, rtol=rtol, atol=atol)

    elif isinstance(actual, float):
        np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)

    else:
        assert actual == expected

def assert_or_update_json(
    actual: dict,
    golden_path: Path,
    update: bool,
):
    if update:
        golden_path.parent.mkdir(parents=True, exist_ok=True)
        golden_path.write_text(json.dumps(actual, indent=2, sort_keys=True))
        return

    expected = json.loads(golden_path.read_text())
    assert_json_close(actual, expected)


def _is_numeric_list(value):
    return (
        isinstance(value, list)
        and len(value) > 0
        and all(isinstance(x, (int, float)) for x in value)
    )


def normalize_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Drop unstable PyRadiomics diagnostics if present
    if "feature_name" in df.columns:
        df = df[~df["feature_name"].astype(str).str.contains("diagnostics")]

    # Stable ordering
    sort_cols = [
        col for col in ["patient_id", "modality", "VOI", "feature_name"]
        if col in df.columns
    ]

    if sort_cols:
        df = df.sort_values(sort_cols)

    return df.reset_index(drop=True)


def assert_or_update_features(
    actual: pd.DataFrame,
    golden_path: Path,
    update: bool,
):
    actual = normalize_features(actual)

    if update:
        golden_path.parent.mkdir(parents=True, exist_ok=True)
        actual.to_parquet(golden_path, index=False)
        return

    expected = pd.read_parquet(golden_path)
    assert_features_close(actual, expected)


def assert_features_close(
    actual: pd.DataFrame,
    expected: pd.DataFrame,
    rtol: float = 1e-4,
    atol: float = 1e-4,
):
    actual = normalize_features(actual)
    expected = normalize_features(expected)

    assert list(actual.columns) == list(expected.columns)
    assert len(actual) == len(expected)

    for col in actual.columns:
        if pd.api.types.is_numeric_dtype(actual[col]):
            np.testing.assert_allclose(
                actual[col].to_numpy(),
                expected[col].to_numpy(),
                rtol=rtol,
                atol=atol,
                equal_nan=True,
            )
        else:
            assert actual[col].astype(str).tolist() == expected[col].astype(str).tolist()