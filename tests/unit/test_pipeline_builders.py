"""Pipeline builders: extraction/preprocessing pipelines and config handling."""

from __future__ import annotations

import inspect
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

from okapy.pipeline import ExtractionPipeline, build_extraction_pipeline

PARAMETERS_DIR = Path(__file__).parents[2] / "parameters"
DEFAULTS_PYRADIOMICS_ONLY = (
    PARAMETERS_DIR / "defaults" / "defaults_pyradiomics_only.yaml"
)

SERIES_UID = "1.2.826.0.1.3680043.8.498.17"


class _FakeRadiomicsFeatureExtractor:
    instances: list = []

    def __init__(self, params=None):
        self.params = params
        type(self).instances.append(self)

    def execute(self, image, mask, **kwargs):
        return {"original_firstorder_Mean": 1.0}


@pytest.fixture
def fake_pyradiomics(monkeypatch):
    """Stand-in for the external pyradiomics package (broken under NumPy 2)."""

    package = types.ModuleType("radiomics")
    module = types.ModuleType("radiomics.featureextractor")
    _FakeRadiomicsFeatureExtractor.instances = []
    module.RadiomicsFeatureExtractor = _FakeRadiomicsFeatureExtractor
    package.featureextractor = module
    monkeypatch.setitem(sys.modules, "radiomics", package)
    monkeypatch.setitem(sys.modules, "radiomics.featureextractor", module)
    return _FakeRadiomicsFeatureExtractor


def write_ct_series(directory: Path, *, n_slices=4, rows=8, cols=8, patient_id="P1"):
    """Write a tiny axial CT series (1 x 1 x 2 mm voxels, HU = slice index)."""

    directory.mkdir(parents=True, exist_ok=True)
    study_uid = generate_uid()
    frame_uid = generate_uid()
    for index in range(n_slices):
        sop_uid = generate_uid()
        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = CTImageStorage
        meta.MediaStorageSOPInstanceUID = sop_uid
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        path = directory / f"ct{index:03d}.dcm"
        ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
        ds.SOPClassUID = CTImageStorage
        ds.SOPInstanceUID = sop_uid
        ds.Modality = "CT"
        ds.PatientID = patient_id
        ds.StudyInstanceUID = study_uid
        ds.SeriesInstanceUID = SERIES_UID
        ds.FrameOfReferenceUID = frame_uid
        ds.ImagePositionPatient = [0.0, 0.0, 2.0 * index]
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        ds.PixelSpacing = [1.0, 1.0]
        ds.SliceThickness = 2.0
        ds.Rows = rows
        ds.Columns = cols
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 1
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.RescaleSlope = 1
        ds.RescaleIntercept = -1000
        ds.InstanceNumber = index + 1
        ds.PixelData = np.full((rows, cols), 1000 + index, dtype=np.int16).tobytes()
        ds.save_as(str(path), enforce_file_format=True)
    return study_uid


def _new_config(**geometry):
    return {
        "general": {},
        "geometry_preprocessing": {
            "common": {"spacing": [2.0, 2.0, 2.0], "crop_to_masks": False, **geometry}
        },
    }


# ---------------------------------------------------------------------------
# build_preprocessing_pipeline / PreprocessingPipeline
# ---------------------------------------------------------------------------


def test_preprocessing_pipeline_is_exported_from_okapy_pipeline():
    import okapy.pipeline as pipeline

    assert callable(pipeline.build_preprocessing_pipeline)
    assert inspect.isclass(pipeline.PreprocessingPipeline)
    assert "build_preprocessing_pipeline" in pipeline.__all__
    assert "PreprocessingPipeline" in pipeline.__all__


def test_preprocessing_pipeline_has_no_feature_result_options():
    from okapy.pipeline import PreprocessingPipeline

    parameters = inspect.signature(PreprocessingPipeline).parameters

    assert "result_format" not in parameters
    assert "include_feature_backend" not in parameters


def test_build_preprocessing_pipeline_ignores_feature_result_options():
    from okapy.pipeline import PreprocessingPipeline, build_preprocessing_pipeline

    config = _new_config()
    config["feature_extraction"] = {"result_format": "wide", "include_backend": True}

    pipeline = build_preprocessing_pipeline(config)

    assert isinstance(pipeline, PreprocessingPipeline)
    assert not hasattr(pipeline, "result_format")
    assert not hasattr(pipeline, "include_feature_backend")


def test_preprocessing_pipeline_end_to_end_resamples_synthetic_ct(tmp_path):
    from okapy.pipeline import build_preprocessing_pipeline

    study_uid = write_ct_series(tmp_path / "dicom")
    pipeline = build_preprocessing_pipeline(_new_config())

    result = pipeline.run(tmp_path / "dicom", work_dir=tmp_path / "work")

    assert len(result.studies) == 1
    assert result.studies[0].study_instance_uid == study_uid
    images = result.images
    assert len(images) == 1
    image = images[0]
    assert image.series_instance_uid == SERIES_UID
    assert image.image.GetSpacing() == (2.0, 2.0, 2.0)
    assert image.image.GetSize() == (4, 4, 4)
    assert image.path.is_file()
    assert image.path.resolve().is_relative_to((tmp_path / "work").resolve())
    assert image.path.name.endswith(".nii.gz")


def test_preprocessing_pipeline_accepts_str_paths(tmp_path):
    from okapy.pipeline import build_preprocessing_pipeline

    write_ct_series(tmp_path / "dicom")
    pipeline = build_preprocessing_pipeline(_new_config())

    result = pipeline.run(str(tmp_path / "dicom"), work_dir=str(tmp_path / "work"))

    assert len(result.images) == 1


def test_build_preprocessing_pipeline_migrates_legacy_config(tmp_path):
    from okapy.pipeline import build_preprocessing_pipeline

    write_ct_series(tmp_path / "dicom")
    legacy = {
        "general": {"padding": "whole_image"},
        "volume_preprocessing": {
            "common": {
                "bspline_resampler": {"resampling_spacing": [2, 2, 2], "order": 1}
            }
        },
    }

    result = build_preprocessing_pipeline(legacy).run(
        tmp_path / "dicom", work_dir=tmp_path / "work"
    )

    assert result.images[0].image.GetSpacing() == (2.0, 2.0, 2.0)


# ---------------------------------------------------------------------------
# build_extraction_pipeline
# ---------------------------------------------------------------------------


def test_build_extraction_pipeline_reads_result_options():
    config = _new_config()
    config["feature_extraction"] = {"result_format": "wide", "include_backend": True}

    pipeline = build_extraction_pipeline(config)

    assert isinstance(pipeline, ExtractionPipeline)
    assert pipeline.result_format == "wide"
    assert pipeline.include_feature_backend is True


def test_build_extraction_pipeline_from_shipped_legacy_defaults_path(
    fake_pyradiomics,
):
    pipeline = build_extraction_pipeline(DEFAULTS_PYRADIOMICS_ONLY)

    assert isinstance(pipeline, ExtractionPipeline)
    backends = pipeline.feature_extraction.backends_by_selector
    assert sorted(backends) == ["CT", "MR", "PT"]
    assert [b.name for b in backends["CT"]] == [
        "pyradiomics_original",
        "pyradiomics_log",
    ]
    assert "pet" in [b.name for b in backends["PT"]]
    assert len(fake_pyradiomics.instances) == 6


def test_build_extraction_pipeline_from_shipped_legacy_defaults_mapping(
    fake_pyradiomics,
):
    raw = yaml.safe_load(DEFAULTS_PYRADIOMICS_ONLY.read_text())

    pipeline = build_extraction_pipeline(raw)

    assert sorted(pipeline.feature_extraction.backends_by_selector) == [
        "CT",
        "MR",
        "PT",
    ]


def test_build_extraction_pipeline_from_shipped_defaults_with_real_pyradiomics():
    pytest.importorskip("radiomics.featureextractor", exc_type=ImportError)

    pipeline = build_extraction_pipeline(DEFAULTS_PYRADIOMICS_ONLY)

    assert isinstance(pipeline, ExtractionPipeline)


def test_build_extraction_pipeline_tolerates_null_feature_sections():
    config = _new_config()
    config["feature_extraction"] = {"common": None, "CT": None}

    pipeline = build_extraction_pipeline(config)

    assert isinstance(pipeline, ExtractionPipeline)


def test_extraction_pipeline_runs_with_str_work_dir(tmp_path):
    write_ct_series(tmp_path / "dicom")
    config = _new_config()
    config["feature_extraction"] = {"common": [{"type": "pet"}]}
    pipeline = build_extraction_pipeline(config)

    df = pipeline.run(str(tmp_path / "dicom"), work_dir=str(tmp_path / "work"))

    assert isinstance(df, pd.DataFrame)
    assert (tmp_path / "work" / "converted").is_dir()
    assert (tmp_path / "work" / "preprocessed").is_dir()
