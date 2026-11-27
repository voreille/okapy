# okapy

Modular biomedical image conversion, preprocessing and feature extraction:

- **DICOM → NIfTI conversion**: CT, MR and PET (with SUV), Enhanced PET, RTSTRUCT and SEG.
- **Geometry preprocessing**: crop to masks and/or the common field of view, and resampling to a target spacing.
- **Feature extraction**: pyradiomics, native PET features (SUVmax/SUVpeak/…), or any external command.

## Installation

okapy needs Python ≥ 3.10.

```bash
python -m pip install -e .                # core
python -m pip install -e ".[dev]"         # + pytest, ruff, mypy, build, pyarrow
```

### pyradiomics

pyradiomics 3.0.1 only installs on Python 3.10, and its wheels are built against NumPy 1.x. The
`radiomics` extra therefore pins `numpy<2`:

```bash
conda create -n okapy-pyradiomics python=3.10
conda activate okapy-pyradiomics
python -m pip install --upgrade pip setuptools wheel
python -m pip install "numpy<2" SimpleITK
python -m pip install pyradiomics==3.0.1 --no-build-isolation
python -m pip install -e ".[dev,radiomics,cli]"
```

If pyradiomics is installed next to NumPy 2, importing it fails with
`numpy.core.multiarray failed to import`.

## Usage

```python
from okapy.pipeline import build_extraction_pipeline, build_preprocessing_pipeline

# DICOM → NIfTI → preprocessing → features, as a pandas DataFrame.
pipeline = build_extraction_pipeline("parameters/defaults/defaults_pyradiomics_only.yaml")
features = pipeline.run("path/to/dicom", work_dir="path/to/work", labels=["GTV"])

# DICOM → NIfTI → preprocessing only (no feature extraction).
preprocessing = build_preprocessing_pipeline("my_config.yaml")
volumes = preprocessing.run("path/to/dicom", work_dir="path/to/work")
```

Builders accept a YAML path (`str` or `Path`) or a mapping. `okapy.config.load_config(source)`
loads and validates a configuration in the same way.

## Configuration

A configuration has these top-level sections:

- `conversion`: DICOM conversion options, such as `continue_on_error`, `extension` and the
  RTSTRUCT/SEG reference checks.
- `general`: study-level options, such as `combine_segmentation`, `skip_images_without_masks`,
  `write_outputs` and the output subdirectories.
- `geometry_preprocessing`: the crop and resampling grid, set per selector (`common`, a modality
  such as `CT` or `PT`, or `default`).
- `local_preprocessing`: image-only processors.
- `mask_preprocessing`: mask cleanup after resampling.
- `feature_extraction`: the feature backends per selector.

`geometry_preprocessing` keys:

| Key | Default | Meaning |
|---|---|---|
| `spacing` | native | Target spacing `[x, y, z]` in mm. Omit it to keep the native spacing; `-1` on one axis keeps that axis native. |
| `crop_to_masks` | `true` | Crop the grid to the masks' extent. |
| `crop_to_common_fov` | `false` | Crop to the intersection of the study's image FOVs. The intersection is axis-aligned, so for oblique images it is a bounding box of the overlap. |
| `padding_voxels` | `0` | Voxels of the target grid added around the masks. Either an integer or a per-axis `[x, y, z]` list. Padding never extends beyond the common FOV. |
| `image_interpolator` | `linear` | Image interpolator. Masks are always resampled linearly and thresholded at 0.5. |
| `default_image_value` / `default_mask_value` | `0` | Fill values outside the source image. |

Parsing is strict:

- Unknown keys raise `ValueError`.
- Booleans, in every section, must be real booleans or one of `true/false`, `yes/no`, `on/off`
  or `1/0`.
- Study options (`combine_segmentation`, `write_outputs`, …) belong under `general`. A
  top-level copy is rejected, and so is `general.result_format`; use
  `feature_extraction.result_format` instead.
- Empty YAML sections (`general:` with no value) count as empty mappings.

### Images without masks

`general.skip_images_without_masks` controls what happens to an image that has no masks:

- Default (unset): when cropping to masks, the image is skipped with a warning and the rest of the
  study is still processed. When not cropping, the image is processed normally.
- `true`: always skip such images.
- `false`: raise an error when cropping to masks.

Masks that are entirely empty are dropped with a warning before cropping. So are masks that
become empty after resampling, for example a tiny ROI on a coarse grid.

### Legacy configurations

Configurations written for okapy < 0.2 are detected (`okapy.config.is_legacy_config`) and migrated
automatically by `load_config` and every pipeline builder. A configuration that mixes legacy keys
(e.g. `general.padding`, `volume_preprocessing`) with new-shape sections is rejected rather than
silently overwritten. Migration details:

- `general.padding` is in millimetres in legacy configs. It becomes `padding_voxels` per axis as
  `round(padding / target spacing)`.
- `padding: whole_image` becomes `crop_to_masks: false`.
- If `padding` is absent, the legacy default of 10 mm is used. When `common` has no target
  spacing, that implicit default is skipped for `common`, with a warning.
- An explicit `padding` can only be converted when the target spacing is known. If an axis keeps
  its native spacing, or the spacing is missing, migration raises and asks for an explicit
  `padding_voxels`. `padding: 0` is always 0 voxels.
- Rounding is half-to-even, as legacy `np.round` did: 10 mm at 4 mm becomes 2 voxels.
- `general.patch_size` and the standard `study_processor` are dropped with a warning.
  `study_processor: studyprocessor_deep` is not supported.
- A selector whose spacing comes only from the legacy mask resampler inherits the voxel padding
  of `common`.
- Legacy `okapy`, `riesz` and `zrad` feature extractors are not supported and raise a clear error.

### Trusted configurations

The `command` feature backend runs the external command written in the configuration, with the
image, mask, params and output paths as arguments, using your user's permissions. `command` must
be a list of arguments, not a shell string, and it is run without a shell. With
`write_outputs: false`, the backend writes the in-memory image and mask into its work directory
first. Treat any
configuration that uses a `command` backend as trusted input, like a script. Only run
configurations you wrote yourself or have reviewed, and never load one from an untrusted source.

Output file and directory names derived from DICOM headers are sanitized: only `[A-Za-z0-9._-]`
is kept, with at most 128 characters per part. A joined name longer than 255 bytes is truncated,
and a short hash keeps it unique. Every written path is checked to stay inside the output
directory.

## Supported inputs and limitations

okapy prefers failing loudly to writing a wrong volume. These inputs are rejected with a clear
`ValueError` (or `SUVComputationError` for SUV), naming the series and the reason:

- Series in which several slices share a position: 4D, dynamic or multi-echo acquisitions.
- Slice gaps that remain non-uniform by more than 1% after missing-slice handling, and
  near-duplicate slices.
- Tilted or oblique stacking, where positions are not along the slice normal (e.g. gantry
  tilt). An oblique but consistent series converts normally.
- IODs without image geometry, such as DX, CR, MG, US and XA, and Enhanced CT/MR.
- A SeriesInstanceUID shared by files of different modalities. The rest of the input is still
  collected.
- A mask whose referenced image is missing or failed to convert. The fallback to "the only
  image of the study" applies only to masks without reference UIDs. It also applies when the
  reference check is disabled (`conversion.check_rtstruct_reference_uid: false` /
  `check_seg_reference_uid: false`), for example with anonymised data; a warning is logged.
- Invalid or missing SUV inputs, for example malformed DICOM TM/DT values, or injection and
  scan times that carry different UTC offsets.

With `conversion.continue_on_error: true`, these errors are logged and the series is skipped instead of
aborting the run (`feature_extraction.continue_on_error` does the same for feature backends).

Other conversion details:

- A single missing slice is interpolated.
- A larger single gap is filled with the modality's background value.
- Single-slice series and single-frame Enhanced PET are supported.
- MR and other non-CT modalities apply `RescaleSlope` and `RescaleIntercept`.
- RTSTRUCT contours nested at an odd depth make holes. Touching or overlapping contours of one
  ROI fill their union. Contours partly outside the image grid are clipped with a warning.

SUV conversion:

- DICOM TM/DT values with reduced precision (`HH`, `HHMM`, …), fractional seconds and UTC
  offsets are parsed by length. Offsets are validated, then dropped; if both the injection and
  the scan times carry an offset, the offsets must match.
- GE vendors are matched by prefix only (`GE MEDICAL…`, `GEMS`, `GE …`, `GENERAL ELECTRIC`).
- A `PatientSize` above 3 is taken as centimetres and converted with a warning.
- Vendor-private scan-start tags are read only when their private creator is present.

## Running the tests

```bash
pytest                          # unit tests; integration tests skip without data
pytest -m "not integration"     # what CI runs
ruff check src/okapy tests && ruff format --check src/okapy tests
```

Integration tests need external data, passed through environment variables. They skip when the
variable is unset or points to a missing directory.

- `SUV_COMPUTATION_TEST_DATA`: a directory of SUV DRO cases (`DRO_*` subdirectories).
- `OKAPY_TEST_DATA`: the curated DICOM collections (`curated-v0/`).

Tests that need pyradiomics skip when it cannot be imported. To (re)create golden snapshot files,
run with `UPDATE_GOLDEN=1`.

`src/okapy_legacy` is the previous implementation. It stays in the repository as a frozen
reference for the parity tests, but is not part of the installed package.
