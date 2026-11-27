=======
History
=======

0.3.0 (2026-11-27)
------------------

Quality pass after the major restructure (#37). Behaviour changes:

* Unsupported or inconsistent DICOM input now fails loudly with a ``ValueError``
  naming the series: 4D/multi-echo duplicate positions, non-uniform slice gaps,
  tilted stacks, IODs without image geometry, mixed-modality series, and masks
  whose referenced image is missing. ``conversion.continue_on_error`` skips them.
* SUV: DICOM TM/DT values are parsed by length (``HHMM`` was misread, giving a
  ~27 min decay error), and fractional seconds are kept. Mismatched UTC offsets
  between injection and scan raise. ``PatientSize`` in cm is converted, GE is
  matched by vendor prefix, and private tags need their private creator.
* Preprocessing: images without masks are skipped by default when cropping;
  empty masks are dropped; tie voxels are kept; padding respects the common
  FOV. ``spacing`` is optional (native), and ``padding_voxels`` may be per axis.
* Config: unknown keys, ambiguous booleans and misplaced options are rejected.
  Legacy configs are auto-migrated by ``load_config``, with padding converted
  from mm to voxels. Mixed legacy/new configs are rejected.
* RTSTRUCT holes are decided by contour nesting; contours outside the grid are
  clipped with a warning. MR and other modalities apply rescale slope/intercept.
* Header-derived file names are sanitized and contained in the output directory.
* ``PreprocessingPipeline`` / ``build_preprocessing_pipeline`` are exported.
  ``PETFeatureBackend`` moved to ``okapy.features.backends`` (the old path still
  works). ``RTStructMaskConverter(check_reference=...)`` is deprecated in favour
  of ``check_reference_uid``.
* Packaging: ``okapy.dicom`` is now shipped; ``okapy_legacy`` is no longer part
  of the wheel. ``scipy`` was dropped from the core dependencies. The
  ``radiomics`` extra pins ``numpy<2``.
* Removed the obsolete parameter files ``param.yaml``, ``pyradiomics/param.yaml``
  and ``defaults_zrad.yaml``, and the ``riesz`` extractor.

0.1.0 (2019-05-24)
------------------

* First release on PyPI.
* First Commit
