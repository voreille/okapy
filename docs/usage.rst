=====
Usage
=====

OkaPy is designed for processing 3D DICOM images.

There are two primary use cases for OkaPy:

1. **Conversion to NIfTI Files:**
   OkaPy allows you to convert unsorted DICOM files to NIfTI files, specifically handling 3D images.

2. **Feature Extraction from DICOM Files:**
   Another usage involves extracting features directly from unsorted DICOM files. In this scenario, OkaPy first converts the DICOM files to NIfTI format and then utilizes `PyRadiomics <https://www.radiomics.io/pyradiomics.html>`_ to perform feature extraction.


Example of the first usage::

    import okapy
