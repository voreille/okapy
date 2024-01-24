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

    from okapy.dicomconverter.converter import NiftiConverter

    path_input = "path/to/DICOM/folder"
    path_output = "path/to/NIfTI/folder"
    converter = NiftiConverter()
    result_conversion = converter(path_input, output_folder=output_folder)

The `result_conversion` is a summary of the conversion. Calling :class:`okapy.dicomconverter.converter.NiftiConverter` like this will
read all the DICOM files from the `path_input` and convert all the images to NIfTI and store it in the `path_output` folder.
If RTSTRUCT are present, all the labels contained in the RTSTRUCT files will be stored in different NIfTI files (this behaviour can be controlled
with the `labels_startswith` parameter that can be passe to the constructor of :class:`okapy.dicomconverter.converter.NiftiConverter`).


Example of the second usage::

    from okapy.dicomconverter.converter import ExtractorConverter

    path_input = "path/to/DICOM/folder"
    path_to_params = "path/to/parameters.yaml"
    converter = ExtractorConverter.from_params(path_to_params)
    result_conversion = converter(path_input)

The `result_conversion` is :class:`pandas.DataFrame` containing all the feature values for each images.
To use the converter this way, one must ensure that segmentation are present for each studies in the form of SEG or RTSTRUCT files.
For PET images, one might want to use a RSTRUCT drawn on the CT, this can be achieved with the parameter :code:`combine_segmentation: True` in the
parameter file. More details are provided below for the parameter file.