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



Parameter File
--------------

The parameter file for OkaPy is a YAML file containing configuration settings for various aspects of the processing pipeline. Here is an example:

.. code-block:: yaml

    general:
      padding: 10 # padding in [mm] around the union of the segmentations to avoid resampling huge image, "whole_image" is a possible value if you want to resample the whole image
      submodalities: False # used for MR, if true okapy parse the SeriesDescription DICOM tag for ` --- ` and append the following string to the modality
      combine_segmentation: False # If the segmentation within a study should be used for all modality (can be useful for PT when the RTSTRUCT is drawn on the CT)
      result_format: "long" # "long" or "multiindex", "long" should be used
      additional_dicom_tags: # Add DICOM tag here, they will be added in the final Dataframe
        - "SeriesInstanceUID"

    volume_preprocessing: # all the preprocessing applied to the images by modality
      common: # to apply to all the images
      PT:
        bspline_resampler: # name defined when subclassing `okapy.dicomconverter.volume_processor.VolumeProcessor`
          resampling_spacing: [2.0, 2.0, 2.0]
          order: 3
      CT:
        bspline_resampler:
          resampling_spacing: [1.0, 1.0, 1.0]
          order: 3
      default: # to apply if the modality is not defined

    mask_preprocessing: # all the preprocessing applied to the segmentation (RTSTRUCT or SEG), the "pixel_spacing" is inferred on the image it corresponds to
      default:
        binary_bspline_resampler:
          order: 3

    feature_extraction: # parameters for the feature extraction, can be defined for each modality/submodality
      MR:
        pyradiomics: # The following are the parameters for pyradiomics, you can paste parameters.yaml from their github with the right indentation 
          original:
            imageType:
              Original: {}
            featureClass:
              shape:
              firstorder:

            setting:
              normalize: False
              normalizeScale: 100  # This allows you to use more or less the same bin width.
              binWidth: 5
              voxelArrayShift: 0
              label: 1
      common: # Feature extraction applied to all modalities
          riesz: # Not supported out of the box, a MATLAB image of QuantImage v1 is used for that
            extractor0:
              RieszOrder: 1
              RieszScales: 4

Other examples can be found in :file:`parameters/`.
The preprocessing applied to the images and segmentations can be tailored to your need. This step is abstracted by the class :class:`okapy.dicomconverter.volume_processor.VolumeProcessor`.
By subclassing this class it is possible to define your own preprocessing step.


Subclassing :class:`VolumeProcessor`
------------------------------------


You can create a custom processor by subclassing :class:`okapy.dicomconverter.volume_processor.VolumeProcessor`. 
When doing so, it's essential to specify a `name` parameter for reference in the parameter YAML file. Here's an example:

.. code-block:: python

    class MyImageProcessor(VolumeProcessor, name="my_image_processor"):

        def __init__(self, *args, my_arg1=None, my_arg2=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.my_arg1 = my_arg1
            self.my_arg2 = my_arg2

        def process(self, volume, *args, mask_file=None, **kwargs):
            # your processing
            return volume


The `MyImageProcessor` example demonstrates subclassing `VolumeProcessor`, specifying initialization arguments through `__init__`, and implementing the processing logic in the `process` method. Ensure to provide the required `name` parameter.

Requirements for the subclass:
- Implement the :meth:`__init__` method specifying preprocessing arguments.
- Implement the :meth:`process` method, which receives the image (`volume` variable) and segmentation (`mask_file` variable).

The base class, :class:`okapy.dicomconverter.volume_processor.VolumeProcessor`, provides access to the segmentation resampler through the `mask_resampler` attribute.

Additional Example: Masked Standardizer
----------------------------------------

Here's an additional example illustrating a processor that standardizes an image within a region defined by a specific label:

.. code-block:: python

    class MaskedStandardizer(VolumeProcessor, name="masked_standardizer"):

        def __init__(self, *args, mask_label="", mask_resampler=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.mask_label = mask_label
            if mask_resampler is None:
                raise TypeError("mask_resampler cannot be None")
            self.mask_resampler = mask_resampler

        def _get_mask_array(self, mask_files, reference_frame=None):
            mask = None
            for f in mask_files:
                if self.mask_label in f.labels:
                    mask = f.get_volume(self.mask_label)
                    break
            if mask is None:
                raise MissingSegmentationException(
                    f"The label {self.mask_label} was not found.")
            return self.mask_resampler(mask, new_reference_frame=reference_frame).array != 0

        def process(self, volume, mask_files=None, **kwargs):
            array = volume.array
            mask_array = self._get_mask_array(
                mask_files, reference_frame=volume.reference_frame)
            mean = np.mean(array[mask_array])
            std = np.std(array[mask_array])
            array = (array - mean) / std
            volume.array = array
            return volume

Now if you want to use the `MaskedStandardizer` in your pipeline, simply write this in your YAML file (under the section preprocesing):

.. code-block:: yaml

    volume_preprocessing: 
      MR:
        masked_standardizer: # the name you defined during suclassing
          mask_label: "edema" # the argument you define, here as an example we took edema