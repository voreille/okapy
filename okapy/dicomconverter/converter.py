from pathlib import Path

import numpy as np
import SimpleITK as sitk

from okapy.dicomconverter.dicom_walker import DicomWalker
from okapy.dicomconverter.dicom_file import EmptyContourException
from okapy.dicomconverter.volume import (BasicResampler, MaskResampler,
                                         IdentityProcessor)


class VolumeResult():
    def __init__(self, study, volume, path):
        self.study_instance_uid = study.study_instance_uid
        self.patient_id = study.patient_id
        self.study_date = study.study_date
        self.modality = volume.modality
        self.path = path

    def __str__(self):
        return self.path


class MaskResult(VolumeResult):
    def __init__(self, study, volume, path):
        super().__init__(study, volume, path)
        self.reference_modality = volume.reference_modality
        self.label = volume.label


class StudyConverter():
    def __init__(self,
                 volume_processor,
                 mask_processor,
                 padding=0,
                 extension='nii.gz',
                 converter_backend='sitk',
                 list_labels=None,
                 volume_dtype=np.float32,
                 mask_dtype=np.uint32):
        self.volume_processor = volume_processor
        self.mask_processor = mask_processor
        self.list_labels = list_labels
        self.padding = padding
        self.extension = extension
        self.converter_backend = converter_backend
        self.volume_dtype = volume_dtype
        self.mask_dtype = mask_dtype

    def extract_volume_of_interest(self, study):
        masks_list = list()
        for f in study.mask_files:
            if self.list_labels is None:
                masks_list.extend([f.get_volume(label) for label in f.labels])
            else:
                label_intersection = list(
                    set(f.labels) & set(self.list_labels))
                for label in label_intersection:
                    try:
                        masks_list.append(f.get_volume(label))
                    except EmptyContourException:
                        continue

        # if self.list_labels is None:
        #     missing_contours = []
        # else:
        #     extracted_labels = [v.label for v in masks_list]
        #     missing_contours = [
        #         s for s in self.list_labels if s not in extracted_labels
        #     ]
        return masks_list

    def get_bounding_box(self, masks_list, volumes_list):
        bb = masks_list[0].padded_bb(self.padding)
        for mask in masks_list:
            bb = mask.bb_union(bb, padding=self.padding)

        for v in volumes_list:
            bb = v.reference_frame.bounding_box_intersection(bb)
        return bb

    def write(self, volume, path, dtype=None):
        if dtype:
            volume = volume.astype(dtype)
        counter = 0
        new_path = path
        while new_path.is_file():
            counter += 1
            new_path = path.with_name(
                path.name.replace('.' + self.extension, '') + '(' +
                str(counter) + ')' + '.' + self.extension)
        if self.converter_backend == 'sitk':
            sitk.WriteImage(volume.sitk_image, str(new_path.resolve()))

    def __call__(self, study, output_folder=None):
        volume_results_list = list()
        mask_results_list = list()
        masks_list = self.extract_volume_of_interest(study)
        volumes_list = [f.get_volume() for f in study.volume_files]

        if self.padding != 'whole_image':
            bb = self.get_bounding_box(masks_list, volumes_list)
        elif self.padding == 'whole_image':
            # The image is not cropped
            bb = None
        else:
            raise ValueError(
                "padding must be a positive integer or 'whole_image'")
        masks_list = map(lambda v: self.mask_processor(v, bb), masks_list)
        volumes_list = map(lambda v: self.volume_processor(v, bb),
                           volumes_list)
        for v in volumes_list:
            name = study.patient_id + '__' + v.modality + '.' + self.extension
            path = Path(output_folder) / name
            self.write(v, path, dtype=self.volume_dtype)
            volume_results_list.append(VolumeResult(study, v, path))

        for v in masks_list:
            name = (study.patient_id + '__' + v.label.replace(' ', '_') +
                    '__' + v.modality + '__' + v.reference_modality + '.' +
                    self.extension)
            path = Path(output_folder) / name
            self.write(v, path, dtype=self.mask_dtype)
            mask_results_list.append(MaskResult(study, v, path))

        return volume_results_list, mask_results_list


class Converter():
    def __init__(self,
                 output_folder,
                 extension='nii.gz',
                 resampling_spacing=(1, 1, 1),
                 order=3,
                 padding=0,
                 list_labels=None,
                 dicom_walker=None,
                 volume_processor=None,
                 mask_processor=None,
                 converter_backend='sitk'):
        self.padding = padding
        self.extension = extension
        self.output_folder = output_folder
        self.list_labels = list_labels
        if dicom_walker is None:
            self.dicom_walker = DicomWalker()
        if resampling_spacing == -1:
            if volume_processor is None:
                self.volume_processor = IdentityProcessor()
            if mask_processor is None:
                self.mask_processor = IdentityProcessor()
        else:
            if volume_processor is None:
                self.volume_processor = BasicResampler(
                    resampling_spacing=resampling_spacing, order=order)
            if mask_processor is None:
                self.mask_processor = MaskResampler(
                    resampling_spacing=resampling_spacing)
        self.converter_backend = converter_backend
        self.study_converter = StudyConverter(
            self.volume_processor,
            self.mask_processor,
            list_labels=list_labels,
            converter_backend=converter_backend,
            extension=extension,
            padding=padding)

    def __call__(self, input_folder, output_folder=None):
        if output_folder is None:
            output_folder = self.output_folder

        studies_list = self.dicom_walker(input_folder)
        result = list()
        for study in studies_list:
            result.append(self.study_converter(study, output_folder))

        return result
