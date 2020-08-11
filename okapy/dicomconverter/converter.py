from os.path import join

import pandas as pd
from scipy import ndimage
import SimpleITK as sitk

from okapy.dicomconverter.dicom_walker import DicomWalker
from okapy.dicomconverter.dicom_file import Study, EmptyContourException
from okapy.dicomconverter.volume import BasicResampler, MaskResampler


class StudyConverterResult():
    def __init__(self, study):
        self.study_instance_uid = study.study_instance_uid
        self.patient_id = study.patient_id
        self.study_date = study.study_date
        self.volume_paths = list()
        self.mask_paths = list()
        self.mask_keys = list()
        self.volume_keys = list()

    def addpath_mask(self, key, path):
        self.mask_keys.append(key)
        self.mask_paths.append(path)

    def addpath_volume(self, key, path):
        self.volume_keys.append(key)
        self.volume_paths.append(path)


class ConverterResult():
    def __init__(self, list_labels):
        self.data_frame = pd.DataFrame()

    def add_result(self, result):
        r = {
            'image_' + str(key): [value]
            for (key, value) in zip(result.volume_keys, result.volume_paths)
        }
        r.update({
            'mask_' + str(key): [value]
            for (key, value) in zip(result.mask_keys, result.mask_paths)
        })
        r.update({'patient_id': [str(result.patient_id)]})
        self.data_frame = pd.concat((self.data_frame, pd.DataFrame(r)),
                                    ignore_index=True)

    def to_csv(self, path):
        self.data_frame.to_csv(path)


class StudyConverter():
    def __init__(self,
                 volume_processor,
                 mask_processor,
                 padding=0,
                 extension='nii.gz',
                 converter_backend='sitk',
                 list_labels=None):
        self.volume_processor = volume_processor
        self.mask_processor = mask_processor
        self.list_labels = list_labels
        self.padding = padding
        self.extension = extension
        self.converter_backend = converter_backend

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

    def write(self, volume, path):
        if self.converter_backend == 'sitk':
            sitk.WriteImage(volume.sitk_image, path)

    def __call__(self, study, output_folder=None):
        result = StudyConverterResult(study)
        masks_list = self.extract_volume_of_interest(study)
        volumes_list = [f.get_volume() for f in study.volume_files]

        bb = self.get_bounding_box(masks_list, volumes_list)
        masks_list = map(lambda v: self.mask_processor(v, bb), masks_list)
        volumes_list = map(lambda v: self.volume_processor(v, bb),
                           volumes_list)
        for v in volumes_list:
            name = study.patient_id + '_' + v.modality + '.' + self.extension
            path = join(output_folder, name)
            self.write(v, path)
            result.addpath_volume(v.modality, path)

        for v in masks_list:
            name = (study.patient_id + '_' + v.label + '_' + v.modality + '_' +
                    v.reference_modality + '.' + self.extension)
            path = join(output_folder, name)
            self.write(v, path)
            result.addpath_mask(v.label, path)

        return result


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
        result = ConverterResult(self.list_labels)
        for study in studies_list:
            result.add_result(self.study_converter(study, output_folder))

        return result
