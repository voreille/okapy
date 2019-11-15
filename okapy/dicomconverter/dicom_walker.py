'''
TODO: Is it better to create a class for just the header and another for the
files?
'''
import os
from os.path import dirname, join
from string import Template
from functools import partial

import re
import numpy as np
import SimpleITK as sitk
import pydicom as pdcm
from skimage.draw import polygon
from pydicom.filereader import read_dicomdir
from pydicom.errors import InvalidDicomError
import SimpleITK as sitk

from .image import ImageCT, ImagePT, Mask, ImageMR


dic_sitk_writer = {
        'nrrd': 'NrrdImageIO',
        'nii': 'NiftiImageIO'
    }

class DicomHeader():
    def __init__(self, patient_id=None,
                 study_instance_uid=None,
                 series_instance_uid=None,
                 modality=None):
        self.patient_id = patient_id
        self.study_instance_uid = study_instance_uid
        self.series_instance_uid = series_instance_uid
        self.modality = modality

    def __str__(self):
        return ('PatientID: {}, StudyInstanceUID: {}, SeriesInstanceUID: {},'
                ' Modality: {}'.format(self.patient_id,
                                       self.study_instance_uid,
                                       self.series_instance_uid,
                                       self.modality))

    def __eq__(self, dcm_header):
        '''
        Could be written more efficiently with a function like dir()
        for another time
        '''
        if isinstance(dcm_header, DicomHeader):
            return (
                self.patient_id == dcm_header.patient_id and
                self.study_instance_uid == dcm_header.study_instance_uid and
                self.series_instance_uid == dcm_header.series_instance_uid and
                self.modality == dcm_header.modality
            )
        else:
            return False


class DicomFile():
    def __init__(self, dicom_header=DicomHeader(), path=None):
        self.dicom_header = dicom_header
        self.path = path

    def __str__(self):
        return str(self.dicom_header)


class DicomWalker():
    def __init__(self, input_dirpath, output_dirpath,
                 template_filename=Template('${patient_id}_'
                                            '${modality}.${ext}'),
                 extension_output='nrrd',
                 list_labels=None):
        self.input_dirpath = input_dirpath
        self.output_dirpath = output_dirpath
        self.template_filename = template_filename
        self.extension_output = extension_output
        self.sitk_writer = sitk.ImageFileWriter()
        self.sitk_writer.SetImageIO(dic_sitk_writer[extension_output])
        self.dicom_files = list()
        self.images = list()
        self.list_labels = list_labels

    def __str__(self):
        dcm_list = [str(dcm) for dcm in self.dicom_files]
        return '\n'.join(dcm_list)


    def _dict_for_modality(self, dcm_header=None):
        out_dict = {
            'CT': ImageCT,
            'PT': ImagePT,
            'MR': ImageMR,
        }
        for im in reversed(self.images):
            if (im.dicom_header.series_instance_uid == dcm_header.series_instance_uid):
                out_dict['RTSTRUCT'] = partial(Mask,
                                               reference_image=im,
                                               list_labels=self.list_labels)
                break

        return out_dict


    def walk(self):
        '''
        Method to walk through the path given and fill the list of DICOM
        headers and sort them
        '''
        for (dirpath, dirnames, filenames) in os.walk(self.input_dirpath):
            for filename in filenames:
                try:
                    data = pdcm.dcmread(join(dirpath, filename),
                                        stop_before_pixels=True)
                except InvalidDicomError:
                    print('This file {} is not recognised as DICOM'.format(
                        filename))
                    continue
                if data.Modality == 'RTSTRUCT':
                    # Adaptation from QuantImage
                    series_instance_uid = (data
                                           .ReferencedFrameOfReferenceSequence[0]
                                           .RTReferencedStudySequence[0]
                                           .RTReferencedSeriesSequence[0]
                                           .SeriesInstanceUID)
                else:
                    series_instance_uid = data.SeriesInstanceUID

                dicom_header = DicomHeader(
                    patient_id=data.PatientID,
                    study_instance_uid=data.StudyInstanceUID,
                    series_instance_uid=series_instance_uid,
                    modality=data.Modality
                )
                self.dicom_files.append(DicomFile(dicom_header=dicom_header,
                                                    path=join(dirpath,
                                                              filename)))


        self.dicom_files.sort(key=lambda x: (x.dicom_header.patient_id,
                                               x.dicom_header.study_instance_uid,
                                               x.dicom_header.series_instance_uid,
                                               x.dicom_header.modality,
                                               x.path))


    def _append_image(self, im_dicom_files, dcm_header):
        try:
            image_modality_dict = self._dict_for_modality(
                dcm_header=dcm_header)

            self.images.append(image_modality_dict[dcm_header.modality](
                sitk_writer=self.sitk_writer,
                dicom_header=dcm_header,
                dicom_paths=[k.path for k in im_dicom_files],
                template_filename=self.template_filename
            ))
        except KeyError:
            print('This modality {} is not yet (?) supported'
                  .format(dcm_header.modality))


    def fill_images(self):
        '''
        Construct the tree-like dependency of the dicom
        It all relies on the fact that the collection of headers has been
        sorted
        '''
        im_dicom_files = list()
        dcm_header = DicomHeader()

        for i, f in enumerate(self.dicom_files):
            if i > 0 and not f.dicom_header == self.dicom_files[i-1].dicom_header:
                self._append_image(im_dicom_files, dcm_header)
                im_dicom_files = list()

            im_dicom_files.append(f)
            dcm_header= f.dicom_header

        self._append_image(im_dicom_files, dcm_header)

    def convert(self):
        for im in self.images:
            im.convert(self.output_dirpath)
