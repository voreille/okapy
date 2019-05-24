'''
TODO: Is it better to create a class for just the header and another for the
files?
'''
import os
from os.path import dirname, join

import re
import numpy as np
import SimpleITK as sitk
import pydicom as pdcm
from skimage.draw import polygon
from pydicom.filereader import read_dicomdir
from pydicom.errors import InvalidDicomError
import SimpleITK as sitk

from image import ImageCT, ImagePT, Mask


class DicomHeader():
    def __init__(self, patient_id, study_instance_uid, series_instance_uid,
                 modality, path):
        self.patient_id = patient_id
        self.study_instance_uid = study_instance_uid
        self.series_instance_uid = series_instance_uid
        self.modality = modality
        self.path = path

    def __str__(self):
        return ('PatientID: {}, StudyInstanceUID: {}, SeriesInstanceUID: {},'
                ' Modality: {}'.format(self.patient_id,
                                       self.study_instance_uid,
                                       self.series_instance_uid,
                                       self.modality))

    def has_same_header(self, dcm_header):
        output = True
        for a in dir(self):
            if not a.startswith('__') and a != 'path':
                output = output and getattr(self, a) == getattr(dcm_header, a)

        return output


class DicomWalker():
    def __init__(self, input_dirpath, output_dirpath,
                 sitk_writer=sitk.ImageFileWriter(),
                 list_labels=None):
        self.input_dirpath = input_dirpath
        self.output_dirpath = output_dirpath
        self.dicom_headers = list()
        self.images = list()
        self.list_labels=list_labels

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

                self.dicom_headers.append(DicomHeader(data.PatientID,
                                                      data.StudyInstanceUID,
                                                      series_instance_uid,
                                                      data.Modality,
                                                      join(dirpath, filename)
                                                      ))

        self.dicom_headers.sort(key=lambda x: (x.patient_id,
                                               x.study_instance_uid,
                                               x.series_instance_uid,
                                               x.modality,
                                               x.path))

        def fill_images(self):
            '''
            Construct the tree-like dependency of the dicom
            It all relies on the fact that the collection of headers has been
            sorted
            '''
            im_dicom_headers = list()
            for i, f in enumerate(self.dicom_headers):
                if i > 0 and f.has_same_header(self.dicom_headers[i-1]):
                    # Could be done more elegantly but the RTSTRUCT fucks
                    # everything, maybe there is a solution with dict of
                    # function
                    if f.modality == 'CT':
                        self.images.append(ImageCT(
                            dicom_headers=im_dicom_headers))
                    elif f.modality == 'PT':
                        self.images.append(ImagePT(
                            dicom_headers=im_dicom_headers))
                    elif f.modality == 'RTSTRUCT':
                        for im in reversed(self.images):
                            if (im.dicom_headers[0].series_instance_uid ==
                                f.series_instance_uid):
                                self.images.append(Mask(
                                    dicom_headers=im_dicom_headers,
                                    reference_image=im,
                                    list_labels=self.list_labels
                                ))
                    else:
                        print('This modality {} is not yet (?) supported'
                              .format(f.modality))

                else:
                    im_dicom_headers = list()

                im_dicom_headers.append(f)

        def convert(self):
            for im in self.images:
                #COMPUTE THE PATH
                im.convert(path)


