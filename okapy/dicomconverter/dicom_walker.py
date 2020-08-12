'''
TODO: Is it better to create a class for just the header and another for the
files?
'''
import os
from os.path import join
from string import Template

import SimpleITK as sitk
import pydicom as pdcm
from pydicom.errors import InvalidDicomError

from okapy.dicomconverter.dicom_file import Study


class DicomHeader():
    def __init__(self,
                 patient_id=None,
                 study_instance_uid=None,
                 study_date=None,
                 series_instance_uid=None,
                 modality=None):
        self.patient_id = patient_id
        self.study_instance_uid = study_instance_uid
        self.study_date = study_date
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
                self.patient_id == dcm_header.patient_id
                and self.study_instance_uid == dcm_header.study_instance_uid
                and self.series_instance_uid == dcm_header.series_instance_uid
                and self.modality == dcm_header.modality)
        else:
            return False


class DicomFile():
    def __init__(self, dicom_header=DicomHeader(), path=None):
        self.dicom_header = dicom_header
        self.path = path

    def __str__(self):
        return str(self.dicom_header)


class DicomWalker():
    def __init__(
        self,
        input_dirpath=None,
    ):
        self.input_dirpath = input_dirpath

    def _walk(self, input_dirpath=None):
        '''
        Method to walk through the path given and fill the list of DICOM
        headers and sort them
        '''
        dicom_files = list()
        for (dirpath, dirnames, filenames) in os.walk(input_dirpath):
            for filename in filenames:
                try:
                    data = pdcm.dcmread(join(dirpath, filename),
                                        stop_before_pixels=True)
                except InvalidDicomError:
                    print('This file {} is not recognised as DICOM'.format(
                        filename))
                    continue
                try:
                    modality = data.Modality
                except AttributeError:
                    print('not reading the DICOMDIR')
                    continue

                if modality == 'RTSTRUCT':
                    # Adaptation from QuantImage
                    series_instance_uid = (
                        data.ReferencedFrameOfReferenceSequence[0].
                        RTReferencedStudySequence[0].
                        RTReferencedSeriesSequence[0].SeriesInstanceUID)
                else:
                    series_instance_uid = data.SeriesInstanceUID

                dicom_header = DicomHeader(
                    patient_id=data.PatientID,
                    study_instance_uid=data.StudyInstanceUID,
                    study_date=data.StudyDate,
                    series_instance_uid=series_instance_uid,
                    modality=data.Modality)
                dicom_files.append(
                    DicomFile(dicom_header=dicom_header,
                              path=join(dirpath, filename)))

        dicom_files.sort(key=lambda x: (
            x.dicom_header.patient_id, x.dicom_header.study_instance_uid, x.
            dicom_header.modality, x.dicom_header.series_instance_uid, x.path))
        return dicom_files

    def _get_studies(self, dicom_files):
        '''
        Construct the tree-like dependency of the dicom
        It all relies on the fact that the collection of headers has been
        sorted
        '''
        im_dicom_files = list()
        studies = list()
        dcm_header = DicomHeader()

        previous_study_uid = None
        for i, f in enumerate(dicom_files):
            # When the image changeschanges we store it as a whole
            current_study_uid = f.dicom_header.study_instance_uid
            if i == 0:
                current_study = Study(study_instance_uid=current_study_uid,
                                      study_date=f.dicom_header.study_date,
                                      patient_id=f.dicom_header.patient_id)

            if i > 0 and not f.dicom_header == dicom_files[i - 1].dicom_header:
                current_study.append_dicom_files(im_dicom_files, dcm_header)
                im_dicom_files = list()

            if i > 0 and not (current_study_uid == previous_study_uid):
                studies.append(current_study)
                current_study = Study(study_instance_uid=current_study_uid,
                                      study_date=f.dicom_header.study_date,
                                      patient_id=f.dicom_header.patient_id)

            im_dicom_files.append(f)
            dcm_header = f.dicom_header
            previous_study_uid = f.dicom_header.study_instance_uid

        current_study.append_dicom_files(im_dicom_files, dcm_header)
        studies.append(current_study)
        return studies

    def __call__(self, input_dirpath=None):
        if input_dirpath:
            dicom_files = self._walk(input_dirpath=input_dirpath)
        else:
            dicom_files = self._walk(input_dirpath=self.input_dirpath)

        return self._get_studies(dicom_files)
