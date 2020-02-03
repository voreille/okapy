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

from okapy.dicomconverter.image import Study


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
                 padding_voi=0,
                 list_labels=None,
                 resampling_spacing_modality=None):
        self.input_dirpath = input_dirpath
        self.output_dirpath = output_dirpath
        self.template_filename = template_filename
        self.extension_output = extension_output
        self.sitk_writer = sitk.ImageFileWriter()
        self.sitk_writer.SetImageIO(dic_sitk_writer[extension_output])
        self.dicom_files = list()
        self.studies = list()
        self.images = list()
        self.list_labels = list_labels
        self.padding_voi = padding_voi
        if resampling_spacing_modality is None:
            self.resampling_spacing_modality = {
                    'CT': (0.75, 0.75, 0.75),
                    'PT': (0.75, 0.75, 0.75),
                    'MR': (0.75, 0.75, 0.75),
                }
        else:
            self.resampling_spacing_modality = self.resampling_spacing_modality

    def __str__(self):
        dcm_list = [str(dcm) for dcm in self.dicom_files]
        return '\n'.join(dcm_list)


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
                try:
                    modality = data.Modality
                except AttributeError:
                    print('not reading the DICOMDIR')
                    continue

                if modality == 'RTSTRUCT':
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
                                               x.dicom_header.modality,
                                               x.dicom_header.series_instance_uid,
                                               x.path))


    def fill_dicom_files(self):
        '''
        Construct the tree-like dependency of the dicom
        It all relies on the fact that the collection of headers has been
        sorted
        '''
        im_dicom_files = list()
        dcm_header = DicomHeader()

        previous_study_uid = None
        for i, f in enumerate(self.dicom_files):
            # When the image changeschanges we store it as a whole
            current_study_uid = f.dicom_header.study_instance_uid
            if i==0:
                current_study = Study(sitk_writer=self.sitk_writer,
                                      padding_voi=self.padding_voi,
                                      study_instance_uid=current_study_uid,
                                      list_labels=self.list_labels,
                                      resampling_spacing_modality=self.resampling_spacing_modality)

            if i > 0 and not f.dicom_header == self.dicom_files[i-1].dicom_header:
                current_study.append_dicom_files(im_dicom_files, dcm_header)
                im_dicom_files = list()

            if i > 0 and not (current_study_uid == previous_study_uid):
                self.studies.append(current_study)
                current_study = Study(sitk_writer=self.sitk_writer,
                                      padding_voi=self.padding_voi,
                                      study_instance_uid=current_study_uid,
                                      list_labels=self.list_labels)

            im_dicom_files.append(f)
            dcm_header= f.dicom_header
            previous_study_uid = f.dicom_header.study_instance_uid

        current_study.append_dicom_files(im_dicom_files, dcm_header)
        self.studies.append(current_study)

    def convert(self):
        for study in self.studies:
            study.process(self.output_dirpath)

