import pydicom as pdcm
from pydicom.dataset import FileDataset


class DicomHeader():
    def __init__(self,
                 patient_id=None,
                 study_instance_uid=None,
                 study_date=None,
                 series_instance_uid=None,
                 series_number=None,
                 instance_number=None,
                 modality=None):
        self.patient_id = patient_id
        self.study_instance_uid = study_instance_uid
        self.study_date = study_date
        self.series_instance_uid = series_instance_uid
        self.series_number = series_number
        self.instance_number = instance_number
        self.modality = modality

    @staticmethod
    def from_file(file):
        if type(file) == FileDataset:
            data = file
        else:
            data = pdcm.filereader.dcmread(file, stop_before_pixels=True)

        return DicomHeader(
            patient_id=data.PatientID,
            study_instance_uid=data.StudyInstanceUID,
            study_date=data.StudyDate,
            series_instance_uid=data.SeriesInstanceUID,
            series_number=data.SeriesNumber,
            instance_number=data.InstanceNumber,
            modality=data.Modality,
        )

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
                # self.patient_id == dcm_header.patient_id
                self.study_instance_uid == dcm_header.study_instance_uid
                and self.series_instance_uid == dcm_header.series_instance_uid
                and self.modality == dcm_header.modality
                and self.instance_number == dcm_header.instance_number
                and self.series_number == dcm_header.series_number)
        else:
            return False

    def same_serie_as(self, dcm_header):
        if isinstance(dcm_header, DicomHeader):
            return (self.study_instance_uid == dcm_header.study_instance_uid
                    and self.series_instance_uid
                    == dcm_header.series_instance_uid
                    and self.modality == dcm_header.modality
                    and self.series_number == dcm_header.series_number)
        else:
            return False
