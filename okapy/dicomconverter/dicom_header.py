from datetime import datetime

import pydicom as pdcm


class DicomHeader():
    def __init__(self,
                 patient_name=None,
                 patient_id=None,
                 study_instance_uid=None,
                 study_date=None,
                 series_instance_uid=None,
                 series_number=None,
                 instance_number=None,
                 image_type=None,
                 modality=None,
                 series_datetime=None):
        self.patient_name = patient_name
        self.patient_id = patient_id
        self.study_instance_uid = study_instance_uid
        self.study_date = study_date
        self.series_instance_uid = series_instance_uid
        self.series_number = series_number
        self.instance_number = instance_number
        self.modality = modality
        self.image_type = image_type
        self.series_datetime = series_datetime

    @staticmethod
    def from_file(file):
        data = pdcm.filereader.dcmread(file, stop_before_pixels=True)
        return DicomHeader.from_pydicom(data)

    @staticmethod
    def from_pydicom(data):
        if hasattr(data, "SeriesDate") and hasattr(data, "SeriesTime"):
            series_datetime = datetime.strptime(
                data.SeriesDate + data.SeriesTime.split('.')[0],
                "%Y%m%d%H%M%S")
        else:
            series_datetime = -1
        return DicomHeader(
            patient_id=data.get("PatientID", -1),
            study_instance_uid=data.get("StudyInstanceUID", -1),
            study_date=data.get("StudyDate", -1),
            series_instance_uid=data.get("SeriesInstanceUID", -1),
            series_number=data.get("SeriesNumber", -1),
            instance_number=data.get("InstanceNumber", -1),
            modality=data.get("Modality", -1),
            image_type=data.get("ImageType", ["-1"]),
            series_datetime=series_datetime,
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
                and self.series_number == dcm_header.series_number
                and self.image_type == dcm_header.image_type
                and self.series_datetime == dcm_header.series_datetime)
        else:
            return False

    def same_serie_as(self, dcm_header):
        if isinstance(dcm_header, DicomHeader):
            return (self.study_instance_uid == dcm_header.study_instance_uid
                    and self.series_instance_uid
                    == dcm_header.series_instance_uid
                    and self.modality == dcm_header.modality
                    and self.series_number == dcm_header.series_number
                    and self.image_type == dcm_header.image_type)
        else:
            return False
