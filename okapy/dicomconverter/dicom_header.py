from datetime import datetime
import re

import pydicom as pdcm


def camel_to_snake_case(str):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', str)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


DEFAULT_DICOM_TAGS = [
    "PatientID",
    "PatientName",
    "StudyInstanceUID",
    "StudyDate",
    "SeriesInstanceUID",
    "SeriesNumber",
    "InstanceNumber",
    "Modality",
    "ImageType",
    "SeriesDate",
    "SeriesTime",
]


class DicomHeader():
    def __init__(self,
                 patient_id=None,
                 patient_name=None,
                 study_instance_uid=None,
                 study_date=None,
                 series_instance_uid=None,
                 series_number=None,
                 instance_number=None,
                 modality=None,
                 image_type=None,
                 series_date=None,
                 series_time=None,
                 additional_data=None):
        self.patient_id = patient_id
        self.patient_name = patient_name
        self.study_instance_uid = study_instance_uid
        self.study_date = study_date
        self.series_instance_uid = series_instance_uid
        self.series_number = series_number
        self.instance_number = instance_number
        self.modality = modality
        self.image_type = image_type
        self.series_date = series_date
        self.series_time = series_time
        if additional_data:
            self.additional_data = additional_data
        else:
            self.additional_data = {}

        try:
            self.series_datetime = datetime.strptime(
                series_date + series_time.split('.')[0], "%Y%m%d%H%M%S")
        except (TypeError, ValueError, AttributeError):
            self.series_datetime = None

    def __getattr__(self, attr):
        attr = camel_to_snake_case(attr)
        try:
            return object.__getattribute__(self, attr)
        except AttributeError as e:
            if not attr.startswith("__"):
                return self.additional_data[attr]
            else:
                raise e

    @staticmethod
    def from_file(file):
        data = pdcm.filereader.dcmread(file, stop_before_pixels=True)
        return DicomHeader.from_pydicom(data)

    @staticmethod
    def from_pydicom(data, additional_tags=None):

        data_dict = {
            camel_to_snake_case(k): data.get(k, "Not found in dicom file")
            for k in DEFAULT_DICOM_TAGS
        }
        if additional_tags:
            additional_data = {
                camel_to_snake_case(k): data.get(k, "Not found in dicom file")
                for k in additional_tags if k not in DEFAULT_DICOM_TAGS
            }
        else:
            additional_data = {}
        return DicomHeader(**data_dict, additional_data=additional_data)

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
            return self.series_instance_uid == dcm_header.series_instance_uid
        else:
            return False
