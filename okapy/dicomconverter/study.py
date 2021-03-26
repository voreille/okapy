from pathlib import Path
import warnings
import pickle

from okapy.dicomconverter.dicom_file import DicomFileBase


class Study():
    def __init__(self,
                 study_instance_uid=None,
                 study_date=None,
                 patient_id=None):
        self.mask_files = list()  # Can have multiple RTSTRUCT and also SEG
        self.volume_files = list()  # Can have multiple RTSTRUCT
        self.study_instance_uid = study_instance_uid
        self.study_date = study_date
        self.patient_id = patient_id

    def append_dicom_files(self, im_dicom_files, dcm_header):
        if dcm_header.modality == 'RTSTRUCT' or dcm_header.modality == 'SEG':
            self.mask_files.append(
                DicomFileBase.get(dcm_header.modality)(
                    dicom_header=dcm_header,
                    dicom_paths=[k.path for k in im_dicom_files],
                    study=self,
                ))

        elif len(im_dicom_files) > 1:
            try:

                self.volume_files.append(
                    DicomFileBase.get(dcm_header.modality)(
                        dicom_header=dcm_header,
                        dicom_paths=[k.path for k in im_dicom_files],
                        study=self,
                    ))
            except KeyError:
                warnings.warn(f"This modality {dcm_header.modality} "
                              f"is not supported")
        else:
            warnings.warn(f"single slice or 2D images are not supported. "
                          f"Patient {dcm_header.patient_id}, "
                          f"image number {dcm_header.series_number}")

    def save(self, dir_path="./tmp/"):
        path = Path(dir_path)
        filename = f"{self.patient_id}__{self.study_date}.pkl"
        file_path = str((path / filename).resolve())
        with open(file_path, "wb") as f:
            pickle.dump(self, f)
