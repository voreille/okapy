from pathlib import Path
import warnings
import pickle

from okapy.dicomconverter.dicom_file import DicomFileBase
from okapy.exceptions import NotHandledModality


class Study():
    def __init__(self,
                 study_instance_uid=None,
                 study_date=None,
                 submodalities=False,
                 patient_id=None):
        self.mask_files = list()  # Can have multiple RTSTRUCT and also SEG
        self.volume_files = list()  # Can have multiple RTSTRUCT
        self.study_instance_uid = study_instance_uid
        self.study_date = study_date
        self.patient_id = patient_id
        self.submodalities = submodalities

    def _is_volume_matched(self, v):
        for m in self.mask_files:
            if m.reference_image_uid == v.dicom_header.series_instance_uid:
                return True
        return False

    def discard_unmatched_volumes(self):
        self.volume_files = [
            v for v in self.volume_files if self._is_volume_matched(v)
        ]

    def append_dicom_files(self, im_dicom_files, dcm_header):
        if dcm_header.Modality == 'RTSTRUCT' or dcm_header.Modality == 'SEG':
            self.mask_files.append(
                DicomFileBase.get(dcm_header.Modality)(
                    dicom_header=dcm_header,
                    dicom_paths=[k.path for k in im_dicom_files],
                    study=self,
                ))

        elif len(im_dicom_files) > 1:
            try:
                self.volume_files.append(
                    DicomFileBase.get(dcm_header.Modality)(
                        dicom_header=dcm_header,
                        dicom_paths=[k.path for k in im_dicom_files],
                        study=self,
                        submodalities=self.submodalities,
                    ))
            except NotHandledModality as e:
                warnings.warn(str(e))
        else:
            warnings.warn(f"single slice or 2D images are not supported. "
                          f"Patient {dcm_header.PatientID}, "
                          f"image number {dcm_header.SeriesNumber}")

    def save(self, dir_path="./tmp/"):
        path = Path(dir_path)
        filename = f"{self.patient_id}__{self.study_date}.pkl"
        file_path = str((path / filename).resolve())
        with open(file_path, "wb") as f:
            pickle.dump(self, f)
