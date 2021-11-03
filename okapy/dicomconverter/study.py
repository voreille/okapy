from pathlib import Path
import warnings
import pickle
import logging

import numpy as np

from okapy.dicomconverter.dicom_file import DicomFileBase
from okapy.exceptions import (EmptyContourException,
                              MissingSegmentationException, NotHandledModality,
                              PETUnitException)

log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)
logger = logging.getLogger(__name__)


class Study():
    def __init__(self,
                 study_instance_uid=None,
                 study_date=None,
                 submodalities=False,
                 patient_id=None):
        self.mask_files = list()
        self.volume_files = list()
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


def bb_union(bbs):
    out = np.array([np.inf, np.inf, np.inf, -np.inf, -np.inf, -np.inf])
    for bb in bbs:
        out[:3] = np.minimum(out[:3], bb[:3])
        out[3:] = np.maximum(out[3:], bb[3:])
    return out


def bb_intersection(bbs):
    out = np.array([-np.inf, -np.inf, -np.inf, np.inf, np.inf, np.inf])
    for bb in bbs:
        out[:3] = np.maximum(out[:3], bb[:3])
        out[3:] = np.minimum(out[3:], bb[3:])
    return out


class StudyProcessor():
    def __init__(
        self,
        volume_processor=None,
        mask_processor=None,
        padding="whole_image",
        combine_segmentation=False,
    ):
        self.volume_processor = volume_processor
        self.mask_processor = mask_processor
        self.padding = padding
        self.combine_segmentation = combine_segmentation

    @staticmethod
    def _extract_volume_of_interest(mask_files, labels=None):
        masks_list = list()
        for f in mask_files:
            if labels is None:
                masks_list.extend([f.get_volume(label) for label in f.labels])
            else:
                label_intersection = list(set(f.labels) & set(labels))
                for label in label_intersection:
                    try:
                        masks_list.append(f.get_volume(label))
                    except EmptyContourException:
                        continue
        return masks_list

    def _get_bounding_box(self, volume, masks_list):
        bb = bb_union([mask.bb for mask in masks_list])

        bb[:3] = bb[:3] - self.padding
        bb[3:] = bb[3:] + self.padding

        return bb_intersection([volume.reference_frame.bb, bb])

    def __call__(self, study, labels=None):
        results = list()
        for f in study.volume_files:
            logger.info(f"Start preprocessing image {f.modality}")
            if self.combine_segmentation:
                mask_files = study.mask_files
            else:
                mask_files = [
                    m for m in study.mask_files if m.reference_image_uid ==
                    f.dicom_header.series_instance_uid
                ]
            if len(mask_files) == 0:
                logger.warning(f"Discarding {f.dicom_header.Modality} "
                               f"image {f.dicom_header.SeriesInstanceUID} "
                               f"since no VOI was found")
                continue

            masks = self._extract_volume_of_interest(mask_files, labels=labels)

            if len(masks) == 0:
                raise MissingSegmentationException(
                    f"No segmentation found for study with"
                    f" StudyInstanceUID {study.study_instance_uid}")

            try:
                volume = f.get_volume()
            except PETUnitException as e:
                print(e)
                continue

            bb = self._get_bounding_box(
                volume, masks) if self.padding != 'whole_image' else None

            logger.info(f"Preprocessing image {f.modality}")
            volume = self.volume_processor(volume,
                                           mask_files=mask_files,
                                           bounding_box=bb)
            logger.info(f"Preprocessing VOIs for {f.modality} image")
            masks = [
                self.mask_processor(m, reference_frame=volume.reference_frame)
                for m in masks
            ]

            logger.info(f"Preprocessing of {f.modality} image is done.")
            results.append((volume, masks))

        return results
