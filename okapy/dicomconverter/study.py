from os import stat
from itertools import product
from pathlib import Path
import warnings
import pickle
import logging

import numpy as np

from okapy.dicomconverter.dicom_file import DicomFileBase
from okapy.dicomconverter.volume import ReferenceFrame
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
        only_segmented_image=False,
    ):
        self.volume_processor = volume_processor
        self.mask_processor = mask_processor
        self.padding = padding
        self.combine_segmentation = combine_segmentation
        self.only_segmented_image = only_segmented_image

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

    @staticmethod
    def _project_bb(bb_vx, reference_frame, new_reference_frame):
        points = [
            np.array([pr, pc, ps, 1])
            for pr, pc, ps in product(*zip(bb_vx[:3], bb_vx[3:]))
        ]
        projection_matrix = np.dot(new_reference_frame.inv_coordinate_matrix,
                                   reference_frame.coordinate_matrix)
        projected_points = np.stack(
            [np.dot(projection_matrix, p)[:3] for p in points], axis=-1)
        bb_proj = np.zeros((6, ))
        bb_proj[:3] = np.min(projected_points, axis=-1)
        bb_proj[3:] = np.max(projected_points, axis=-1)
        return bb_proj

    def _get_new_reference_frame(self, volume, masks_list):
        bb_vx = bb_union([
            self._project_bb(mask.bb_vx, mask.reference_frame,
                             volume.reference_frame) for mask in masks_list
        ])

        bb_vx[:3] = (
            bb_vx[:3] -
            np.round(self.padding / volume.reference_frame.voxel_spacing))
        bb_vx[3:] = (
            bb_vx[3:] +
            np.round(self.padding / volume.reference_frame.voxel_spacing))

        bb_volume = np.concatenate([[0, 0, 0], volume.reference_frame.shape])
        bb_vx = bb_intersection([bb_volume, bb_vx])

        return ReferenceFrame(
            origin=volume.reference_frame.vx_to_mm(bb_vx[:3]),
            orientation_matrix=volume.reference_frame.orientation_matrix,
            voxel_spacing=volume.reference_frame.voxel_spacing,
            last_point_coordinate=volume.reference_frame.vx_to_mm(bb_vx[3:]),
        )

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
            if len(mask_files) == 0 and self.only_segmented_image:
                logger.warning(f"Discarding {f.dicom_header.Modality} "
                               f"image {f.dicom_header.SeriesInstanceUID} "
                               f"since no VOI was found")
                continue

            masks = self._extract_volume_of_interest(mask_files, labels=labels)

            if len(masks) == 0 and len(mask_files) > 0:
                raise MissingSegmentationException(
                    f"No segmentation found for study with"
                    f" StudyInstanceUID {study.study_instance_uid}")

            try:
                volume = f.get_volume()
            except PETUnitException as e:
                print(e)
                continue

            new_reference_frame = self._get_new_reference_frame(volume, masks)

            logger.info(f"Preprocessing image {f.modality}")
            if self.volume_processor:
                volume = self.volume_processor(
                    volume,
                    mask_files=mask_files,
                    new_reference_frame=new_reference_frame)
            logger.info(f"Preprocessing VOIs for {f.modality} image")
            if self.mask_processor and len(masks) > 0:
                masks = [
                    self.mask_processor(
                        m,
                        new_reference_frame=volume.reference_frame,
                    ) for m in masks
                ]

            logger.info(f"Preprocessing of {f.modality} image is done.")
            results.append((volume, masks))

        return results


class SimpleStudyProcessor():
    def __init__(self, volume_processor=None, mask_processor=None):
        self.volume_processor = volume_processor
        self.mask_processor = mask_processor

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

    def __call__(self, study, labels=None):
        results = list()
        volumes = list()
        for f in study.volume_files:
            try:
                volume = f.get_volume()
            except PETUnitException as e:
                print(e)
                continue

            if self.volume_processor:
                logger.info(f"Preprocessing image {f.modality}")
                volume = self.volume_processor(
                    volume,
                    mask_files=mask_files,
                    new_reference_frame=new_reference_frame)
            volumes.append(volume)

        mask_output = list()
        for m in study.mask_files:
            if self.mask_processor and len(masks) > 0:
                logger.info(f"Preprocessing VOIs for {f.modality} image")
                masks = [
                    self.mask_processor(
                        m,
                        new_reference_frame=volume.reference_frame,
                    ) for m in masks
                ]
            mask_output.extend(masks)

        logger.info(f"Preprocessing of {f.modality} image is done.")
        results.append((volume, masks))

        return results
