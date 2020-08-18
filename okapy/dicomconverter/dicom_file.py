"""
TODO: Make it DRYer line 220
"""

from os.path import join
import warnings
from copy import copy
from datetime import time, datetime

import numpy as np
import pydicom as pdcm
from pydicom.tag import Tag
import pydicom_seg
from skimage.draw import polygon
import pandas as pd
import SimpleITK as sitk

from okapy.dicomconverter.volume import Volume, VolumeMask, ReferenceFrame


class OkapyException(Exception):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class EmptyContourException(OkapyException):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class MissingWeightException(OkapyException):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class DicomFileBase():
    def __init__(
            self,
            dicom_header=None,
            dicom_paths=list(),
            reference_frame=None,
            study=None,
    ):
        self.dicom_header = dicom_header
        self.dicom_paths = dicom_paths
        self.reference_frame = reference_frame
        if study:
            self.study = study
        else:
            study = Study(study_instance_uid=dicom_header.study_instance_uid)

    def get_volume(self, *args):
        raise NotImplementedError('It is an abstract class')

    def read(self):
        raise NotImplementedError('It is an abstract class')

    @property
    def patient_weight(self):
        if self.slices[0].PatientWeight is None:
            if hasattr(self.slices[0], 'PatientsWeight'):
                patient_weight = float(self.slices[0].PatientsWeight)
            else:
                raise MissingWeightException(
                    'Weight is missing in {}'.format(self))

        else:
            patient_weight = float(self.slices[0].PatientWeight)

        return patient_weight


class DicomFileImageBase(DicomFileBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.slices = None
        self.reference_frame = None

    def get_physical_values(self):
        raise NotImplementedError('This is an abstract class')

    def read(self):
        slices = [pdcm.read_file(dcm) for dcm in self.dicom_paths]
        image_orientation = slices[0].ImageOrientationPatient
        n = np.cross(image_orientation[:3], image_orientation[3:])
        slices.sort(
            key=lambda x: np.dot(n, np.asarray(x.ImagePositionPatient)))

        self.slices = slices
        self.reference_frame = ReferenceFrame(
            origin=slices[0].ImagePositionPatient,
            origin_last_slice=slices[-1].ImagePositionPatient,
            orientation=slices[0].ImageOrientationPatient,
            pixel_spacing=slices[0].PixelSpacing,
            shape=(*slices[0].pixel_array.shape, len(slices)))

    def get_dicom_header_df(self):
        return pd.DataFrame.from_dict({
            'Manufacturer': [self.slices[0].Manufacturer],
            'ManufacturerModelName': [self.slices[0].ManufacturerModelName],
            'InstitutionName': [self.slices[0].InstitutionName],
        })

    def get_volume(self):
        if self.slices is None:
            self.read()
        image = self.get_physical_values()
        image = np.transpose(image, (1, 0, 2))

        return Volume(image,
                      reference_frame=copy(self.reference_frame),
                      modality=self.dicom_header.modality)


class DicomFileCT(DicomFileImageBase):
    def get_physical_values(self):
        image = list()
        for s in self.slices:
            image.append(
                float(s.RescaleSlope) * s.pixel_array +
                float(s.RescaleIntercept))
        return np.stack(image, axis=-1)


class DicomFileMR(DicomFileImageBase):
    def get_physical_values(self):
        image = list()
        for s in self.slices:
            image.append(s.pixel_array)
        return np.stack(image, axis=-1)


class DicomFilePT(DicomFileImageBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def patient_weight(self):
        try:
            patient_weight = super().patient_weight
        except MissingWeightException:
            list_images = self.study.volume_files.extend(self.study.mask_files)
            weight_found = False
            for f in list_images:
                try:
                    patient_weight = f.patient_weight
                    weight_found = True
                    break
                except MissingWeightException:
                    continue
            if not weight_found:
                warnings.warn("Estimation of patient weight by 75.0 kg")
                patient_weight = 75.0

        return patient_weight

    def get_physical_values(self):
        s = self.slices[0]
        units = s.Units
        if units == 'BQML':
            decay_time = self._get_decay_time()
            return self._get_suv_from_bqml(decay_time)
        elif units == 'CNTS':
            return self._get_suv_philips()
        else:
            raise ValueError('The {} units is not handled'.format(units))

    def _get_decay_time(self):
        s = self.slices[0]
        acquisition_datetime = datetime.strptime(
            s[Tag(0x00080022)].value + s[Tag(0x00080032)].value.split('.')[0],
            "%Y%m%d%H%M%S")
        serie_datetime = datetime.strptime(
            s[Tag(0x00080021)].value + s[Tag(0x00080031)].value.split('.')[0],
            "%Y%m%d%H%M%S")

        try:
            if (serie_datetime <= acquisition_datetime) and (
                    serie_datetime > datetime(1950, 1, 1)):
                scan_datetime = serie_datetime
            else:
                scan_datetime_value = s[Tag(0x0009100d)].value
                if isinstance(scan_datetime_value, bytes):
                    scan_datetime_str = scan_datetime_value.decode(
                        "utf-8").split('.')[0]
                elif isinstance(scan_datetime_value, str):
                    scan_datetime_str = scan_datetime_value.split('.')[0]
                else:
                    raise ValueError(
                        "The value of scandatetime is not handled")
                scan_datetime = datetime.strptime(scan_datetime_str,
                                                  "%Y%m%d%H%M%S")

            start_time_str = s.RadiopharmaceuticalInformationSequence[
                0].RadiopharmaceuticalStartTime
            start_time = time(int(start_time_str[0:2]),
                              int(start_time_str[2:4]),
                              int(start_time_str[4:6]))
            start_datetime = datetime.combine(scan_datetime.date(), start_time)
            decay_time = (scan_datetime - start_datetime).total_seconds()
        except KeyError:
            warnings.warn("Estimation of time decay for SUV"
                          " computation from average parameters")
            decay_time = 1.75 * 3600  # From Martin's code

        return decay_time

    def _get_suv_philips(self):
        image = list()
        suv_scale_factor_tag = Tag(0x70531000)
        for s in self.slices:
            im = (float(s.RescaleSlope) * s.pixel_array + float(
                s.RescaleIntercept)) * float(s[suv_scale_factor_tag].value)
            image.append(im)
        return np.stack(image, axis=-1)

    def _get_suv_from_bqml(self, decay_time):
        # Get SUV from raw PET
        image = list()
        for s in self.slices:
            pet = float(s.RescaleSlope) * s.pixel_array + float(
                s.RescaleIntercept)
            half_life = float(s.RadiopharmaceuticalInformationSequence[0].
                              RadionuclideHalfLife)
            total_dose = float(s.RadiopharmaceuticalInformationSequence[0].
                               RadionuclideTotalDose)
            decay = 2**(-decay_time / half_life)
            actual_activity = total_dose * decay

            im = pet * self.patient_weight * 1000 / actual_activity
            image.append(im)
        return np.stack(image, axis=-1)

    def get_dicom_header_df(self):
        return pd.DataFrame.from_dict({
            'Manufacturer': [self.slices[0].Manufacturer],
            'ManufacturerModelName': [self.slices[0].ManufacturerModelName],
            'Units': [self.slices[0].Units],
            'InstitutionName': [self.slices[0].InstitutionName],
        })


class MaskFile(DicomFileBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._labels = None
        self.reference_modality = ""

    @property
    def labels(self):
        if self._labels is None:
            self.read()
        return self._labels

    def get_volumes(self, list_labels):
        raise NotImplementedError('This is an abstract class')


class SegFile(MaskFile):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reference_frame = None
        self.raw_volume = None
        self.label_to_num = dict()
        if len(self.dicom_paths) != 1:
            raise RuntimeError('SEG has more than one file')

    def read(self):
        dcm = pdcm.dcmread(self.dicom_paths[0])
        self.raw_volume = pydicom_seg.SegmentReader().read(dcm)
        coordinate_matrix = np.zeros((4, 4))
        coordinate_matrix[:3, :3] = self.raw_volume.direction
        coordinate_matrix[:3, 3] = self.raw_volume.origin
        coordinate_matrix[3, 3] = 1
        self.reference_frame = ReferenceFrame(
            coordinate_matrix=coordinate_matrix, shape=self.raw_volume.size)
        self._labels = list()
        for segment_number in self.raw_volume.available_segments:
            label = self.raw_volume.segment_infos[segment_number][Tag(
                0x620006)].value
            self._labels.append(label)
            self.label_to_num[label] = segment_number

    def get_volume(self, label):
        if self.raw_volume is None:
            self.read()

        trans = (2, 1, 0)
        np_volume = np.transpose(
            self.raw_volume.segment_data(self.label_to_num[label]), trans)

        return VolumeMask(np_volume,
                          reference_frame=copy(self.reference_frame),
                          label=label,
                          reference_modality=self.reference_modality,
                          modality=self.dicom_header.modality)


class RtstructFile(MaskFile):
    def __init__(self, *args, reference_image=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.contours = None
        self.reference_frame = None
        self._reference_image = None
        self.reference_image_uid = None
        self.error_list = list()

    @property
    def reference_image(self):
        if self.reference_image_uid is None:
            self.read()
        if self._reference_image is None:
            found = False
            for f in self.study.volume_files:
                if (self.reference_image_uid ==
                        f.dicom_header.series_instance_uid):
                    self._reference_image = f
                    found = True
                    break

            if not found:
                warnings.warn(
                    "The Reference image was not found for"
                    " the RTSTRUCT {}. The CT image will be taken as reference."
                    .format(str(self)))
                for f in self.study.volume_files:
                    if f.dicom_header.modality == 'CT':
                        self._reference_image = f
            self.reference_modality = (
                self._reference_image.dicom_header.modality)
        return self._reference_image

    @property
    def labels(self):
        if self._labels is None:
            self.read()
        return self._labels

    def read(self):
        self._labels = list()
        if len(self.dicom_paths) != 1:
            raise RuntimeError('RTSTRUCT has more than one file')
        structure = pdcm.read_file(self.dicom_paths[0])
        self.reference_image_uid = (
            structure.ReferencedFrameOfReferenceSequence[0].
            RTReferencedStudySequence[0].RTReferencedSeriesSequence[0].
            SeriesInstanceUID)
        self.contours = {}
        for i, roi_seq in enumerate(structure.StructureSetROISequence):

            assert structure.ROIContourSequence[
                i].ReferencedROINumber == roi_seq.ROINumber

            try:
                self.contours[roi_seq.ROIName] = [
                    s.ContourData
                    for s in structure.ROIContourSequence[i].ContourSequence
                ]
            except AttributeError:
                warnings.warn(f"{roi_seq.ROIName} is empty")
                continue

            self._labels.append(roi_seq.ROIName)

    def get_volume(self, label):
        if self.contours is None:
            self.read()
        if self.reference_image.reference_frame is None:
            self.reference_image.read()
            self.reference_frame = self.reference_image.reference_frame

        mask = np.zeros(self.reference_frame.shape, dtype=np.uint8)
        cond_empty = True
        for current in self.contours[label]:
            cond_empty = False
            nodes = np.array(current).reshape((-1, 3))
            assert np.amax(np.abs(np.diff(nodes[:, 2]))) == 0
            vx_indices = np.stack([
                self.reference_frame.mm_to_vx(
                    [nodes[k, 0], nodes[k, 1], nodes[k, 2]])
                for k in range(nodes.shape[0])
            ],
                                  axis=0)
            rr, cc = polygon(vx_indices[:, 0], vx_indices[:, 1])
            if len(rr) > 0 and len(cc) > 0:
                if np.max(rr) > 512 or np.max(cc) > 512:
                    raise Exception("The RTSTRUCT file is compromised")

            mask[rr, cc, np.round(vx_indices[0, 2]).astype(int)] = 1
        if cond_empty:
            raise EmptyContourException()

        return VolumeMask(
            mask,
            reference_frame=self.reference_frame,
            reference_modality=self.reference_image.dicom_header.modality,
            label=label,
            modality=self.dicom_header.modality)


class Study():
    image_modality_dict = {
        'CT': DicomFileCT,
        'PT': DicomFilePT,
        'MR': DicomFileMR,
        'RTSTRUCT': RtstructFile,
        'SEG': SegFile,
    }

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
                Study.image_modality_dict[dcm_header.modality](
                    dicom_header=dcm_header,
                    dicom_paths=[k.path for k in im_dicom_files],
                    study=self,
                ))

        else:
            try:

                self.volume_files.append(
                    Study.image_modality_dict[dcm_header.modality](
                        dicom_header=dcm_header,
                        dicom_paths=[k.path for k in im_dicom_files],
                        study=self,
                    ))
            except KeyError:
                print('This modality {} is not yet (?) supported'.format(
                    dcm_header.modality))
