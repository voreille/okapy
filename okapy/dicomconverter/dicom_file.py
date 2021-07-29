from copy import copy
from datetime import time, datetime
import logging

import numpy as np
import pydicom as pdcm
from pydicom.dataset import FileDataset
import pydicom_seg
from skimage.draw import polygon

from okapy.dicomconverter.volume import Volume, VolumeMask, ReferenceFrame
from okapy.exceptions import (EmptyContourException, MissingWeightException,
                              PETUnitException)

log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)
logger = logging.getLogger(__name__)


def is_approx_equal(x, y, tolerance=0.05):
    return abs(x - y) <= tolerance


class DicomFileBase():
    _registry = {}  # class var that store the different daughter

    def __init_subclass__(cls, name, **kwargs):
        cls.name = name
        DicomFileBase._registry[name] = cls
        super().__init_subclass__(**kwargs)

    @classmethod
    def get(cls, name: str):
        return DicomFileBase._registry[name]

    @classmethod
    def from_dicom_paths(cls, dicom_paths):
        if isinstance(dicom_paths[0], FileDataset):
            modality = dicom_paths[0].Modality
        else:
            modality = pdcm.filereader.dcmread(
                dicom_paths[0], stop_before_pixels=True).Modality
        return DicomFileBase._registry[modality](dicom_paths=dicom_paths)

    def __init__(
            self,
            dicom_header=None,
            dicom_paths=list(),
            reference_frame=None,
            study=None,
    ):
        self._dicom_header = dicom_header
        self.dicom_paths = dicom_paths
        self._reference_frame = reference_frame
        self.study = study
        self.slices = None

    def get_volume(self, *args):
        raise NotImplementedError('It is an abstract class')

    def read(self):
        raise NotImplementedError('It is an abstract class')

    @property
    def dicom_header(self):
        if self._dicom_header is None and not type(
                self.dicom_paths[0]) == FileDataset:
            self._dicom_header = pdcm.read_file(self.dicom_paths[0],
                                                stop_before_pixels=True)
        elif self._dicom_header is None and type(
                self.dicom_paths[0]) == FileDataset:
            self._dicom_header = self.dicom_paths[0]

        return self._dicom_header

    @property
    def patient_weight(self):
        patient_weight = getattr(self.slices[0], "PatientWeight", None)
        if patient_weight is None:
            raise MissingWeightException(
                'Weight is missing in {}'.format(self))

        return float(patient_weight)

    @property
    def reference_frame(self):
        if self._reference_frame is None:
            self.read(stop_before_pixel=False)
        return self._reference_frame


class DicomFileImageBase(DicomFileBase, name="image_base"):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_physical_values(self):
        raise NotImplementedError('This is an abstract class')

    def read(self, stop_before_pixel=False):
        if type(self.dicom_paths[0]) == FileDataset:
            slices = self.dicom_paths
        else:
            slices = [
                pdcm.filereader.dcmread(dcm,
                                        stop_before_pixels=stop_before_pixel)
                for dcm in self.dicom_paths
            ]
        image_orientation = slices[0].ImageOrientationPatient
        n = np.cross(image_orientation[:3], image_orientation[3:])

        orthogonal_positions = [
            np.dot(n, np.asarray(x.ImagePositionPatient)) for x in slices
        ]
        # Sort the slices accordind to orthogonal_postions,
        slices_pos = list(zip(slices, orthogonal_positions))
        slices_pos.sort(key=lambda x: x[1])
        slices, orthogonal_positions = zip(*slices_pos)
        # Check shape consistency
        columns = [s.Columns for s in slices]
        val, counts = np.unique(columns, return_counts=True)
        valcounts = list(zip(val, counts))
        valcounts.sort(key=lambda x: x[1])
        val, counts = zip(*valcounts)
        ind2rm = [
            ind for ind in range(len(slices))
            if slices[ind].Columns in val[:-1]
        ]

        if len(ind2rm) > 0:
            slices = [k for i, k in enumerate(slices) if i not in ind2rm]
            orthogonal_positions = [
                k for i, k in enumerate(orthogonal_positions)
                if i not in ind2rm
            ]

        rows = [s.Rows for s in slices]
        val, counts = np.unique(rows, return_counts=True)
        valcounts = list(zip(val, counts))
        valcounts.sort(key=lambda x: x[1])
        val, counts = zip(*valcounts)
        ind2rm = [
            ind for ind in range(len(slices))
            if slices[ind].Columns in val[:-1]
        ]

        if len(ind2rm) > 0:
            slices = [k for i, k in enumerate(slices) if i not in ind2rm]
            orthogonal_positions = [
                k for i, k in enumerate(orthogonal_positions)
                if i not in ind2rm
            ]

        # Compute redundant slice positions and possibly weird slice
        ind2rm = [
            ind for ind in range(len(orthogonal_positions))
            if orthogonal_positions[ind] == orthogonal_positions[ind - 1]
        ]
        # Check if there is redundancy in slice positions and remove them
        if len(ind2rm) > 0:
            slices = [k for i, k in enumerate(slices) if i not in ind2rm]

        self.slices = slices
        self.orthogonal_positions = orthogonal_positions
        self.expected_n_slices = self._check_missing_slices()
        self._reference_frame = ReferenceFrame(
            origin=slices[0].ImagePositionPatient,
            origin_last_slice=slices[-1].ImagePositionPatient,
            orientation=slices[0].ImageOrientationPatient,
            pixel_spacing=slices[0].PixelSpacing,
            shape=(*slices[0].pixel_array.shape, self.expected_n_slices))

    def _check_missing_slices(self):
        d_slices = np.array([
            self.orthogonal_positions[ind + 1] - self.orthogonal_positions[ind]
            for ind in range(len(self.slices) - 1)
        ])
        self.d_slices = d_slices
        slice_spacing = np.min(np.unique(d_slices))
        condition_missing_slice = np.abs(d_slices - slice_spacing) > 0.05
        n_missing_slices = np.sum(condition_missing_slice)
        if n_missing_slices == 1:
            # If only one slice is missing
            logger.warning(f"One slice is missing, we replaced "
                           f"it by linear interpolation for patient"
                           f"{self.dicom_header.PatientID}")
            logger.warning(f"One slice is missing, "
                           f"for patient "
                           f"{self.dicom_header.PatientID}"
                           f" and modality "
                           f"{self.dicom_header.Modality}")
            expected_n_slices = len(self.slices) + 1
        elif n_missing_slices > 1:
            raise RuntimeError("Multiple slices are missing")
        else:
            expected_n_slices = len(self.slices)

        return expected_n_slices

    def _interp_missing_slice(self, image):
        mean_slice_spacing = np.mean(self.d_slices)
        errors = np.abs(self.d_slices - mean_slice_spacing)
        diff = np.asarray([e < 0.1 * mean_slice_spacing for e in errors])
        ind2interp = int(np.where(diff)[0])
        new_slice = (image[:, :, ind2interp] +
                     image[:, :, ind2interp + 1]) * 0.5
        new_slice = new_slice[..., np.newaxis]
        image = np.concatenate(
            (image[..., :ind2interp], new_slice, image[..., ind2interp:]),
            axis=2)
        return image

    def get_volume(self):
        if self.slices is None:
            self.read()
        image = self.get_physical_values()
        image = np.transpose(image, (1, 0, 2))
        if self.expected_n_slices != len(self.slices):
            image = self._interp_missing_slice(image)

        return Volume(image,
                      reference_frame=copy(self.reference_frame),
                      dicom_header=self.dicom_header)


class DicomFileCT(DicomFileImageBase, name="CT"):
    def get_physical_values(self):
        image = list()
        for s in self.slices:
            image.append(
                float(s.RescaleSlope) * s.pixel_array +
                float(s.RescaleIntercept))
        return np.stack(image, axis=-1)


class DicomFileMR(DicomFileImageBase, name="MR"):
    def get_physical_values(self):
        image = list()
        for s in self.slices:
            image.append(s.pixel_array)
        return np.stack(image, axis=-1)


class DicomFilePT(DicomFileImageBase, name="PT"):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def patient_weight(self):
        try:
            patient_weight = super().patient_weight
        except MissingWeightException:
            list_images = [
                f for f in self.study.volume_files + self.study.mask_files
                if f is not self
            ]
            weight_found = False
            for f in list_images:
                try:
                    patient_weight = f.patient_weight
                    weight_found = True
                    break
                except MissingWeightException:
                    continue
            if not weight_found:
                logger.warning(f"Estimation of patient weight by 75.0 kg"
                               f" for patient {self.dicom_header.PatientID}")
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
            raise PETUnitException('The {} units is not handled'.format(units))

    def _acquistion_datetime(self):
        times = [
            datetime.strptime(
                s[0x00080022].value + s[0x00080032].value.split('.')[0],
                "%Y%m%d%H%M%S") for s in self.slices
        ]
        times.sort()
        return times[0]

    def _get_decay_time(self):
        s = self.slices[0]
        acquisition_datetime = self._acquistion_datetime()
        serie_datetime = datetime.strptime(
            s[0x00080021].value + s[0x00080031].value.split('.')[0],
            "%Y%m%d%H%M%S")

        try:
            if (serie_datetime <= acquisition_datetime) and (
                    serie_datetime > datetime(1950, 1, 1)):
                scan_datetime = serie_datetime
            elif 0x0009100d in s:
                scan_datetime_value = s[0x0009100d].value
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
            else:
                scan_datetime = acquisition_datetime

            start_time_str = s.RadiopharmaceuticalInformationSequence[
                0].RadiopharmaceuticalStartTime
            start_time = time(int(start_time_str[0:2]),
                              int(start_time_str[2:4]),
                              int(start_time_str[4:6]))
            start_datetime = datetime.combine(scan_datetime.date(), start_time)
            decay_time = (scan_datetime - start_datetime).total_seconds()
        except KeyError:
            decay_time = 1.75 * 3600  # From Martin's code
            logger.warning(
                f"Estimation of time decay for SUV"
                f" for patient {self.dicom_header.PatientID}"
                f" computation from average parameters, "
                f"i.e. with an estimated decay time of {decay_time} [s]")
        logger.debug(f"Computed decay time for patient "
                     f"{self.dicom_header.PatientName} is {decay_time} [s]")
        return decay_time

    def _get_suv_philips(self):
        image = list()
        for s in self.slices:
            im = (float(s.RescaleSlope) * s.pixel_array +
                  float(s.RescaleIntercept)) * float(s[0x70531000].value)
            image.append(im)
        return np.stack(image, axis=-1)

    def _get_suv_from_bqml(self, decay_time):
        # Get SUV from raw PET
        image = list()
        patient_weight = self.patient_weight
        for s in self.slices:
            pet = float(s.RescaleSlope) * s.pixel_array + float(
                s.RescaleIntercept)
            half_life = float(s.RadiopharmaceuticalInformationSequence[0].
                              RadionuclideHalfLife)
            total_dose = float(s.RadiopharmaceuticalInformationSequence[0].
                               RadionuclideTotalDose)
            decay = 2**(-decay_time / half_life)
            actual_activity = total_dose * decay

            im = pet * patient_weight * 1000 / actual_activity
            image.append(im)
        return np.stack(image, axis=-1)


class MaskFile(DicomFileBase, name="mask_base"):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._labels = None

    @property
    def labels(self):
        if self._labels is None:
            self.read()
        return self._labels


class SegFile(MaskFile, name="SEG"):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.raw_volume = None
        self.label_to_num = dict()
        if len(self.dicom_paths) != 1:
            raise RuntimeError('SEG has more than one file')

    def read(self):
        if type(self.dicom_paths[0]) == FileDataset:
            dcm = self.dicom_paths[0]
        else:
            dcm = pdcm.dcmread(self.dicom_paths[0])
        self.raw_volume = pydicom_seg.SegmentReader().read(dcm)
        coordinate_matrix = np.zeros((4, 4))
        coordinate_matrix[:3, :3] = self.raw_volume.direction * np.tile(
            self.raw_volume.spacing, [3, 1])
        coordinate_matrix[:3, 3] = self.raw_volume.origin
        coordinate_matrix[3, 3] = 1
        self._reference_frame = ReferenceFrame(
            coordinate_matrix=coordinate_matrix, shape=self.raw_volume.size)
        self._labels = list()
        for segment_number in self.raw_volume.available_segments:
            label = self.raw_volume.segment_infos[segment_number][
                0x620006].value
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
                          reference_dicom_header=None,
                          dicom_header=self.dicom_header)


class RtstructFile(MaskFile, name="RTSTRUCT"):
    def __init__(self,
                 *args,
                 reference_image=None,
                 reference_frame=None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self._reference_frame = reference_frame
        self._reference_image = reference_image
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
                        f.dicom_header.SeriesInstanceUID):
                    self._reference_image = f
                    found = True
                    break

            if not found:
                self.study.volume_files.sort(
                    key=lambda x: x.dicom_header.Modality)

                self._reference_image = self.study.volume_files[0]
                logger.warning(f"The Reference image was not found for"
                               f" the RTSTRUCT {str(self)}. The "
                               f"{self._reference_image.dicom_header.Modality}"
                               f" image will be"
                               f" taken as reference.")
        return self._reference_image

    @property
    def labels(self):
        if self._labels is None:
            self.read()
        return self._labels

    @staticmethod
    def get_reference_image_uid(dcm):
        return (dcm.ReferencedFrameOfReferenceSequence[0].
                RTReferencedStudySequence[0].RTReferencedSeriesSequence[0].
                SeriesInstanceUID)

    def read(self):
        self._labels = list()
        if len(self.dicom_paths) > 1:
            logger.warning(
                "there is multiple instances of the same RTSTRUCT file")
        if type(self.dicom_paths[0]) == FileDataset:
            self.slices = self.dicom_paths
        else:
            self.slices = [pdcm.read_file(p) for p in self.dicom_paths]
        self.reference_image_uid = RtstructFile.get_reference_image_uid(
            self.slices[0])
        self.label_number_mapping = {}
        for dcm in self.slices:
            if (self.reference_image_uid !=
                    RtstructFile.get_reference_image_uid(dcm)):
                raise RuntimeError(
                    "The different instances of the rtstruct do not"
                    " point to the same reference image")
            for i, roi_seq in enumerate(dcm.StructureSetROISequence):

                assert dcm.ROIContourSequence[
                    i].ReferencedROINumber == roi_seq.ROINumber

                label = roi_seq.ROIName

                self.label_number_mapping[label] = i
                self._labels.append(label)

    def _initialize(self):
        if self._labels is None:
            self.read()
        if self._reference_frame is None:
            if self.reference_image.reference_frame is None:
                self.reference_image.read()
            self._reference_frame = self.reference_image.reference_frame

    def _compute_mask(self, contour_sequence):
        mask = np.zeros(self.reference_frame.shape, dtype=np.uint8)
        for s in contour_sequence:
            current = s.ContourData

            nodes = np.array(current).reshape((-1, 3))
            # assert np.amax(np.abs(np.diff(nodes[:, 2]))) == 0
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
        return mask

    def get_volume(self, label):
        self._initialize()
        try:
            contour_sequence = self.slices[0].ROIContourSequence[
                self.label_number_mapping[label]].ContourSequence
        except AttributeError:
            logger.warning(f"{label} is empty")
            raise EmptyContourException()

        mask = self._compute_mask(contour_sequence)

        return VolumeMask(
            mask,
            reference_frame=self.reference_frame,
            reference_dicom_header=self.reference_image.dicom_header,
            label=label,
            dicom_header=self.dicom_header)
