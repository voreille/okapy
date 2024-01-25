"""
TODO: check NumberOfSlices as dicom tag
"""

from copy import copy
from datetime import time, datetime
import logging
from re import U
from statistics import mode
from tracemalloc import stop

import numpy as np
import pydicom as pdcm
from pydicom.dataset import FileDataset
import pydicom_seg
from skimage.draw import polygon

from okapy.dicomconverter.volume import Volume, BinaryVolume, ReferenceFrame
from okapy.dicomconverter.dicom_header import DicomHeader
from okapy.exceptions import (EmptyContourException, MissingWeightException,
                              NotHandledModality, PETUnitException)

log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)
logger = logging.getLogger(__name__)


def is_approx_equal(x, y, tolerance=0.05):
    return abs(x - y) <= tolerance


class DicomFileBase():
    """
    
    """
    _registry = {}  # class var that store the different daughter

    def __init_subclass__(cls, name, **kwargs):
        cls.name = name
        DicomFileBase._registry[name] = cls
        super().__init_subclass__(**kwargs)

    @classmethod
    def get(cls, name: str):
        try:
            return DicomFileBase._registry[name]
        except KeyError:
            raise NotHandledModality(f"The modality {name} is not handled.")

    @staticmethod
    def _check_modality(dcm_header):
        try:
            units = dcm_header.Units
        except AttributeError:
            if dcm_header.Modality == "PT":
                logger.warning(
                    f"Units for series{dcm_header.SeriesInstanceUID} "
                    f" is not present, since modality is PT we change it to CT"
                    f" in hope that it will be right.")
            return dcm_header.Modality if dcm_header.Modality != "PT" else "CT"

        if (units in ["BQML", "CNTS"] and dcm_header.Modality != "PT"):
            logger.warning(
                f"Modality for series{dcm_header.SeriesInstanceUID} "
                f"was changed from {dcm_header.Modality} to PT since the unit does not"
                f" match the modality (unit was {dcm_header.Units})")
            return "PT"
        if units == "HU" and dcm_header.Modality != "CT":
            logger.warning(
                f"Modality for series{dcm_header.SeriesInstanceUID} "
                f"was changed from {dcm_header.Modality} to CT since the unit does not"
                f" match the modality (unit was {dcm_header.Units})")
            return "CT"

        return dcm_header.Modality

    @staticmethod
    def from_dicom_slices(dicoms,
                          study=None,
                          additional_dicom_tags=None,
                          submodalities=False):
        if isinstance(dicoms[0], FileDataset):
            modality = DicomFileBase._check_modality(dicoms[0])

        else:
            modality = DicomFileBase._check_modality(
                pdcm.filereader.dcmread(
                    dicoms[0],
                    stop_before_pixels=True,
                ))
        return DicomFileBase.get(modality)(
            dicom_paths=dicoms,
            study=study,
            additional_dicom_tags=additional_dicom_tags,
            submodalities=submodalities)

    def __init__(
        self,
        dicom_header=None,
        dicom_paths=list(),
        reference_frame=None,
        study=None,
        additional_dicom_tags=None,
        submodalities=False,
        slices=None,
    ):
        self._dicom_header = dicom_header
        self.dicom_paths = dicom_paths
        self._reference_frame = reference_frame
        self.study = study
        self.slices = slices
        if submodalities:
            self.additional_dicom_tags = [
                "SeriesDescription"
            ] if additional_dicom_tags is None else additional_dicom_tags + [
                "SeriesDescription"
            ]
        else:
            self.additional_dicom_tags = additional_dicom_tags
        self.modality = self.name
        self._patient_weight = None
        if submodalities:
            self.modality = self.modality + self._parse_submodalities()

    def _parse_submodalities(self):
        try:
            return "_" + self.dicom_header.SeriesDescription.split(" --- ")[1].strip()
        except IndexError:
            return ""

    def get_volume(self, *args):
        raise NotImplementedError('It is an abstract class')

    def read(self):
        raise NotImplementedError('It is an abstract class')

    @property
    def dicom_header(self):
        if self._dicom_header is None and not type(
                self.dicom_paths[0]) == FileDataset:
            self._dicom_header = DicomHeader.from_file(
                self.dicom_paths[0],
                additional_tags=self.additional_dicom_tags)

        elif self._dicom_header is None and type(
                self.dicom_paths[0]) == FileDataset:
            self._dicom_header = DicomHeader.from_pydicom(
                self.dicom_paths[0],
                additional_tags=self.additional_dicom_tags)

        return self._dicom_header

    @property
    def patient_weight(self):
        if self._patient_weight:
            return self._patient_weight
        if self.slices is None:
            self.read()
        patient_weight = getattr(self.slices[0], "PatientWeight", None)
        if patient_weight is None:
            raise MissingWeightException(
                'Weight is missing in {}'.format(self))
        patient_weight = float(patient_weight)
        self._patient_weight = patient_weight
        return patient_weight

    @property
    def reference_frame(self):
        if self._reference_frame is None:
            self.read()
        return self._reference_frame


class DicomFileImageBase(DicomFileBase, name="image_base"):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_physical_values(self):
        raise NotImplementedError('This is an abstract class')

    def _check_dicom_paths(self):
        if type(self.dicom_paths[0]) == FileDataset:
            slices = self.dicom_paths
        else:
            slices = [
                pdcm.filereader.dcmread(dcm, stop_before_pixels=True)
                for dcm in self.dicom_paths
            ]
        return slices, self.dicom_paths

    def _sort_slices(self, slices, dicom_paths):
        image_orientation = slices[0].ImageOrientationPatient
        n = np.cross(image_orientation[:3], image_orientation[3:])
        orthogonal_positions = [
            np.dot(n, np.asarray(x.ImagePositionPatient)) for x in slices
        ]
        # Sort the slices accordind to orthogonal_postions,
        slices_pos = list(zip(slices, dicom_paths, orthogonal_positions))
        slices_pos.sort(key=lambda x: x[2])
        slices, dicom_paths, orthogonal_positions = zip(*slices_pos)
        return slices, dicom_paths, orthogonal_positions

    def _check_shape_consistency(self,
                                 slices,
                                 dicom_paths,
                                 orthogonal_positions,
                                 dimension="Columns"):
        # Check shape consistency
        columns = [getattr(s, dimension) for s in slices]
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
            dicom_paths = [
                k for i, k in enumerate(dicom_paths) if i not in ind2rm
            ]
            orthogonal_positions = [
                k for i, k in enumerate(orthogonal_positions)
                if i not in ind2rm
            ]

        return slices, dicom_paths, orthogonal_positions

    def _check_redundancy(self, slices, dicom_paths, orthogonal_positions):
        # Compute redundant slice positions and possibly weird slice
        ind2rm = [
            ind for ind in range(len(orthogonal_positions))
            if orthogonal_positions[ind] == orthogonal_positions[ind - 1]
        ]
        # Check if there is redundancy in slice positions and remove them
        if len(ind2rm) > 0:
            slices = [k for i, k in enumerate(slices) if i not in ind2rm]
            dicom_paths = [
                k for i, k in enumerate(dicom_paths) if i not in ind2rm
            ]
            orthogonal_positions = [
                k for i, k in enumerate(orthogonal_positions)
                if i not in ind2rm
            ]

        return slices, dicom_paths, orthogonal_positions

    def _check_missing_slices(self):
        if self.slice_spacing == 0:
            raise RuntimeError(
                "The most frequent slice spacing computed"
                " is 0, probably due to multi-channel image (e.g. DWI).")
        elif np.min(self.slice_spacing) == 0:
            raise RuntimeError("Some slices have the same position")

        slice_discontinuities = (np.abs(self.d_slices - self.slice_spacing) >
                                 0.9 * self.slice_spacing)
        n_missing_slices = np.round(
            np.sum(self.d_slices[slice_discontinuities] / self.slice_spacing -
                   1)).astype(int)
        return n_missing_slices, slice_discontinuities

    def read(self):
        slices, dicom_paths = self._check_dicom_paths()
        slices, dicom_paths, orthogonal_positions = self._sort_slices(
            slices, dicom_paths)

        (slices, dicom_paths,
         orthogonal_positions) = self._check_shape_consistency(
             slices, dicom_paths, orthogonal_positions, dimension="Rows")

        (slices, dicom_paths,
         orthogonal_positions) = self._check_shape_consistency(
             slices, dicom_paths, orthogonal_positions, dimension="Columns")

        (slices, dicom_paths, orthogonal_positions) = self._check_redundancy(
            slices, dicom_paths, orthogonal_positions)

        self.slices = slices
        self.dicom_paths = dicom_paths
        self.orthogonal_positions = orthogonal_positions
        self.d_slices = np.array([
            self.orthogonal_positions[ind + 1] - self.orthogonal_positions[ind]
            for ind in range(len(self.slices) - 1)
        ])
        self.slice_spacing = mode(np.round(self.d_slices, decimals=5))

        self.n_missing_slices, self.slice_discontinuities = self._check_missing_slices(
        )
        expected_n_slices = len(self.slices) + self.n_missing_slices
        slice_shape = (slices[0].Rows, slices[0].Columns)  # TO CHECK
        self._reference_frame = ReferenceFrame.from_slice_info(
            origin=slices[0].ImagePositionPatient,
            origin_last_slice=slices[-1].ImagePositionPatient,
            orientation=slices[0].ImageOrientationPatient,
            pixel_spacing=slices[0].PixelSpacing,
            shape=slice_shape + (expected_n_slices, ))

    def _interp_missing_slice(self, image):
        mean_slice_spacing = np.mean(self.d_slices)
        errors = np.abs(self.d_slices - mean_slice_spacing)
        diff = np.asarray([e > 0.5 * mean_slice_spacing for e in errors])
        ind2interp = int(np.where(diff)[0])
        new_slice = (image[:, :, ind2interp] +
                     image[:, :, ind2interp + 1]) * 0.5
        new_slice = new_slice[..., np.newaxis]
        image = np.concatenate(
            (image[..., :ind2interp], new_slice, image[..., ind2interp:]),
            axis=2)
        return image

    def _fill_discontinuity(self, image):
        fill_constant = {
            "PT": 0,
            "CT": -1000,
            "MR": 0,
        }
        idx_discontinuity = np.where(self.slice_discontinuities)[0][0]
        fill_shape = image.shape[:2] + (self.n_missing_slices, )
        return np.concatenate(
            (
                image[..., :idx_discontinuity + 1],
                fill_constant.get(self.name, 0) * np.ones(fill_shape),
                image[..., idx_discontinuity + 1:],
            ),
            axis=2,
        )

    def get_volume(self):
        if self.slices is None:
            self.read()

        if np.sum(self.slice_discontinuities) > 1:
            raise ValueError("Too many discontinuities in the slices")
        image = self.get_physical_values()
        image = np.transpose(image, (1, 0, 2))
        if self.n_missing_slices == 1:
            logger.warning(f"One slice is missing, we replace "
                           f"it by linear interpolation for patient"
                           f"{self.dicom_header.PatientID}"
                           f" and modality "
                           f"{self.dicom_header.Modality}")
            image = self._interp_missing_slice(image)

        if (np.sum(self.slice_discontinuities) == 1
                and self.n_missing_slices > 1):
            logger.warning(f"Multiple slices are missing for patient"
                           f"{self.dicom_header.PatientID}"
                           f" and modality "
                           f"{self.dicom_header.Modality} "
                           "filling the void with zeros")
            image = self._fill_discontinuity(image)
            # if idx_discontinuity < len(self.slices) // 2:
            #     self.slices = self.slices[idx_discontinuity + 1:]
            # else:
            #     self.slices = self.slices[:idx_discontinuity + 1]

        return Volume(image,
                      reference_frame=copy(self.reference_frame),
                      dicom_header=self.dicom_header,
                      modality=self.modality)


class DicomFileCT(DicomFileImageBase, name="CT"):

    @staticmethod
    def _get_physical_per_slice(s):
        return float(s.RescaleSlope) * s.pixel_array + float(
            s.RescaleIntercept)

    def get_physical_values(self):
        if not hasattr(self.slices[0], "pixel_array"):
            image = [
                DicomFileCT._get_physical_per_slice(pdcm.read_file(p))
                for p in self.dicom_paths
            ]
        else:
            image = [
                DicomFileCT._get_physical_per_slice(s) for s in self.slices
            ]
        return np.stack(image, axis=-1)


class DicomFileMR(DicomFileImageBase, name="MR"):

    def get_physical_values(self):
        if not hasattr(self.slices[0], "pixel_array"):
            image = [pdcm.read_file(p).pixel_array for p in self.dicom_paths]
        else:
            image = [s.pixel_array for s in self.slices]
        return np.stack(image, axis=-1)


class DicomFilePT(DicomFileImageBase, name="PT"):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def patient_weight(self):
        if self._patient_weight:
            return self._patient_weight
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

        if patient_weight < 1.0:
            logger.warning(
                f"The weight found in the dicom is {patient_weight}, "
                "there is probably an error, "
                "thus estimation of patient weight by 75.0 kg"
                f" for patient {self.dicom_header.PatientID}")
            patient_weight = 75.0
        self._patient_weight = patient_weight

        return patient_weight

    def _get_physical_values_func(self, units):
        if units == 'BQML':
            decay_time = self._get_decay_time()
            return lambda x: self._get_suv_from_bqml(x, decay_time)
        elif units == 'CNTS':
            return self._get_suv_philips
        elif units == 'GML':
            return self._get_suv_from_gml
        else:
            raise PETUnitException('The {} units is not handled'.format(units))

    def get_physical_values(self):
        s = self.slices[0]
        units = s.Units
        phys_value_func = self._get_physical_values_func(units)

        if not hasattr(self.slices[0], "pixel_array"):
            image = [
                phys_value_func(pdcm.read_file(p)) for p in self.dicom_paths
            ]
        else:
            image = [phys_value_func(s) for s in self.slices]
        return np.stack(image, axis=-1)

    def _acquistion_datetime(self):
        times = [
            datetime.strptime(
                s[0x00080022].value +
                s[0x00080032].value.split('.')[0].replace(":", ""),
                "%Y%m%d%H%M%S") for s in self.slices
        ]
        times.sort()
        return times[0]

    def _parse_time(self, t):
        for fmt in ["%H:%M:%S", "%H%M%S", "%H%M%S.%f"]:
            try:
                return datetime.strptime(t, fmt).time()
            except ValueError:
                pass
        raise ValueError("Could not parse time {}".format(t))

    def _get_decay_time(self):
        s = self.slices[0]
        acquisition_datetime = self._acquistion_datetime()
        try:
            serie_datetime = datetime.strptime(
                s[0x00080021].value + s[0x00080031].value.split('.')[0],
                "%Y%m%d%H%M%S")
        except KeyError:
            logger.warning(
                f"No SeriesDate found for {self.dicom_header.PatientID}, AcquisitionDate is used"
            )
            serie_datetime = acquisition_datetime

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
            try:
                start_time = self._parse_time(start_time_str)
            except ValueError:
                raise ValueError(
                    "There is an issue with the data for the DICOM"
                    " tag 'RadiopharmaceuticalStartTime'")
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

    def _get_suv_philips(self, s):
        return (float(s.RescaleSlope) * s.pixel_array +
                float(s.RescaleIntercept)) * float(s[0x70531000].value)

    def _get_suv_from_gml(self, s):
        return (float(s.RescaleSlope) * s.pixel_array +
                float(s.RescaleIntercept))

    def _get_suv_from_bqml(self, s, decay_time):
        # Get SUV from raw PET
        patient_weight = self.patient_weight
        pet = float(s.RescaleSlope) * s.pixel_array + float(s.RescaleIntercept)
        half_life = float(
            s.RadiopharmaceuticalInformationSequence[0].RadionuclideHalfLife)
        total_dose = float(
            s.RadiopharmaceuticalInformationSequence[0].RadionuclideTotalDose)
        decay = 2**(-decay_time / half_life)
        actual_activity = total_dose * decay

        return pet * patient_weight * 1000 / actual_activity


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

    def __init__(self, *args, reference_image=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.raw_volume = None
        self.label_to_num = dict()
        self._reference_image = reference_image
        self._reference_image_uid = None
        if len(self.dicom_paths) != 1:
            raise RuntimeError('SEG has more than one file')

    @property
    def reference_image_uid(self):
        if self._reference_image_uid is None:
            self.read()
        return self._reference_image_uid

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
                logger.warning(f"The reference image was not found for"
                               f" the SEG {str(self)}. The "
                               f"{self._reference_image.dicom_header.Modality}"
                               f" image will be"
                               f" taken as reference.")
        return self._reference_image

    def read(self):
        if type(self.dicom_paths[0]) == FileDataset:
            dcm = self.dicom_paths[0]
        else:
            dcm = pdcm.dcmread(self.dicom_paths[0])
        self._reference_image_uid = dcm.ReferencedSeriesSequence[
            0].SeriesInstanceUID
        self.raw_volume = pydicom_seg.SegmentReader().read(dcm)
        coordinate_matrix = np.zeros((4, 4))
        coordinate_matrix[:3, :3] = self.raw_volume.direction * np.tile(
            self.raw_volume.spacing, [3, 1])
        coordinate_matrix[:3, 3] = self.raw_volume.origin
        coordinate_matrix[3, 3] = 1
        self._labels = list()
        for segment_number in self.raw_volume.available_segments:
            label = self.raw_volume.segment_infos[segment_number][
                0x620006].value
            self._labels.append(label)
            self.label_to_num[label] = segment_number
        shape = self.raw_volume.segment_data(segment_number).shape
        shape = (shape[2], shape[1], shape[0])
        self._reference_frame = ReferenceFrame.from_coordinate_matrix(
            coordinate_matrix, shape=shape)

    def get_volume(self, label):
        if self.raw_volume is None:
            self.read()

        trans = (2, 1, 0)
        array = np.transpose(
            self.raw_volume.segment_data(self.label_to_num[label]), trans)

        return BinaryVolume(
            array,
            reference_frame=copy(self.reference_frame),
            reference_modality=self.reference_image.modality,
            modality="SEG",
            label=label,
            reference_dicom_header=self.reference_image.dicom_header,
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
        self._reference_image_uid = None
        self.error_list = list()

    @property
    def reference_image_uid(self):
        if self._reference_image_uid is None:
            self.read()
        return self._reference_image_uid

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
                logger.warning(
                    f"The reference image was not found for"
                    f" the RTSTRUCT of patient {self.dicom_header.patient_id}"
                    f" and study_id {self.dicom_header.study_instance_uid} "
                    f"which contains the labels {self.labels}. The "
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
        try:
            return (dcm.ReferencedFrameOfReferenceSequence[0].
                    RTReferencedStudySequence[0].RTReferencedSeriesSequence[0].
                    SeriesInstanceUID)
        except IndexError as e:
            raise RuntimeError(
                "The RTSTRUCT has no valid ReferencedFrameOfReferenceSequence")

    def read(self):
        self._labels = list()
        if len(self.dicom_paths) > 1:
            logger.warning(
                "there is multiple instances of the same RTSTRUCT file")
        if type(self.dicom_paths[0]) == FileDataset:
            self.slices = self.dicom_paths
        else:
            self.slices = [pdcm.read_file(p) for p in self.dicom_paths]
        self._reference_image_uid = RtstructFile.get_reference_image_uid(
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

    def _compute_mask(self, contour_sequence, label=""):
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
                if (np.min(rr) < 0 or np.min(cc) < 0
                        or np.max(rr) > mask.shape[0]
                        or np.min(cc) > mask.shape[1]):
                    raise Exception(f"The RTSTRUCT file is compromised, "
                                    f"it seems that the contour with "
                                    f"label {label} is out of bound")

            mask[rr, cc, np.round(vx_indices[0, 2]).astype(int)] = 1
        return mask

    def get_volume(self, label):
        self._initialize()
        try:
            contour_sequence = self.slices[0].ROIContourSequence[
                self.label_number_mapping[label]].ContourSequence
        except AttributeError:
            logger.warning(
                f"The RTSTRUCT for patient {self.dicom_header.patient_id}"
                f" and study {self.dicom_header.study_instance_uid} has a label '{label}' "
                f"which is empty")
            raise EmptyContourException()

        mask = self._compute_mask(contour_sequence, label=label)

        return BinaryVolume(
            mask,
            reference_frame=copy(self.reference_frame),
            reference_dicom_header=self.reference_image.dicom_header,
            reference_modality=self.reference_image.modality,
            label=label,
            modality="RTSTRUCT",
            dicom_header=self.dicom_header)
