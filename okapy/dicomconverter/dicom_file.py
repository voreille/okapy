"""
TODO: Make it DRYer line 220
"""

from os.path import join
import warnings
from copy import copy

import numpy as np
import pydicom as pdcm
from pydicom.tag import Tag
import pydicom_seg
from skimage.draw import polygon
import pandas as pd
import SimpleITK as sitk

from okapy.dicomconverter.volume import Volume, VolumeMask, ReferenceFrame


class DicomFileBase():
    def __init__(
            self,
            dicom_header=None,
            dicom_paths=list(),
            extension='nii.gz',
            reference_frame=None,
    ):
        self.dicom_header = dicom_header
        self.dicom_paths = dicom_paths
        self.extension = extension

    @property
    def reference_frame(self):
        raise NotImplementedError('This is an abstract class')

    @reference_frame.setter
    def reference_frame(self, value):
        raise NotImplementedError('This is an abstract class')


class DicomFileImageBase(DicomFileBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.slices = None
        self._reference_frame = None

    @property
    def reference_frame(self):
        return self._reference_frame

    @reference_frame.setter
    def reference_frame(self, value):
        self._reference_frame = value

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
        if self.slices is None:
            self.read()
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

        return Volume(image, reference_frame=copy(self.reference_frame))


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
        self._is_philips = None  # Should find a better name

    def read(self):
        super().read()
        if (self.slices[0].Manufacturer.upper().startswith('PHILIPS')
                and self.slices[0].Units.upper() == 'CNTS'):
            self._is_philips = True
        else:
            self._is_philips = False

    def _get_physical_values_philips(self):
        image = list()
        t1 = Tag(0x70531000)  # You can put this in other placs
        for s in self.slices:
            im = (float(s.RescaleSlope) * s.pixel_array +
                  float(s.RescaleIntercept)) * float(s[t1].value)
            image.append(im)
        return np.stack(image, axis=-1)

    def _get_physical_values_not_philips(self):
        # Get SUV from raw PET
        image = list()
        for s in self.slices:
            pet = float(s.RescaleSlope) * s.pixel_array + float(
                s.RescaleIntercept)
            half_life = float(s.RadiopharmaceuticalInformationSequence[0].
                              RadionuclideHalfLife)
            total_dose = float(s.RadiopharmaceuticalInformationSequence[0].
                               RadionuclideTotalDose)
            scan_time = s.SeriesTime
            scan_t = float(scan_time[0:2])*3600 + \
                float(scan_time[2:4])*60 + float(scan_time[4:])
            measured_time = s.RadiopharmaceuticalInformationSequence[
                0].RadiopharmaceuticalStartTime
            measured_t = float(measured_time[0:2]) * 3600 + float(
                measured_time[2:4]) * 60 + float(measured_time[4:])
            decay = 2**(-(scan_t - measured_t) / half_life)
            actual_activity = total_dose * decay
            im = pet * float(s.PatientWeight) * 1000 / actual_activity
            image.append(im)
        return np.stack(image, axis=-1)

    def get_physical_values(self):
        if self._is_philips:
            return self._get_physical_values_philips()
        else:
            return self._get_physical_values_not_philips()

    def get_dicom_header_df(self):
        if self.slices is None:
            self.read()
        return pd.DataFrame.from_dict({
            'Manufacturer': [self.slices[0].Manufacturer],
            'ManufacturerModelName': [self.slices[0].ManufacturerModelName],
            'Units': [self.slices[0].Units],
            'InstitutionName': [self.slices[0].InstitutionName],
        })


class MaskFile(DicomFileBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_volumes(self, list_labels):
        raise NotImplementedError('This is an abstract class')


class SegFile(MaskFile):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reader = pydicom_seg.SegmentReader()
        self._reference_frame = None

    @property
    def reference_frame(self):
        return self._reference_frame

    @reference_frame.setter
    def reference_frame(self, value):
        self._reference_frame = value

    def get_volumes(self, list_labels):
        volume_masks = list()
        if len(self.dicom_paths) != 1:
            raise RuntimeError('RTSTRUCT has more than one file')
        dcm = pdcm.dcmread(self.dicom_paths[0])
        result = self.reader.read(dcm)
        trans = (2, 1, 0)
        coordinate_matrix = np.zeros((4, 4))
        coordinate_matrix[:3, :3] = result.direction
        coordinate_matrix[:3, 3] = result.origin
        coordinate_matrix[3, 3] = 1
        compute_ref = True
        for segment_number in result.available_segments:
            np_volume = np.transpose(result.segment_data(segment_number),
                                     trans)
            if compute_ref:
                self.reference_frame = ReferenceFrame(
                    coordinate_matrix=coordinate_matrix, shape=np_volume.shape)
                compute_ref = False
            volume_masks.append(
                VolumeMask(np_volume,
                           reference_frame=copy(self.reference_frame),
                           name=result.segment_infos[segment_number][Tag(
                               0x620006)].value))

        return volume_masks


class RtstructFile(MaskFile):
    def __init__(self, *args, reference_image=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.contours = None
        self.reference_image = reference_image

    @property
    def reference_frame(self):
        return self.reference_image.reference_frame

    def read_structure(self, list_labels):
        if len(self.dicom_paths) != 1:
            raise RuntimeError('RTSTRUCT has more than one file')
        structure = pdcm.read_file(self.dicom_paths[0])
        self.contours = []
        for i, roi_seq in enumerate(structure.StructureSetROISequence):
            contour = {}
            if list_labels is None:
                contour['color'] = structure.ROIContourSequence[
                    i].ROIDisplayColor
                contour['number'] = structure.ROIContourSequence[
                    i].ReferencedROINumber
                contour['name'] = roi_seq.ROIName
                assert contour['number'] == roi_seq.ROINumber
                try:
                    contour['contours'] = [
                        s.ContourData for s in
                        structure.ROIContourSequence[i].ContourSequence
                    ]
                except AttributeError:
                    warnings.warn("No contour found for label {}.".format(
                        roi_seq.ROIName))
                    continue

                self.contours.append(contour)

            else:
                for label in list_labels:
                    if roi_seq.ROIName.startswith(label):
                        contour['color'] = structure.ROIContourSequence[
                            i].ROIDisplayColor
                        contour['number'] = structure.ROIContourSequence[
                            i].ReferencedROINumber
                        contour['name'] = roi_seq.ROIName
                        assert contour['number'] == roi_seq.ROINumber
                        try:
                            contour['contours'] = [
                                s.ContourData for s in
                                structure.ROIContourSequence[i].ContourSequence
                            ]
                        except AttributeError:
                            warnings.warn(
                                "No contour found for label {}.".format(
                                    roi_seq.ROIName))
                            continue

                        self.contours.append(contour)

    def get_volumes(self, list_labels):
        self.read_structure(list_labels)
        volume_masks = list()
        if self.reference_image.slices is None:
            self.reference_image.read()

        for con in self.contours:
            mask = np.zeros(self.reference_frame.shape, dtype=np.uint8)
            for current in con['contours']:
                nodes = np.array(current).reshape((-1, 3))
                assert np.amax(np.abs(np.diff(nodes[:, 2]))) == 0
                vx_positions = np.stack([
                    self.reference_frame.mm_to_vx(
                        [nodes[k, 1], nodes[k, 0], nodes[k, 2]])
                    for k in range(nodes.shape[0])
                ],
                                        axis=0)
                rr, cc = polygon(vx_positions[:, 1], vx_positions[:, 0])
                if len(rr) > 0 and len(cc) > 0:
                    if np.max(rr) > 512 or np.max(cc) > 512:
                        raise Exception("The RTSTRUCT file is compromised")

                mask[rr, cc, np.round(vx_positions[0, 2]).astype(int)] = 1

            name = con['name']
            volume_name = (self.dicom_header.patient_id + '__from_' +
                           self.reference_image.dicom_header.modality +
                           '_mask__' + name.replace(' ', '_'))

            volume_masks.append(
                VolumeMask(mask,
                           reference_frame=self.reference_frame,
                           name=volume_name))
        return volume_masks


class Study():
    image_modality_dict = {
        'CT': DicomFileCT,
        'PT': DicomFilePT,
        'MR': DicomFileMR,
    }

    def __init__(self,
                 study_instance_uid=None,
                 padding_voi=0,
                 resampling_spacing_modality=None,
                 extension_output='nii.gz',
                 list_labels=None,
                 csv_info=False):
        self.volume_masks = list()
        self.mask_files = list()  # Can have multiple RTSTRUCT and also SEG
        self.dicom_file_images = list()  # Can have multiple RTSTRUCT
        self.study_instance_uid = study_instance_uid
        self.padding_voi = padding_voi
        self.bounding_box = None
        self.extension_output = extension_output
        self.list_labels = list_labels
        self.csv_info = csv_info
        if resampling_spacing_modality is None:
            self.resampling_spacing_modality = {
                'CT': (0.75, 0.75, 0.75),
                'PT': (0.75, 0.75, 0.75),
                'MR': (0.75, 0.75, 0.75),
            }
        else:
            self.resampling_spacing_modality = resampling_spacing_modality
        self.current_modality_list = list()

    def append_dicom_files(self, im_dicom_files, dcm_header):
        if dcm_header.modality == 'RTSTRUCT':
            for im in reversed(self.dicom_file_images):
                if (im.dicom_header.series_instance_uid ==
                        dcm_header.series_instance_uid):
                    self.mask_files.append(
                        RtstructFile(
                            reference_image=im,
                            extension=self.extension_output,
                            dicom_header=dcm_header,
                            dicom_paths=[k.path for k in im_dicom_files],
                        ))

                    break

            if not self.mask_files:
                for im in reversed(self.dicom_file_images):
                    if (im.dicom_header.modality == 'CT'
                            and im.dicom_header.patient_id
                            == dcm_header.patient_id):
                        self.mask_files.append(
                            RtstructFile(
                                reference_image=im,
                                extension=self.extension_output,
                                dicom_header=dcm_header,
                                dicom_paths=[k.path for k in im_dicom_files],
                            ))
                        print('Taking the CT as ref for patient: {}'.format(
                            im.dicom_header.patient_id))

                        break
        elif dcm_header.modality == 'SEG':
            self.mask_files.append(
                SegFile(extension=self.extension_output,
                        dicom_header=dcm_header,
                        dicom_paths=[k.path for k in im_dicom_files]))

        else:
            try:

                self.dicom_file_images.append(
                    Study.image_modality_dict[dcm_header.modality](
                        extension=self.extension_output,
                        dicom_header=dcm_header,
                        dicom_paths=[k.path for k in im_dicom_files],
                    ))
                self.current_modality_list.append(dcm_header.modality)
            except KeyError:
                print('This modality {} is not yet (?) supported'.format(
                    dcm_header.modality))

    def process(self, output_dirpath):
        # Compute the mask
        for mask_file in self.mask_files:
            # we extend since get_volumes return a list
            self.volume_masks.extend(mask_file.get_volumes(self.list_labels))

        # Compute the bounding box

        bb = self.volume_masks[0].padded_bb(self.padding_voi)
        for mask in self.volume_masks:
            bb = mask.bb_union(bb, self.padding_voi)

        # process the images i. to np ii. resampling iii. saving
        for dcm_file in self.dicom_file_images:
            image = dcm_file.get_volume()
            image.resample(
                self.resampling_spacing_modality[
                    dcm_file.dicom_header.modality], bb)
            filename = (dcm_file.dicom_header.patient_id + '__' +
                        dcm_file.dicom_header.modality + image.str_resampled +
                        '.' + self.extension_output)

            filename = filename.replace(' ', '_')
            filepath = join(output_dirpath, filename)

            sitk.WriteImage(image.get_sitk_image(), filepath)

            # Save the minimal dicom info relevant for the stantdardisation
            if self.csv_info:
                dicom_header_df = dcm_file.get_dicom_header_df()
                filename = (dcm_file.dicom_header.patient_id + '__' +
                            dcm_file.dicom_header.modality +
                            image.str_resampled + '.csv')
                filepath = join(output_dirpath, filename)
                dicom_header_df.to_csv(filepath)

        # Keeping only the present modalities
        resampling_spacing_modality = {
            key: self.resampling_spacing_modality[key]
            for key in self.current_modality_list
        }

        for mask in self.volume_masks:
            for key, item in resampling_spacing_modality.items():
                image = mask.get_resampled_volume(item, bb)
                filename = (mask.name + '__resampled_for__' + key + '.' +
                            self.extension_output)
                filepath = join(output_dirpath, filename)
                sitk.WriteImage(image.get_sitk_image(), filepath)
