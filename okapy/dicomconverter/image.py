'''
TODO: Check the MR of the rescale and slope and stuff
TODO: remove the method read, just put all the thing in write
TODO: Direction cosine in sitk
'''
from os.path import join
from string import Template

import numpy as np
from scipy import ndimage
import SimpleITK as sitk
import pydicom as pdcm
from skimage.draw import polygon


class VolumeBase():
    def __init__(self, sitk_writer=None, dicom_header=None, dicom_paths=list(),
                 extension='nrrd', resampling_px_spacing=None):
        self.sitk_writer = sitk_writer # TODO: make it simpler
        self.dicom_header = dicom_header
        self.dicom_paths = dicom_paths
        self.extension = extension
        self.resampling_px_spacing = resampling_px_spacing

    def convert(self, path):
        self.read()
        self.write(path)
        # DELETE ?!

    def read(self):
        raise NotImplementedError('This is an abstract class')

    def write(self, path):
        raise NotImplementedError('This is an abstract class')

    def resample(self):
        raise NotImplementedError('This is an abstract class')

    def __str__(self):
        return '''Image :''' + str(self.dicom_header)


class ImageBase(VolumeBase):
    def __init__(self, *args,
                 template_filename=Template('{patient_id}_${modality}.${ext}'),
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.slices_z_position = None
        self.image_pos_patient = None
        self.pixel_spacing = None
        self.np_image = None
        self.filename = template_filename.substitute(
            patient_id=self.dicom_header.patient_id,
            modality=self.dicom_header.modality,
            ext=self.extension
        )

    def get_physical_values(self, slices):
        raise NotImplementedError('This is an abstract class')

    def read(self):
        slices = [pdcm.read_file(dcm) for dcm in self.dicom_paths]
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
        image = self.get_physical_values(slices)

        slice_spacing = (slices[1].ImagePositionPatient[2] -
          slices[0].ImagePositionPatient[2])

        pixel_spacing = np.asarray([slices[0].PixelSpacing[0],
                                    slices[0].PixelSpacing[1],
                                    slice_spacing,
                                    ])
#        image.SetDirection([float(k) for k in slices[0].ImageOrientationPatient])
        image_pos_patient = [float(k) for k in slices[0].ImagePositionPatient]
        self.slices_z_position = [float(s.ImagePositionPatient[2]) for s in slices]
        self.pixel_spacing = pixel_spacing
        self.image_pos_patient = image_pos_patient
        self.shape = image.shape
        self.np_image = image

    def resample(self):
        if self.resampling_px_spacing is not None:
            zooming_matrix = np.identity(3)
            zooming_factor_x = (self.resampling_px_spacing[0] / self.pixel_spacing[0]
                                if self.resampling_px_spacing[0]>0 else 1)
            zooming_factor_y = (self.resampling_px_spacing[1] / self.pixel_spacing[1]
                                if self.resampling_px_spacing[1]>0 else 1)
            zooming_factor_z = (self.resampling_px_spacing[2] / self.pixel_spacing[2]
                                if self.resampling_px_spacing[2]>0 else 1)
            zooming_matrix[0, 0] = zooming_factor_x
            zooming_matrix[1, 1] = zooming_factor_y
            zooming_matrix[2, 2] = zooming_factor_z
            output_shape = (int(self.shape[0] / zooming_factor_x),
                            int(self.shape[1] / zooming_factor_y),
                            int(self.shape[2] / zooming_factor_z))

            self.np_image = ndimage.affine_transform(self.np_image, zooming_matrix, mode='mirror',
                                            output_shape=output_shape)
            self.pixel_spacing = self.resampling_px_spacing


    def write(self, path):
        trans = (2,0,1)
        sitk_image = sitk.GetImageFromArray(np.transpose(self.np_image,
                                                         trans))
        sitk_image.SetSpacing(self.pixel_spacing)
        sitk_image.SetOrigin(self.image_pos_patient)
        path = join(path, self.filename)
        self.sitk_writer.SetFileName(path)
        self.sitk_writer.Execute(sitk_image)
        del sitk_image

    def convert(self, path):
        if self.np_image is None:
            self.read()
        self.resample()
        self.write(path)


class ImageCT(ImageBase):
    def get_physical_values(self, slices):
        image = list()
        for s in slices:
            image.append(float(s.RescaleSlope) * s.pixel_array +
                         float(s.RescaleIntercept))
        return np.stack(image, axis=-1)


class ImageMR(ImageBase):
    def get_physical_values(self, slices):
        image = list()
        for s in slices:
            image.append(s.pixel_array)
        return np.stack(image, axis=-1)



class ImagePT(ImageBase):
    def get_physical_values(self, slices):
        # Get SUV from raw PET
        image = list()
        for s in slices:
            pet = float(s.RescaleSlope) * s.pixel_array + float(s.RescaleIntercept)
            half_life = float(
                s.RadiopharmaceuticalInformationSequence[0].RadionuclideHalfLife)
            total_dose = float(
                s.RadiopharmaceuticalInformationSequence[0].RadionuclideTotalDose)
            scan_time = s.SeriesTime
            scan_t = float(scan_time[0:2])*3600 + \
                float(scan_time[2:4])*60 + float(scan_time[4:])
            measured_time = s.RadiopharmaceuticalInformationSequence[0].RadiopharmaceuticalStartTime
            measured_t = float(
                measured_time[0:2])*3600 + float(measured_time[2:4])*60 + float(measured_time[4:])
            decay = 2**(-(scan_t-measured_t)/half_life)
            actual_activity = total_dose * decay
            im = pet * float(s.PatientWeight)*1000 / actual_activity
            image.append(im)
        return np.stack(image, axis=-1)


class Mask(VolumeBase):
    def __init__(self, *args,
                 template_filename=Template('{patient_id}_${modality}.${ext}'),
                 reference_image=None,
                 list_labels=None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.list_labels = list_labels
        self.contours = None
        self.reference_image = reference_image
        self.np_masks = list()
        self.filenames = list()
        self.template_filename = template_filename


    def _compute_and_append_name(self, name_contour):
        self.filenames.append(
            self.template_filename.substitute(
            patient_id=self.dicom_header.patient_id,
            modality=self.dicom_header.modality + '_{}'.format(
                name_contour.replace(' ','_')),
            ext=self.extension)
        )


    def read_structure(self):
        if len(self.dicom_paths) != 1:
            raise Exception('RTSTRUCT has more than one file')
        structure = pdcm.read_file(self.dicom_paths[0])
        self.contours = []
        for i, roi_seq in enumerate(structure.StructureSetROISequence):
            contour = {}
            if self.list_labels is None:
                contour['color'] = structure.ROIContourSequence[i].ROIDisplayColor
                contour['number'] = structure.ROIContourSequence[i].ReferencedROINumber
                contour['name'] = roi_seq.ROIName
                assert contour['number'] == roi_seq.ROINumber
                contour['contours'] = [
                    s.ContourData for s in structure.ROIContourSequence[i].ContourSequence
                ]
                self.contours.append(contour)

            else:
                for label in self.list_labels:
                    if roi_seq.ROIName.startswith(label):
                        contour['color'] = structure.ROIContourSequence[i].ROIDisplayColor
                        contour['number'] = structure.ROIContourSequence[i].ReferencedROINumber
                        contour['name'] = roi_seq.ROIName
                        assert contour['number'] == roi_seq.ROINumber
                        contour['contours'] = [
                            s.ContourData for s in structure.ROIContourSequence[i].ContourSequence
                        ]
                        self.contours.append(contour)


    def compute_mask(self):
        if self.reference_image.np_image is None:
            self.reference_image.read()
        z = np.asarray(self.reference_image.slices_z_position)
        pos_r = self.reference_image.image_pos_patient[1]
        spacing_r = self.reference_image.pixel_spacing[1]
        pos_c = self.reference_image.image_pos_patient[0]
        spacing_c = self.reference_image.pixel_spacing[0]
        shape = self.reference_image.shape
        self.pixel_spacing = self.reference_image.pixel_spacing
        self.image_pos_patient = self.reference_image.image_pos_patient

        for con in self.contours:
            mask = np.zeros(shape, dtype=np.uint8)
            for current in con['contours']:
                nodes = np.array(current).reshape((-1, 3))
                assert np.amax(np.abs(np.diff(nodes[:, 2]))) == 0
                z_index = np.where((nodes[0, 2]- 0.001 < z) &
                                   (z < nodes[0, 2]+0.001))[0][0]
                r = (nodes[:, 1] - pos_r) / spacing_r
                c = (nodes[:, 0] - pos_c) / spacing_c
                rr, cc = polygon(r, c)
                if len(rr) > 0 and len(cc) > 0:
                    if np.max(rr) > 512 or np.max(cc) > 512:
                        raise Exception("The RTSTRUCT file is compromised")

                mask[rr, cc, z_index] = 1

            name = con['name']
            self.np_masks.append(mask)
            self._compute_and_append_name(name)

    def read(self):
        self.read_structure()
        self.compute_mask()

    def resample(self, resampling_px_spacing):
        if self.resampling_px_spacing is not None:
            zooming_matrix = np.identity(3)
            zooming_factor_x = (self.resampling_px_spacing[0] / self.pixel_spacing[0]
                                if self.resampling_px_spacing[0]>0 else 1)
            zooming_factor_y = (self.resampling_px_spacing[1] / self.pixel_spacing[1]
                                if self.resampling_px_spacing[1]>0 else 1)
            zooming_factor_z = (self.resampling_px_spacing[2] / self.pixel_spacing[2]
                                if self.resampling_px_spacing[2]>0 else 1)
            zooming_matrix[0, 0] = zooming_factor_x
            zooming_matrix[1, 1] = zooming_factor_y
            zooming_matrix[2, 2] = zooming_factor_z
            output_shape = (int(self.shape[0] / zooming_factor_x),
                            int(self.shape[1] / zooming_factor_y),
                            int(self.shape[2] / zooming_factor_z))

            for mask in self.np_masks:
                mask = ndimage.affine_transform(mask, zooming_matrix,
                                                mode='mirror',
                                                output_shape=output_shape)

            self.pixel_spacing = self.resampling_px_spacing


    def write(self, path):
        for i, mask in enumerate(self.np_masks):
            sitk_mask = sitk.GetImageFromArray(np.moveaxis(mask, 2, 0))
            sitk_mask.SetSpacing(self.pixel_spacing)
            sitk_mask.SetOrigin(self.image_pos_patient)
            self.sitk_writer.SetFileName(join(path, self.filenames[i]))
            self.sitk_writer.Execute(sitk_mask)
