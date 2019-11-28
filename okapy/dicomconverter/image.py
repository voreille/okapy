'''
TODO: Manage when multiple RTSTRUCT
TODO: Direction cosine in sitk
TODO: Change the name of the class for instance DicomBase instead of VolumeBase
TODO: Add check of the bounding_box, if it goes beyond the image domain
'''
from os.path import join
from string import Template
from functools import partial

import numpy as np
from scipy import ndimage
import SimpleITK as sitk
import pydicom as pdcm
from radiomics.imageoperations import resampleImage
from skimage.draw import polygon




class VolumeBase():
    def __init__(self, sitk_writer=None, dicom_header=None, dicom_paths=list(),
                 extension='nrrd', resampling_px_spacing=None):
        self.sitk_writer = sitk_writer # TODO: make it simpler
        self.dicom_header = dicom_header
        self.dicom_paths = dicom_paths
        self.extension = extension

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

class Volume():
    def __init__(self, np_image=None, image_pos_patient=None,
                 pixel_spacing=None):
        self.np_image = np_image
        self.image_pos_patient = image_pos_patient
        self.pixel_spacing = pixel_spacing
        self.shape = np_image.shape

    def resample(self, resampling_px_spacing, bounding_box):
        """
        Resample the 3D volume to the resampling_px_spacing according to
        the bounding_boc in cm (x1, y1, z1, x2, y2, z2)
        """

        zooming_matrix = np.identity(3)
        zooming_matrix[0, 0] = resampling_px_spacing[0] / self.pixel_spacing[0]
        zooming_matrix[1, 1] = resampling_px_spacing[1] / self.pixel_spacing[1]
        zooming_matrix[2, 2] = resampling_px_spacing[2] / self.pixel_spacing[2]

        #Check the UNITS
        offset = ((bounding_box[0] - self.image_pos_patient[0] *
                   resampling_px_spacing[0]) / self.pixel_spacing[0],
                  (bounding_box[1] - self.image_pos_patient[1] *
                   resampling_px_spacing[1]) / self.pixel_spacing[1],
                  (bounding_box[2] - self.image_pos_patient[2] *
                   resampling_px_spacing[2]) / self.pixel_spacing[2])

        output_shape = np.ceil([
            bounding_box[3] - bounding_box[0],
            bounding_box[4] - bounding_box[1],
            bounding_box[5] - bounding_box[2],
        ]) / resampling_px_spacing

        self.np_image = ndimage.affine_transform(self.np_image, zooming_matrix,
                                                 offset=offset, mode='mirror',
                                                 output_shape=output_shape)

        self.shape = output_shape
        self.pixel_spacing = resampling_px_spacing
        self.image_pos_patient = np.asarray([bounding_box[0], bounding_box[1],
                                             bounding_box[2]])

    def get_sitk_image(self):
        trans = (2,0,1)
        sitk_image = sitk.GetImageFromArray(np.transpose(self.np_image,
                                                        trans))
        sitk_image.SetSpacing(self.pixel_spacing)
        sitk_image.SetOrigin(self.image_pos_patient)
        return sitk_image


class ImageBase(VolumeBase):
    def __init__(self, *args,
                 template_filename=Template('{patient_id}_${modality}.${ext}'),
                 resampling_px_spacing=(1.0, 1.0, 1.0),
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.resampling_px_spacing = resampling_px_spacing
        self.slices_z_position = None
        self.image_pos_patient = None # Maybe to get rid of
        self.pixel_spacing = None # Maybe to get rid of
        self.volume = None
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
        self.volume = Volume(image, pixel_spacing=pixel_spacing,
                               image_pos_patient=image_pos_patient)

    def resample(self):
        if self.resampling_px_spacing is not None:
            self.volume.resample(self.resampling_px_spacing)


    def write(self, path):
        path = join(path, self.filename)
        self.sitk_writer.SetFileName(path)
        self.sitk_writer.Execute(self.volume.get_sitk_image())

    def convert(self, path):
        if self.volume is None:
            self.read()
        self.resample()
        self.write(path)


class ImageCT(ImageBase):
    def get_physical_values(self, slices):
        image = list()
        dtype = slices[0].pixel_array.dtype
        for s in slices:
            image.append(np.asarray(s.RescaleSlope, dtype=dtype) * s.pixel_array +
                         np.asarray(s.RescaleIntercept, dtype=dtype))
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
            scan_t = float(scan_time[1:2])*3600 + \
                float(scan_time[3:4])*60 + float(scan_time[5:6])
            measured_time = s.RadiopharmaceuticalInformationSequence[0].RadiopharmaceuticalStartTime
            measured_t = float(
                measured_time[1:2])*3600 + float(measured_time[3:4])*60 + float(measured_time[5:6])
            decay = 2**(-(scan_t-measured_t)/half_life)
            actual_activity = total_dose * decay
            im = pet * float(s.PatientWeight)*1000 / actual_activity
            image.append(im)
        return np.stack(image, axis=-1)


class Rtstruct(VolumeBase):
    def __init__(self, *args,
                 template_filename=Template('{patient_id}_${modality}.${ext}'),
                 reference_image=None,
                 list_labels=None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.list_labels = list_labels
        self.contours = None
        self.reference_image = reference_image
        self.masks = list()
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
        if self.reference_image.volume is None:
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


class Study():
    image_modality_dict = {
        'CT': ImageCT,
        'PT': ImagePT,
        'MR': ImageMR,
    }

    def __init__(self, sitk_writer=None, study_instance_uid=None,
                 padding_voi=None, resampling_spacing_modality=None):
        self.images = list()
        self.rtstruct = None # Can have multiple RTSTRUCT
        self.sitk_writer = sitk_writer
        self.study_instance_uid = study_instance_uid
        self.padding_voi = padding_voi
        self.voi_tot = None
        if resampling_spacing_modality is None:
            self.resampling_spacing_modality = {
                    'CT': (1.0, 1.0, 1.0),
                    'PT': (1.0, 1.0, 1.0),
                    'MR': (1.0, 1.0, 1.0),
                }
        else:
            self.resampling_spacing_modality = resampling_spacing_modality


    def append_image(self, im_dicom_files, dcm_header):
        if dcm_header.modality == 'RTSTRUCT':
            for im in reversed(self.images):
                if (im.dicom_header.series_instance_uid == dcm_header.series_instance_uid):
                    self.rtstruct = Rtstruct(
                        reference_image=im,
                        list_labels=self.list_labels,
                        extension=self.extension_output,
                        sitk_writer=self.sitk_writer,
                        dicom_header=dcm_header,
                        dicom_paths=[k.path for k in im_dicom_files],
                        template_filename=self.template_filename,
                    )

                    break
        else:
            try:

                self.images.append(Study.image_modality_dict[dcm_header.modality](
                    extension=self.extension_output,
                    sitk_writer=self.sitk_writer,
                    dicom_header=dcm_header,
                    dicom_paths=[k.path for k in im_dicom_files],
                    template_filename=self.template_filename,
                    resampling_px_spacing=self.resampling_px_spacing
                ))
            except KeyError:
                print('This modality {} is not yet (?) supported'
                    .format(dcm_header.modality))

    def read(self):
        self.rtstruct.read()

        self.voi_tot = np.zeros(self.rtstruct.np_masks[0].shape,
                                dtype=np.uint8)
        for mask in self.rtstruct.np_masks:
            self.voi_tot += mask

        for image in self.images:
            image.read()


    def write(self, path):
        pass

    def resample(self):
        pass

    def convert(self, path):
        self.read()
        if self.voi_tot is not None:
            self.resample()
        self.write(path)


