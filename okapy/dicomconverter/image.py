'''
TODO: Direction cosine in sitk
TODO: Change the name of the class for instance DicomBase instead of DicomFileBase
TODO: Add check of the bounding_box, if it goes beyond the image domain
TODO: Investingat getattr for the nested value like shape in volume
'''
from os.path import join

import numpy as np
from scipy import ndimage
import SimpleITK as sitk
import pydicom as pdcm
from skimage.draw import polygon


class Volume():
    def __init__(self, np_image=None, image_pos_patient=None,
                 pixel_spacing=None, name='', str_resampled=''):
        self.np_image = np_image
        self.image_pos_patient = image_pos_patient
        self.pixel_spacing = pixel_spacing
        self.shape = np_image.shape
        self.str_resampled = str_resampled
        self.name = name
        self.total_bb = self._get_total_bb()

    def _get_total_bb(self):
        out = np.asarray([*self.image_pos_patient, *self.image_pos_patient])
        out += np.asarray([0,0,0, self.shape[0]*self.pixel_spacing[0],
                           self.shape[1]*self.pixel_spacing[1],
                           self.shape[2]*self.pixel_spacing[2]])
        return out

    def get_zeros_like(self):
        return Volume(np_image=np.zeros_like(self.np_image),
                      image_pos_patient=self.image_pos_patient,
                      pixel_spacing=self.pixel_spacing)

    def _check_resampling_spacing(self, resampling_px_spacing):
        if len(resampling_px_spacing) == 3:
            cond = True
            for i in range(3):
                cond *= (self.pixel_spacing[i] == resampling_px_spacing[i])
        else:
            cond = False

        return cond

    def _get_resampled_np(self, resampling_px_spacing, bounding_box, order=1):

            zooming_matrix = np.identity(3)
            zooming_matrix[0, 0] = resampling_px_spacing[0] / self.pixel_spacing[0]
            zooming_matrix[1, 1] = resampling_px_spacing[1] / self.pixel_spacing[1]
            zooming_matrix[2, 2] = resampling_px_spacing[2] / self.pixel_spacing[2]

            offset = ((bounding_box[0] - self.image_pos_patient[0]) / self.pixel_spacing[0],
                      (bounding_box[1] - self.image_pos_patient[1]) / self.pixel_spacing[1],
                      (bounding_box[2] - self.image_pos_patient[2]) / self.pixel_spacing[2])

#            offset = ((-bounding_box[0] + self.image_pos_patient[0]) / resampling_px_spacing[0],
#                      (-bounding_box[1] + self.image_pos_patient[1]) / resampling_px_spacing[1],
#                      (-bounding_box[2] + self.image_pos_patient[2]) / resampling_px_spacing[2])


            output_shape = np.ceil([
                bounding_box[3] - bounding_box[0],
                bounding_box[4] - bounding_box[1],
                bounding_box[5] - bounding_box[2],
            ]) / resampling_px_spacing

            np_image = ndimage.affine_transform(self.np_image, zooming_matrix,
                                                    offset=offset, mode='mirror',
                                                    order=order,
                                                    output_shape=output_shape.astype(int))

            return np_image, output_shape


    def resample(self, resampling_px_spacing, bounding_box, order=1):
        """
        Resample the 3D volume to the resampling_px_spacing according to
        the bounding_boc in cm (x1, y1, z1, x2, y2, z2)
        """
        if not self._check_resampling_spacing(resampling_px_spacing):

            self.str_resampled = '__resampled'
            np_image, output_shape = self._get_resampled_np(resampling_px_spacing, bounding_box, order)

            self.np_image = np_image
            self.shape = output_shape
            self.pixel_spacing = resampling_px_spacing
            self.total_bb = bounding_box
            self.image_pos_patient = np.asarray([bounding_box[0], bounding_box[1],
                                                bounding_box[2]])


    def get_sitk_image(self):
        trans = (2,0,1)
        sitk_image = sitk.GetImageFromArray(np.transpose(self.np_image,
                                                        trans))
        sitk_image.SetSpacing(self.pixel_spacing)
        sitk_image.SetOrigin(self.image_pos_patient)
        return sitk_image


class VolumeMask(Volume):
    def __init__(self, *args,**kwargs):
        super().__init__(*args, **kwargs)
        self.bb = None
        self.np_image[self.np_image!=0] = 1

    def get_absolute_bb(self, padding=0):
        if self.bb is None:
            indices = np.where(self.np_image > 0)
            x1 = (np.min(indices[0]) * self.pixel_spacing[0] +
                self.image_pos_patient[0]  - padding)
            x2 = (np.max(indices[0]) * self.pixel_spacing[0] +
                self.image_pos_patient[0]  + padding)
            y1 = (np.min(indices[1]) * self.pixel_spacing[1] +
                self.image_pos_patient[1]  - padding)
            y2 = (np.max(indices[1]) * self.pixel_spacing[1] +
                self.image_pos_patient[1]  + padding)
            z1 = (np.min(indices[2]) * self.pixel_spacing[2] +
                self.image_pos_patient[2]  - padding)
            z2 = (np.max(indices[2]) * self.pixel_spacing[2] +
                self.image_pos_patient[2]  + padding)

            bb = np.array([x1, y1, z1, x2, y2, z2,])
            bb[0:3] = np.maximum(bb[0:3], self.total_bb[0:3])
            bb[3:] = np.minimum(bb[3:], self.total_bb[3:])
            self.bb = bb

        return self.bb

    def resample(self, resampling_px_spacing, bounding_box, order=1):
        super().resample(resampling_px_spacing=resampling_px_spacing,
                         bounding_box=bounding_box, order=order)
        self.np_image[self.np_image>0.5] = 1
        self.np_image[self.np_image<0.5] = 0

    def get_resampled_volume(self, resampling_px_spacing, bounding_box, order=1):
        if not self._check_resampling_spacing(resampling_px_spacing):

            np_image, output_shape = self._get_resampled_np(resampling_px_spacing, bounding_box, order=order)
            image_pos_patient = np.asarray([bounding_box[0], bounding_box[1],
                                                bounding_box[2]])
            return VolumeMask(np_image=np_image,
                              image_pos_patient=image_pos_patient,
                              pixel_spacing=resampling_px_spacing,
                              name=self.name,
                              str_resampled='_resampled_')

        else:
            return self





class DicomFileBase():
    def __init__(self, dicom_header=None, dicom_paths=list(),
                 extension='nrrd', resampling_px_spacing=None):
        self.dicom_header = dicom_header
        self.dicom_paths = dicom_paths
        self.extension = extension


class DicomFileImageBase(DicomFileBase):
    def __init__(self, *args,
                 resampling_px_spacing=(1.0, 1.0, 1.0),
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.resampling_px_spacing = resampling_px_spacing
        self.slices_z_position = None
        self.image_pos_patient = None # Maybe to get rid of
        self.pixel_spacing = None # Maybe to get rid of
        self.slices=None

    def get_physical_values(self, slices):
        raise NotImplementedError('This is an abstract class')

    def read(self):
        slices = [pdcm.read_file(dcm) for dcm in self.dicom_paths]
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))

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
        self.slices = slices
        self.shape = (*slices[0].pixel_array.shape, len(slices))

    def get_volume(self):
        if self.slices is None:
            self.read()
        image = self.get_physical_values(self.slices)

        return Volume(image, pixel_spacing=self.pixel_spacing,
                      image_pos_patient=np.asarray([
                          self.image_pos_patient[1],
                          self.image_pos_patient[0],
                          self.image_pos_patient[2]]))


class DicomFileCT(DicomFileImageBase):
    def get_physical_values(self, slices):
        image = list()
        for s in slices:
            image.append(float(s.RescaleSlope) * s.pixel_array +
                         float(s.RescaleIntercept))
        return np.stack(image, axis=-1)


class DicomFileMR(DicomFileImageBase):
    def get_physical_values(self, slices):
        image = list()
        for s in slices:
            image.append(s.pixel_array)
        return np.stack(image, axis=-1)



class DicomFilePT(DicomFileImageBase):
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


class RtstructFile(DicomFileBase):
    def __init__(self, *args,
                 reference_image=None,
                 list_labels=None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.list_labels = list_labels
        self.contours = None
        self.reference_image = reference_image


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


    def get_volumes(self):
        self.read_structure()
        volume_masks = list()
        if self.reference_image.slices_z_position is None:
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
            volume_name = (self.dicom_header.patient_id + '__from_' +
                           self.reference_image.dicom_header.modality
                           +'_mask__' + name.replace(' ', '_'))

            volume_masks.append(VolumeMask(mask,
                                           image_pos_patient=np.asarray([
                                               self.image_pos_patient[1],
                                               self.image_pos_patient[0],
                                               self.image_pos_patient[2],
                                           ]),
                                           pixel_spacing=self.pixel_spacing,
                                           name=volume_name
                                           ))
        return volume_masks



class Study():
    image_modality_dict = {
        'CT': DicomFileCT,
        'PT': DicomFilePT,
        'MR': DicomFileMR,
    }

    def __init__(self, sitk_writer=None, study_instance_uid=None,
                 padding_voi=0, resampling_spacing_modality=None,
                 extension_output='nii', list_labels=None):
        self.volume_masks = list()
        self.rtstruct_files = list() # Can have multiple RTSTRUCT
        self.dicom_file_images = list() # Can have multiple RTSTRUCT
        self.sitk_writer = sitk_writer
        self.study_instance_uid = study_instance_uid
        self.padding_voi = padding_voi
        self.bounding_box = None
        self.extension_output = extension_output
        self.list_labels = list_labels
        if resampling_spacing_modality is None:
            self.resampling_spacing_modality = {
                    'CT': (0.75, 0.75, 0.75),
                    'PT': (0.75, 0.75, 0.75),
                }
        else:
            self.resampling_spacing_modality = resampling_spacing_modality


    def append_dicom_files(self, im_dicom_files, dcm_header):
        if dcm_header.modality == 'RTSTRUCT':
            for im in reversed(self.dicom_file_images):
                if (im.dicom_header.series_instance_uid == dcm_header.series_instance_uid):
                    self.rtstruct_files.append(RtstructFile(
                        reference_image=im,
                        list_labels=self.list_labels,
                        extension=self.extension_output,
                        dicom_header=dcm_header,
                        dicom_paths=[k.path for k in im_dicom_files],
                    ))

                    break

            if not self.rtstruct_files:
                for im in reversed(self.dicom_file_images):
                    if (im.dicom_header.modality == 'CT' and
                        im.dicom_header.patient_id == dcm_header.patient_id):
                        self.rtstruct_files.append(RtstructFile(
                            reference_image=im,
                            list_labels=self.list_labels,
                            extension=self.extension_output,
                            dicom_header=dcm_header,
                            dicom_paths=[k.path for k in im_dicom_files],
                        ))
                        print('Taking the CT as ref for patient: {}'.format(
                            im.dicom_header.patient_id))

                        break


        else:
            try:

                self.dicom_file_images.append(Study.image_modality_dict[dcm_header.modality](
                    extension=self.extension_output,
                    dicom_header=dcm_header,
                    dicom_paths=[k.path for k in im_dicom_files],
                    resampling_px_spacing=self.resampling_spacing_modality[dcm_header.modality]
                ))
            except KeyError:
                print('This modality {} is not yet (?) supported'
                    .format(dcm_header.modality))

    def process(self, output_dirpath):
        # Compute the mask
        for rtstruct in self.rtstruct_files:
            self.volume_masks.extend(rtstruct.get_volumes())

        #Compute the bounding box

        bb = self.volume_masks[0].get_absolute_bb(self.padding_voi)
        for mask in self.volume_masks:
            bb_m = mask.get_absolute_bb(self.padding_voi)
            bb[0:3] = np.minimum(bb[0:3], bb_m[0:3])
            bb[3:] = np.maximum(bb[3:], bb_m[3:])


        #process the images i. to np ii. resampling iii. saving
        for dcm_file in self.dicom_file_images:
            image = dcm_file.get_volume()
            image.resample(self.resampling_spacing_modality[dcm_file.dicom_header.modality], bb)
            filename = (dcm_file.dicom_header.patient_id + '__' + dcm_file.dicom_header.modality +
                        image.str_resampled + '.' + self.extension_output)

            filepath = join(output_dirpath, filename)

            self.sitk_writer.SetFileName(filepath)
            self.sitk_writer.Execute(image.get_sitk_image())

        for mask in self.volume_masks:
            for key, item in self.resampling_spacing_modality.items():
                image = mask.get_resampled_volume(item, bb)
                filename = mask.name + '__resampled_for__'+ key +'.' + self.extension_output
                filepath = join(output_dirpath, filename)
                self.sitk_writer.SetFileName(filepath)
                self.sitk_writer.Execute(image.get_sitk_image())

