'''
TODO: remove the method read, just put all the thing in write
'''
import os
from os.path import dirname, join

import numpy as np
import SimpleITK as sitk
import pydicom as pdcm
from skimage.draw import polygon


class VolumeBase():
    def __init__(self, sitk_writer=None, dicom_headers=list()):
        self.sitk_writer = sitk_writer
        self.dicom_headers = dicom_headers # we can retrieve path here

    def convert(self, path):
        self.read()
        self.write(path)
        # DELETE ?!

    def read(self):
        raise NotImplementedError('This is an abstract class')

    def write(self, path):
        raise NotImplementedError('This is an abstract class')


class ImageBase(VolumeBase):
    def __init__(self, *args, sitk_image=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.slices_z_position = None
        self.image_pos_patient = None
        self.pixel_spacing = None
        self.sitk_image = sitk_image

    def get_physical_values(slices):
        raise NotImplementedError('This is an abstract class')

    def read(self):
        slices = [pdcm.read_file(dcm) for dcm in self.dicom_paths]
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
        image = sitk.GetImageFromArray(self.get_physical_values(slices))

        slice_spacing = (slices[1].ImagePositionPatient[2] -
          slices[0].ImagePositionPatient[2])

        pixel_spacing = np.asarray([slices[0].PixelSpacing[0],
                                    slices[0].PixelSpacing[1],
                                    slice_spacing,
                                    ])
        image.SetSpacing(pixel_spacing)
        image.SetDirection(slices[0].ImageOrientationPatient)
        image_pos_patient = slices[0].ImagePositionPatient
        image.SetOrigin(image_pos_patient)
        self.slices_z_position = [s.ImagePositionPatient[2] for s in slices]
        self.pixel_spacing = pixel_spacing
        self.image_pos_patient = image_pos_patient
        self.sitk_image = image

    def write(self, path):
        self.sitk_writer.SetFileName(path)
        self.sitk_writer.Execute(self.sitk_image)

    def convert(self, path):
        self.read()
        self.write(path)
        del self.sitk_image


class ImageCT(ImageBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_physical_values(slices):
        image = list()
        for s in slices:
            image.append(float(s.RescaleSlope) * s.pixel_array +
                         float(s.RescaleIntercept))
        return np.stack(image, axis=-1)


class ImagePT(ImageBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_physical_values(slices):
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


class Mask(VolumeBase):
    def __init__(self, *args,reference_image=None, list_labels=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.list_labels = list_labels
        self.contours = None
        self.reference_image = reference_image
        self.masks = list()

    def read_structure(self):
        if len(self.dicom_paths) != 1:
            raise Exception('RTSTRUCT has more than one file')
        structure = pdcm.read_file(self.dicom_paths[0])
        for roi_seq in structure.ROIContourSequence:
            contours = []
            contour = {}
            for label in self.list_labels:
                if roi_seq.ROIName.startswith(label):
                    contour['color'] = roi_seq.ROIDisplayColor
                    contour['number'] = roi_seq.ReferencedROINumber
                    contour['name'] = roi_seq.ROIName
                    assert contour['number'] == roi_seq.ROINumber
                    contour['contours'] = [
                        s.ContourData for s in roi_seq.ContourSequence]
                    contours.append(contour)

        self.contours = contours

    def compute_mask(self):
        z = self.reference_image.slices_z_position
        pos_r = self.reference_image.image_pos_patient[1]
        spacing_r = self.reference_image.pixel_spacing[1]
        pos_c = self.reference_image.image_pos_patient[0]
        spacing_c = self.reference_image.pixel_spacing[0]
        shape = self.reference_image.GetSize()

        for i, con in enumerate(self.contours):
            label = np.zeros(shape, dtype=bool)
            for current in con['contours']:
                nodes = np.array(current).reshape((-1, 3))
                assert np.amax(np.abs(np.diff(nodes[:, 2]))) == 0
                z_index = z.index(nodes[0, 2])
                r = (nodes[:, 1] - pos_r) / spacing_r
                c = (nodes[:, 0] - pos_c) / spacing_c
                rr, cc = polygon(r, c)
                if len(rr) > 0 and len(cc) > 0:
                    if np.max(rr) > 512 or np.max(cc) > 512:
                        raise Exception("The RTSTRUCT file is compromised")

                label[rr, cc, z_index] = True
                name = con['name']

            self.masks.append({'mask':label, 'label_name': name})

    def read(self):
        self.read_structure()
        self.compute_mask()

    def write(self):
        pass
