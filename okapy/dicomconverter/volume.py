'''
TODO: Is SliceLocation always a thing, if yes, use it instead of
      ImagePositionPatient[2]
TODO: Resampling with nearest neighbor for VolumeMask
TODO: Change how pixelspacing and origin fields are initialized in RtstructFile
TODO: Direction cosine in sitk
TODO: Change the name of the class for instance DicomBase instead of DicomFileBase
TODO: Add check of the bounding_box, if it goes beyond the image domain
'''

from copy import copy

import numpy as np
from scipy import ndimage
import SimpleITK as sitk


class ReferenceFrame():
    def __init__(self,
                 origin=None,
                 origin_last_slice=None,
                 orientation=None,
                 pixel_spacing=None,
                 coordinate_matrix=None,
                 shape=None):
        super().__init__()
        if coordinate_matrix is not None:
            self.origin = np.dot(coordinate_matrix, [0, 0, 0, 1])[:3]
            self.origin_last_slice = np.dot(coordinate_matrix,
                                            [0, 0, shape[2] - 1, 1])[:3]
            self.pixel_spacing = (np.dot(coordinate_matrix, [1, 1, 1, 1]) -
                                  np.dot(coordinate_matrix, [0, 0, 0, 1]))[:2]
            self.orientation = (coordinate_matrix[:3, :2] /
                                self.pixel_spacing).T.flatten()
            self.shape = np.array(shape)
        else:
            self.origin = origin
            self.origin_last_slice = origin_last_slice
            self.orientation = orientation
            self.pixel_spacing = pixel_spacing
            self.shape = np.array(shape)

    @staticmethod
    def compute_coordinate_matrix(origin=None,
                                  origin_last_slice=None,
                                  pixel_spacing=None,
                                  orientation=None,
                                  shape=None):
        return np.array(
            [[
                orientation[0] * pixel_spacing[0],
                orientation[3] * pixel_spacing[1],
                (origin_last_slice[0] - origin[0]) / (shape[2] - 1), origin[0]
            ],
             [
                 orientation[1] * pixel_spacing[0],
                 orientation[4] * pixel_spacing[1],
                 (origin_last_slice[1] - origin[1]) / (shape[2] - 1), origin[1]
             ],
             [
                 orientation[2] * pixel_spacing[0],
                 orientation[5] * pixel_spacing[1],
                 (origin_last_slice[2] - origin[2]) / (shape[2] - 1), origin[2]
             ], [0, 0, 0, 1]])

    @property
    def coordinate_matrix(self):
        return ReferenceFrame.compute_coordinate_matrix(
            origin=self.origin,
            origin_last_slice=self.origin_last_slice,
            pixel_spacing=self.pixel_spacing,
            orientation=self.orientation,
            shape=self.shape)

    @property
    def voxel_spacing(self):
        delta_r = np.sqrt(
            np.sum(((np.dot(self.coordinate_matrix, [1, 0, 0, 1]) -
                     np.dot(self.coordinate_matrix, [0, 0, 0, 1]))[:3])**2))
        delta_c = np.sqrt(
            np.sum(((np.dot(self.coordinate_matrix, [0, 1, 0, 1]) -
                     np.dot(self.coordinate_matrix, [0, 0, 0, 1]))[:3])**2))
        delta_s = np.sqrt(
            np.sum(((np.dot(self.coordinate_matrix, [0, 0, 1, 1]) -
                     np.dot(self.coordinate_matrix, [0, 0, 0, 1]))[:3])**2))
        return np.array([delta_r, delta_c, delta_s])

    @property
    def inv_coordinate_matrix(self):
        return np.linalg.inv(self.coordinate_matrix)

    @property
    def bounding_box(self):
        return np.array(
            [*self.vx_to_mm([0, 0, 0]), *self.vx_to_mm(self.shape - 1)])

    def bounding_box_intersection(self, bb):
        bb_vx_1 = self.mm_to_vx(bb[:3])
        bb_vx_2 = self.mm_to_vx(bb[3:])
        bb_vx = np.zeros((6, ))
        bb_vx[0:3] = np.maximum([0, 0, 0], bb_vx_1)
        bb_vx[3:] = np.minimum(
            [self.shape[0] - 1, self.shape[1] - 1, self.shape[2] - 1], bb_vx_2)
        return np.array([*self.vx_to_mm(bb_vx[:3]), *self.vx_to_mm(bb_vx[3:])])

    def direction_vector(self, vx_vector):
        v = self.vx_to_mm(vx_vector) - self.vx_to_mm([0, 0, 0])
        return v / np.linalg.norm(v)

    @property
    def direction_vector_dict(self):
        return {
            'row': self.direction_vector([1, 0, 0]),
            'col': self.direction_vector([0, 1, 0]),
            'slice': self.direction_vector([0, 0, 1]),
        }

    def positions(self, key='slice'):
        return np.array([
            np.dot(self.vx_to_mm([0, 0, k]), self.direction_vector_dict[key])
            for k in range(self.shape[2])
        ])

    def get_bb_vx(self, bb):
        bb_vx = np.zeros((6, ), dtype=int)
        bb_vx[:3] = np.round(
            np.dot(self.inv_coordinate_matrix,
                   list(bb[:3]) + [1])[:3]).astype(int)
        bb_vx[3:] = np.round(
            np.dot(self.inv_coordinate_matrix,
                   list(bb[3:]) + [1])[:3]).astype(int)
        return bb_vx

    def vx_to_mm(self, a):
        return np.dot(self.coordinate_matrix, list(a) + [1])[:3]

    def mm_to_vx(self, a):
        return np.dot(self.inv_coordinate_matrix, list(a) + [1])[:3]

    def get_new_reference_frame(self, bb, new_voxel_spacing):
        # You need to add one since the last pixel of the bb is in the domain
        if bb is None:
            bb = self.bounding_box
        output_shape = np.ceil(
            (bb[3:] - bb[:3]) / new_voxel_spacing).astype(int)
        new_coordinate_matrix = np.zeros((4, 4))
        new_coordinate_matrix[:3, :3] = (self.coordinate_matrix[:3, :3] /
                                         self.voxel_spacing *
                                         new_voxel_spacing)
        new_coordinate_matrix[:3, 3] = bb[:3]
        new_coordinate_matrix[3, 3] = 1
        return ReferenceFrame(coordinate_matrix=new_coordinate_matrix,
                              shape=output_shape)


class Volume():
    def __init__(self, np_image=None, reference_frame=None, dicom_header=None):
        self.np_image = np_image
        self.reference_frame = reference_frame
        self.dicom_header = dicom_header

    def __getattr__(self, name):
        return getattr(self.dicom_header, name)

    def astype(self, dtype):
        self.np_image = self.np_image.astype(dtype)
        return self

    def zeros_like(self):
        return Volume(
            np_image=np.zeros_like(self.np_image),
            dicom_header=self.dicom_header,
            reference_frame=copy(self.reference_frame),
        )

    def _check_resampling_spacing(self, resampling_vx_spacing):
        if len(resampling_vx_spacing) == 3:
            cond = True
            for i in range(3):
                cond *= (self.reference_frame.voxel_spacing[i] ==
                         resampling_vx_spacing[i])
        else:
            cond = False

        return cond

    def _get_resampled_np(self, resampling_vx_spacing, bounding_box, order=3):
        new_reference_frame = self.reference_frame.get_new_reference_frame(
            bounding_box, resampling_vx_spacing)
        matrix = np.dot(self.reference_frame.inv_coordinate_matrix,
                        new_reference_frame.coordinate_matrix)

        np_image = ndimage.affine_transform(
            self.np_image,
            matrix[:3, :3],
            offset=matrix[:3, 3],
            mode='mirror',
            order=order,
            output_shape=new_reference_frame.shape)

        return np_image, new_reference_frame

    def resample(self, resampling_vx_spacing, bounding_box, order=3):
        """
        Resample the 3D volume to the resampling_vx_spacing according to
        the bounding_boc in cm (x1, y1, z1, x2, y2, z2)
        """
        if not self._check_resampling_spacing(resampling_vx_spacing):

            np_image, reference_frame = self._get_resampled_np(
                resampling_vx_spacing, bounding_box, order)

            self.np_image = np_image
            self.reference_frame = reference_frame

    @property
    def sitk_image(self):
        trans = (2, 1, 0)
        sitk_image = sitk.GetImageFromArray(np.transpose(self.np_image, trans))
        sitk_image.SetSpacing(self.reference_frame.voxel_spacing)
        sitk_image.SetOrigin(self.reference_frame.origin)
        sitk_image.SetDirection(
            self.reference_frame.coordinate_matrix[:3, :3].flatten() /
            np.tile(self.reference_frame.voxel_spacing, 3))
        return sitk_image


class VolumeMask(Volume):
    def __init__(self,
                 *args,
                 label=None,
                 reference_dicom_header=None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.np_image[self.np_image != 0] = 1
        self.reference_dicom_header = reference_dicom_header
        self.label = label

    def zeros_like(self):
        return VolumeMask(np_image=np.zeros_like(self.np_image),
                          reference_frame=copy(self.reference_frame),
                          reference_dicom_header=self.reference_dicom_header,
                          dicom_header=self.dicom_header,
                          label=self.label)

    def __getattr__(self, name):
        if "reference_" in name:
            if self.reference_dicom_header:
                return getattr(self.reference_dicom_header,
                               name.replace("reference_", ""))
            else:
                return None
        else:
            return getattr(self.dicom_header, name)

    @property
    def bb_vx(self):
        indices = np.where(self.np_image != 0)
        return np.array([
            np.min(indices[0]),
            np.min(indices[1]),
            np.min(indices[2]),
            np.max(indices[0]),
            np.max(indices[1]),
            np.max(indices[2])
        ])

    @property
    def bb(self):
        indices = np.where(self.np_image != 0)
        return np.array([
            *self.reference_frame.vx_to_mm(
                [np.min(indices[0]),
                 np.min(indices[1]),
                 np.min(indices[2])]),
            *self.reference_frame.vx_to_mm(
                [np.max(indices[0]),
                 np.max(indices[1]),
                 np.max(indices[2])]),
        ])

    def padded_bb(self, padding):
        bb = self.bb
        bb[:3] = bb[:3] - padding
        bb[3:] = bb[3:] + padding
        return bb

    def bb_union(self, bb, padding=0):
        bb_vx_1 = self.reference_frame.mm_to_vx(bb[:3])
        bb_vx_2 = self.reference_frame.mm_to_vx(bb[3:])
        bb_vx = self.bb_vx
        bb_vx[0:3] = np.minimum(bb_vx[0:3], bb_vx_1)
        bb_vx[3:] = np.maximum(bb_vx[3:], bb_vx_2)
        return np.array([
            *(self.reference_frame.vx_to_mm(bb_vx[:3]) - padding),
            *(self.reference_frame.vx_to_mm(bb_vx[3:]) + padding)
        ])

    def resample(self, resampling_vx_spacing, bounding_box, order=1):
        super().resample(resampling_vx_spacing=resampling_vx_spacing,
                         bounding_box=bounding_box,
                         order=order)
        self.np_image[self.np_image > 0.5] = 1
        self.np_image[self.np_image < 0.5] = 0

    def get_resampled_volume(self,
                             resampling_vx_spacing,
                             bounding_box,
                             order=1):
        if not self._check_resampling_spacing(resampling_vx_spacing):

            np_image, reference_frame = self._get_resampled_np(
                resampling_vx_spacing, bounding_box, order=order)
            np_image[np_image > 0.5] = 1
            np_image[np_image < 0.5] = 0

            return VolumeMask(
                np_image=np_image,
                reference_frame=reference_frame,
                reference_dicom_header=self.reference_dicom_header,
                label=self.label,
                dicom_header=self.dicom_header,
            )

        else:
            return self


class VolumeProcessor():
    def __init__(self, *args, **kwargs):
        super().__init__()

    def process(self, volume, *args, **kwargs):
        raise NotImplementedError('abstract class')

    def __call__(self, volume, *args, **kwargs):
        return self.process(volume, *args, **kwargs)


class IdentityProcessor(VolumeProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def process(self, volume, *args, **kwargs):
        return volume


class MRStandardizer(VolumeProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def process(self, volume, *args, **kwargs):
        return super().process(volume, *args, **kwargs)


class BasicResampler(VolumeProcessor):
    def __init__(self, resampling_spacing=(1, 1, 1), order=3):
        self.resampling_spacing = resampling_spacing
        self.order = order

    def process(self, volume, bounding_box):
        new_reference_frame = volume.reference_frame.get_new_reference_frame(
            bounding_box, self.resampling_spacing)
        matrix = np.dot(volume.reference_frame.inv_coordinate_matrix,
                        new_reference_frame.coordinate_matrix)

        np_image = ndimage.affine_transform(
            volume.np_image,
            matrix[:3, :3],
            offset=matrix[:3, 3],
            mode='mirror',
            order=self.order,
            output_shape=new_reference_frame.shape)

        v = volume.zeros_like()
        v.np_image = np_image
        v.reference_frame = new_reference_frame
        return v


class MaskResampler(BasicResampler):
    def __init__(self, *args, threshold=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.t = threshold

    def threshold(self, volume):
        volume.np_image[volume.np_image < self.t] = 0
        volume.np_image[volume.np_image >= self.t] = 1
        return volume

    def process(self, volume, *args, **kwargs):
        return self.threshold(super().process(volume, *args, **kwargs))
