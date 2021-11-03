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
from itertools import product
from os import stat

import numpy as np
from scipy import ndimage
import SimpleITK as sitk


def correct_bb(bb):
    bb_corrected = np.zeros((6, ))
    bb_corrected[:3] = np.minimum(bb[:3], bb[3:])
    bb_corrected[3:] = np.maximum(bb[:3], bb[3:])
    return bb_corrected


def get_bb_diagonal_frame(mask, resampling_spacing=(1, 1, 1)):
    bb_vx = mask.bb_vx
    points = [
        np.array([pr, pc, ps]) for pr, pc, ps in product(bb_vx[:3], bb_vx[3:])
    ]
    projected_points = np.stack(
        [mask.reference_frame.vx_to_mm(p) for p in points], axis=-1)
    bb_diag = np.zeros((6, ))
    bb_diag[:3] = np.min(projected_points, axis=-1)
    bb_diag[3:] = np.max(projected_points, axis=-1)
    return bb_diag


# class FlexibleReferenceFrame():
#     def __init__(self,
#                  slices_origin=None,
#                  orientation=None,
#                  pixel_spacing=None,
#                  slice_shape=None):

#         self.e_r = np.array(orientation[:3])
#         self.e_c = np.array(orientation[3:])
#         self.e_s = np.cross(orientation[:3], orientation[3:])
#         self.slices_origin = np.stack(slices_origin, axis=-1)
#         self.d_r = pixel_spacing[0]
#         self.d_c = pixel_spacing[1]
#         self.slice_shape = slice_shape

#     def vx_to_mm(self, a):
# return (self.slices_origin[:,0] + self.e_r * self.d_r * a[0] + self.e_c * self.d_c * a[1])


class ReferenceFrame():
    def __init__(
        self,
        origin=None,
        origin_last_slice=None,
        orientation=None,
        pixel_spacing=None,
        shape=None,
    ):
        super().__init__()
        self.origin = origin
        self.origin_last_slice = origin_last_slice
        n = np.cross(orientation[:3], orientation[3:])
        self.orientation_matrix = np.stack(
            [orientation[:3], orientation[3:], n], axis=-1)
        self.pixel_spacing = pixel_spacing
        self.shape = np.array(shape)
        self.slice_spacing = np.dot(
            np.array(self.origin_last_slice) - np.array(self.origin),
            n) / (self.shape[2] - 1)

    @staticmethod
    def from_coordinate_matrix(matrix, shape=None):
        origin = np.dot(matrix, [0, 0, 0, 1])[:3]
        origin_last_slice = np.dot(matrix, [0, 0, shape[2] - 1, 1])[:3]
        pixel_spacing = ReferenceFrame.get_voxel_spacing(matrix)[:2]
        orientation = (matrix[:3, :2] / pixel_spacing).T.flatten()
        return ReferenceFrame(origin=origin,
                              origin_last_slice=origin_last_slice,
                              orientation=orientation,
                              pixel_spacing=pixel_spacing,
                              shape=shape)

    @staticmethod
    def get_diagonal_reference_frame(
            pixel_spacing=(1, 1, 1),
            origin=(0, 0, 0),
            shape=None,
    ):
        matrix = np.eye(4)
        matrix[:3, :3] = matrix[:3, :3] * pixel_spacing
        matrix[:3, 3] = origin
        return ReferenceFrame.from_coordinate_matrix(matrix, shape=shape)

    @staticmethod
    def get_voxel_spacing(coordinate_matrix):
        delta_r = np.sqrt(
            np.sum(((np.dot(coordinate_matrix, [1, 0, 0, 1]) -
                     np.dot(coordinate_matrix, [0, 0, 0, 1]))[:3])**2))
        delta_c = np.sqrt(
            np.sum(((np.dot(coordinate_matrix, [0, 1, 0, 1]) -
                     np.dot(coordinate_matrix, [0, 0, 0, 1]))[:3])**2))
        delta_s = np.sqrt(
            np.sum(((np.dot(coordinate_matrix, [0, 0, 1, 1]) -
                     np.dot(coordinate_matrix, [0, 0, 0, 1]))[:3])**2))
        return np.array([delta_r, delta_c, delta_s])

    @property
    def voxel_spacing(self):
        return np.array(
            [self.pixel_spacing[0], self.pixel_spacing[1], self.slice_spacing])

    @property
    def coordinate_matrix(self):
        matrix = np.zeros((4, 4))
        matrix[:3, :3] = self.orientation_matrix * self.voxel_spacing
        matrix[:3, 3] = self.origin
        matrix[3, 3] = 1
        return matrix

    @property
    def inv_coordinate_matrix(self):
        return np.linalg.inv(self.coordinate_matrix)

    @property
    def domain(self):
        bb = np.array(
            [*self.vx_to_mm([0, 0, 0]), *self.vx_to_mm(self.shape - 1)])
        return correct_bb(bb)

    @property
    def bb(self):
        points = [
            np.array([pr, pc, ps])
            for pr, pc, ps in product(*zip([0, 0, 0], self.shape - 1))
        ]
        projected_points = np.stack([self.vx_to_mm(p) for p in points],
                                    axis=-1)
        bb_diag = np.zeros((6, ))
        bb_diag[:3] = np.min(projected_points, axis=-1)
        bb_diag[3:] = np.max(projected_points, axis=-1)
        return bb_diag

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
            np.linalg.inv(self.coordinate_matrix[:3, :3]) @ (bb[3:] - bb[:3]) /
            new_voxel_spacing).astype(int)
        new_coordinate_matrix = np.zeros((4, 4))
        new_coordinate_matrix[:3, :3] = (self.coordinate_matrix[:3, :3] /
                                         self.voxel_spacing *
                                         new_voxel_spacing)
        new_coordinate_matrix[:3, 3] = bb[:3]
        new_coordinate_matrix[3, 3] = 1
        return ReferenceFrame.from_coordinate_matrix(new_coordinate_matrix,
                                                     shape=output_shape)

    def get_matching_grid_bb(self, bb):
        or_vx = self.mm_to_vx(bb[:3])
        origin = np.minimum(self.vx_to_mm(np.ceil(or_vx)),
                            self.vx_to_mm(np.floor(or_vx)))

        end_vx = self.mm_to_vx(bb[3:])
        end = np.maximum(self.vx_to_mm(np.ceil(end_vx)),
                         self.vx_to_mm(np.floor(end_vx)))
        return np.concatenate([origin, end], axis=0)


class VolumeBase():
    def __init__(self,
                 array=None,
                 reference_frame=None,
                 modality=None,
                 dicom_header=None):
        self.modality = modality
        self.array = array
        self.reference_frame = reference_frame
        self.dicom_header = dicom_header
        self.series_datetime = dicom_header.series_datetime

    def __getattr__(self, name):
        return getattr(self.dicom_header, name)

    def astype(self, dtype):
        self.array = self.array.astype(dtype)
        return self

    def zeros_like(self):
        raise NotImplementedError("This is an abstract class")

    @property
    def sitk_image(self):
        trans = (2, 1, 0)
        sitk_image = sitk.GetImageFromArray(np.transpose(self.array, trans))
        sitk_image.SetSpacing(self.reference_frame.voxel_spacing)
        sitk_image.SetOrigin(self.reference_frame.origin)
        sitk_image.SetDirection(
            self.reference_frame.coordinate_matrix[:3, :3].flatten() /
            np.tile(self.reference_frame.voxel_spacing, 3))
        return sitk_image

    def contains_bb(self, bb):
        volume_bb = self.reference_frame.bb
        return (np.all(volume_bb[:3] <= bb[:3])
                and np.all(volume_bb[3:] >= bb[3:]))


class Volume(VolumeBase):
    def __init__(self,
                 array=None,
                 reference_frame=None,
                 modality=None,
                 dicom_header=None):
        super().__init__(array=array,
                         reference_frame=reference_frame,
                         modality=modality,
                         dicom_header=dicom_header)

    def zeros_like(self):
        return Volume(
            array=np.zeros_like(self.array),
            modality=self.modality,
            dicom_header=self.dicom_header,
            reference_frame=copy(self.reference_frame),
        )


class BinaryVolume(VolumeBase):
    def __init__(self,
                 *args,
                 label=None,
                 reference_dicom_header=None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.array[self.array != 0] = 1
        self.reference_dicom_header = reference_dicom_header
        self.label = label

    def zeros_like(self):
        return BinaryVolume(array=np.zeros_like(self.array),
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
        indices = np.where(self.array != 0)
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
        bb_vx = self.bb_vx
        points = [
            np.array([pr, pc, ps])
            for pr, pc, ps in product(*zip(bb_vx[:3], bb_vx[3:]))
        ]
        projected_points = np.stack(
            [self.reference_frame.vx_to_mm(p) for p in points], axis=-1)
        bb_diag = np.zeros((6, ))
        bb_diag[:3] = np.min(projected_points, axis=-1)
        bb_diag[3:] = np.max(projected_points, axis=-1)
        return bb_diag
