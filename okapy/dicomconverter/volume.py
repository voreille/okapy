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
    def get_diagonal_reference_frame(
            pixel_spacing=(1, 1, 1),
            origin=(0, 0, 0),
            shape=None,
    ):
        matrix = np.eye(4)
        matrix[:3, :3] = matrix[:3, :3] * pixel_spacing
        matrix[:3, 3] = origin
        return ReferenceFrame(coordinate_matrix=matrix, shape=shape)

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
        return ReferenceFrame(coordinate_matrix=new_coordinate_matrix,
                              shape=output_shape)

    def get_matching_grid_bb(self, bb):
        or_vx = self.mm_to_vx(bb[:3])
        origin = np.minimum(self.vx_to_mm(np.ceil(or_vx)),
                            self.vx_to_mm(np.floor(or_vx)))

        end_vx = self.mm_to_vx(bb[3:])
        end = np.maximum(self.vx_to_mm(np.ceil(end_vx)),
                         self.vx_to_mm(np.floor(end_vx)))
        return np.concatenate([origin, end], axis=0)


class Volume():
    def __init__(self,
                 np_image=None,
                 reference_frame=None,
                 modality=None,
                 dicom_header=None):
        self.modality = modality
        self.np_image = np_image
        self.reference_frame = reference_frame
        self.dicom_header = dicom_header
        self.series_datetime = dicom_header.series_datetime

    def __getattr__(self, name):
        return getattr(self.dicom_header, name)

    def astype(self, dtype):
        self.np_image = self.np_image.astype(dtype)
        return self

    def zeros_like(self):
        return Volume(
            np_image=np.zeros_like(self.np_image),
            modality=self.modality,
            dicom_header=self.dicom_header,
            reference_frame=copy(self.reference_frame),
        )

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

    def contains_bb(self, bb):
        volume_bb = self.reference_frame.bb
        return (np.all(volume_bb[:3] <= bb[:3])
                and np.all(volume_bb[3:] >= bb[3:]))


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
    def __init__(self,
                 resampling_spacing=(1, 1, 1),
                 order=3,
                 mode="mirror",
                 cval=0):
        self.resampling_spacing = np.array(resampling_spacing)
        self.order = order
        self.mode = mode
        self.cval = cval

    def process(self, volume, bounding_box):
        bounding_box = volume.reference_frame.get_matching_grid_bb(
            bounding_box)
        original_spacing = volume.reference_frame.voxel_spacing
        resampling_spacing = (original_spacing *
                              (self.resampling_spacing <= 0) +
                              self.resampling_spacing *
                              (self.resampling_spacing > 0))

        bounding_box = np.array(bounding_box)
        output_shape = np.ceil((bounding_box[3:] - bounding_box[:3]) /
                               resampling_spacing).astype(int)
        new_reference_frame = ReferenceFrame.get_diagonal_reference_frame(
            pixel_spacing=resampling_spacing,
            origin=bounding_box[:3],
            shape=output_shape,
        )
        matrix = np.dot(volume.reference_frame.inv_coordinate_matrix,
                        new_reference_frame.coordinate_matrix)

        np_image = ndimage.affine_transform(
            volume.np_image,
            matrix[:3, :3],
            offset=matrix[:3, 3],
            mode=self.mode,
            order=self.order,
            cval=self.cval,
            output_shape=new_reference_frame.shape)

        v = volume.zeros_like()
        v.np_image = np_image
        v.reference_frame = new_reference_frame
        return v


class MaskResampler(BasicResampler):
    def __init__(self, *args, threshold=0.5, mode="constant", **kwargs):
        super().__init__(*args, mode=mode, **kwargs)
        self.t = threshold

    def threshold(self, volume):
        volume.np_image[volume.np_image < self.t] = 0
        volume.np_image[volume.np_image >= self.t] = 1
        return volume

    def process(self, volume, *args, **kwargs):
        return self.threshold(super().process(volume, *args, **kwargs))
