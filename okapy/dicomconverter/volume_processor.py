import numpy as np
from scipy import ndimage


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
