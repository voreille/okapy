import yaml
import numpy as np
from scipy import ndimage

from okapy.dicomconverter.volume import ReferenceFrame


class VolumeProcessorStack():
    def __init__(self, stacks=None):
        self.stacks = stacks

    @staticmethod
    def from_params(params_path):
        if type(params_path) == dict:
            params = params_path
        else:
            with open(params_path, 'r') as f:
                params = yaml.safe_load(f)

        stacks = {}
        for key, params_stack in params.items():
            stacks[key] = []
            if params_stack is None:
                params_stack = {}
            for name, p in params_stack.items():
                stacks[key].append(VolumeProcessor.get(name=name)(**p))

        return VolumeProcessorStack(stacks=stacks)

    def __call__(self, volume, **kwargs):
        for processor in self.stacks.get("common", []):
            volume = processor(volume, **kwargs)
        for processor in self.stacks.get(volume.modality,
                                         self.stacks["default"]):
            volume = processor(volume, **kwargs)
        return volume


class VolumeProcessor():
    _registry = {}  # class var that store the different daughter

    def __init_subclass__(cls, name, **kwargs):
        cls.name = name
        VolumeProcessor._registry[name] = cls
        super().__init_subclass__(**kwargs)

    @classmethod
    def get(cls, name: str):
        try:
            return VolumeProcessor._registry[name]
        except KeyError:
            raise ValueError(f"The VolumeProcessor {name} is not defined.")

    def __init__(self, *args, **kwargs):
        super().__init__()

    def process(self, volume, *args, **kwargs):
        raise NotImplementedError('abstract class')

    def __call__(self, volume, *args, **kwargs):
        return self.process(volume, *args, **kwargs)


class IdentityProcessor(VolumeProcessor, name="identity_processor"):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def process(self, volume, *args, **kwargs):
        return volume


class Standardizer(VolumeProcessor, name="standardizer"):
    def __init__(self, *args, threshold=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.threshold = threshold

    def process(self, volume, **kwargs):
        array = volume.np_image
        mean = np.mean(array[array > self.threshold])
        std = np.std(array[array > self.threshold])
        array = (array - mean) / std
        volume.np_image = array
        return volume


class BSplineResampler(VolumeProcessor, name="bspline_resampler"):
    def __init__(self,
                 resampling_spacing=(1, 1, 1),
                 order=3,
                 mode="mirror",
                 cval=0):
        self.resampling_spacing = resampling_spacing
        self.order = order
        self.mode = mode
        self.cval = cval

    def process(self, volume, bounding_box=None, **kwargs):
        bounding_box = np.array(bounding_box)
        output_shape = np.ceil((bounding_box[3:] - bounding_box[:3]) /
                               self.resampling_spacing).astype(int)
        new_reference_frame = ReferenceFrame.get_diagonal_reference_frame(
            pixel_spacing=self.resampling_spacing,
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


class BinaryBSplineResampler(BSplineResampler,
                             name="binary_bspline_resampler"):
    def __init__(self, *args, threshold=0.5, mode="constant", **kwargs):
        super().__init__(*args, mode=mode, **kwargs)
        self.t = threshold

    def threshold(self, volume):
        volume.np_image[volume.np_image < self.t] = 0
        volume.np_image[volume.np_image >= self.t] = 1
        return volume

    def process(self, volume, *args, **kwargs):
        return self.threshold(super().process(volume, *args, **kwargs))
