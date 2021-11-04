from copy import copy

from okapy.exceptions import MissingSegmentationException
import yaml
import numpy as np
from scipy import ndimage

from okapy.dicomconverter.volume import ReferenceFrame


class VolumeProcessorStack():
    def __init__(self, stacks=None):
        self.stacks = stacks

    @staticmethod
    def from_params(params_path, mask_resampler=None):
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
                stacks[key].append(
                    VolumeProcessor.get(name=name)(
                        mask_resampler=mask_resampler, **p))

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
        array = volume.array
        mean = np.mean(array[array > self.threshold])
        std = np.std(array[array > self.threshold])
        array = (array - mean) / std
        volume.array = array
        return volume


class MaskedStandardizer(VolumeProcessor, name="masked_standardizer"):
    def __init__(self, *args, mask_label="", mask_resampler=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask_label = mask_label
        if mask_resampler is None:
            raise TypeError("mask_resamper cannot be None")
        self.mask_resampler = mask_resampler

    def _get_mask_array(self, mask_files, reference_frame=None):
        mask = None
        for f in mask_files:
            if self.mask_label in f.labels:
                mask = f.get_volume(self.mask_label)
                break
        if mask is None:
            raise MissingSegmentationException(
                f"The label was {self.mask_labe} was not found")
        return self.mask_resampler(
            mask, new_reference_frame=reference_frame).array != 0

    def process(self, volume, mask_files=None, **kwargs):
        array = volume.array
        mask_array = self._get_mask_array(
            mask_files, reference_frame=volume.reference_frame)
        mean = np.mean(array[mask_array])
        std = np.std(array[mask_array])
        array = (array - mean) / std
        volume.array = array
        return volume


class BSplineResampler(VolumeProcessor, name="bspline_resampler"):
    def __init__(self,
                 resampling_spacing=None,
                 order=3,
                 mode="mirror",
                 cval=0,
                 diagonalize_reference_frame=False,
                 **kwargs):
        self.resampling_spacing = np.array(
            resampling_spacing) if resampling_spacing is not None else None
        self.order = order
        self.mode = mode
        self.cval = cval
        self.diagonalize_reference_frame = diagonalize_reference_frame

    def process(self, volume, new_reference_frame=None, **kwargs):
        new_reference_frame = copy(new_reference_frame)
        if self.resampling_spacing is not None:
            new_resampling_spacing = np.where(
                self.resampling_spacing > 0, self.resampling_spacing,
                volume.reference_frame.voxel_spacing)
            new_reference_frame.voxel_spacing = new_resampling_spacing
        matrix = np.dot(volume.reference_frame.inv_coordinate_matrix,
                        new_reference_frame.coordinate_matrix)

        array = ndimage.affine_transform(
            volume.array,
            matrix[:3, :3],
            offset=matrix[:3, 3],
            mode=self.mode,
            order=self.order,
            cval=self.cval,
            output_shape=new_reference_frame.shape)

        v = volume.zeros_like()
        v.array = array
        v.reference_frame = new_reference_frame
        return v


class BinaryBSplineResampler(BSplineResampler,
                             name="binary_bspline_resampler"):
    def __init__(self, *args, threshold=0.5, mode="constant", **kwargs):
        super().__init__(*args, mode=mode, **kwargs)
        self.t = threshold

    def threshold(self, volume):
        volume.array[volume.array < self.t] = 0
        volume.array[volume.array >= self.t] = 1
        return volume

    def process(self, volume, *args, **kwargs):
        return self.threshold(super().process(volume, *args, **kwargs))
