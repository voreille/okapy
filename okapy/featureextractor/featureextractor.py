import six
from abc import ABC, abstractmethod
from collections import OrderedDict
import warnings
import yaml

import numpy as np
import SimpleITK as sitk
from radiomics.featureextractor import RadiomicsFeatureExtractor


class OkapyExtractors():
    def __init__(self, params_path):
        if type(params_path) == dict:
            params = params_path
        else:
            with open(params_path, 'r') as f:
                params = yaml.safe_load(f)
        self.feature_extractors = dict()
        self.default_extractor = FeatureExtractor()
        for modality, sub_dict in params.items():
            for extractor_type, list_params in sub_dict:
                self.feature_extractors[modality].extend([
                    create_extractor(modality, extractor_type, params=p)
                    for p in list_params
                ])

    def __call__(self, image, mask, modality=None):
        results = OrderedDict()
        for extractor in self.feature_extractors.get(modality,
                                                     [self.default_extractor]):
            results.update(extractor(image, mask))


def create_extractor(modality, extractor_type="pyradiomics", params=None):
    if extractor_type == "pyradiomics":
        if modality == 'PT':
            return FeatureExtractorPT(params)
        else:
            return FeatureExtractorPyradiomics(params)
    else:
        raise ValueError(f"The type {extractor_type} is not recongised")


def to_np(image):
    return np.transpose(sitk.GetArrayFromImage(image), (2, 1, 0))


def ellipsoid_window(radius):
    neighborhood = np.zeros((
        2 * radius[0] + 1,
        2 * radius[1] + 1,
        2 * radius[2] + 1,
    ))
    x = np.arange(-radius[0], radius[0] + 1)
    y = np.arange(-radius[1], radius[1] + 1)
    z = np.arange(-radius[2], radius[2] + 1)
    x, y, z = np.meshgrid(x, y, z, indexing='ij')
    neighborhood[x**2 / radius[0]**2 + y**2 / radius[1]**2 +
                 z**2 / radius[2]**2 <= 1] = 1
    return neighborhood


def check_image(image):
    if type(image) == sitk.Image:
        return image
    else:
        return sitk.ReadImage(image)


class FeatureExtractor(ABC):
    @abstractmethod
    def __init__(self, params=None):
        pass

    @abstractmethod
    def __call__(self, image, mask):
        pass


class FeatureExtractorPyradiomics(FeatureExtractor):
    def __init__(self, params=None):
        self.params = params
        self.radiomics_extractor = RadiomicsFeatureExtractor(params)

    def __call__(self, image_path, mask_path, **kwargs):
        image = check_image(image_path)
        mask = check_image(mask_path)
        results = self.radiomics_extractor(image, mask, **kwargs)
        return results


class FeatureExtractorPT(FeatureExtractor):
    def __init__(self, params=None):
        super().__init__(params)

    @staticmethod
    def translate_radiomics_output(results):
        results_copy = results.copy()
        for key, item in six.iteritems(results_copy):
            if key.startswith('original_firstorder'):
                new_key = key.replace('original_firstorder', 'SUV')
                results[new_key] = results.pop(key)
        return results

    @staticmethod
    def pet_features(image, mask, threshold=0.4, relative=True):

        np_image = to_np(image)
        np_mask = to_np(mask)
        spacing = np.array(image.GetSpacing())
        positions = np.where(np_mask != 0)
        if threshold != 0:
            if relative:
                t = threshold * np.max(np_image[positions])
                string_output = f"_T_{threshold*100}rel"
            else:
                t = threshold
                string_output = f"_T_{threshold}abs"
        else:
            t = 0
            string_output = ""

        mtv = np.sum(
            np_image[positions] > t) * spacing[0] * spacing[1] * spacing[2]
        positions = np.where((np_mask != 0) & (np_image > t))
        # compute SUVpeak
        if len(positions[0]) != 0:
            ind_max = np.argmax(np_image[positions])
            pos_max = np.array([
                positions[0][ind_max], positions[1][ind_max],
                positions[2][ind_max]
            ])
            # Sphere of 12 mm
            radius = np.round(6 / spacing).astype(int)
            max_neighborhood = np_image[pos_max[0] - radius[0]:pos_max[0] +
                                        radius[0] + 1, pos_max[1] -
                                        radius[1]:pos_max[1] + radius[1] + 1,
                                        pos_max[2] - radius[2]:pos_max[2] +
                                        radius[2] + 1, ]
            # it's an ellipsoid since isotropy is not assumed
            spherical_mask = ellipsoid_window(radius)
            try:
                suv_peak = np.mean(max_neighborhood[spherical_mask != 0])
            except IndexError:
                suv_peak = np.nan
                warnings.warn(
                    "The SUVpeak cannot be computed since the mask"
                    "is too close of the border, add more padding if you want"
                    "this feature.")
            tlg = mtv * np.mean(np_image[positions])
        else:
            suv_peak = np.nan
            tlg = np.nan
        return OrderedDict({
            "MTV" + string_output: mtv,
            "TLG" + string_output: tlg,
            "SUVpeak" + string_output: suv_peak,
        })

    def __call__(self, image_path, mask_path, **kwargs):

        if type(image_path) != sitk.SimpleITK.Image:
            image = sitk.ReadImage(str(image_path))
        else:
            image = image_path

        if type(mask_path) != sitk.SimpleITK.Image:
            mask = sitk.ReadImage(str(mask_path))
        else:
            mask = mask_path

        results = FeatureExtractorPT.translate_radiomics_output(
            self.radiomics_extractor.execute(image, mask, **kwargs))

        for threshold, relative in zip([0, 0.3, 0.4, 0.42, 1, 2.5],
                                       [True] * 4 + [False] * 3):
            results.update(
                FeatureExtractorPT.pet_features(image,
                                                mask,
                                                threshold=threshold,
                                                relative=relative))
        return results
