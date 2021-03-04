import six
from collections import OrderedDict

import numpy as np
import SimpleITK as sitk
from radiomics.featureextractor import RadiomicsFeatureExtractor


def create_extractor(modality, params):
    if modality == 'PT':
        return FeatureExtractorPT(params)
    else:
        return RadiomicsFeatureExtractor(params)


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


def compute_mtv(image, mask, threshold=0.4, relative=True):
    """Return the Metabolic Tumor Volume (MTV) which is computed by
    resegmenting within the mask the tumor base on a relative or
    absolute threshold.

    Args:
        image (SimpleITK):
        mask ([type]): [description]
        threshold (float, optional): [description]. Defaults to 0.4.
        relative (bool, optional): [description]. Defaults to True.
    """
    np_image = to_np(image)
    np_mask = to_np(mask)
    positions = np.where(np_mask != 0)
    if relative:
        t = threshold * np.max(np_image[positions])
    new_mask = np_image


class FeatureExtractor():
    def __init__(self, params):
        self.params = params
        self.radiomics_extractor = RadiomicsFeatureExtractor(params)

    def execute(self, image_path, mask_path, **kwargs):
        image = sitk.ReadImage(image_path)
        mask = sitk.ReadImage(mask_path)
        results = self.radiomics_extractor(image, mask)
        return results


class FeatureExtractorPT(FeatureExtractor):
    def __init__(self, params):
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
            suv_peak = np.mean(max_neighborhood[spherical_mask != 0])
            tlg = mtv * np.mean(np_image[positions])
        else:
            suv_peak = np.nan
            tlg = np.nan
        return OrderedDict({
            "MTV" + string_output:
            mtv,
            "TLG" + string_output:
            tlg,
            "SUVpeak" + string_output:
            suv_peak,
        })

    def execute(self,
                image_path,
                mask_path,
                label=None,
                label_channel=None,
                voxelBased=False):

        if type(image_path) != sitk.SimpleITK.Image:
            image = sitk.ReadImage(str(image_path))
        else:
            image = image_path

        if type(mask_path) != sitk.SimpleITK.Image:
            mask = sitk.ReadImage(str(mask_path))
        else:
            mask = mask_path

        results = FeatureExtractorPT.translate_radiomics_output(
            self.radiomics_extractor.execute(image,
                                             mask,
                                             label=label,
                                             label_channel=label_channel,
                                             voxelBased=voxelBased))

        for threshold, relative in zip([0, 0.3, 0.4, 0.42, 1, 2.5],
                                       [True] * 4 + [False] * 3):
            results.update(
                FeatureExtractorPT.pet_features(image,
                                                mask,
                                                threshold=threshold,
                                                relative=relative))
        return results
