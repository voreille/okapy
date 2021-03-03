import six

import numpy as np
import SimpleITK as sitk
from radiomics.featureextractor import RadiomicsFeatureExtractor


def create_extractor(modality, params):
    if modality == 'PT':
        return OkapyFeatureExtractorPT(params)
    else:
        return RadiomicsFeatureExtractor(params)


def to_np(image):
    return np.transpose(sitk.GetArrayFromImage(image), (2, 1, 0))


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


class OkapyFeatureExtractorPT():
    def __init__(self, params):
        self.params = params
        self.radiomics_extractor = RadiomicsFeatureExtractor(params)

    @staticmethod
    def translate_radiomics_output(results):
        results_copy = results.copy()
        for key, item in six.iteritems(results_copy):
            if key.startswith('original_firstorder'):
                new_key = key.replace('original_firstorder', 'SUV')
                results[new_key] = results.pop(key)
        return results

    def execute(self,
                image_path,
                mask_path,
                label=None,
                label_channel=None,
                voxelBased=False):
        image = sitk.ReadImage(image_path)
        mask = sitk.ReadImage(mask_path)
        results = OkapyFeatureExtractorPT.translate_radiomics_output(
            self.radiomics_extractor.execute(image,
                                             mask,
                                             label=label,
                                             label_channel=label_channel,
                                             voxelBased=voxelBased
                                             )
        )

        return results
