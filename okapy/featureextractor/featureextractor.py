import six

import SimpleITK as sitk
from radiomics.featureextractor import RadiomicsFeatureExtractor


def create_extractor(modality, params):
    if modality == 'PT':
        return OkapyFeatureExtractorPT(params)
    else:
        return RadiomicsFeatureExtractor(params)


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

    def execute(self, image_path, mask_path):
        sitk_image = sitk.ReadImage(image_path)
        sitk_mask = sitk.ReadImage(mask_path)
        results = OkapyFeatureExtractorPT.translate_radiomics_ouput(
            self.radiomics_extractor(sitk_image, sitk_mask))
        return results
