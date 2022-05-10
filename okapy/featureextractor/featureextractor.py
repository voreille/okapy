from calendar import month_abbr
from re import M
import subprocess
import json
import six
from pathlib import Path
from abc import ABC, abstractmethod
from collections import OrderedDict
import warnings
import yaml
import logging

import numpy as np
import SimpleITK as sitk
import pandas as pd

from okapy.utils import make_temp_directory

log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)
logger = logging.getLogger(__name__)

try:
    from radiomics.featureextractor import RadiomicsFeatureExtractor
except ImportError:
    logger.warning("Pyradiomics is not installed")

try:
    from zrad.interface import compute_zrad_features
except ImportError:
    logger.warning("Zrad is not installed")


class OkapyExtractors():

    def __init__(self, params_path):
        if type(params_path) == dict:
            params = params_path
        else:
            with open(params_path, 'r') as f:
                params = yaml.safe_load(f)
        self.feature_extractors = dict()
        self.default_extractor = FeatureExtractorPyradiomics(name="DEFAULT")
        for modality, sub_dict in params.items():
            self.feature_extractors[modality] = list()
            for extractor_type, list_extractors in sub_dict.items():
                self.feature_extractors[modality].extend([
                    create_extractor(modality=modality,
                                     extractor_type=extractor_type,
                                     name=extractor_name,
                                     params=params)
                    for extractor_name, params in list_extractors.items()
                ])

    @staticmethod
    def _update_results(image, mask, extractor, results, modality=None):
        try:
            results.update(extractor(image, mask, modality=modality))
        except Exception as e:
            logger.error(f"Features were not extracted for "
                         f"extractor {extractor.name}, image {image} "
                         f"and mask {mask}. The error message is {e}")

    def __call__(self, image, mask, modality=None):
        results = OrderedDict()
        for extractor in self.feature_extractors.get(modality,
                                                     [self.default_extractor]):
            OkapyExtractors._update_results(
                image,
                mask,
                extractor,
                results,
                modality=modality,
            )
        for extractor in self.feature_extractors.get("common", []):
            OkapyExtractors._update_results(
                image,
                mask,
                extractor,
                results,
                modality=modality,
            )
        return results


def create_extractor(modality=None,
                     name="",
                     extractor_type="pyradiomics",
                     params=None):
    if extractor_type == "pyradiomics":
        if modality == 'PT':
            return FeatureExtractorPyradiomicsPT(name=name, params=params)
        else:
            return FeatureExtractorPyradiomics(name=name, params=params)
    elif extractor_type == "riesz":
        return RieszFeatureExtractor(name=name, params=params)
    elif extractor_type == "zrad":
        return ZradFeatureExtractor(name=name, params=params)
    else:
        raise ValueError(f"The type {extractor_type} is not recongised")


def check_image(image):
    if isinstance(image, sitk.Image):
        return image
    elif isinstance(image, Path):
        return sitk.ReadImage(str(image.resolve()))
    elif isinstance(image, str):
        return sitk.ReadImage(image)
    else:
        raise ValueError(f"{image} is not accepted for the feature extractor")


class FeatureExtractor(ABC):

    def __init__(self, name="", params=None):
        self.name = name
        self.params = params

    @abstractmethod
    def __call__(self, image, mask, modality=None):
        pass

    @staticmethod
    def to_np(image):
        return np.transpose(sitk.GetArrayFromImage(image), (2, 1, 0))


class FeatureExtractorPyradiomics(FeatureExtractor):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.radiomics_extractor = RadiomicsFeatureExtractor(
            kwargs.get("params"))

    def __call__(self, image_path, mask_path, **kwargs):
        kwargs = {k: i for k, i in kwargs.items() if k is not "modality"}
        image = check_image(image_path)
        mask = check_image(mask_path)
        results = self.radiomics_extractor.execute(image, mask, **kwargs)
        return results


class FeatureExtractorPyradiomicsPT(FeatureExtractorPyradiomics):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
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

    @staticmethod
    def translate_radiomics_output(results):
        results_copy = results.copy()
        for key, item in six.iteritems(results_copy):
            if key.startswith('original_firstorder'):
                new_key = key.replace('original_firstorder', 'PET_SUV')
                results[new_key] = results.pop(key)
        return results

    @staticmethod
    def pet_features(image, mask, threshold=0.4, relative=True):

        np_image = FeatureExtractor.to_np(image)
        np_mask = FeatureExtractor.to_np(mask)
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
            spherical_mask = FeatureExtractorPyradiomicsPT.ellipsoid_window(
                radius)
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
            "PET_MTV" + string_output: mtv,
            "PET_TLG" + string_output: tlg,
            "PET_SUVpeak" + string_output: suv_peak,
        })

    def __call__(self, image_path, mask_path, **kwargs):
        kwargs = {k: i for k, i in kwargs.items() if k is not "modality"}
        if type(image_path) != sitk.SimpleITK.Image:
            image = sitk.ReadImage(str(image_path))
        else:
            image = image_path

        if type(mask_path) != sitk.SimpleITK.Image:
            mask = sitk.ReadImage(str(mask_path))
        else:
            mask = mask_path

        # ROGER - Don't rename first order features anymore at the extraction level
        # results = FeatureExtractorPyradiomicsPT.translate_radiomics_output(
        #     self.radiomics_extractor.execute(image, mask, **kwargs))
        results = self.radiomics_extractor.execute(image, mask, **kwargs)

        for threshold, relative in zip([0], [False]):
            results.update(
                FeatureExtractorPyradiomicsPT.pet_features(image,
                                                           mask,
                                                           threshold=threshold,
                                                           relative=relative))
        return results


class RieszFeatureExtractor(FeatureExtractor):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, image_path, labels_path, **kwargs):

        path_of_this_file = os.path.dirname(os.path.abspath(__file__))
        path_of_executable = os.path.join(path_of_this_file, 'matlab_bin/RieszExtractor')

        completed_matlab_process = subprocess.run(
            [
                path_of_executable,
                image_path, labels_path,
                json.dumps(self.params)
            ],
            stdout=subprocess.PIPE,
            encoding="utf-8",
        )
        print("MATLAB STDOUT!!!!!!!!!!!!!!")
        output = completed_matlab_process.stdout
        output_lines = output.splitlines()
        results = json.loads(output_lines[-1])

        return results


class ZradFeatureExtractor(FeatureExtractor):

    def __call__(self, images_path, labels_path, modality=None):
        modality = modality[:2]
        modality = modality.replace("PT", "PET")
        with make_temp_directory() as output_dir:
            path_to_features = compute_zrad_features(
                images_path,
                labels_path,
                modality,
                output_dir,
                patient_id='xx',
                save_name='patient',  # only for tmp output in p_out
                **self.params,
            )

            results = pd.read_csv(path_to_features,
                                  delimiter="\t",
                                  index_col=False)

        results = results.drop(["patient", "organ"], axis=1)
        results = results.to_dict("records")[0]
        results = {"zrad_" + k: i for k, i in results.items()}
        return results
