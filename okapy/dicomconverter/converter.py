from abc import abstractmethod
from pathlib import Path
from itertools import product
from functools import partial
from tempfile import mkdtemp
from shutil import rmtree
from multiprocessing import Pool
import logging

import yaml
import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm

from okapy.dicomconverter.dicom_walker import DicomWalker
from okapy.dicomconverter.dicom_file import (EmptyContourException,
                                             PETUnitException)
from okapy.dicomconverter.volume_processor import VolumeProcessorStack
from okapy.featureextractor.featureextractor import OkapyExtractors
from okapy.exceptions import MissingSegmentationException

log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)
logger = logging.getLogger(__name__)


def bb_union(bbs):
    out = np.array([np.inf, np.inf, np.inf, -np.inf, -np.inf, -np.inf])
    for bb in bbs:
        out[:3] = np.minimum(out[:3], bb[:3])
        out[3:] = np.maximum(out[3:], bb[3:])
    return out


def bb_intersection(bbs):
    out = np.array([-np.inf, -np.inf, -np.inf, np.inf, np.inf, np.inf])
    for bb in bbs:
        out[:3] = np.maximum(out[:3], bb[:3])
        out[3:] = np.minimum(out[3:], bb[3:])
    return out


class VolumeFile():
    def __init__(self, path=None, dicom_header=None):
        self.path = path
        self.dicom_header = dicom_header

    def __getattr__(self, name):
        return getattr(self.dicom_header, name)


class VolumeResult():
    def __init__(self, study, volume, path):
        self.study_instance_uid = study.study_instance_uid
        self.patient_id = study.patient_id
        self.study_date = study.study_date
        self.modality = volume.Modality
        self.series_instance_uid = volume.SeriesInstanceUID
        self.series_number = volume.SeriesNumber
        self.path = path

    def __str__(self):
        return self.path


class MaskResult(VolumeResult):
    def __init__(self, study, volume, path):
        super().__init__(study, volume, path)
        self.reference_Modality = volume.reference_Modality
        self.label = volume.label


class BaseConverter():
    def __init__(
        self,
        padding=10,
        list_labels=None,
        dicom_walker=None,
        volume_processor=None,
        mask_processor=None,
        volume_dtype=np.float32,
        mask_dtype=np.uint32,
        extension='nii.gz',
        naming=0,
        cores=None,
        converter_backend='sitk',
    ):
        self.padding = BaseConverter._check_padding(padding)
        self.list_labels = list_labels
        self.naming = naming
        if dicom_walker is None:
            # self.dicom_walker = DicomWalker()
            self.dicom_walker = DicomWalker(cores=cores)
        else:
            self.dicom_walker = dicom_walker
        self.volume_processor = volume_processor
        self.mask_processor = mask_processor
        self.converter_backend = converter_backend
        self.volume_dtype = volume_dtype
        self.mask_dtype = mask_dtype
        self.extension = extension
        self.output_folder = None
        self.cores = cores

    @staticmethod
    def _check_padding(padding):
        if padding == "whole_image":
            return padding
        if not isinstance(padding, str):
            if padding > 0.0:
                return padding
            else:
                raise ValueError(
                    "padding must be a positive float or 'whole_image'")
        else:
            raise ValueError(
                "padding must be a positive float or 'whole_image'")

    def extract_volume_of_interest(self, study):
        masks_list = list()
        for f in study.mask_files:
            if self.list_labels is None:
                masks_list.extend([f.get_volume(label) for label in f.labels])
            else:
                label_intersection = list(
                    set(f.labels) & set(self.list_labels))
                if len(label_intersection) == 0:
                    logger.warning(
                        f"The label(s) {self.list_labels} was/were"
                        f" not found in the file {f.dicom_paths[0]}")
                for label in label_intersection:
                    try:
                        masks_list.append(f.get_volume(label))
                    except EmptyContourException:
                        continue
        return masks_list

    def get_bounding_box(self, masks_list, volumes_list):
        bb = bb_union([mask.bb for mask in masks_list])

        bb[:3] = bb[:3] - self.padding
        bb[3:] = bb[3:] + self.padding

        return bb_intersection([v.reference_frame.bb
                                for v in volumes_list] + [bb])

    def get_path(self, volume, is_mask=False, ouput_folder=None):
        if is_mask:
            name = self._get_name_mask(volume)
        else:
            name = self._get_name_volume(volume)
        if ouput_folder is None:
            return Path(self.output_folder) / name
        else:
            return Path(ouput_folder) / name

    def _get_name_mask(self, volume):
        if self.naming == 0:
            return (f"{volume.PatientID}__{volume.label.replace(' ', '_')}__"
                    f"{volume.Modality}__{volume.reference_Modality}"
                    f".{self.extension}")

        elif self.naming == 1:
            return (f"{volume.PatientID}__{volume.label.replace(' ', '_')}__"
                    f"{volume.Modality}__{volume.SeriesNumber}__"
                    f"{volume.reference_Modality}"
                    f"__{volume.reference_SeriesNumber}"
                    f".{self.extension}")

        elif self.naming == 2:
            return (
                f"{volume.PatientID}__{volume.label.replace(' ', '_')}__"
                f"{volume.Modality}__{volume.SeriesNumber}__"
                f"{volume.reference_Modality}"
                f"__{volume.reference_SeriesNumber}__"
                f"{str(volume.series_datetime).replace(' ', '_').replace(':', '-')}"
                f".{self.extension}")

    def _get_name_volume(self, volume):
        if self.naming == 0:
            return (f"{volume.PatientID}__{volume.Modality}"
                    f".{self.extension}")
        elif self.naming == 1:
            return (f"{volume.PatientID}__{volume.Modality}__"
                    f"{volume.SeriesNumber}.{self.extension}")
        elif self.naming == 2:
            return (
                f"{volume.PatientID}__{volume.Modality}__"
                f"{volume.SeriesNumber}__"
                f"{str(volume.series_datetime).replace(' ', '_').replace(':', '-')}"
                f".{self.extension}")

    def write(self, volume, is_mask=False, dtype=None, output_folder=None):
        if dtype:
            volume = volume.astype(dtype)
        counter = 0
        path = self.get_path(volume,
                             is_mask=is_mask,
                             ouput_folder=output_folder)
        new_path = path
        while new_path.is_file():
            counter += 1
            new_path = path.with_name(
                path.name.replace('.' + self.extension, '') + '(' +
                str(counter) + ')' + '.' + self.extension)
        if self.converter_backend == 'sitk':
            sitk.WriteImage(volume.sitk_image, str(new_path.resolve()))
        return VolumeFile(path=path, dicom_header=volume.dicom_header)

    @abstractmethod
    def process_study(self, study, output_folder=None):
        pass

    @abstractmethod
    def __call__(self, input_folder, output_folder=None):
        pass


class NiftiConverter(BaseConverter):
    def __init__(
        self,
        output_folder=".",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.output_folder = Path(output_folder).resolve()

    def process_study(self, study, output_folder=None):
        logger.debug(
            f"Start of processing study for patient {study.patient_id}")
        volume_results_list = list()
        mask_results_list = list()
        masks_list = self.extract_volume_of_interest(study)
        volumes_list = list()
        for f in study.volume_files:
            try:
                volumes_list.append(f.get_volume())
            except PETUnitException as e:
                print(e)
                continue

        if self.padding != 'whole_image':
            bb = self.get_bounding_box(masks_list, volumes_list)
        else:
            # The image is not cropped
            bb = None
        masks_list = map(lambda v: self.mask_processor(v, bb), masks_list)
        volumes_list = map(lambda v: self.volume_processor(v, bb),
                           volumes_list)
        for v in volumes_list:
            volume_file = self.write(v,
                                     is_mask=False,
                                     dtype=self.volume_dtype,
                                     output_folder=output_folder)
            volume_results_list.append(VolumeResult(study, v,
                                                    volume_file.path))

        for v in masks_list:
            volume_file = self.write(v,
                                     is_mask=True,
                                     dtype=self.mask_dtype,
                                     output_folder=output_folder)
            mask_results_list.append(MaskResult(study, v, volume_file.path))

        logger.debug(f"End of processing study for patient {study.patient_id}")
        return volume_results_list, mask_results_list

    def __call__(self, input_folder, output_folder=None):
        if output_folder is not None:
            self.output_folder = output_folder

        studies_list = self.dicom_walker(input_folder, cores=self.cores)
        if self.cores:
            with Pool(self.cores) as p:
                result = list(
                    tqdm(p.imap(self.process_study, studies_list),
                         total=len(studies_list)))
        else:
            result = list()
            for study in tqdm(studies_list):
                result.append(self.process_study(study))

        return result


class ExtractorConverter(BaseConverter):
    def __init__(self,
                 okapy_extractors=None,
                 result_format="long",
                 additional_dicom_tags=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.okapy_extractors = okapy_extractors
        self.output_folder = None
        self.result_format = ExtractorConverter._check_result_format(
            result_format)
        self.dicom_walker.additional_dicom_tags = additional_dicom_tags
        self._additional_dicom_tags = additional_dicom_tags
        self.core = None  # TODO: fix multiprocessing

    @property
    def additional_dicom_tags(self):
        return self._additional_dicom_tags

    @additional_dicom_tags.setter
    def additional_dicom_tags(self, tags):
        self._additional_dicom_tags = tags
        self.dicom_walker.additional_dicom_tags = tags

    @staticmethod
    def _check_result_format(result_format):
        if result_format not in ["long", "multiindex"]:
            raise ValueError(f"The format {result_format} for the argument "
                             f"result_format is not supported.")
        return result_format

    @staticmethod
    def from_params(params_path):
        if type(params_path) == dict:
            params = params_path
        else:
            with open(params_path, 'r') as f:
                params = yaml.safe_load(f)

        additional_dicom_tags = params["general"].get("additional_dicom_tags",
                                                      [])
        dicom_walker = DicomWalker(
            additional_dicom_tags=additional_dicom_tags,
            submodalities=params["general"].get("submodalities", False),
        )

        volume_processor = VolumeProcessorStack.from_params(
            params["volume_preprocessing"])

        mask_processor = VolumeProcessorStack.from_params(
            params["mask_preprocessing"])

        okapy_extractors = OkapyExtractors(params["feature_extraction"])

        return ExtractorConverter(
            dicom_walker=dicom_walker,
            volume_processor=volume_processor,
            mask_processor=mask_processor,
            okapy_extractors=okapy_extractors,
            additional_dicom_tags=additional_dicom_tags,
        )

    def get_empty_results_df(self):
        if self.result_format == "mutltiindex":
            return pd.DataFrame(
                index=pd.MultiIndex(levels=[[], []],
                                    codes=[[], []],
                                    names=["patient_id", "VOI"]),
                columns=pd.MultiIndex(levels=[[], []],
                                      codes=[[], []],
                                      names=["modality", "features"]))
        elif self.result_format == "long":
            return pd.DataFrame()

    def extract_volume_of_interest(self, study, labels=None):
        masks_list = list()
        for f in study.mask_files:
            if labels is None:
                masks_list.extend([f.get_volume(label) for label in f.labels])
            else:
                label_intersection = list(set(f.labels) & set(labels))
                for label in label_intersection:
                    try:
                        masks_list.append(f.get_volume(label))
                    except EmptyContourException:
                        continue
        return masks_list

    def process_study(self, study, labels=None):
        masks_list = self.extract_volume_of_interest(study, labels=labels)
        if not masks_list:
            raise MissingSegmentationException(
                f"No segmentation found for study with"
                f" StudyInstanceUID {study.study_instance_uid}")
        volumes_list = list()
        for f in study.volume_files:
            try:
                volumes_list.append(f.get_volume())
            except PETUnitException as e:
                print(e)
                continue

        if self.padding != 'whole_image':
            bb = self.get_bounding_box(masks_list, volumes_list)
        else:
            # The image is not cropped
            bb = None
        masks_list = list(
            map(lambda v: self.mask_processor(v, bounding_box=bb), masks_list))
        volumes_list = list(
            map(lambda v: self.volume_processor(v, bounding_box=bb),
                volumes_list))

        modalities_list = list(map(lambda v: v.modality, volumes_list))
        labels_list = list(map(lambda v: v.label, masks_list))
        volumes_list = list(
            map(
                lambda v: self.write(v, is_mask=False, dtype=self.volume_dtype
                                     ), volumes_list))
        masks_list = list(
            map(lambda v: self.write(v, is_mask=True, dtype=self.mask_dtype),
                masks_list))

        results_df = self.get_empty_results_df()

        for (volume, modality), (mask, label) in product(
                zip(volumes_list, modalities_list),
                zip(masks_list, labels_list),
        ):
            result = self.okapy_extractors(volume.path,
                                           mask.path,
                                           modality=modality)
            for key, val in result.items():
                if "diagnostics" in key:
                    continue
                if self.result_format == "multiindex":
                    results_df.loc[(study.patient_id, label),
                                   (modality, key)] = val
                elif self.result_format == "long":
                    result_dict = {
                        "patient_id": study.patient_id,
                        "modality": modality,
                        "VOI": label,
                        "feature_name": key,
                        "feature_value": val,
                    }
                    result_dict.update({
                        k: getattr(volume.dicom_header, k)
                        for k in self.additional_dicom_tags
                    })
                    results_df = results_df.append(
                        result_dict,
                        ignore_index=True,
                    )

        return results_df

    def __call__(self, input_folder, labels=None):
        try:
            self.output_folder = mkdtemp()
            studies_list = self.dicom_walker(input_folder, cores=self.cores)
            if self.cores is None:
                result_dfs = list()
                for study in studies_list:
                    result_dfs.append(self.process_study(study, labels=labels))
            else:

                with Pool(self.cores) as p:
                    result_dfs = list(
                        tqdm(p.imap(partial(self.process_study, labels=labels),
                                    studies_list),
                             total=len(studies_list)))

            rmtree(self.output_folder, True)
            self.output_folder = None
        except Exception as e:
            rmtree(self.output_folder, True)
            self.output_folder = None
            raise e

        return pd.concat(result_dfs, axis=0, ignore_index=True)
