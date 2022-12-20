from abc import abstractmethod
from pathlib import Path
from functools import partial
from tempfile import mkdtemp
from shutil import rmtree
from multiprocessing import Pool
import logging

import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm

from okapy.dicomconverter.dicom_walker import DicomWalker
from okapy.dicomconverter.volume_processor import VolumeProcessorStack
from okapy.dicomconverter.study import StudyProcessor, StudyProcessorDeep
from okapy.featureextractor.featureextractor import OkapyExtractors
import okapy.yaml.yaml as yaml

log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)
logger = logging.getLogger(__name__)


class BaseConverter():

    def __init__(
        self,
        padding=10,
        list_labels=None,
        dicom_walker=None,
        study_processor=None,
        volume_dtype=np.float32,
        mask_dtype=np.uint8,
        extension='nii.gz',
        naming=0,
        cores=None,
    ):
        self.padding = BaseConverter._check_padding(padding)
        self.list_labels = list_labels
        self.naming = naming
        self.dicom_walker = dicom_walker if dicom_walker else DicomWalker()
        self.study_processor = (study_processor
                                if study_processor else StudyProcessor())
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

    def get_path(self, volume, include_roi=False, is_mask=False, ouput_folder=None):
        if is_mask:
            name = self._get_name_mask(volume)
        else:
            name = self._get_name_volume(volume, include_roi)
        if ouput_folder is None:
            return Path(self.output_folder) / name
        else:
            return Path(ouput_folder) / name

    def _get_name_mask(self, volume):
        if self.naming == 0:
            return (f"{volume.PatientID}__{volume.label.replace(' ', '_')}__"
                    f"{volume.modality}__{volume.reference_modality}"
                    f".{self.extension}")

        elif self.naming == 1:
            return (f"{volume.PatientID}__{volume.label.replace(' ', '_')}__"
                    f"{volume.modality}__{volume.SeriesNumber}__"
                    f"{volume.reference_modality}"
                    f"__{volume.reference_SeriesNumber}"
                    f".{self.extension}")

        elif self.naming == 2:
            return (
                f"{volume.PatientID}__{volume.label.replace(' ', '_')}__"
                f"{volume.modality}__{volume.SeriesNumber}__"
                f"{volume.reference_modality}"
                f"__{volume.reference_SeriesNumber}__"
                f"{str(volume.series_datetime).replace(' ', '_').replace(':', '-')}"
                f".{self.extension}")

    def _get_name_volume(self, volume, include_roi=False):
        if self.naming == 0:
            return (f"{volume.PatientID}__{volume.modality}{'__' if include_roi else ''}{include_roi if include_roi else ''}"
                    f".{self.extension}")
        elif self.naming == 1:
            return (f"{volume.PatientID}__{volume.modality}{'__' if include_roi else ''}{include_roi if include_roi else ''}__"
                    f"{volume.SeriesNumber}.{self.extension}")
        elif self.naming == 2:
            return (
                f"{volume.PatientID}__{volume.modality}{'__' if include_roi else ''}{include_roi if include_roi else ''}__"
                f"{volume.SeriesNumber}__"
                f"{str(volume.series_datetime).replace(' ', '_').replace(':', '-')}"
                f".{self.extension}")

    def write(self, volume, include_roi=False, is_mask=False, dtype=None, output_folder=None):
        if dtype:
            volume = volume.astype(dtype)
        elif self.volume_dtype and not is_mask:
            volume = volume.astype(self.volume_dtype)
        elif self.mask_dtype and is_mask:
            volume = volume.astype(self.mask_dtype)

        counter = 0
        path = self.get_path(volume,
                             include_roi=include_roi,
                             is_mask=is_mask,
                             ouput_folder=output_folder)
        new_path = path
        while new_path.is_file():
            counter += 1
            new_path = path.with_name(
                path.name.replace('.' + self.extension, '') + '(' +
                str(counter) + ')' + '.' + self.extension)
        sitk.WriteImage(volume.sitk_image, str(new_path.resolve()))

        if is_mask:
            return dict(
                path=new_path,
                dicom_header=volume.dicom_header,
                modality=volume.modality,
                label=volume.label,
                reference_dicom_header=volume.reference_dicom_header,
            )
        else:
            return dict(path=new_path,
                        modality=volume.modality,
                        dicom_header=volume.dicom_header)

    @abstractmethod
    def process_study(self, study, output_folder=None):
        pass

    @abstractmethod
    def __call__(self, input_folder, output_folder=None):
        pass


class ExtractorConverter(BaseConverter):

    def __init__(self,
                 okapy_extractors=None,
                 result_format="long",
                 additional_dicom_tags=None,
                 combine_segmentation=False,
                 **kwargs):
        super().__init__(**kwargs)
        self.okapy_extractors = okapy_extractors
        self.output_folder = None
        self.result_format = ExtractorConverter._check_result_format(
            result_format)
        self.dicom_walker.additional_dicom_tags = additional_dicom_tags
        self._additional_dicom_tags = additional_dicom_tags
        self.core = None  # TODO: fix multiprocessing
        self._combine_segmentation = False
        self.combine_segmentation = combine_segmentation

    @property
    def combine_segmentation(self):
        return self._combine_segmentation

    @combine_segmentation.setter
    def combine_segmentation(self, cond):
        self._combine_segmentation = cond
        self.study_processor.combine_segmentation = cond

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
                params = yaml.load(f)

        additional_dicom_tags = params["general"].get("additional_dicom_tags",
                                                      [])
        combine_segmentation = params["general"].get("combine_segmentation",
                                                     False)

        dicom_walker = DicomWalker(
            additional_dicom_tags=additional_dicom_tags,
            submodalities=params["general"].get("submodalities", False),
        )

        mask_processor = VolumeProcessorStack.from_params(
            params["mask_preprocessing"])

        volume_processor = VolumeProcessorStack.from_params(
            params["volume_preprocessing"], mask_resampler=mask_processor)

        # TODO - Don't hard-code this here
        study_processor_class = StudyProcessorDeep if params["general"].get("study_processor", None) == "studyprocessor_deep" else StudyProcessor

        study_processor = study_processor_class(
            volume_processor=volume_processor,
            mask_processor=mask_processor,
            padding=params["general"].get("padding", 10),
            combine_segmentation=combine_segmentation,
            convert_image_without_seg=False,
            patch_size=params["general"].get("patch_size", None)
        )

        okapy_extractors = OkapyExtractors(params["feature_extraction"]) if "feature_extraction" in params else None

        return ExtractorConverter(
            dicom_walker=dicom_walker,
            study_processor=study_processor,
            okapy_extractors=okapy_extractors,
            additional_dicom_tags=additional_dicom_tags,
            combine_segmentation=combine_segmentation,
        )

    def process_study(self, study, labels=None):
        results_df = pd.DataFrame() if self.okapy_extractors else None
        index = 0
        for volume, masks in self.study_processor(study, labels=labels):

            # TODO - Handle this more elegantly
            if isinstance(self.study_processor, StudyProcessorDeep):
                volume_info = self.write(volume, include_roi=masks[0].label, output_folder=self.output_folder)
            else:
                volume_info = self.write(volume, output_folder=self.output_folder)

            for mask in masks:
                mask.reference_modality = volume.modality
                mask_info = self.write(mask,
                                       is_mask=True,
                                       output_folder=self.output_folder)
                if self.okapy_extractors:
                    result = self.okapy_extractors(
                        volume_info["path"],
                        mask_info["path"],
                        modality=volume_info["modality"])
                    for key, val in result.items():
                        if "diagnostics" in key:
                            continue

                        result_dict = {
                            "patient_id": study.patient_id,
                            "modality": volume_info["modality"],
                            "VOI": mask_info["label"],
                            "feature_name": key,
                            "feature_value": val,
                        }
                        result_dict.update({
                            k: getattr(volume.dicom_header, k)
                            for k in self.additional_dicom_tags
                        })
                        results_df = pd.concat([
                            results_df,
                            pd.DataFrame(result_dict, index=[index]),
                        ])
                        index = index + 1

        return results_df

    def __call__(self, input_folder, labels=None, output_folder=None):
        try:
            self.output_folder = mkdtemp() if output_folder is None else output_folder
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

            # If a temporary directory was used, clean up at the end
            if output_folder is None:
                rmtree(self.output_folder, True)
            self.output_folder = None
        except Exception as e:
            # In case of an exception, always clean up at the end
            rmtree(self.output_folder, True)
            self.output_folder = None
            raise e

        # If no results were computed, return None
        if None in [x.to_string() if isinstance(x, pd.DataFrame) else x for x in result_dfs]:
            return None

        return pd.concat(result_dfs, axis=0, ignore_index=True)
