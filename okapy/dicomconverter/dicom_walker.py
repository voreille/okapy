'''
TODO: Is it better to create a class for just the header and another for the
files?
'''
from pathlib import Path
from multiprocessing import Pool
import logging

from tqdm import tqdm
import pydicom as pdcm
from pydicom.errors import InvalidDicomError

from okapy.dicomconverter.study import Study
from okapy.dicomconverter.dicom_header import DicomHeader

log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)
logger = logging.getLogger(__name__)


class DicomFile():

    def __init__(self, dicom_header=None, path=None):
        self.dicom_header = dicom_header
        self.path = path


class DicomWalker():

    def __init__(
        self,
        input_dirpath=None,
        cores=None,
        additional_dicom_tags=None,
        submodalities=False,
    ):
        self.input_dirpath = input_dirpath
        self.cores = cores
        self.additional_dicom_tags = additional_dicom_tags
        self.submodalities = submodalities

    def _parse_file(self, file):
        try:
            data = pdcm.filereader.dcmread(str(file.resolve()),
                                           stop_before_pixels=True)

        except InvalidDicomError:
            logger.debug(f"The file {str(file)} is not recognised as dicom")
            return None

        if not hasattr(data, "Modality"):
            logger.debug(f"DICOMDIR are not read, filepath: {str(file)}")
            return None

        return DicomFile(dicom_header=DicomHeader.from_pydicom(
            data, additional_tags=self.additional_dicom_tags),
                         path=str(file.resolve()))

    def _get_files(self, input_dirpath):
        if type(input_dirpath) == list:
            return [
                f for path in input_dirpath for f in Path(path).rglob("*")
                if f.is_file()
            ]

        try:
            output = [f for f in Path(input_dirpath).rglob("*") if f.is_file()]
        except Exception as e:
            raise TypeError(
                f"input_dirpath must be a path or a list of paths, "
                f"string or pathlib.Path, not {type(input_dirpath)}")
        return output

    def _walk(self, input_dirpath, cores=None):
        '''
        Method to walk through the path given and fill the list of DICOM
        headers and sort them
        '''
        dicom_files = list()
        files = self._get_files(input_dirpath)
        # to test wether a slice appear multiple times
        logger.info("Parsing the DICOM files")
        if cores is None:
            for file in tqdm(files, desc="Walking through all the files"):
                dicom_file = self._parse_file(file)
                if dicom_file:
                    dicom_files.append(dicom_file)
        else:
            with Pool(self.cores) as pool:
                dicom_files = list(
                    tqdm(pool.imap(self._parse_file, files, chunksize=500),
                         total=len(files)))
            dicom_files = [f for f in dicom_files if f]

        logger.info("Parsing - END")
        dicom_files.sort(key=lambda x: (
            x.dicom_header.StudyInstanceUID,
            x.dicom_header.Modality,
            x.dicom_header.SeriesInstanceUID,
            x.dicom_header.InstanceNumber,
            x.dicom_header.PatientID,
        ))
        return dicom_files

    def _get_studies(self, dicom_files):
        '''
        Construct the tree-like dependency of the dicom
        It all relies on the fact that the collection of headers has been
        sorted
        '''
        im_dicom_files = list()
        studies = list()
        previous_dcm_header = DicomHeader()

        previous_study_uid = None
        for i, f in enumerate(dicom_files):
            # When the image changeschanges we store it as a whole
            current_study_uid = f.dicom_header.StudyInstanceUID
            if i == 0:
                current_study = Study(
                    study_instance_uid=current_study_uid,
                    study_date=f.dicom_header.StudyDate,
                    patient_id=f.dicom_header.PatientID,
                    additional_dicom_tags=self.additional_dicom_tags,
                    submodalities=self.submodalities)

            if i > 0 and not (f.dicom_header.SeriesInstanceUID
                              == previous_dcm_header.SeriesInstanceUID and
                              (len(im_dicom_files) > 0)):
                current_study.append_dicom_files(im_dicom_files,
                                                 previous_dcm_header)
                im_dicom_files = list()

            if i > 0 and not (current_study_uid == previous_study_uid):
                # check if the study contains images, in case of lone RTSTRUCT
                if len(current_study.volume_files) > 0:
                    studies.append(current_study)
                current_study = Study(
                    study_instance_uid=current_study_uid,
                    study_date=f.dicom_header.StudyDate,
                    patient_id=f.dicom_header.PatientID,
                    additional_dicom_tags=self.additional_dicom_tags,
                    submodalities=self.submodalities)

            # Only keeping files that are different
            if f.dicom_header != previous_dcm_header:
                im_dicom_files.append(f)
                previous_dcm_header = f.dicom_header
                previous_study_uid = f.dicom_header.StudyInstanceUID

        current_study.append_dicom_files(im_dicom_files, previous_dcm_header)
        if len(current_study.volume_files) > 0:
            studies.append(current_study)
        return studies

    def __call__(self, input_dirpath=None, cores=None):
        if cores is None:
            cores = self.cores
        if input_dirpath:
            dicom_files = self._walk(input_dirpath, cores=cores)
        else:
            dicom_files = self._walk(self.input_dirpath, cores=cores)

        if len(dicom_files) == 0:
            raise RuntimeError(
                "No valid DICOM files found in the input directory")

        return self._get_studies(dicom_files)
