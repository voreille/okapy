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
# from okapy.dicomconverter.dicom_header import DicomHeader

log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)
logger = logging.getLogger(__name__)


class DicomFile():
    def __init__(self, dicom_header=None, path=None):
        self.dicom_header = dicom_header
        self.path = path

    def __str__(self):
        return str(self.dicom_header)


class DicomWalker():
    def __init__(
        self,
        input_dirpath=None,
        cores=None,
    ):
        self.input_dirpath = input_dirpath
        self.cores = cores

    @staticmethod
    def _parse_file(file):
        try:
            data = pdcm.filereader.dcmread(str(file.resolve()),
                                           stop_before_pixels=True)

        except InvalidDicomError:
            logger.debug(f"The file {str(file)} is not recognised as dicom")
            return None

        if not hasattr(data, "Modality"):
            logger.debug(f"DICOMDIR are not read, filepath: {str(file)}")
            return None

        return DicomFile(dicom_header=data, path=str(file.resolve()))

    def _walk(self, input_dirpath):
        '''
        Method to walk through the path given and fill the list of DICOM
        headers and sort them
        '''
        dicom_files = list()
        files = [f for f in Path(input_dirpath).rglob("*") if f.is_file()]
        # to test wether a slice appear multiple times
        logger.info("Parsing the DICOM files")
        if self.cores is None:
            for file in tqdm(files, desc="Walking through all the files"):
                dicom_file = DicomWalker._parse_file(file)
                if dicom_file:
                    dicom_files.append(dicom_file)
        else:
            with Pool(self.cores) as pool:
                dicom_files = pool.map(DicomWalker._parse_file,
                                       files,
                                       chunksize=500)
            dicom_files = [f for f in dicom_files if f]

        logger.info("Parsing - END")
        dicom_files.sort(key=lambda x: (
            x.dicom_header.StudyInstanceUID, x.dicom_header.Modality, x.
            dicom_header.SeriesInstanceUID, x.dicom_header.InstanceNumber, x.
            dicom_header.PatientID))
        return dicom_files

    def _get_studies(self, dicom_files):
        '''
        Construct the tree-like dependency of the dicom
        It all relies on the fact that the collection of headers has been
        sorted
        '''
        im_dicom_files = list()
        studies = list()
        previous_dcm_header = dict()

        previous_study_uid = None
        for i, f in enumerate(dicom_files):
            # When the image changeschanges we store it as a whole
            current_study_uid = f.dicom_header.StudyInstanceUID
            if i == 0:
                current_study = Study(study_instance_uid=current_study_uid,
                                      study_date=f.dicom_header.StudyDate,
                                      patient_id=f.dicom_header.PatientID)

            if i > 0 and not (f.dicom_header.SeriesInstanceUID
                              == previous_dcm_header.get(
                                  "SeriesInstanceUID",
                                  -1)) and (len(im_dicom_files) > 0):
                current_study.append_dicom_files(im_dicom_files,
                                                 previous_dcm_header)
                im_dicom_files = list()

            if i > 0 and not (current_study_uid == previous_study_uid):
                studies.append(current_study)
                current_study = Study(study_instance_uid=current_study_uid,
                                      study_date=f.dicom_header.StudyDate,
                                      patient_id=f.dicom_header.PatientID)

            # Only keeping files that are different
            if dict(f.dicom_header) != dict(previous_dcm_header):
                im_dicom_files.append(f)
                previous_dcm_header = f.dicom_header
                previous_study_uid = f.dicom_header.StudyInstanceUID

        current_study.append_dicom_files(im_dicom_files, previous_dcm_header)
        studies.append(current_study)
        return studies

    def __call__(self, input_dirpath=None):
        if input_dirpath:
            dicom_files = self._walk(input_dirpath)
        else:
            dicom_files = self._walk(self.input_dirpath)

        if len(dicom_files) == 0:
            raise RuntimeError(
                "No valid DICOM files found in the input directory")

        return self._get_studies(dicom_files)
