import unittest

from okapy.dicomconverter.converter import ExtractorConverter


class TestOkapy(unittest.TestCase):
    # def test_extractor1(self):

    #     # input_path = '/home/val/Documents/check_hecktor_anna_tmp/HN-CHUS-047'
    #     # input_path = '/home/val/Documents/check_hecktor_anna_tmp/P9'
    #     # input_path = '/home/val/python_wkspce/lcnn_radiomic/data/raw/Head-Neck-PET-CT/DICOM/HN-CHUS-047'
    #     # input_path = '/mnt/nas4/datasets/ToReadme/ORL_RennesCHUV_Castelli/TEP_RENNES/P9'
    #     # input_path = '/home/val/python_wkspce/lymphangitis3.0/data/raw/PatientLC_51'
    #     input_path = "/mnt/nas4/datasets/ToReadme/HECKTOR/training/dicom/HN-HMR-029"
    #     # input_path = "/home/val/Documents/test_okapy/TCGA-06-6389"
    #     # input_path1 = '/mnt/nas2/data/Personal/Roger/IMAGINE/NIFTI-SEG/'
    #     # input_path2 = ('/mnt/nas4/datasets/ToReadme/TCIA-Head-Neck-Radi'
    #     # 'omics-HN1/HEAD-NECK-RADIOMICS-HN1-NORTSTRUCT/HN1026')
    #     extraction_params = "/home/val/python_wkspce/okapy/parameters/defaults/defaults.yaml"
    #     converter = ExtractorConverter.from_params(extraction_params)
    #     result = converter(input_path, labels=["GTVt", "GTVn"])
    #     print(result)

    def test_extractor2(self):

        # input_path = '/home/val/Documents/check_hecktor_anna_tmp/HN-CHUS-047'
        # input_path = '/home/val/Documents/check_hecktor_anna_tmp/P9'
        # input_path = '/home/val/python_wkspce/lcnn_radiomic/data/raw/Head-Neck-PET-CT/DICOM/HN-CHUS-047'
        # input_path = '/mnt/nas4/datasets/ToReadme/ORL_RennesCHUV_Castelli/TEP_RENNES/P9'
        # input_path = '/home/val/python_wkspce/lymphangitis3.0/data/raw/PatientLC_51'
        # input_path = "/mnt/nas4/datasets/ToReadme/HECKTOR/training/dicom/HN-HMR-029"
        input_path = "/mnt/nas4/datasets/ToReadme/HECKTOR/training/dicom/HN-CHUS-035"
        # input_path = "/home/val/Documents/test_okapy/dicom_without_seg/"
        # input_path1 = '/mnt/nas2/data/Personal/Roger/IMAGINE/NIFTI-SEG/'
        # input_path2 = ('/mnt/nas4/datasets/ToReadme/TCIA-Head-Neck-Radi'
        # 'omics-HN1/HEAD-NECK-RADIOMICS-HN1-NORTSTRUCT/HN1026')
        extraction_params = "/home/valentin/python_wkspce/okapy/parameters/defaults/defaults_pyradiomics_only.yaml"
        converter = ExtractorConverter.from_params(extraction_params)
        converter.cores = 5
        converter.additional_dicom_tags = [
            "AcquisitionDate", "StudyInstanceUID"
        ]
        result = converter(input_path, labels=["GTVt"])
        print(result)

    # def test_extractor2(self):

    #     input_path = "/mnt/nas4/datasets/ToReadme/HECKTOR/training/dicom/HN-HMR-029"
    #     extraction_params = "/home/val/python_wkspce/okapy/parameters/defaults/defaults_pyradiomics_only.yaml"
    #     converter = ExtractorConverter.from_params(extraction_params)
    #     result = converter(input_path, labels=["GTVt", "GTVn"])
    #     print(result)


if __name__ == '__main__':
    unittest.main()
