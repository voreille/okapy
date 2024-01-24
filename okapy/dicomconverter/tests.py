import unittest
from pathlib import Path

import SimpleITK as sitk

from okapy.dicomconverter.dicom_walker import DicomWalker
from okapy.dicomconverter.converter import NiftiConverter, ExtractorConverter


class TestOkapy(unittest.TestCase):
    """Tests for `okapy` package."""

    def test_conversion(self):
        dicom_path = "data/test_files/dicom/PTCT"
        nii_path = Path("data/output/nii/PTCT")
        nii_path.mkdir(exist_ok=True, parents=True)
        suv_probed_values = {
            "HN-CHUS-005": (60, 56, 208, 3.629),
            "HN-CHUM-017": (54, 60, 48, 9.994),
            "HN-HMR-023": (62, 46, 52, 12.074)
        }
        nii_paths = [f for f in Path(nii_path).rglob("*.nii.gz")]
        # Remove the files from previous tests:
        for f in nii_paths:
            f.unlink()
        converter = NiftiConverter(
            padding="whole_image",
            labels_startswith="GTV",
            dicom_walker=DicomWalker(),
            cores=2,
            naming=2,
        )
        conversion_results = converter(dicom_path, output_folder=nii_path)
        nii_paths = [f for f in Path(nii_path).rglob("*.nii.gz")]

        # Assert if the right number of files were converted
        assert len(nii_paths) == 22

        # Assert if the SUV are computed correctly, hardcoded but no choice
        nii_pt_paths = [f for f in nii_paths if "PT" in f.name]
        assert len(nii_pt_paths) == 3
        for f in nii_pt_paths:
            patient_id = f.name.split("__")[0]
            image = sitk.ReadImage(str(f.resolve()))
            self.assertAlmostEqual(
                image.GetPixel(*suv_probed_values[patient_id][:3]),
                suv_probed_values[patient_id][3],
                places=3,
            )
        print("Test conversion is finished here an overview of the output")
        print(conversion_results)

    def test_feature_extraction(self):
        dicom_path = "data/test_files/dicom/MR"
        nii_path = Path("data/output/nii/MR")
        nii_path.mkdir(exist_ok=True, parents=True)

        extractor = ExtractorConverter.from_params(
            "parameters/tests/test_mr.yaml")
        results = extractor(dicom_path)
        results[results["feature_name"] == "original_firstorder_Mean"]
        mean_edema_flair = results[
            (results["feature_name"] == "original_firstorder_Mean")
            & (results["VOI"] == " edema") &
            (results["modality"] == "MR_FLAIR")]["feature_value"].values[0]
        mean_edema_t2 = results[
            (results["feature_name"] == "original_firstorder_Mean")
            & (results["VOI"] == " edema") &
            (results["modality"] == "MR_T2")]["feature_value"].values[0]
        self.assertAlmostEqual(mean_edema_flair, 0)
        self.assertAlmostEqual(mean_edema_t2, 0)
        print(
            "Test feature extraction is finished here an overview of the output"
        )
        print(results)


if __name__ == '__main__':
    unittest.main()
