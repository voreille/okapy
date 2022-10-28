"""
TODO: test nifti converter: with and withouth mask, resampling, etc.
TODO: test MR, MR submodality, CT, CT submodality
TODO: EXtract image with and without mask

"""

from pathlib import Path
import unittest

from okapy.dicomconverter.converter import ExtractorConverter

project_dir = Path(__file__).resolve().parents[1]


class TestOkapy(unittest.TestCase):

    def test_extractor1(self):
        """Test the extractor with a single file."""
        input_path = project_dir / "data/test_files/images/HN-HMR-029"
        extraction_params = project_dir / "parameters/defaults/defaults.yaml"
        converter = ExtractorConverter.from_params(extraction_params)
        result = converter(input_path, labels=["GTVt", "GTVn"])
        print(result)
        result.to_csv("test_extractor1.csv")

    def test_extractor2(self):
        """Test the extractor with a single file."""
        input_path = project_dir / "data/test_files/images/HN-HMR-029"
        extraction_params = project_dir / "parameters/defaults/defaults.yaml"
        converter = ExtractorConverter.from_params(extraction_params)
        converter.combine_segmentation = True
        converter.cores = None
        result = converter(input_path, labels=["GTVt"])
        print(result)
        result.to_csv("test_extractor1.csv")


if __name__ == '__main__':
    unittest.main()
