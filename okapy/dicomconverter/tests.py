import unittest
from click.testing import CliRunner

from dicom_walker import DicomWalker


class TestOkapy(unittest.TestCase):
    """Tests for `okapy` package."""

#    def test_walker(self):
#        """Test the DicomWalker class"""
#        input_path = ('/home/val/Documents/data_radiomics/lymphangite/'
#        'final-v3/L0/L0_1')
#        output_path = '/home/val/python_wkspce/output_okapy/'
#        print('Running test on the walker')
#        walker = DicomWalker(input_path, output_path,
#                             list_labels=['GTV L', 'GTV T'])
#        walker.walk()
#        walker.fill_images()
#        walker.convert()

#    def test_command_line_interface(self):
#        """Test the CLI."""
#        runner = CliRunner()
#        result = runner.invoke(cli.main)
#        assert result.exit_code == 0
#        assert 'okapy.cli.main' in result.output
#        help_result = runner.invoke(cli.main, ['--help'])
#        assert help_result.exit_code == 0
#        assert '--help  Show this message and exit.' in help_result.output
#

    def test_convert_mr(self):
        """
        The walker must extract all the MR and also all the VOIs with their
        label in the name of the file and resample at 0.75 mm
        """

        input_path = '/home/val/python_wkspce/okapy/okapy/dicomconverter/examples/MR/'
        output_path = '/home/val/python_wkspce/okapy/okapy/dicomconverter/examples/MR_output/'
        print('Running test on the walker for MR conversiont')
        walker = DicomWalker(input_path, output_path)
        walker.walk()
        walker.fill_images()
        walker.convert()


if __name__ == '__main__':
    unittest.main()
