import unittest
from pathlib import Path

from click.testing import CliRunner
import numpy as np

from okapy.dicomconverter.dicom_walker import DicomWalker
from okapy.dicomconverter.volume import ReferenceFrame
from okapy.dicomconverter.dicom_file import DicomFileMR
from okapy.dicomconverter.converter import Converter


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

    def test_converter(self):
        """
        The walker must extract all the MR and also all the VOIs with their
        label in the name of the file and resample at 0.75 mm
        """

        # input_path = '/home/val/Documents/check_hecktor_anna_tmp/HN-CHUS-047'
        # input_path = '/home/val/Documents/check_hecktor_anna_tmp/P9'
        input_path = '/home/val/python_wkspce/lcnn_radiomic/data/raw/Head-Neck-PET-CT/DICOM/HN-CHUS-047'
        # input_path = '/mnt/nas4/datasets/ToReadme/ORL_RennesCHUV_Castelli/TEP_RENNES/P9'
        # input_path1 = '/mnt/nas2/data/Personal/Roger/IMAGINE/NIFTI-SEG/'
        # input_path2 = ('/mnt/nas4/datasets/ToReadme/TCIA-Head-Neck-Radi'
        # 'omics-HN1/HEAD-NECK-RADIOMICS-HN1-NORTSTRUCT/HN1026')
        output_path = '/home/val/Documents/output_okapy'
        converter = Converter(output_folder=output_path,
                              list_labels=['GTVt'],
                              padding=-1,
                              resampling_spacing=-1)
        result = converter(input_path)
        print(result)

    # def test_dicomfilemr(self):
    #     input_path = Path(
    #         '/mnt/nas2/data/Personal/Roger/IMAGINE/NIFTI-SEG/dicoms/')
    #     dcm_paths = [str(k.resolve()) for k in input_path.rglob('*.dcm')]
    #     dicom_mr = DicomFileMR(dicom_paths=dcm_paths)
    #     volume = dicom_mr.get_volume()
    #     assert True

    def test_ref_frame(self):
        orientation = [0, 1, 0, 0, 0, -1]
        origin = [82.615730762482, -143.29116344452, 153.0847454071]
        origin_last_slice = [
            -92.384269237518, -143.29116344452, 153.0847454071
        ]
        pixel_spacing = [0.5, 0.5]
        shape = [256, 256, 176]
        ref_frame = ReferenceFrame(origin=origin,
                                   origin_last_slice=origin_last_slice,
                                   orientation=orientation,
                                   pixel_spacing=pixel_spacing,
                                   shape=shape)
        assert np.equal(ref_frame.vx_to_mm([0, 0, 0]), np.array(origin)).all()
        assert np.equal(ref_frame.vx_to_mm([0, 0, shape[-1] - 1]),
                        np.array(origin_last_slice)).all()

        assert np.equal(ref_frame.mm_to_vx(origin), np.array([0, 0, 0])).all()

        assert np.equal(ref_frame.mm_to_vx(origin_last_slice),
                        np.array([0, 0, shape[-1] - 1])).all()

        bb_vx = np.array([128, 64, 80, 172, 128, 150], dtype=float)
        bb_mm = np.zeros_like(bb_vx)
        bb_mm[:3] = ref_frame.vx_to_mm(bb_vx[:3])
        bb_mm[3:] = ref_frame.vx_to_mm(bb_vx[3:])
        new_resampling = np.array([1.5, 1, 0.5])
        new_ref_frame = ref_frame.get_new_reference_frame(
            bb_mm, new_resampling)
        assert (new_ref_frame.vx_to_mm([0, 0, 0]) == bb_mm[:3]).all()
        assert np.sqrt(
            np.sum((new_ref_frame.voxel_spacing - new_resampling)**2)) < 0.01


if __name__ == '__main__':
    unittest.main()
