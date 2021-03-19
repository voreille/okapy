import unittest
from pathlib import Path

import pydicom as pdcm
import SimpleITK as sitk

from okapy.dicomconverter.utils import get_sitk_image, get_mask_file


class TestOkapy(unittest.TestCase):
    """Tests for `okapy` package."""
    def test_utils_get_mask_atlas(self):
        image_path = Path(
            "/mnt/nas2/data/Personal/Vincent/brain_mets/atlas_luis/ATLAS-BRAIN"
        )
        mask_path = Path(
            "/mnt/nas2/data/Personal/Vincent/brain_mets/atlas_luis/"
            "struct_set_2021-03-18_16-10-32.dcm")

        output_path_im = '/home/val/Documents/output_okapy/image_test2.nii.gz'
        output_path_mask = '/home/val/Documents/output_okapy/mask_test2.nii.gz'
        image_dcm_paths = [str(f.resolve()) for f in image_path.rglob("*")]

        mask_file = get_mask_file(str(mask_path.resolve()), image_dcm_paths)
        print(mask_file.labels)
        sitk_mask = mask_file.get_volume(mask_file.labels[2]).sitk_image
        sitk.WriteImage(sitk_mask, output_path_mask)

        slices = [pdcm.filereader.dcmread(dcm) for dcm in image_dcm_paths]
        sitk_image = get_sitk_image(slices)
        sitk.WriteImage(sitk_image, output_path_im)


if __name__ == '__main__':
    unittest.main()
