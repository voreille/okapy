import os
import logging

import click
import SimpleITK as sitk
from radiomics.imageoperations import resampleImage

from dicom_walker import DicomWalker


@click.command()
@click.argument('image_path', type=click.Path(exists=True))
@click.argument('mask_path', type=click.Path(exists=True))
@click.option('-o', '--output_filepath', required=False, type=click.Path())
@click.option('-s', '--out_px_spacing', required=False,
              type=(float, float, float), default=(0.75,0.75,0.75))
@click.option('-p', '--pad_distance', required=False, type=float,
              default=-1)
def main(image_path, mask_path, output_filepath, out_px_spacing, pad_distance,
         str_interp):
    """
    Convert to dicom to the right format based on extension
    """

    logger = logging.getLogger(__name__)
    logger.info('Resampling Dicom')

    extension_dict = {
        '.nii': 'NiftiImageIO',
        '.nrrd': 'NrrdImageIO'
    }
    image_name, image_extension = os.path.splitext(image_path)
    mask_name, mask_extension = os.path.splitext(mask_path)

    reader = sitk.ImageFileReader()
    reader.SetImageIO(extension_dict[image_extension])
    reader.SetFileName(image_path)
    image = reader.Execute()

    reader.SetImageIO(extension_dict[mask_extension])
    reader.SetFileName(mask_path)
    mask = reader.Execute()

    image, mask = resampleImage(image, mask,
                                resampledPixelSpacing=out_px_spacing,
                                padDistance=pad_distance,
                                interpolator=interpolator)





if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
