import os
import yaml

import click
import logging

from okapy.dicomconverter.converter import Converter
from okapy.featureextractor.featureextractor import create_extractor

input_directory = "/home/val/Documents/image_to_process/Mario_20jan2021/patient18"
output_directory = "/home/val/Documents/image_to_process/Mario_20jan2021/processed/"
nii_directory = "/home/val/Documents/image_to_process/Mario_20jan2021/nii/"


@click.command()
@click.argument('input_directory',
                type=click.Path(exists=True),
                default=input_directory)
@click.option('-o',
              '--output_filepath',
              required=True,
              type=click.Path(),
              default=output_directory)
@click.option('-l', '--list_labels', default=None, type=click.STRING)
@click.option('-s', '--spacing', default=0.75, type=click.FLOAT)
@click.option('-p', '--padding', default=100, type=click.FLOAT)
def main(input_directory, output_filepath, list_labels, spacing, padding):
    """
    Convert to dicom to the right format based on extension
    """
    logger = logging.getLogger(__name__)
    logger.info('Loading Dicom')
    if padding == -1:
        padding = "whole_image"

    if not os.path.exists(output_filepath):
        os.makedirs(output_filepath)
    if not os.path.exists(nii_directory):
        os.makedirs(nii_directory)

    converter = Converter(nii_directory,
                          padding=padding,
                          resampling_spacing=spacing,
                          list_labels=list_labels,
                          extension='nii.gz')
    results = converter(input_directory)
    with open('/home/val/python_wkspce/okapy/parameters/param.yaml', 'r') as f:
        params = yaml.safe_load(f)

    modalities = set(
        [image.modality for study in results for image in study[0]])

    feature_extractor_dict = {
        modality: create_extractor(modality, params[modality])
        for modality in modalities
    }
    for study in results:
        images, masks = study[0], study[1]
        for im in images:
            for mask in masks:
                features = feature_extractor_dict[im.modality](im.path,
                                                               mask.path)

    print(results)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
