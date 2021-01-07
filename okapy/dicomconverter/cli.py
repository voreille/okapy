import os

import click
import logging

from okapy.dicomconverter.converter import Converter


@click.command()
@click.argument('input_directory', type=click.Path(exists=True))
@click.option('-o', '--output_filepath', required=True, type=click.Path())
@click.option('-l', '--list_labels', default=None, type=click.STRING)
@click.option('-e', '--extension', default='nii.gz', type=click.STRING)
@click.option('-s', '--spacing', default=-1, type=click.FLOAT)
@click.option('-p', '--padding', default=-1, type=click.FLOAT)
def main(input_directory, output_filepath, list_labels, extension, spacing,
         padding):
    """
    Convert to dicom to the right format based on extension
    """
    logger = logging.getLogger(__name__)
    logger.info('Loading Dicom')
    if padding == -1:
        padding = "whole_image"

    if not os.path.exists(output_filepath):
        os.makedirs(output_filepath)

    converter = Converter(output_filepath,
                          padding=padding,
                          resampling_spacing=spacing,
                          list_labels=list_labels,
                          extension=extension)
    result = converter(input_directory)
    print(result)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
