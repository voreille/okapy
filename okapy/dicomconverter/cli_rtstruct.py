import os
from pathlib import Path

import click
import logging
import pandas as pd

from okapy.dicomconverter.converter import Converter


@click.command()
@click.argument('input_directory', type=click.Path(exists=True))
@click.option('-o', '--output_filepath', required=True, type=click.Path())
@click.option('-l', '--list_labels', default=None, type=click.STRING)
@click.option('-e', '--extension', default='nii.gz', type=click.STRING)
@click.option('-j', '--cores', default=1, type=click.INT)
def main(input_directory, output_filepath, list_labels, extension, cores):
    """
    Convert to dicom to the right format based on extension
    """
    logger = logging.getLogger(__name__)
    logger.info('Loading Dicom')

    if not os.path.exists(output_filepath):
        os.makedirs(output_filepath)

    converter = Converter(output_filepath,
                          padding=-1,
                          resampling_spacing=-1,
                          list_labels=list_labels)
    converter(input_directory)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
