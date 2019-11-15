import os

import click
import logging
import pandas as pd

from .dicom_walker import DicomWalker


@click.command()
@click.argument('input_directory', type=click.Path(exists=True))
@click.option('-o', '--output_filepath', required=True, type=click.Path())
@click.option('-l', '--list_labels', default=None, type=str)
@click.option('-r', '--resampling_px_spacing', default=None,
              type=(float, float, float), required=False)
def main(input_directory, output_filepath, list_labels, resampling_px_spacing):
    """
    Convert to dicom to the right format based on extension
    """
    logger = logging.getLogger(__name__)
    logger.info('Loading Dicom')

    if not os.path.exists(output_filepath):
        os.makedirs(output_filepath)

    walker = DicomWalker(input_directory, output_filepath,
                            list_labels=list_labels)
    walker.walk()
    walker.fill_images()
    walker.resample_images(resampling_px_spacing)
    walker.convert()




if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
