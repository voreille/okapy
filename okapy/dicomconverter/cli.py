from operator import mul
import os

import click
import logging

from okapy.dicomconverter.converter import NiftiConverter

log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)
logger = logging.getLogger(__name__)


@click.command()
@click.argument('input_directory', type=click.Path(exists=True))
@click.option('-o', '--output_filepath', required=True, type=click.Path())
@click.option('-l',
              '--list_labels',
              default=None,
              type=click.STRING,
              multiple=True)
@click.option('-s', '--spacing', default=-1, type=click.FLOAT)
@click.option('-p', '--padding', default=-1, type=click.FLOAT)
@click.option('-p', '--padding', default=-1, type=click.FLOAT)
@click.option('-j', '--cores', default=None, type=click.INT)
def main(input_directory, output_filepath, list_labels, spacing, padding,
         cores):
    """
    Convert to dicom to the right format based on extension
    """
    logger.info('Loading Dicom')
    if padding == -1:
        padding = "whole_image"
    if list_labels is not None:
        list_labels = list(list_labels)

    if not os.path.exists(output_filepath):
        logger.info(f"Creating folder {output_filepath}")
        os.makedirs(output_filepath)

    converter = NiftiConverter(
        padding=padding,
        resampling_spacing=spacing,
        list_labels=list_labels,
        cores=cores,
    )
    _ = converter(input_directory, output_folder=output_filepath)
    logger.info("End of the conversion")


if __name__ == '__main__':
    main()
