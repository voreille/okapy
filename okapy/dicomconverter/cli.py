import click
import logging
import pandas as pd

from dicom_walker import DicomWalker


@click.command()
@click.argument('input_directory', type=click.Path(exists=True), nargs=-1)
@click.option('-o', '--output_filepath', required=True, type=click.Path())
def main(input_directory, output_filepath):
    """
    Convert to dicom to the right format based on extension
    """
    logger = logging.getLogger(__name__)
    logger.info('Loading Dicom')

    walker = DicomWalker(input_directory, output_filepath,
                            list_labels=['GTV L', 'GTV T'])
    walker.walk()
    walker.fill_images()
    walker.convert()




if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)



    main()
