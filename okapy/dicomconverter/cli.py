import click
import logging
import pandas as pd

from dicom_walker import DicomWalker


@click.command()
@click.argument('input_directory', type=click.Path(exists=True))
@click.option('-o', '--output_filepath', required=True, type=click.Path())
@click.option('-l', '--list_labels', default='GTV T', required=True,
              type=click.Path())
def main(input_directory, output_filepath, list_labels):
    """
    Convert to dicom to the right format based on extension
    """
    logger = logging.getLogger(__name__)
    logger.info('Loading Dicom')

    import ipdb; ipdb.set_trace()  # XXX BREAKPOINT
    walker = DicomWalker(input_directory, output_filepath,
                            list_labels=list_labels)
    walker.walk()
    walker.fill_images()
    walker.convert()




if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
