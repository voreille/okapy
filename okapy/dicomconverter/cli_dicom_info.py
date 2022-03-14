import click
import logging

import pandas as pd

from okapy.dicomconverter.dicom_walker import DicomWalker

log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)
logger = logging.getLogger(__name__)


@click.command()
@click.argument('input_directory', type=click.Path(exists=True))
@click.option('-o', '--output_filepath', required=True, type=click.Path())
@click.option('-t', '--tags', default=None, type=click.STRING, multiple=True)
def main(input_directory, output_filepath, tags):
    """
    Convert to dicom to the right format based on extension
    """
    logger.info('Loading Dicom')

    list_labels = list(tags)
    if len(list_labels) == 0:
        list_labels = None

    walker = DicomWalker(additional_dicom_tags=tags)
    studies = walker(input_dirpath=input_directory)
    df = pd.DataFrame()
    for s in studies:
        for v in s.volume_files:
            d = {
                key: str(item)
                for key, item in v.dicom_header.__dict__.items()
                if key != "additional_data"
            }
            d.update(v.dicom_header.additional_data)
            df = df.append(d, ignore_index=True)
    df.to_csv(output_filepath)
    logger.info("End")


if __name__ == '__main__':
    main()
