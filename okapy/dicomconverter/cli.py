import click
import logging
import pandas as pd


@click.command()
@click.argument('input_directory', type=click.Path(exists=True), nargs=-1)
@click.option('-o', '--output_filepath', required=True, type=click.Path())
@click.option('-s', '--source_features', default='Pyradiomics', required=True,
              type=click.Path())
def main(input_directory, output_filepath, source_features):
    """
    Convert to dicom to the right format based on extension
    """
    logger = logging.getLogger(__name__)
    logger.info('Loading Dicom')

    slices = [pdcm.read_file(dcm) for dcm in patient_files.ct_files]
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    image_ct = get_hounsfield(slices)



if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)



    main()
