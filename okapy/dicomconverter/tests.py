import click
import pandas as pd

from dicom_walker import DicomWalker


@click.command()
@click.argument('input_filepath', nargs=-1)
@click.option('-o', '--output_filepath', required=True, type=click.Path())
def main(input_filepath, output_filepath):
    """
    Test of the dicom walker
    """
    walker = DicomWalker(input_filepath[0])
    walker.walk()
    for k in walker.dicom_headers:
        print(k)

if __name__ == '__main__':
    main()
