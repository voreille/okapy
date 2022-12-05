import os
import shutil
from okapy.dicomconverter.converter import ExtractorConverter

base_path = "/data/Datasets/okapy-conversion-test"

images_path = f"{base_path}/DICOM/DICOM/"
output_path = f"{base_path}/output"
params_path = f"{base_path}/params.yml"

# Delete previous output
shutil.rmtree(output_path, ignore_errors=True)
os.mkdir(output_path)

converter = ExtractorConverter.from_params(params_path)

converter(images_path, output_folder=output_path, labels=["GTVn", "GTVt"])
