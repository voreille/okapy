from okapy.dicomconverter.converter import ExtractorConverter


# okapy defaults - pyradiomics
params = "/data/imagine-data/feature-presets-samples/PETCT_pyradiomics.yaml"
folder = "/data/Datasets/Sample/Clark Kent - CLARKKENT"

# imagine defaults - pyradiomics
# params = "/data/imagine-data/feature-preset-samples/MR_all.yaml"
# folder = "/data/imagine-data/brats-data/00834-568"

converter = ExtractorConverter.from_params(params)

result = converter(folder, labels=None)

print(result)
