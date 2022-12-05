from okapy.dicomconverter.converter import ExtractorConverter


# okapy defaults - pyradiomics
params = "/data/ownCloud/Projects/IMAGINE/FeaturePresets/PETCT_pyradiomics.yaml"
folder = "/data/Datasets/Sample/Clark Kent - CLARKKENT"

# imagine defaults - pyradiomics
# params = "/data/imagine-data/feature-preset-samples/MR_all.yaml"
# folder = "/data/imagine-data/brats-data/00834-568"

# MRI imagine
# params = "/data/ownCloud/Projects/IMAGINE/FeaturePresets/qi_config_leomed.yaml"
# folder = "/data/Datasets/BRATS-MGMT-test/00008-6/"

converter = ExtractorConverter.from_params(params)

result = converter(folder, labels=None)

print(result)
