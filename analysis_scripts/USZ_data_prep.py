
import numpy as np
import pathlib as pl

from okapy.dicomconverter.dicom_walker import DicomWalker
from okapy.dicomconverter.volume import ReferenceFrame
from okapy.dicomconverter.dicom_file import DicomFileMR
from okapy.dicomconverter.converter import Converter


base_path   = pl.Path('/Users/dabler/Documents/repositories/okapy/test_data/lymphagitis')
input_path  = base_path.joinpath('DICOM')

#base_path   = pl.Path('/Users/dabler/Documents/repositories/okapy/test_data/TCGA-GBM/')
#input_path  = base_path.joinpath('TCGA-02-0046').joinpath('1.3.6.1.4.1.14519.5.2.1.1706.4001.667069087302553300405296434177')

padding = 100
output_path = base_path.joinpath('out_GTV-V_pad-%i_res-1-1-1'%padding)
output_path.mkdir(parents=True, exist_ok=True)

converter = Converter(output_folder=output_path,
                      #list_labels=['GTV L', 'GTV N', 'GTV T'],
                      list_labels=['GTV T'],
                      resampling_spacing=(1, 1, 1),
                      padding=padding+1,
                      #padding='whole_image'
                      )

result = converter(input_path)

print(result)


import SimpleITK as sitk
import numpy as np

for path in output_path.iterdir():
    file_name, ext = path.name.split('.')[0], '.'.join(path.name.split('.')[1:])
    if path.is_file() and ext=='nii.gz':
        try:
            print("- Converting '%s' to npy"%(path))
            image = sitk.ReadImage(path.as_posix())
            image_nda = sitk.GetArrayFromImage(image)
            path_to_npy = path.parent.joinpath(file_name + '.npy')
            print("  -> '%s'"%path_to_npy)
            np.save(path_to_npy.as_posix(), image_nda)
        except Exception as E:
            print(E)


for path in output_path.iterdir():
    file_name, ext = path.name.split('.')[0], '.'.join(path.name.split('.')[1:])
    if path.is_file() and ext=='npy':
        try:
            print("- Reading '%s'"%(path))
            image_nda = np.load(path)
        except Exception as E:
            print(E)


#Comparison
p_nii = "/Users/dabler/Documents/repositories/okapy/test_data/lymphagitis/out_GTV-V_pad-100/PatientLC_61__GTV_T__RTSTRUCT__CT.nii.gz"
image_nii = sitk.ReadImage(p_nii)
array_nii = sitk.GetArrayFromImage(image_nii)
p_npy = "/Users/dabler/Documents/repositories/okapy/test_data/lymphagitis/out_GTV-V_pad-100/PatientLC_61__GTV_T__RTSTRUCT__CT.npy"
p_npy = "/Users/dabler/Documents/repositories/okapy/test_data/lymphagitis/out_GTV-V_pad-50_res-1-1-1/PatientLC_61__PT.npy"
array_npy = np.load(p_npy)

np.where(array_nii-array_npy!=0)

