import SimpleITK as sitk
import pathlib as pl

from numpy.testing._private.parameterized import param

import copy

def apply_N4BiasFieldCorrection(image, n_fitting_levels=4, n_iterations=5, mask=None, shrink_factor=None):
    image = copy.copy(image)
    if isinstance(mask, sitk.Image):
        mask_image = mask
    else:
        print("No mask image specified, trying to guess")
        mask_image = sitk.OtsuThreshold(image, 0, 1, 10)
    if shrink_factor:
        image = sitk.Shrink(image, [shrink_factor] * image.GetDimension())
        image = sitk.Shrink(mask_image, [shrink_factor] * image.GetDimension())
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations([n_iterations]*n_fitting_levels)
    image_corrected = corrector.Execute(image, mask_image)
    return image_corrected, mask_image


p_img = pl.Path('/home/daniel/mnt/ultrafast_home/data/TCIA-GBM/BIDS/original/sub-TCGA-02-0006/ses-1996-08-23/T2w/sub-TCGA-02-0006_ses-1996-08-23_original_T2w.nii')
inputImage = sitk.ReadImage(p_img.as_posix())
inputImage = sitk.Cast(inputImage, sitk.sitkFloat32)

p_out_base = pl.Path('/home/daniel/Downloads/sitk-tests')
p_out_base.mkdir(exist_ok=True, parents=True)

p_img_in = p_out_base.joinpath('img_in.nii')
p_img_corrected = p_out_base.joinpath('img_corrected.nii')
p_mask_out = p_out_base.joinpath('mask.nii')


img_cor, mask = apply_N4BiasFieldCorrection(inputImage)

sitk.WriteImage(inputImage, p_img_in.as_posix())
sitk.WriteImage(mask, p_mask_out.as_posix())
sitk.WriteImage(img_cor, p_img_corrected.as_posix())


# if isinstance(image, str) or isinstance(image, pl.Path):
#     p_image = pl.Path(image)
#     image_in = sitk.ReadImage(p_image.as_posix())
# elif isinstance(image, sitk.Image):
#     image_in = image



label_tissue_map_bratumia = {
    1 : 'CSF',
    2 : 'GrayMatter',
    3 : 'WhiteMatter',
    4 : 'Necrosis',
    5 : 'Edema',
    6 : 'NonEnhancingTumor',
    7 : 'EnhancingTumor',
    #--- extended, see bratumia merge map
    8 : 'T1c',
    9 : 'T2'
}


label_tissue_map_manual = {
    8 : 'T1c',
    9 : 'T2'
}

bratumia_merge_map_T1c = {
    8 : [4, 6, 7],
    0 : [1, 2, 3, 5]
}

bratumia_merge_map_T2 = {
    9 : [4, 5, 6, 7],
    0 : [1, 2, 3]
}

