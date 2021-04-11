import pathlib as pl
import pandas as pd
from okapy.featureextractor.featureextractor import OkapyExtractors
from okapy.dicomconverter.dicom_file import DicomFileMR
from okapy.dicomconverter.converter import ExtractorConverter
import SimpleITK as sitk
from shutil import rmtree

import numpy as np
import copy
import itertools
def merge_segmentation_labels(input_img, label_map={}):
    """
    label_map: {<target value 1>: [<val to change 1>, <val to change 2>, ...],
                <target value 2>: [...]}
    """
    img_np = sitk.GetArrayFromImage(input_img)
    out_img_np = copy.deepcopy(img_np)
    for target_label, source_label_list in label_map.items():
        out_img_np[np.isin(img_np, source_label_list)]=target_label # NOTE: we check label values in input image img_np,
                                                                    #       not output image out_img_np!
    out_img = sitk.GetImageFromArray(out_img_np)
    out_img.SetSpacing(input_img.GetSpacing())
    out_img.SetDirection(input_img.GetDirection())
    out_img.SetOrigin(input_img.GetOrigin())
    return out_img


def create_label_permutations(label_list):
    labels_combinations = []
    for n in range(1,len(label_list)+1):
        combinations = list(itertools.combinations(label_list,n))
        labels_combinations = labels_combinations + combinations
    return labels_combinations

def create_label_dict(label_permutation_list):
    label_dict={}
    for labelset in label_permutation_list:
        labelset_name = "ROI_"+"-".join(map(str,labelset))
        label_dict[labelset_name] = list(labelset)
    return label_dict

def generate_label_merge_map(label_list_orig, label_list_selected, roi_label_id=1, background_label_id=0):
    labels_to_suppress = list(set(label_list_orig).difference(set(label_list_selected)))
    label_merge_map = { roi_label_id : label_list_selected,
                        background_label_id : labels_to_suppress}
    return label_merge_map

def reorient_image(path_img, orientation='LPS', outpath=None, data_type=None):
    path_img = pl.Path(path_img)
    img_name, img_ext = path_img.name.split('.')

    img = sitk.ReadImage(path_img.as_posix())
    orientation_filter = sitk.DICOMOrientImageFilter()
    orientation_filter.SetDesiredCoordinateOrientation(orientation)
    img_reoriented = orientation_filter.Execute(img)

    if not outpath:
        outpath = path_img.parent.joinpath("%s_reoriented.%s"%(img_name, img_ext))
    else:
        outpath = pl.Path(outpath)
    outpath.parent.mkdir(exist_ok=True, parents=True)

    if data_type:
        img_reoriented = sitk.Cast(img_reoriented, data_type)
    sitk.WriteImage(img_reoriented, outpath.as_posix())
    return path_img

def convert_dcm_dir_nii_okapy(path_to_dcm_dir, path_to_nii, glob_str='*.dcm'):
    dcm_paths = [p.as_posix() for p in pl.Path(path_to_dcm_dir).glob(glob_str)]
    dcm = DicomFileMR(dicom_paths=dcm_paths)
    dcm_volume = dcm.get_volume()
    dcm_sitk = dcm_volume.sitk_image
    path_to_nii = pl.Path(path_to_nii)
    path_to_nii.parent.mkdir(exist_ok=True, parents=True)
    sitk.WriteImage(dcm_sitk, path_to_nii.as_posix())

def extract_features_roi(okapy_extractor, path_to_img, path_to_mask, attribute_dict={}):
    results_df = pd.DataFrame()
    path_to_img = pl.Path(path_to_img)
    path_to_mask = pl.Path(path_to_mask)
    result = okapy_extractor(path_to_img.as_posix(), path_to_mask.as_posix())
    for key, val in result.items():
        if "diagnostics" in key or ("glcm" in key
                                    and "original" not in key):
            continue
        attribute_dict_int = { "feature_name": key,
                           "feature_value": val}
        attribute_dict_int.update(attribute_dict)
        results_df = results_df.append(attribute_dict_int, ignore_index=True)
    return results_df

def extract_features_dcmdir_segmentation_pair(okapy_extractor, path_to_dcm_imgs, path_to_segmentation, tmp_dir,
                                                attribute_dict={}):
    path_to_dcm_imgs = pl.Path(path_to_dcm_imgs)
    path_to_segmentation = pl.Path(path_to_segmentation)
    tmp_dir = pl.Path(tmp_dir)
    if tmp_dir.exists():
        rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # == Fix up segmentation
    seg_path = tmp_dir.joinpath('label_ROIs_LPS.nii')
    reorient_image(seg_path_orig, 'LPS', outpath=seg_path, data_type=sitk.sitkUInt8)

    # == convert dicom to nii using okapy
    dcm_img_path = tmp_dir.joinpath('dcm_image_okapy.nii.gz')
    convert_dcm_dir_nii_okapy(dcm_path, dcm_img_path)

    # == define ROI label combinations
    mask_path_dict = create_masks(seg_path)
    results_df = pd.DataFrame()
    for roi_name, path_to_mask in mask_path_dict.items():
        attribute_dict_int = {
            "VOI": roi_name
        }
        attribute_dict_int.update(attribute_dict)
        results_df_tmp = extract_features_roi(okapy_extractor, dcm_img_path, path_to_mask, attribute_dict=attribute_dict_int)
        results_df = results_df.append(results_df_tmp, ignore_index=True)

    if tmp_dir.exists():
        rmtree(tmp_dir)
    return results_df


def create_masks(path_to_mask):
    path_to_mask = pl.Path(path_to_mask)
    #- get unique labels
    mask = sitk.ReadImage(path_to_mask.as_posix())
    mask_np = sitk.GetArrayFromImage(mask)
    unique = np.unique(mask_np)
    label_list = list(unique[unique!=0])
    #- create label permutations
    label_permutation_list = create_label_permutations(label_list)
    label_dict = create_label_dict(label_permutation_list)
    # generate images for each label combination
    mask_path_dict = {}
    for roi_name, roi_labels in label_dict.items():
        label_merge_map = generate_label_merge_map(label_list, roi_labels)
        mask_selected = merge_segmentation_labels(mask, label_merge_map)
        mask_path_selected = path_to_mask.parent.joinpath('%s.nii'%roi_name)
        sitk.WriteImage(sitk.Cast(mask_selected, sitk.sitkUInt8), mask_path_selected.as_posix())
        mask_path_dict[roi_name] = mask_path_selected.as_posix()
    return mask_path_dict


#== PATHS
okapy_path = pl.Path('/Users/dabler/Documents/repositories/okapy')
data_base_path  = okapy_path.joinpath('test_data/MR-okapi-test')

data_tmp_path   = data_base_path.joinpath('tmp')
dcm_path = data_base_path.joinpath('dcm-image')
seg_path_orig = data_base_path.joinpath('label_ROIs.nii')

config_path = okapy_path.joinpath('analysis_scripts/gbm_test_config.yaml')
extr = OkapyExtractors(config_path.as_posix())

patient_id = 1111
modality = 'MR-T1'
attribute_dict = {
    "patient_id": patient_id,
    "modality": modality
}

# loop
# - modalities
# - resampling
# - normalization
# - labelmap smoothing

results_df = extract_features_dcmdir_segmentation_pair(extr, dcm_path, seg_path_orig, data_tmp_path,
                                                           attribute_dict=attribute_dict)

#
# # == Fix up segmentation
# seg_path_orig = data_base_path.joinpath('label_ROIs.nii')
#
# seg_path = data_tmp_path.joinpath('label_ROIs_LPS.nii')
# reorient_image(seg_path_orig, 'LPS', outpath=seg_path, data_type=sitk.sitkUInt8)
#
# # == convert dicom to nii using okapy
# dcm_path = data_base_path.joinpath('dcm-image')
# dcm_img_path = data_tmp_path.joinpath('dcm_image_okapy.nii.gz')
# convert_dcm_dir_nii_okapy(dcm_path, dcm_img_path)
# #== EXTRACTION
#
# patient_id = 1111
# modality = 'MR-T1'
#
# results_df = pd.DataFrame()
#
# config_path = okapy_path.joinpath('analysis_scripts/gbm_test_config.yaml')
# extr = OkapyExtractors(config_path.as_posix())
#
# for roi_name, path_to_mask in mask_path_dict.items():
#     attribute_dict = {
#                         "patient_id": patient_id,
#                         "modality": modality,
#                         "VOI": roi_name
#                       }
#     results_df_tmp = extract_features_roi(extr, dcm_img_path, path_to_mask, attribute_dict=attribute_dict)
#     results_df = results_df.append(results_df_tmp, ignore_index=True)

# reshape results
results_df.feature_value = results_df.feature_value.astype(float)
a = results_df.set_index(['patient_id', 'modality', 'VOI', 'feature_name']).unstack(['VOI'])
p_features = data_base_path.joinpath('features.xlsx')
a.to_excel(p_features.as_posix())











#pd.pivot_table(results_df, values=['feature_value'], columns=['feature_name'], index=['patient_id', 'modality', 'VOI'])


#
# # smoothing operations
# seg_path = data_base_path.joinpath('ROI_1-2-4.nii')
# seg = sitk.ReadImage(seg_path_selected.as_posix())
# # seg_cleaned = sitk.BinaryOpeningByReconstruction(seg, [10, 10, 1])
# # seg_cleaned = sitk.BinaryClosingByReconstruction(seg_cleaned, [10, 10, 1])
# seg_cleaned = sitk.BinaryMedian(seg, [1,1,1])
# seg_path_cleaned = data_base_path.joinpath('ROI_1-2-4_cleaned.nii')
# sitk.WriteImage(seg_cleaned, seg_path_cleaned.as_posix())



# ec = ExtractorConverter(config_path.as_posix())
# ec.__call__(data_base_path.as_posix())



# p_base  = pl.Path('/Users/dabler/Downloads/GBM-okapi-test/compilation/')
# p = p_base.joinpath('label_merged_LPS_smoothed_no-islands.nii')
# i = sitk.ReadImage(p.as_posix())
# sitk.WriteImage(sitk.Cast(i, sitk.sitkUInt8),p.as_posix())

#
#
# import pydicom
# import pydicom_seg
#
# dcm = pydicom.dcmread(p_base.joinpath('dcm_img').joinpath('sub-TCGA-02-0006_ses-1996-08-23_original-dcm_T1w_0000.dcm').as_posix())
# dcm = pydicom.dcmread(p_base.joinpath('seg_merged_LPS.dcm').as_posix(), force=True)
#
# reader = pydicom_seg.SegmentReader()
# result = reader.read(dcm)
#


#-> check ExtractorConverter for compilation of results

#
#
# import numpy as np
# img = sitk.ReadImage(img_path.as_posix())
# mask = sitk.ReadImage(seg_path.as_posix())
#
# mask_all = sitk.ReadImage("/Users/dabler/Downloads/GBM-okapi-test/registered/sub-TCGA-02-0006/ses-1996-08-23/brats/sub-TCGA-02-0006_ses-1996-08-23_registered_brats_reg-rigid_seg-manual.nii")
#
#
# a = sitk.Resample(img, mask)
# path_test = data_base_path.joinpath('test.nii')
# sitk.WriteImage(a, path_test.as_posix())
#
# shape_stats = sitk.LabelShapeStatisticsImageFilter()
# shape_stats.ComputeOrientedBoundingBoxOn()
# shape_stats.Execute(mask)
#
#
# def get_bounding_box_in_coords(mask, label_id):
#     shape_stats = sitk.LabelShapeStatisticsImageFilter()
#     shape_stats.ComputeOrientedBoundingBoxOn()
#     shape_stats.Execute(mask)
#     bbox = shape_stats.GetBoundingBox(label_id)
#     bbox_start = np.array(bbox[:3])
#     bbox_width = np.array(bbox[3:])
#     spacing = np.array(mask.GetSpacing())
#     origin  = np.array(mask.GetOrigin())
#     bbox_origin_coords = origin + bbox_start*spacing
#     bbox_width_coords  = bbox_width * spacing
#     bbox_end_coords = bbox_origin_coords + bbox_width_coords
#     bbox_coords = bbox_origin_coords.tolist() + bbox_end_coords.tolist()
#     return bbox_coords
#
# bbox = get_bounding_box_in_coords(mask, 1)





# #
# import yaml
#
# with open('/Users/dabler/Documents/repositories/okapy/analysis_scripts/gbm_test_config.yaml', 'r') as f:
#     params = yaml.safe_load(f)


