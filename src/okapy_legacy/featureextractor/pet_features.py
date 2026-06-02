import numpy as np
import SimpleITK as sitk

pet_features_dict = dict()


def pet_features(f, name=None):
    if name is None:
        name = "PET_feature_num_" + str(len(pet_features_dict.keys()))
    pet_features_dict[name] = f
    return f


def features_mtv(np_image,
                 np_mask,
                 voxel_volume=1,
                 threshold=0.4,
                 relative=True):
    positions = np.where(np_mask != 0)
    if relative:
        t = threshold * np.max(np_image[positions])
    else:
        t = threshold
    return np.sum(np_image[positions] > t) * voxel_volume


def suv_peak():
    pass


def tlg():
    pass