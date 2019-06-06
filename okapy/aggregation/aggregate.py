import SimpleITK as sitk
import pandas as pd
import numpy as np
import os


class aggreagate():
    def __init__(self,imagein=None, mask=None, listout=None):
        self.imagein = imagein
        self.mask = mask
        self.listout = listout


    def listfeaturesmap(self,relevant_path):
        listfiles = []
        for r, d, f in os.walk(relevant_path):
            for file in f:
                if '.nii.gz' in file and not ('ABS' in file) and 'ClusterProminence.nii.gz' in file:
                   listfiles.append(os.path.join(r, file))
        return listfiles

    def update(self,listimages,mask,listpixelout=None):
        Mask = sitk.ReadImage(mask)
        castfilter = sitk.CastImageFilter()
        castfilter.SetOutputPixelType(pixelID=sitk.sitkUInt8)
        Mask = castfilter.Execute(Mask)
        filter = sitk.BinaryErodeImageFilter()
        filter.SetKernelRadius(3)
        MaskErode = filter.Execute(Mask)
        ndaMask = sitk.GetArrayFromImage(MaskErode)
        idx = np.where(ndaMask.flatten() == 1)[0]
        VoxelFeatures = np.zeros(idx.shape)

        for idxImages in listimages:
            ResponseMap = sitk.ReadImage(idxImages)
            ndaResponseMap = sitk.GetArrayFromImage(ResponseMap)
            ValueResponseMap = np.take(ndaResponseMap, idx)
            VoxelFeatures = np.vstack((VoxelFeatures, ValueResponseMap))

        df = pd.DataFrame(VoxelFeatures.T)

        return df
