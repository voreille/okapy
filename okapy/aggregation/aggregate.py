"""
TODO: Maybe optimize the code
"""
import SimpleITK as sitk
import pandas as pd
import numpy as np
import re
import os
import sys


class aggreagate():
    def __init__(self, imagefolderin, mask, filemame="Patient", folderout="./"):
        try:
            mask = sitk.ReadImage(mask)
        except:
            print("An error occured while opening the mask image")

        castfilter = sitk.CastImageFilter()
        castfilter.SetOutputPixelType(pixelID=sitk.sitkUInt8)
        self.mask = castfilter.Execute(mask)

        if os.path.isdir(imagefolderin):
            self.relevant_path = imagefolderin
        else:
            sys.exit("(In __init__) imagefolderin is not a folder's path")

        self.filename = filemame
        self. folderout = folderout

    def listfeaturesmap(self):
        listfiles = []
        for r, d, f in os.walk(self.relevant_path):
            for file in f:
                if '.nii.gz' in file and not ('ABS' in file):
                   listfiles.append(os.path.join(r, file))
        return listfiles

    def update(self, listimages=None):

        if listimages is None:
            sys.exit("(In update) A list of images path is mandatory!")

        ndaMask = sitk.GetArrayFromImage(self.mask)
        idx = np.where(ndaMask.flatten() == 1)[0]
        VoxelFeatures = np.zeros(idx.shape)

        header = []
        i = 0

        for idxImages in listimages:
            #id_patient="54_Besancon"
            m = re.search(self.filename+"_(.+?).nii.gz", idxImages)
            print(m.group(1))
            header.append(m.group(1))
            try:
                ResponseMap = sitk.ReadImage(idxImages)
            except:
                print("(In update) An error occured while opening the image")
            if not(self.mask.GetOrigin() == ResponseMap.GetOrigin()):
                sys.exit("the origin does not correspond")
            if not(self.mask.GetSpacing() == ResponseMap.GetSpacing()):
                sys.exit("the spacing does not correspond")
            if not(self.mask.GetSize() == ResponseMap.GetSize()):
                sys.exit("the size does not correspond")

            ndaResponseMap = sitk.GetArrayFromImage(ResponseMap)
            ValueResponseMap = np.take(ndaResponseMap, idx)
            VoxelFeatures = np.vstack((VoxelFeatures, ValueResponseMap))
            i = i + 1

        df = pd.DataFrame(np.delete(VoxelFeatures,0,0).T, columns=header)
        df.to_csv(self.folderout+"/"+self.filename+".csv")

        return df
