import SimpleITK as sitk
import pandas as pd
import math
import os



class meanfilter():
    def __init__(self,imagein,maskin,folderout=None,fileout=None,name=None):
        self.imagein = sitk.ReadImage(imagein)
        if maskin != None:
            maskin = sitk.ReadImage(maskin)
            castfilter = sitk.CastImageFilter()
            castfilter.SetOutputPixelType(pixelID=sitk.sitkUInt8)
            binaryimagetolabelmapfilter = sitk.BinaryImageToLabelMapFilter()
            labelmap = binaryimagetolabelmapfilter.Execute(castfilter.Execute(maskin))
            labelmaptolabelimagefilter = sitk.LabelMapToLabelImageFilter()
            self.mask = labelmaptolabelimagefilter.Execute(labelmap)
        if fileout != None:
            self.outputfile = fileout
        self.folderout = folderout
        self.name = name

    def update(self):
        labelstatisticsimagefilter = sitk.LabelStatisticsImageFilter()
        labelstatisticsimagefilter.Execute(self.imagein, self.mask)
        data = {"ID": [self.name],
                "Mean": [labelstatisticsimagefilter.GetMean(1)]} #,
               #"Std.": [math.sqrt(labelstatisticsimagefilter.GetVariance(1))]}
        df = pd.DataFrame(data=data)
        df.set_index("ID", inplace=True)
        df.to_csv(os.path.join(self.folderout,self.outputfile))
