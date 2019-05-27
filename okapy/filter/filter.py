import SimpleITK as sitk
import csv
import math

class SobelFilter():
    def __init__(self,imagein,mask,imageout,fileout):
        self.imagein = sitk.ReadImage(imagein)

        maskin = sitk.ReadImage(mask)
        castfilter = sitk.CastImageFilter()
        castfilter.SetOutputPixelType(pixelID=sitk.sitkUInt8)
        self.mask = castfilter.Execute(maskin)
        self.imageout = imageout
        self.outputfile = fileout

    def update(self):

        sobelfilter = sitk.SobelEdgeDetectionImageFilter()
        maskfilter = sitk.MaskImageFilter()
        maskfilter.SetMaskingValue(0)
        maskedimage=maskfilter.Execute(self.imagein, self.mask)


        binaryimagetolabelmapfilter = sitk.BinaryImageToLabelMapFilter()
        binaryimagetolabelmapfilter.SetInputForegroundValue(1)

        labelmaptolabelimagefilter = sitk.LabelMapToLabelImageFilter()
        labeloutput = labelmaptolabelimagefilter.Execute(binaryimagetolabelmapfilter.Execute(self.mask))

        labelstatisticsimagefilter = sitk.LabelStatisticsImageFilter()
        labelstatisticsimagefilter.Execute(sobelfilter.Execute(maskedimage), labeloutput)
        csvData = [['First Order Statistics'],
                   ['Min', '=', labelstatisticsimagefilter.GetMinimum(1)],
                   ['Max', '=', labelstatisticsimagefilter.GetMaximum(1)],
                   ['Median', '=', labelstatisticsimagefilter.GetMedian(1)],
                   ['Mean', '=', labelstatisticsimagefilter.GetMean(1)],
                   ['Std.', '=', math.sqrt(labelstatisticsimagefilter.GetVariance(1))]]
        with open(self.outputfile, 'w') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerows(csvData)
        csvFile.close()

        sitk.WriteImage(sobelfilter.Execute(maskedimage), self.imageout)

