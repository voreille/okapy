import SimpleITK as sitk
import csv
import math

class SobelFilter():
    def __init__(self,imagein,mask=None,imageout=None, outputfile=None):
        self.imagein = sitk.ReadImage(imagein)
        self.mask = mask
        self.imageout = imageout
        self.outputfile = outputfile
        if mask!=None:
            maskin = sitk.ReadImage(mask)
            castfilter = sitk.CastImageFilter()
            castfilter.SetOutputPixelType(pixelID=sitk.sitkUInt8)
            self.mask = castfilter.Execute(maskin)
        if imageout!=None:
            self.imageout = imageout
        if outputfile!=None:
            self.outputfile = outputfile

    def update(self):

        sobelfilter = sitk.SobelEdgeDetectionImageFilter()
        if self.mask != None:

            maskfilter = sitk.MaskImageFilter()
            maskfilter.SetMaskingValue(0)
            maskedimage=maskfilter.Execute(self.imagein, self.mask)
            imageout = sobelfilter.Execute(maskedimage)
        else:
            imageout= sobelfilter.Execute(self.imagein)

        if self.outputfile!=None:
            if self.mask!=None:
                binaryimagetolabelmapfilter = sitk.BinaryImageToLabelMapFilter()
                binaryimagetolabelmapfilter.SetInputForegroundValue(1)

                labelmaptolabelimagefilter = sitk.LabelMapToLabelImageFilter()
                labeloutput = labelmaptolabelimagefilter.Execute(binaryimagetolabelmapfilter.Execute(self.mask))

                statisticsimagefilter = sitk.LabelStatisticsImageFilter()
                statisticsimagefilter.Execute(imageout, labeloutput)
            else:
                statisticsimagefilter = sitk.StatisticsImageFilter()
                statisticsimagefilter.Execute(imageout)

            csvData = [['First Order Statistics'],
                    ['Min', '=', statisticsimagefilter.GetMinimum(1)],
                    ['Max', '=', statisticsimagefilter.GetMaximum(1)],
                    ['Median', '=', statisticsimagefilter.GetMedian(1)],
                    ['Mean', '=', statisticsimagefilter.GetMean(1)],
                    ['Std.', '=', math.sqrt(statisticsimagefilter.GetVariance(1))]]
            with open(self.outputfile, 'w') as csvFile:
                writer = csv.writer(csvFile)
                writer.writerows(csvData)
            csvFile.close()

        if self.imageout!=None:
            sitk.WriteImage(imageout, self.imageout)

        return imageout


class LaplacianOfGaussianFilter():
    def __init__(self,imagein,mask=None, imageout=None, outputfile=None, sigma=11/3):
        self.imagein = sitk.ReadImage(imagein)
        self.mask = mask
        self.imageout = imageout
        self.outputfile = outputfile
        if mask!=None:
            maskin = sitk.ReadImage(mask)
            castfilter = sitk.CastImageFilter()
            castfilter.SetOutputPixelType(pixelID=sitk.sitkUInt8)
            self.mask = castfilter.Execute(maskin)
        if imageout!=None:
            self.imageout = imageout
        if outputfile!=None:
            self.outputfile = outputfile
        self.sigma=sigma

    def update(self):
        logfilter = sitk.LaplacianRecursiveGaussianImageFilter()
        logfilter.SetSigma(self.sigma)
        if self.mask != None:

            maskfilter = sitk.MaskImageFilter()
            maskfilter.SetMaskingValue(0)
            maskedimage = maskfilter.Execute(self.imagein, self.mask)
            imageout = logfilter.Execute(maskedimage)
        else:
            imageout = logfilter.Execute(self.imagein)

        if self.outputfile != None:
            if self.mask != None:
                binaryimagetolabelmapfilter = sitk.BinaryImageToLabelMapFilter()
                binaryimagetolabelmapfilter.SetInputForegroundValue(1)

                labelmaptolabelimagefilter = sitk.LabelMapToLabelImageFilter()
                labeloutput = labelmaptolabelimagefilter.Execute(binaryimagetolabelmapfilter.Execute(self.mask))

                statisticsimagefilter = sitk.LabelStatisticsImageFilter()
                statisticsimagefilter.Execute(imageout, labeloutput)
            else:
                statisticsimagefilter = sitk.StatisticsImageFilter()
                statisticsimagefilter.Execute(imageout)

            csvData = [['First Order Statistics'],
                       ['Min', '=', statisticsimagefilter.GetMinimum(1)],
                       ['Max', '=', statisticsimagefilter.GetMaximum(1)],
                       ['Median', '=', statisticsimagefilter.GetMedian(1)],
                       ['Mean', '=', statisticsimagefilter.GetMean(1)],
                       ['Std.', '=', math.sqrt(statisticsimagefilter.GetVariance(1))]]
            with open(self.outputfile, 'w') as csvFile:
                writer = csv.writer(csvFile)
                writer.writerows(csvData)
            csvFile.close()

        if self.imageout != None:
            sitk.WriteImage(imageout, self.imageout)

        return imageout
