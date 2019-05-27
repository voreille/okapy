import SimpleITK as sitk
import csv


class StatisticsFilter():
    def __init__(self, imagein, mask, outputfile):
        self.imagein = sitk.ReadImage(imagein)

        maskin = sitk.ReadImage(mask)
        castfilter = sitk.CastImageFilter()
        castfilter.SetOutputPixelType(pixelID=sitk.sitkUInt8)
        self.mask = castfilter.Execute(maskin)

        self.outputfile = outputfile

    def update(self):
        print('test')
        binaryimagetolabelmapfilter = sitk.BinaryImageToLabelMapFilter()
        binaryimagetolabelmapfilter.SetInputForegroundValue(1)

        labelmaptolabelimagefilter = sitk.LabelMapToLabelImageFilter()
        labeloutput = labelmaptolabelimagefilter.Execute(binaryimagetolabelmapfilter.Execute(self.mask))

        labelstatisticsimagefilter = sitk.LabelStatisticsImageFilter()
        labelstatisticsimagefilter.Execute(self.imagein, labeloutput)
        csvData = [['First Order Statistics'],
                   ['Min', '=', labelstatisticsimagefilter.GetMinimum(1)],
                   ['Max', '=', labelstatisticsimagefilter.GetMaximum(1)],
                   ['Median', '=', labelstatisticsimagefilter.GetMedian(1)],
                   ['Mean', '=', labelstatisticsimagefilter.GetMean(1)],
                   ['Variance', '=', labelstatisticsimagefilter.GetVariance(1)]]
        with open(self.outputfile, 'w') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerows(csvData)

        csvFile.close()

