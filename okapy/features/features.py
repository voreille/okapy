from radiomics import glcm
import radiomics as rad
import SimpleITK as sitk
import six

class haralick():
    def __init__(self,imagein,mask,imageout=None,outputfile=None):
        self.imagein = sitk.ReadImage(imagein)
        self.mask = mask
        self.imageout = imageout
        self.outputfile = outputfile
        if mask != None:
            maskin = sitk.ReadImage(mask)
            castfilter = sitk.CastImageFilter()
            castfilter.SetOutputPixelType(pixelID=sitk.sitkUInt8)
            self.mask = castfilter.Execute(maskin)
        if imageout != None:
            self.imageout = imageout
        if outputfile != None:
            self.outputfile = outputfile

    def update(self):
        rad.setVerbosity(0)
        haralictexture = glcm.RadiomicsGLCM(self.imagein, self.mask, voxelBased=True, binWidth=1)
        haralictexture.disableAllFeatures()
        haralictexture.enableFeatureByName('JointEntropy')
        haralictexture.enableFeatureByName('Contrast')
        haralictexture.enableFeatureByName('JointEnergy')
        haralictexture.enableFeatureByName('Correlation')
        haralictexture.enableFeatureByName('SumSquares')
        haralictexture.enableFeatureByName('Idm')
        haralictexture.enableFeatureByName('SumAverage')
        haralictexture.enableFeatureByName('ClusterTendency')
        haralictexture.enableFeatureByName('SumEntropy')
        haralictexture.enableFeatureByName('DifferenceVariance')
        haralictexture.enableFeatureByName('DifferenceEntropy')
        haralictexture.enableFeatureByName('Imc1')
        haralictexture.enableFeatureByName('Imc2')
        #haralictexture.enableFeatureByName('MCC')
        haralictexture.execute()
        """
        for (key, val) in six.iteritems(haralictexture.featureValues):
            print("\t%s: %s" % (key, val))
        """

        for key, val in six.iteritems(haralictexture.featureValues):
            if isinstance(val, sitk.Image):  # Feature map
                sitk.WriteImage(val, key + '.nii.gz', True)
                print("Stored feature %s in %s" % (key, key + ".nii.gz"))
            else:  # Diagnostic information
                print("\t%s: %s" % (key, val))

