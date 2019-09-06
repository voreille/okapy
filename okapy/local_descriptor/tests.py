import unittest

import numpy as np
import matplotlib.pyplot as plt
#from click.testing import CliRunner
from scipy.ndimage import geometric_transform

from create_data import create_synthetic_dataset
from detector import (generate_dog_space, generate_scale_space,
                     get_candidate_keypoints, discarding_low_contrast,
                      apply_gaussian_blur3D, keypoint_interpolation)
from sh_utils import get_harmonics
from descriptors import sph2cart


def place_template(im, t, position):
    nx, ny, nz = t.shape
    indx = np.s_[-nx//2+1 +position[0]:nx//2+1+position[0]]
    indy = np.s_[-ny//2+1 +position[1]:ny//2+1+position[1]]
    indz = np.s_[-nz//2+1 +position[2]:nz//2+1+position[2]]
    im[indx, indy, indz] = t
    return im


def get_gaussian_kernel(ksize):
    sigma = ksize/6-1
    x = np.arange(-ksize//2+1, ksize//2+1)
    x, y, z = np.meshgrid(x,x,x)
    g = np.exp(-((x**2+y**2+z**2)/(2.0*sigma**2)))
    return g/g.sum()


class TestSIFT(unittest.TestCase):
    """Tests for `okapy` package."""
#
#    def test_dog_octave(self):
#        """Test the DicomWalker class"""
#        synthetic_data = create_synthetic_dataset(n_samples=1, positions=[
#        (24,24,24), (8,24,8), (3,24,3)])
#        im = synthetic_data[0][0,:,:,:,0]
#        scale_space = generate_scale_space(im, sigma_in=0.5, sigma_min=0.8)
#        dog_space = generate_dog_space(scale_space)
#        plt.subplot(311)
#        plt.imshow(im[:,24,:])
#        plt.subplot(312)
#        plt.imshow(scale_space[1][1][:,24,:])
#        plt.subplot(313)
#        plt.imshow(dog_space[1][1,:,24,:])
#        plt.show()
#
#    def test_get_candidate_kp(self):
#        synthetic_data = create_synthetic_dataset(n_samples=1, positions=[
#        (24,24,24), (8,24,8), (3,24,3)])
#        im = synthetic_data[0][0,:,:,:,0]
#        im = apply_gaussian_blur3D(im, 2.0)
#
#        scale_space = generate_scale_space(im, sigma_in=0.5, sigma_min=0.8)
#        dog_space = generate_dog_space(scale_space)
#        kp = get_candidate_keypoints(dog_space)
#        kp = discarding_low_contrast(dog_space, kp, c_dog=0.015)
#        true_candidates = keypoint_interpolation(kp, dog_space, [0.5, 1, 2, 4, 8])
#        print(true_candidates)
#        print(kp)

    def test_extract_patch_cat2sph(self):
        im = np.zeros((32,32,32))
        degreeMax = 3
        ksize = 7
        harmonics = np.reshape(get_harmonics(degreeMax, ksize),
                               (ksize, ksize, ksize, (degreeMax+1)**2))


        im = place_template(im, np.real(harmonics[..., 2]), (24, 4, 24))
        plt.imshow(im[:,:,24])
        plt.show()
        output_shape=(5,16,16)
        patch_sh = geometric_transform(im, sph2cart,
                                       output_shape=output_shape,
                                       extra_keywords={
                                           'output_shape': output_shape,
                                           'delta_px': 1,
                                           'scale': 1,
                                           'origin': (24, 4, 24)
                                       })
        print('shape of output:', patch_sh.shape)
        plt.imshow(patch_sh[:,0,:])
        plt.show()






if __name__ == '__main__':
    unittest.main()
