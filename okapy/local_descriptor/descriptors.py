from math import pi

import numpy as np
from scipy.ndimage import geometric_transform
from sh_utils import get_harmonics_sph


def get_patch_sph_coord(image, scale, keypoint):
    """ Return a patch around keypoint in sph coordinate
    The Radius of the patch is 6*scale the number of
    """
    pass

def get_sh_scalar_product():
    pass

def sph2cart(C, output_shape=(7,48,27), delta_px=1, scale=1, origin=(0,0,0)):
    delta_phi = (output_shape[1])/2/pi
    delta_theta = (output_shape[2])/pi
    x = (C[0] * np.sin(C[2]*delta_theta) * np.cos(C[1]*delta_phi) *
         scale*delta_px + origin[0])
    y = (C[0] * np.sin(C[2]*delta_theta) * np.sin(C[1]*delta_phi) *
         scale*delta_px + origin[1])
    z = C[0] * np.cos(C[2]*delta_theta) * scale + origin[2]

    return (x, y, z)


def get_sph_patch(image, location=(0, 0, 0), scale=1.0, delta_px=1.0,
                  radius=6):

    output_shape=(radius+1,8*radius,4*radius+3),
    patch_sph = geometric_transform(image, sph2cart,
                                   output_shape=output_shape,
                                   extra_keywords={
                                       'output_shape': output_shape,
                                       'delta_px': delta_px,
                                       'scale': scale,
                                       'origin': location
                                   })
    return patch_sph





