import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import convolve1d
from scipy.ndimage import zoom
from skimage.feature import peak_local_max


def compute_sigma_transition(s_p, s, sigma_min, delta_min, n_oct):
    return sigma_min/delta_min * np.sqrt(2**(2*s/n_oct)-2**(2*s_p/n_oct))

def gaussian_filter1D(sigma):
    '''
    Return 1D gaussian filter to use on the 3 axis, maybe we should use some
    kind of normalisation?
    '''
    size = 2*np.ceil(3*sigma)+1
    x = np.arange(-size//2+1, size//2+1)
    g = np.exp(-((x**2)/(2.0*sigma**2)))
    return g/g.sum()

def apply_gaussian_blur3D(im, sigma):
    kernel = gaussian_filter1D(sigma)
    out = convolve1d(im, kernel, axis=0)
    out = convolve1d(out, kernel, axis=1)
    out = convolve1d(out, kernel, axis=2)
    return out


def generate_octave(init_level, n_spo=3, sigma_min=0.5, delta_min=0.5):
    octave = [init_level]
    for i in range(n_spo+2):
        sigma = compute_sigma_transition(i, i+1, sigma_min, delta_min, n_spo)
        next_level = apply_gaussian_blur3D(octave[-1], sigma)
        octave.append(next_level)
    return octave


def generate_scale_space(im, num_octave=5, n_spo=3, sigma_min=0.8, sigma_in=0.5,
                         delta_min=0.5):
    scale_space = []
    im = zoom(im, 1/delta_min, order=1)
    im = apply_gaussian_blur3D(im, 1/delta_min*np.sqrt(
        sigma_min**2-sigma_in**2))
    for _ in range(num_octave):
        octave = generate_octave(im, n_spo=n_spo, sigma_min=sigma_min,
                                 delta_min=delta_min)
        scale_space.append(octave)
        im = octave[-3][::2, ::2, ::2]

    return scale_space


def compute_dog_octave(gaussian_octave):
    octave = []

    for i in range(1, len(gaussian_octave)):
        octave.append(gaussian_octave[i] - gaussian_octave[i-1])

    return np.concatenate([o[np.newaxis, :, :, :] for o in octave], axis=0)

def generate_dog_space(scale_space):
    pyr = []

    for gaussian_octave in scale_space:
        pyr.append(compute_dog_octave(gaussian_octave))

    return pyr

def get_candidate_keypoints(dog_space):
    '''
    Return a tuple of the indices of the extrema in the dog space
    (o, s, x, y, z), o for octave, s for scale and x,y,z you get it.
    The peak_local_max has been tested on 4D arrays and it gives the expected
    behaviour
    '''
    candidates = []

    for i, o in enumerate(dog_space):
        list_candidate = peak_local_max(o)
        candidates.extend(np.concatenate((np.ones((list_candidate.shape[0], 1))*i, list_candidate),axis=1).astype(int))
        list_candidate = peak_local_max(-o)
        candidates.extend(np.concatenate((np.ones((list_candidate.shape[0], 1))*i, list_candidate),axis=1).astype(int))

    return candidates

def discarding_low_contrast(dog_space, candidates, c_dog=0.015):
    '''
    Threshold the candidates, maybe read more carefully how to choose c_dog
    '''
    new_candidates = []
    threshold = c_dog*0.8
    for ind in candidates:
        if np.abs(dog_space[ind[0]][ind[1], ind[2], ind[3], ind[4]]) >= threshold:
            new_candidates.append(ind)

    return new_candidates


def quadratic_interpolation(dog_space, coordinate):
    octave = dog_space[coordinate[0]]
    s, x, y, z = (coordinate[1], coordinate[2], coordinate[3], coordinate[4])
    grad = np.asarray([octave[s+1, x, y, z] - octave[s-1, x, y, z],
                       octave[s, x+1, y, z] - octave[s, x-1, y, z],
                       octave[s, x, y+1, z] - octave[s, x, y-1, z],
                       octave[s, x, y, z+1] - octave[s, x, y, z-1]]) / 2

    h_ss = octave[s+1, x, y, z] + octave[s-1, x, y, z] - 2 * octave[s, x, y, z]
    h_xx = octave[s, x+1, y, z] + octave[s, x-1, y, z] - 2 * octave[s, x, y, z]
    h_yy = octave[s, x, y+1, z] + octave[s, x, y-1, z] - 2 * octave[s, x, y, z]
    h_zz = octave[s, x, y, z+1] + octave[s, x, y, z-1] - 2 * octave[s, x, y, z]

    h_sx = (octave[s+1, x+1, y, z] - octave[s+1, x-1, y, z] - octave[s-1, x+1, y, z] +
            octave[s-1, x-1, y, z]) / 4

    h_sy = (octave[s+1, x, y+1, z] - octave[s-1, x, y+1, z] - octave[s+1, x, y-1, z] +
            octave[s-1, x, y-1, z]) / 4

    h_sz = (octave[s+1, x, y, z+1] - octave[s-1, x, y, z+1] - octave[s+1, x, y, z-1] +
            octave[s-1, x, y, z-1]) / 4

    h_xy = (octave[s, x+1, y+1, z] - octave[s, x+1, y-1, z] - octave[s, x-1, y+1, z] +
            octave[s, x-1, y-1, z]) / 4

    h_xz = (octave[s, x+1, y, z+1] - octave[s, x+1, y, z-1] - octave[s, x-1, y, z+1] +
            octave[s, x-1, y, z-1]) / 4

    h_yz = (octave[s, x, y+1, z+1] - octave[s, x, y+1, z-1] - octave[s, x, y-1, z+1] +
            octave[s, x, y-1, z-1]) / 4

    hessian = np.asarray([[h_ss, h_sx, h_sy, h_sz],
                          [h_sx, h_xx, h_xy, h_xz],
                          [h_sy, h_xy, h_yy, h_yz],
                          [h_sz, h_xz, h_yz, h_zz]])

    inv_hessian = np.linalg.inv(hessian)
    alpha = np.matmul(-inv_hessian, grad)
    w = octave[s,x,y,z] - 0.5 *np.matmul(grad.T, np.matmul(inv_hessian, grad))
    return alpha, w


def keypoint_interpolation(candidates, dog_space, list_delta, delta_min=0.5,
                           sigma_min=0.8, n_spo=3, n_iter=5):
    new_candidates = list()
    for c in candidates:
        coord = c[1:]
        for _ in range(n_iter):
            delta_o = list_delta[c[0]]
            alpha, w = quadratic_interpolation(dog_space, c)
            absolute_c = np.asarray([(delta_o / delta_min * sigma_min  *
                            2 **((alpha[0]+coord[0])/n_spo)),
                        delta_o * (alpha[1] + coord[1]),
                        delta_o * (alpha[2] + coord[2]),
                        delta_o * (alpha[3] + coord[3])])
            coord = np.round(coord + alpha)
            if np.max(np.abs(alpha)) < 0.6:
                new_candidates.append(np.concatenate((np.asarray([c[0]]),
                                                      coord, absolute_c,
                                                      np.asarray([w]))))
                break

    return new_candidates


























