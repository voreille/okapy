import numpy as np
from scipy.ndimage import geometric_transform
from matplotlib.image import imread
import matplotlib.pyplot as plt


def cart2pol(x, y):
    r = np.sqrt(x**2 + y**2)
    theta = np.atan2(y/x)
    return r, theta

def pol2cart(output_coord):
    x = output_coord[0]*np.cos(output_coord[1])+64
    y = output_coord[0]*np.sin(output_coord[1])+64
    return x, y


def main():

    x = np.linspace(-1, 1, 128)
    y = np.linspace(-1, 1, 128)
    f = lambda x1, x2: np.sqrt(x1**2 + x2**2)
    x, y = np.meshgrid(x,y)
    im = f(x,y)
    plt.imshow(im)
    plt.show()
#    r, theta = cart2pol(x,y)
    im_sh = geometric_transform(im, pol2cart)
    plt.imshow(im_sh)
    plt.show()






if __name__ == '__main__':
    main()
