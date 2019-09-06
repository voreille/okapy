import sys

import numpy as np
import scipy
from sklearn import preprocessing


def transform_matrix_offset_center_fixed(matrix, x, y, z):
    # Based on keras implementation that is wrong. It should be - 0.5
    o_x = float(x) / 2 - 0.5
    o_y = float(y) / 2 - 0.5
    o_z = float(z) / 2 - 0.5
    offset_matrix = np.array([[1, 0, 0, o_x],
                              [0, 1, 0, o_y],
                              [0, 0, 1, o_z],
                              [0, 0, 0, 1]])
    reset_matrix = np.array([[1, 0, 0, -o_x],
                             [0, 1, 0, -o_y],
                             [0, 0, 1, -o_z],
                             [0, 0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix


def apply_affine_transform_fixed(x, theta_xyz=(0,0,0), tx=0, ty=0, tz=0, shear_xy=0, shear_xz=0, shear_yz=0,
                           zx=1, zy=1, zz=1, row_axis=0, col_axis=1, z_axis=2,
                           channel_axis=3, fill_mode='nearest', cval=0., order=1):
    """Applies an affine transformation specified by the parameters given.

    # Arguments
        x: 3D numpy array, single image.
        theta_xyz: rotation angles
        theta: Azimutal rotation angle in degrees.
        phi: Polar rotation angle in degrees.
        tx: Width shift.
        ty: Heigh shift.
        tz: depth shift.
        shear_xy: Shear angle in degrees on the xy plane.
        shear_xz: Shear angle in degrees on the xz plane.
        zx: Zoom in x direction.
        zy: Zoom in y direction
        zz: Zoom in z direction
        row_axis: Index of axis for rows in the input image.
        col_axis: Index of axis for columns in the input image.
        z_axis: Index of axis for depth in the input image.
        channel_axis: Index of axis for channels in the input image.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.

    # Returns
        The transformed version of the input.
    """
    theta_x = theta_xyz[0]
    theta_y = theta_xyz[1]
    theta_z = theta_xyz[2]
    if scipy is None:
        raise ImportError('Image transformations require SciPy. '
                          'Install SciPy.')
    transform_matrix = None
    if theta_x != 0:
        theta = np.deg2rad(theta_x)
        rotation_matrix = np.array([[1, 0, 0, 0],
                                    [0, np.cos(theta), -np.sin(theta), 0],
                                    [0, np.sin(theta), np.cos(theta), 0],
                                    [0, 0, 0, 1]])
        transform_matrix = rotation_matrix
        #import pdb;pdb.set_trace()

    if theta_y != 0:

        theta = np.deg2rad(theta_y)
        rotation_matrix = np.asarray([[np.cos(theta), 0,  np.sin(theta), 0],
                                    [0, 1, 0, 0],
                                    [-np.sin(theta), 0, np.cos(theta), 0],
                                    [0, 0, 0, 1]])
        if transform_matrix is None:
            transform_matrix = rotation_matrix
        else:
            transform_matrix = np.dot(transform_matrix, rotation_matrix)

    if theta_z != 0:
        theta = np.deg2rad(theta_z)
        rotation_matrix = np.asarray([[np.cos(theta), -np.sin(theta), 0, 0],
                                    [np.sin(theta), np.cos(theta), 0, 0],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]])
        if transform_matrix is None:
            transform_matrix = rotation_matrix
        else:
            transform_matrix = np.dot(transform_matrix, rotation_matrix)
        #import pdb;pdb.set_trace()
    if tx != 0 or ty != 0 or tz != 0:
        shift_matrix = np.array([[1, 0, 0, tx],
                                 [0, 1, 0, ty],
                                 [0, 0, 1, tz],
                                 [0, 0, 0, 1]])
        if transform_matrix is None:
            transform_matrix = shift_matrix
        else:
            transform_matrix = np.dot(transform_matrix, shift_matrix)

    if shear_xy != 0:
        shear = np.deg2rad(shear_xy)
        shear_matrix = np.array([[1, -np.sin(shear), 0, 0],
                                 [0, np.cos(shear), 0, 0],
                                 [0, 0, 1, 0],
                                 [0, 0, 0, 1]])
        if transform_matrix is None:
            transform_matrix = shear_matrix
        else:
            transform_matrix = np.dot(transform_matrix, shear_matrix)

    if shear_xz != 0:
        shear = np.deg2rad(shear_xz)
        shear_matrix = np.array([[1, 0, -np.sin(shear), 0],
                                 [0, 1, 0, 0],
                                 [0, 0, np.cos(shear), 0],
                                 [0, 0, 0, 1]])
        if transform_matrix is None:
            transform_matrix = shear_matrix
        else:
            transform_matrix = np.dot(transform_matrix, shear_matrix)

    if shear_yz != 0:
        shear = np.deg2rad(shear_yz)
        shear_matrix = np.array([[1, 0, 0, 0],
                                 [0, 1, -np.sin(shear), 0],
                                 [0, 0, np.cos(shear), 0],
                                 [0, 0, 0, 1]])
        if transform_matrix is None:
            transform_matrix = shear_matrix
        else:
            transform_matrix = np.dot(transform_matrix, shear_matrix)

    if zx != 1 or zy != 1 or zz != 1:
        zoom_matrix = np.array([[zx, 0, 0, 0],
                                [0, zy, 0, 0],
                                [0, 0, zz, 0],
                                [0, 0, 0, 1]])
        if transform_matrix is None:
            transform_matrix = zoom_matrix
        else:
            transform_matrix = np.dot(transform_matrix, zoom_matrix)

    if transform_matrix is not None:
        h, w, d = x.shape[row_axis], x.shape[col_axis], x.shape[z_axis]
        #import pdb;pdb.set_trace()
        transform_matrix = transform_matrix_offset_center_fixed(
            transform_matrix, h, w, d)
        final_affine_matrix = transform_matrix[:3, :3]
        final_offset = transform_matrix[:3, 3]

        x = scipy.ndimage.interpolation.affine_transform(
                x,
                final_affine_matrix,
                final_offset,
                order=order,
                mode=fill_mode,
                cval=cval)

    return x


def copy_template(cube,t,pos):
    cube_size = cube.shape[0]
    # Copy the template t at position pos in cube
    margin = int(t.shape[0]/2) # if the position is on the side of the cube with this margin, the template won't be copied entirely or it's outside the cube
    x_out1 = max(0,margin-pos[0])
    x_out2 = max(0,pos[0]-(cube_size-margin-1))
    y_out1 = max(0,margin-pos[1])
    y_out2 = max(0,pos[1]-(cube_size-margin-1))
    z_out1 = max(0,margin-pos[2])
    z_out2 = max(0,pos[2]-(cube_size-margin-1))
    cube[max(0,pos[0]-margin):min(cube_size,pos[0]+margin+1),max(0,pos[1]-margin):min(cube_size,pos[1]+margin+1),max(0,pos[2]-margin):min(cube_size,pos[2]+margin+1)] = t[x_out1:t.shape[0]-x_out2,y_out1:t.shape[1]-y_out2,z_out1:t.shape[2]-z_out2]
    return cube



def create_synthetic_dataset(n_samples=500,cube_size=32,template_size=7,density_min=.1,density_max=.5,proportions=[0.3,0.7],
                             positions=None):
    '''
    Creates a basic 3D texture synthetic dataset.
    Returns the volumes X and labels y.
    n_samples: number of samples per class (default 500)
    cube_size: size of the cubes (default 32)
    template_size: size of the template rotated and pasted in the volumes (default 7)
    density_min: minimum density of patterns (default 0.1)
    density_max: maximum density of patterns (default 0.5)
    proportions: proportion of template 1 for classes (the proportion of template 2 is 1-p) (default= [0.3,0.7])
    '''
    np.random.seed(seed=0)
    # number of classes
    n_class=2
    # Rotation range
    rot = 360
    range_rot = [0,rot]
    # Generate empty templates
    template = np.zeros((2,template_size,template_size,template_size))
    # Fill the templates
    # For now a simple line for t1
    template[0,int(template_size/2)-1:int(template_size/2)+1,int(template_size/2)-1:int(template_size/2)+1,:] = 1
    # And a cross for t2
    template[1,int(template_size/2)-1:int(template_size/2)+1,int(template_size/2)-1:int(template_size/2)+1,int(template_size/4):int(3*template_size/4)+1] = 1
    template[1,int(template_size/2)-1:int(template_size/2)+1,int(template_size/4):int(3*template_size/4),int(template_size/2)-1:int(template_size/2)+1] = 1
    # Initialize dataset lists
    X = []
    y = []

    for c in range(n_class):
        for s in range(n_samples):
            # Generate an empty 64x64x64 cube
            cube = np.zeros((cube_size,cube_size,cube_size))
            # Generate random density
            density = np.random.uniform(density_min, density_max)
            # Number of patterns in volume based on the density
            n_templates = int((cube_size**3)/(template_size**3)*density)
            # Crop size after rotation:
            crop_size = int(template_size*np.sqrt(3))
            # place the rotated patterns in the cube
            if positions is not None:
                for position in positions:
                    position = np.asarray(position)
                    # random position
                    # is it template 1 or 2:
                    template_type = np.random.choice(2, p=[proportions[c],1-proportions[c]])
                    # Rotate the template 1 or 2
                    random_angles = [np.random.uniform(range_rot[0], range_rot[1]) for i in range(3)]
                    rotated_template = apply_affine_transform_fixed(template[template_type],random_angles)
                    # copy the rotated template in the cube
                    cube = copy_template(cube,rotated_template,position)

            else:
                for t in range(n_templates):
                    # random position
                    position = np.array([np.random.choice(cube_size),np.random.choice(cube_size),np.random.choice(cube_size)])
                    # is it template 1 or 2:
                    template_type = np.random.choice(2, p=[proportions[c],1-proportions[c]])
                    # Rotate the template 1 or 2
                    random_angles = [np.random.uniform(range_rot[0], range_rot[1]) for i in range(3)]
                    rotated_template = apply_affine_transform_fixed(template[template_type],random_angles)
                    # copy the rotated template in the cube
                    cube = copy_template(cube,rotated_template,position)
            X.append(cube)
            y.append(c)
    X = np.expand_dims(np.asarray(X),axis=-1)
    y = np.asarray(y)
    le = preprocessing.LabelEncoder()
    le.fit(np.unique(y))
    y = le.transform(y)
    return X,y
