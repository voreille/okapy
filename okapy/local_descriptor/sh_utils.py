"""
Spherical Harmonics convolution core implementation
"""

import numpy as np
from math import pi
from scipy import special as sp
from sympy.physics.quantum.spin import Rotation
from sympy.physics.quantum.cg import CG


def get_harmonics(degreeMax, ksize):
    """
    Returns the spherical harmonics for all degrees (n) and orders (m) specified by the maximum degree (.N)
    output: complex tensor of shape (ksize**3,(N+1)**2)
    The center of the filters is set to 0 for m==0 and n!=0 for the Hermitian symmetry
    """
    _,theta,phi = getSphCoordArrays(ksize)
    harmonics = []
    for n in range(degreeMax+1):
        P_legendre = legendre(n,np.cos(theta))
        for m in range(-n,n+1):
            # get the spherical harmonics (without radial profile)
            sh = spherical_harmonics(m,n,P_legendre,phi)
            # Reshape for multiplication with radial profile r
            harmonics.append(sh)

    return np.stack(harmonics,axis=-1)


def get_harmonics_sph(degreeMax, output_shape):
    """
    Returns the spherical harmonics for all degrees (n) and orders (m) specified by the maximum degree (.N)
    output: complex tensor of shape (ksize**3,(N+1)**2)
    The center of the filters is set to 0 for m==0 and n!=0 for the Hermitian symmetry
    """
    r = np.arange(0, output_shape[0])
    phi = np.linspace(0, 2*pi, output_shape[1], endpoint=False)
    theta = np.linspace(0, pi, output_shape[2], endpoint=False)
    r,phi,theta = np.meshgrid(r,phi,theta)
    harmonics = []
    for n in range(degreeMax+1):
        P_legendre = legendre(n,np.cos(theta))
        for m in range(-n,n+1):
            # get the spherical harmonics (without radial profile)
            sh = spherical_harmonics(m,n,P_legendre,phi)
            # Reshape for multiplication with radial profile r
            harmonics.append(sh)

    return np.stack(harmonics,axis=-1)


def get_steering_matrixBD(N,ksize,M):
    '''
    Returns tensor of a block diagonal matrix Sr for each angle a: a matrix of weights of size (N+1)^2,(N+1)^2 for each orientation.
    So Sr has size (M,(N+1)**2,(N+1)**2)).
    (N+1)**2 is the total number of Ynm, it comes from the sum_n(2n+1)
    N: maximum degree of the SHs
    M: number of orientations
    '''
    # define search space for angles.
    zyz = get_euler_angles(M)
    _,theta,phi = getSphCoordArrays(ksize)

    Sr = np.zeros((M,(N+1)**2,(N+1)**2),dtype=np.complex64)
    # scan through angles
    for a in range(zyz.shape[0]):
        alpha, beta, gamma = zyz[a]*pi/180
        # Import Wigner D matrix directly from simpy
        for n in range(N+1):
            for k1 in range(n*2+1):
                m1 = k1-n
                for k2 in range(n*2+1):
                    m2 = k2-n
                    Sr[a,n**2+k1,n**2+k2] = np.complex(Rotation.D(n, m1, m2, alpha, beta, gamma).doit())

    return Sr

def legendre(n,X):
    '''
    Legendre polynomial used to define the SHs
    '''
    res = np.zeros(((n+1,)+(X.shape)))
    for m in range(n+1):
        res[m] = sp.lpmv(m,n,X)
    return res

def spherical_harmonics(m,n,P_legendre,phi):
    '''
    Returns the SH of degree n, order m
    '''
    P_n_m = np.squeeze(P_legendre[np.abs(m)])
    sign = (-1)**((m+np.abs(m))/2)
    # Normalization constant
    A = sign * np.sqrt( (2*n+1)/(4*pi) * np.math.factorial(n-np.abs(m))/np.math.factorial(n+np.abs(m)))
    # Spherical harmonics
    sh = A * np.exp(1j*m*phi) * P_n_m
    # normalize the SH to unit norm
    sh /= np.sqrt(np.sum(sh*np.conj(sh)))
    return sh.astype(np.complex64)


def getSphCoordArrays(ksize):
    '''
    Returns spherical coordinates (rho,theta,phi) from the kernel size
    '''
    if np.mod(ksize,2): # ksize odd
        x = np.linspace(-1,1,ksize)
    else: # ksize even
        x = np.linspace(-1,1,ksize)#
    X, Y, Z = np.meshgrid(x, x, x)
    rho = np.sqrt(np.sum(np.square([X,Y,Z]),axis=0))
    phi = np.squeeze(np.arctan2(Y,X))
    theta = np.nan_to_num(np.squeeze(np.arccos(Z/rho)))
    rho = np.squeeze(rho)
    return [rho,theta,phi]

def get_euler_angles(M):
    '''
    Returns the zyz Euler angles with shape (M, 3) for the defined number of orientations M.
    (intrinsic Euler angles in the zyz convention)
    '''
    if M==1:
        zyz=np.array([[0,0,0]])
    elif M==2:
        zyz=np.array([[0,0,0],[180,0,0]])
    elif M==4: # Implement Klein's four group see Worrall and Brostow 2018
        zyz=np.array([[0,0,0],[180,0,0],[0,180,0],[180,180,0]])
    elif M==8: # Test of theta and phi.
        zyz=np.array([[0,0,0],[0,45,315],[0,90,270],[0,135,225],[0,180,180],[0,225,135],[0,270,90],[0,315,45]])
    elif M==24: # as represented in Worrall and Brostow 2018, derived from the Caley's table
        # For intrinsic Euler angles (each row is for one of the six points on the sphere (theta,phi angles))
        zyz=np.array([[0,0,0],[0,0,90],[0,0,180],[0,0,270],
                     [0,90,0],[0,90,90],[0,90,180],[0,90,270],
                     [0,180,0],[0,180,90],[0,180,180],[0,180,270],
                     [0,270,0],[0,270,90],[0,270,180],[0,270,270],
                     [90,90,0],[90,90,90],[90,90,180],[90,90,270],
                     [90,270,0],[90,270,90],[90,270,180],[90,270,270]
                     ])

    elif M==72: # as represented in Worrall and Brostow 2018, derived from the Caley's table
        # For intrinsic Euler angles (each row is for one of the six points on the sphere (theta,phi angles))
        zyz=np.array([[0,0,0],[0,0,90],[0,0,180],[0,0,270],
                     [0,90,0],[0,90,90],[0,90,180],[0,90,270],
                     [0,180,0],[0,180,90],[0,180,180],[0,180,270],
                     [0,270,0],[0,270,90],[0,270,180],[0,270,270],
                     [90,90,0],[90,90,90],[90,90,180],[90,90,270],
                     [90,270,0],[90,270,90],[90,270,180],[90,270,270],

                     [0,45,0],[0,45,90],[0,45,180],[0,45,270],
                     [0,135,0],[0,135,90],[0,135,180],[0,135,270],
                     [0,225,0],[0,225,90],[0,225,180],[0,225,270],
                     [0,315,0],[0,315,90],[0,315,180],[0,315,270],

                     [90,45,0],[90,45,90],[90,45,180],[90,45,270],
                     [90,135,0],[90,135,90],[90,135,180],[90,135,270],
                     [90,225,0],[90,225,90],[90,225,180],[90,225,270],
                     [90,315,0],[90,315,90],[90,315,180],[90,315,270],

                     [45,90,0],[45,90,90],[45,90,180],[45,90,270],
                     [135,90,0],[135,90,90],[135,90,180],[135,90,270],
                     [45,270,0],[45,270,90],[45,270,180],[45,270,270],
                     [135,270,0],[135,270,90],[135,270,180],[135,270,270]
                     ])

    else: # TO DO
        raise ValueError( "M = "+str(M)+" not yet implemented. Try 1, 4, 24 or 72" )
        ''' XXX
        # Parametrized uniform triangulation of 3D circle/sphere:
        n_gamma = 4

        # No need for stlPoints AND A, B, C
        stlPoints, _, _, _ = sphereTriangulation(M,n_gamma)
        # Then do spherical coordinates to get the alpha and beta angles uniformly sampled on the sphere.
        alpha,beta = change_vars(np.swapaxes(stlPoints,0,1)) # The Euler angles alpha and beta are respectively theta and phi in spherical coord.
        # Then sample uniformly on gamma
        step_gamma = 2*pi/n_gamma
        gamma = np.tile(np.linspace(0,2*pi-step_gamma,n_gamma),alpha.shape[0])
        alpha = np.repeat(alpha,n_gamma)
        beta = np.repeat(beta,n_gamma)
        zyz = np.stack((alpha,beta,gamma),axis=1)*180.0/pi
        '''
    return zyz

def degreeToIndexes_range(n):
    return range(n*n,n*n+2*n+1)
def degreeToIndexes_slice(n):
    return slice(n*n,n*n+2*n+1)

def compute_cg_matrix(k,l):
    '''
    Computes the matrix that block-diagonilizes the Kronecker product of Wigned D matrices of degree k and l respectively
    Output size (2k+1)(2l+1)x(2k+1)(2l+1)
    '''
    c_kl = np.zeros([(2*k+1)*(2*l+1), (2*k+1)*(2*l+1)])

    n_off=0
    for J in range(abs(k-l), k+l+1):
        m_off=0
        for m1_i in range(2*k+1):
            m1 = m1_i-k
            for m2_i in range(2*l+1):
                m2 = m2_i-l
                for n_i in range(2*J+1):
                    n = n_i-J
                    if m1+m2==n:
                        c_kl[m_off+m2_i,n_off + n_i] = CG(k,m1,l,m2,J,m1+m2).doit()
            m_off=m_off+2*l+1
        n_off=n_off+2*J+1

    return c_kl


## NOT USED FOR NOW ##

def sphereTriangulation(M,n_gamma):
    '''
    Defines points on the sphere that we use for alpha (z) and beta (y') Euler angles sampling. We can have 24 points (numIterations=0), 72 (numIterations=1), 384 (numIterations=2) etc.
    Copied from the matlab function https://ch.mathworks.com/matlabcentral/fileexchange/38909-parametrized-uniform-triangulation-of-3d-circle-sphere
    M is the number total of orientation, i.e. number of points on the sphere + number of angles for the gamma angle (n_gamma).

    '''
    #
    numIter = int((M/24)**(1/n_gamma)-1)
    # function returns stlPoints fromat and ABC format if its needed,if not - just delete it and adapt to your needs
    radius = 1
    # basic Octahedron reff:http://en.wikipedia.org/wiki/Octahedron
    # ( ?1, 0, 0 )
    # ( 0, ?1, 0 )
    # ( 0, 0, ?1 )
    A=np.asarray([1, 0, 0])*radius
    B=np.asarray([0, 1, 0])*radius
    C=np.asarray([0, 0, 1])*radius
    # from +-ABC create initial triangles which define oxahedron
    triangles=np.asarray([ A,  B,  C,
            A,  B, -C,
          # -x, +y, +-Z quadrant
           -A,  B,  C,
           -A,  B, -C,
          # -x, -y, +-Z quadrant
           -A, -B,  C,
           -A, -B, -C,
          # +x, -y, +-Z quadrant
            A, -B,  C,
            A, -B, -C])# -----STL-similar format
    # for simplicity lets break into ABC points...
    selector = np.arange(0,len(triangles[:,1])-2 ,3)
    Apoints = triangles[selector  ,:]
    Bpoints = triangles[selector+1,:]
    Cpoints = triangles[selector+2,:]
    # in every of numIterations
    for iteration in range(numIter):
        # devide every of triangle on three new
        #        ^ C
        #       / \
        # AC/2 /_4_\CB/2
        #     /\ 3 /\
        #    / 1\ /2 \
        # A /____V____\B           1st              2nd              3rd               4th
        #        AB/2
        #new triangleSteck is [ A AB/2 AC/2;     AB/2 B CB/2;     AC/2 AB/2 CB/2    AC/2 CB/2 C]
        AB_2 = (Apoints+Bpoints)/2
        # do normalization of vector
        AB_2 = arsUnit(AB_2,radius) # same for next 2 lines
        AC_2 = (Apoints+Cpoints)/2
        AC_2   =  arsUnit(AC_2,radius)
        CB_2 = (Cpoints+Bpoints)/2
        CB_2   =  arsUnit(CB_2,radius)
        Apoints = np.concatenate(( Apoints, # A point from 1st triangle
                    AB_2 ,   #    A point from 2nd triangle
                    AC_2 ,   #    A point from 3rd triangle
                    AC_2))    #   A point from 4th triangle..same for B and C
        Bpoints = np.concatenate((AB_2, Bpoints, AB_2, CB_2))
        Cpoints = np.concatenate((AC_2, CB_2   , CB_2, Cpoints))
    # now tur points back to STL-like format....
    numPoints = np.shape(Apoints)[0]
    selector  = np.arange(numPoints)
    selector  = np.stack((selector, selector+numPoints, selector+2*numPoints))

    selector  = np.swapaxes(selector,0,1)
    selector  = np.concatenate(selector)
    stlPoints = np.concatenate((Apoints, Bpoints, Cpoints))
    stlPoints = stlPoints[selector,:]

    return stlPoints, Apoints, Bpoints, Cpoints

def change_vars(MG):
    '''
    MG: np array of shape (3,...) containing 3D cartesian coordinates.
    returns spherical coordinates theta and phi (could return rho if needed)
    '''
    rho = np.sqrt(np.sum(np.square(MG),axis=0))
    phi = np.squeeze(np.arctan2(MG[1,...],MG[0,...])) + pi
    theta = np.squeeze(np.arccos(MG[2,...]/rho))
    # The center value is Nan due to the 0/0. So make it 0.
    theta[np.isnan(theta)] = 0
    rho = np.squeeze(rho)

    return theta,phi

def arsNorm(A):
    # vectorized norm() function
    rez = A[:,0]**2 + A[:,1]**2 + A[:,2]**2
    rez = np.sqrt(rez)
    return rez

def arsUnit(A,radius):
    # vectorized unit() functon
    normOfA = arsNorm(A)
    rez = A / np.stack((normOfA, normOfA, normOfA),1)
    rez = rez * radius
    return rez
