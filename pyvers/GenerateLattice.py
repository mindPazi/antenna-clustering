import numpy as np

def GenerateLattice(Ny, Nz, x1, x2, A=None, B=None):
    """
    Version: 1.0;
    Developed by Milan RC. delivered on 06/30/2017.
    
    This function calculates a regular lattice based on the generation
    vectors x1 and x2. In particular, the function assumes that y-axis
    referes to the horizontal plane while z-axis refers to the vertical
    plane: then x1 is assumed to be aligned with the y-axis, while x2 can any
    other vector of the yz-plane different from x1.
    
    Please NOTE:
    
    - RECTANGULAR LATTICE for a regular rectangular lattice x1 and x2 must be
      orthogonal. Example x1=[1,0] and x2=[0,1];
    
    - TRIANGULAR LATTICE for a triangular atice x1 and x2 must be
      45° inclined. Example x1=[1,0] and x2=[1,1] for a 45° lattice
    
    - HEXAGONAL LATTICE. Example x1=[1,0] and x2=[1,1] for a 30° lattice
    
    
    INPUT:
    Nz: number of rows [Scalar];
    Ny: number of columns [Scalar];
    x1,x2: distance between elements is z and y coordinates according to
       lattice type [Scalar];
    A and B: refers to the truncation of the aperture to an ellipsoid
    A refers to y-axis while B to z-axis.
    
    
    OUTPUT:
    Y: matrix [Nz x Ny] with element y coordinates;
    Z: matrix [Nz x Ny] with element z coordinates;
    NN: matrix [Nz x Ny] with element index in y coordinates;
    MM: matrix [Nz x Ny] with element index in z coordinates;
    DY: maximum size of array in y in mm [Scalar];
    DZ: maximum size of array in z in mm [Scalar];
    Mask: truncation mask;
    I: truncation index;
    """
    
    if A is None:
        A = np.inf
        B = np.inf
    elif B is None:
        B = A
    
    # Generate array indexes
    if (Nz % 2):
        M = np.arange(-(Nz-1)//2, (Nz-1)//2 + 1)
    else:
        M = np.arange(-Nz//2 + 1, Nz//2 + 1)
    
    if (Ny % 2):
        N = np.arange(-(Ny-1)//2, (Ny-1)//2 + 1)
    else:
        N = np.arange(-Ny//2 + 1, Ny//2 + 1)
    
    NN, MM = np.meshgrid(N, M)
    
    dz = x2[1]
    dy = x1[0]
    DELTA = max(x2[0], x1[1])
    
    Y = NN * dy
    Z = MM * dz
    
    Y[1::2, :] = Y[1::2, :] + DELTA
    
    DZ = (np.max(Z) - np.min(Z))
    DY = (np.max(Y) - np.min(Y))
    
    ### Aperture truncation
    
    I = np.where((Y/A)**2 + (Z/B)**2 > 1)
    Mask = np.ones(Y.shape)
    Mask[I] = 0
    
    return Y, Z, NN, MM, DY, DZ, Mask, I
