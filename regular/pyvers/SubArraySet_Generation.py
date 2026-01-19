import numpy as np

def SubArraySet_Generation(B, N, M):
    """
    Version: 1.0;
    Developed by Milan RC. delivered on 06/30/2017.
    
    B is assumed to be a Lsub x 2 matrix where Lsub is the size of the
    sub-array and 2 columns for (Iy,Iz) index positions
    """
    
    A = np.sum(B, axis=0)
    if A[0] == 0:  # vertical cluster
        step_M = B.shape[0]
        step_N = 1
    elif A[1] == 0:  # horizontal cluster
        step_N = B.shape[0]
        step_M = 1
    
    S = []
    
    for kk in range(int(np.min(M)), int(np.max(M)) + 1, step_M):
        for hh in range(int(np.min(N)), int(np.max(N)) + 1, step_N):
            Bshift = np.column_stack([B[:, 0] + hh, B[:, 1] + kk])
            check = len(np.where((Bshift[:, 0] > np.max(N)) | (Bshift[:, 0] < np.min(N)) | 
                                  (Bshift[:, 1] > np.max(M)) | (Bshift[:, 1] < np.min(M)))[0]) == 0
            if check:
                if len(S) == 0:
                    S = Bshift
                else:
                    S = np.column_stack([S, Bshift])
    
    Nsub = S.shape[1] // 2
    
    return S, Nsub
