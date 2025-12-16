import numpy as np

def SubArraySet_Generation(B, N, M):
    """
    Version: 1.0;
    Developed by Milan RC. delivered on 06/30/2017.
    Optimized version with pre-allocation and vectorized operations
    
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
    
    # OPTIMIZATION 1: Pre-compute grid dimensions and pre-allocate result array
    min_N = int(np.min(N))
    max_N = int(np.max(N))
    min_M = int(np.min(M))
    max_M = int(np.max(M))
    
    # Calculate maximum number of possible sub-arrays
    n_steps_M = len(range(min_M, max_M + 1, step_M))
    n_steps_N = len(range(min_N, max_N + 1, step_N))
    max_subarrays = n_steps_M * n_steps_N
    
    # Pre-allocate array (columns will be: N0, M0, N1, M1, ..., N_Lsub, M_Lsub for each sub-array)
    Lsub = B.shape[0]
    S_preallocated = np.zeros((Lsub * 2, max_subarrays))
    
    # OPTIMIZATION 2: Vectorize boundary checking where possible
    valid_count = 0
    
    for kk in range(min_M, max_M + 1, step_M):
        for hh in range(min_N, max_N + 1, step_N):
            # Shift the base cluster to current position
            Bshift_N = B[:, 0] + hh
            Bshift_M = B[:, 1] + kk
            
            # OPTIMIZATION 3: Vectorized boundary check (single operation instead of element-wise)
            # Check if all elements are within bounds
            check = np.all((Bshift_N >= min_N) & (Bshift_N <= max_N) & 
                          (Bshift_M >= min_M) & (Bshift_M <= max_M))
            
            if check:
                # Interleave N and M coordinates: [N0, M0, N1, M1, ..., N_Lsub, M_Lsub]
                S_preallocated[0::2, valid_count] = Bshift_N
                S_preallocated[1::2, valid_count] = Bshift_M
                valid_count += 1
    
    # Trim to actual number of valid sub-arrays
    S = S_preallocated[:, :valid_count]
    Nsub = valid_count
    
    return S, Nsub
