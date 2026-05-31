import numpy as np

def _get_product_state(L, index, amplitude):
    """Generates an MPS for a single computational basis state."""
    binary_str = format(index, f'0{L}b')
    bits = [int(b) for b in binary_str]
    
    mps = []
    for i, bit in enumerate(bits):
        # Tensors have shape: (Left_Bond, Physical_Dim, Right_Bond)
        T = np.zeros((1, 2, 1), dtype=complex)
        T[0, bit, 0] = 1.0
        
        # Absorb the amplitude into the first site
        if i == 0:
            T *= amplitude
            
        mps.append(T)
    return mps

def _add_mps(mps1, mps2):
    """Sums two MPS by taking the direct sum of their virtual bonds."""
    L = len(mps1)
    mps_sum = []
    
    for i in range(L):
        A = mps1[i]
        B = mps2[i]
        
        vL_A, d, vR_A = A.shape
        vL_B, _, vR_B = B.shape
        
        if i == 0:
            # First site: Shared left vacuum index (vL = 1)
            T = np.zeros((1, d, vR_A + vR_B), dtype=complex)
            T[0, :, :vR_A] = A[0, :, :]
            T[0, :, vR_A:] = B[0, :, :]
        elif i == L - 1:
            # Last site: Shared right vacuum index (vR = 1)
            T = np.zeros((vL_A + vL_B, d, 1), dtype=complex)
            T[:vL_A, :, 0] = A[:, :, 0]
            T[vL_A:, :, 0] = B[:, :, 0]
        else:
            # Bulk sites: Block-diagonal structure
            T = np.zeros((vL_A + vL_B, d, vR_A + vR_B), dtype=complex)
            T[:vL_A, :, :vR_A] = A
            T[vL_A:, :, vR_A:] = B
            
        mps_sum.append(T)
        
    return mps_sum

def _compress_mps(mps, tol=1e-10):
    """Performs SVD sweeps to truncate minimal singular values."""
    L = len(mps)
    
    # 1. Right-to-Left Sweep: Bring MPS into Right-Orthogonal Form
    for i in range(L - 1, 0, -1):
        T = mps[i]
        vL, d, vR = T.shape
        
        # Reshape to perform SVD on the left bond
        T_mat = T.reshape(vL, d * vR)
        U, S, Vh = np.linalg.svd(T_mat, full_matrices=False)
        
        # Replace current tensor with the orthogonal part
        mps[i] = Vh.reshape(-1, d, vR)
        
        # Absorb singular values and left matrix into the left neighbor
        US = U @ np.diag(S)
        mps[i-1] = np.tensordot(mps[i-1], US, axes=([2], [0]))
        
    # 2. Left-to-Right Sweep: Compress and Truncate
    for i in range(L - 1):
        T = mps[i]
        vL, d, vR = T.shape
        
        # Reshape to perform SVD on the right bond
        T_mat = T.reshape(vL * d, vR)
        U, S, Vh = np.linalg.svd(T_mat, full_matrices=False)
        
        # Truncate values below the tolerance
        keep = np.sum(S > tol)
        if keep == 0: 
            keep = 1  # Retain at least a dimension of 1
            
        U = U[:, :keep]
        S = S[:keep]
        Vh = Vh[:keep, :]
        
        # Update current tensor
        mps[i] = U.reshape(vL, d, keep)
        
        # Pass the remainder into the right neighbor
        SVh = np.diag(S) @ Vh
        mps[i+1] = np.tensordot(SVh, mps[i+1], axes=([1], [0]))
        
    return mps

def build_peak_mps(L, peak_values):
    """
    Constructs a compressed MPS for an array with a localized peak.
    
    Returns:
    List[np.ndarray]: A list of L numpy arrays, where each array is a rank-3 
                      tensor with shape (Virtual_Left, Physical, Virtual_Right).
    """
    k = len(peak_values)
    center_idx = 2**(L - 1)
    start_idx = center_idx - (k // 2)
    
    total_mps = None
    
    for i, amplitude in enumerate(peak_values):
        if amplitude == 0:
            continue
            
        current_idx = start_idx + i
        mps_prod = _get_product_state(L, current_idx, amplitude)
        
        if total_mps is None:
            total_mps = mps_prod
        else:
            total_mps = _add_mps(total_mps, mps_prod)
            
    # Remove redundancies from the direct summation
    final_mps = _compress_mps(total_mps)
    
    return final_mps


def get_gaussian_peak(N, sigma_factor=6.0):
    """Generates a normalized Gaussian-shaped peak of N elements."""
    x = np.arange(N)
    mu = (N - 1) / 2.0
    sigma = max(N / sigma_factor, 1e-10)
    
    # Create the Gaussian shape
    peak = np.exp(-0.5 * ((x - mu) / sigma)**2)
    
    # Normalize it so the sum of squared amplitudes is 1 (quantum state normalization)
    peak = peak / np.sum(peak)
    
    return peak