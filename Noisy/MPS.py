import numpy as np
from scipy.linalg import svd, LinAlgError
import matplotlib.pyplot as plt

def stable_svd(A):
    try:
        U, S, V = svd(A, full_matrices=False, lapack_driver='gesdd')
    except LinAlgError:
        U, S, V = svd(A, full_matrices=False, lapack_driver='gesvd')
    return U, S, V

def left_isometric(Bs):
    As = []

    B = np.squeeze(Bs[0]) # shape (d, chir)
    chil, d, chir = B.shape
    B = B.reshape(chil * d, chir)
    Q, R = np.linalg.qr(B)
    As.append(Q.reshape(chil, d, -1)) 

    for B in Bs[1:-1]:
        B = np.einsum('ab, bjc -> ajc', R, B)
        chil, d, chir = B.shape
        B = B.reshape(chil * d, chir)
        Q, R = np.linalg.qr(B)
        As.append(Q.reshape(chil, d, -1))
    
    As.append(np.einsum('ab, bjc -> ajc', R, Bs[-1]))
    return As

def truncate(As, chi_max=2, svd_min=1e-10, which='right_to_left', renormalize=False):
    S_list = []
    if which == 'right_to_left':
        # sweeps right to left
        # assumes left-canonical form, returns right-canonical form
        for i in range(len(As)-1, 0, -1):
            # two-site tensor
            A0 = As[i-1]
            A1 = As[i]
            theta = np.einsum('ajb, bkc -> ajkc', A0, A1)
            chil, dl, dr, chir = theta.shape
            theta = theta.reshape(chil * dl, dr * chir)
            # singular value decomposition
            X, S, Y = stable_svd(theta)
            S_list.append(S)
            # new bond dimension
            chi_new = min(np.sum(S >= svd_min), chi_max)
            # print(chi_new)
            if renormalize:
                S = S[:chi_new] / np.linalg.norm(S[:chi_new])
            # truncate and save
            As[i-1] = (X[:, :chi_new] * S[None, :chi_new])\
                .reshape(chil, dl, chi_new)
            As[i] = Y[:chi_new, :]\
                .reshape(chi_new, dr, chir)
    elif which == 'left_to_right':
        # sweeps left to right
        # assumes right-canonical form, returns left-canonical form
        for i in range(len(As)-1):
            # two-site tensor
            A0 = As[i]
            A1 = As[i+1]
            theta = np.einsum('ajb, bkc -> ajkc', A0, A1)
            chil, dl, dr, chir = theta.shape
            theta = theta.reshape(chil * dl, dr * chir)
            # singular value decomposition
            X, S, Y = stable_svd(theta)
            S_list.append(S)
            # new bond dimension
            chi_new = min(np.sum(S >= svd_min), chi_max)
            if renormalize:
                S = S[:chi_new] / np.linalg.norm(S[:chi_new])
            # truncate and save
            As[i] = X[:, :chi_new]\
                .reshape(chil, dl, chi_new)
            As[i+1] = (S[:chi_new, None] * Y[:chi_new, :])\
                .reshape(chi_new, dr, chir)
    else:
        raise ValueError("Wrong argument 'which' given for 'truncate'.")
    return As, S_list

def truncate_Gio(As, svd_min=1e-10, threshold=0.1, threshold_big=5, which='right_to_left', renormalize=False):
    S_list = []
    chi_max_list = []
    # sweeps right to left
    # assumes left-canonical form, returns right-canonical form
    for i in range(len(As)-1, 0, -1):
        # two-site tensor
        A0 = As[i-1]
        A1 = As[i]
        theta = np.einsum('ajb, bkc -> ajkc', A0, A1)
        chil, dl, dr, chir = theta.shape
        theta = theta.reshape(chil * dl, dr * chir)
        # singular value decomposition
        X, S, Y = stable_svd(theta)
        S_list.append(S)
        # new bond dimension
        S = np.array(S)
        S_diff = np.abs(S[1:] - S[:-1])
        chi_max = np.where(S_diff < threshold)[0]
        if len(chi_max) == 0:
            chi_max = len(S) + 1
        else:
            index = chi_max[0]
            # if S_diff[index-1] > threshold_big:
            #     chi_max = index
            # else:
            #     chi_max = index + 1 # +1 because chi counts the number of singular values, while S_diff has one less element
            chi_max = index + 1
        chi_max_list.append(chi_max)
        chi_new = min(np.sum(S >= svd_min), chi_max)
        # print(chi_new)
        if renormalize:
            S = S[:chi_new] / np.linalg.norm(S[:chi_new])
        # truncate and save
        As[i-1] = (X[:, :chi_new] * S[None, :chi_new])\
            .reshape(chil, dl, chi_new)
        As[i] = Y[:chi_new, :]\
            .reshape(chi_new, dr, chir)

    return As, S_list, chi_max_list

# ----------------------------------------
# My own stuff

def interpolate_func(As):
    # func_interp = np.squeeze(As[-1]) #now has two legs: ab
    i, j, k = As[-1].shape
    func_interp = As[-1].reshape(i, j * k) # shape (i, jk)

    for A in As[-2:0:-1]:
        func_interp = np.einsum('idk, kj -> idj', A, func_interp)
        func_interp = func_interp.reshape(A.shape[0], -1)

    D = As[0].shape[0]

    #print(As[0].shape)
    func_interp = np.einsum('dak, kj -> daj', np.squeeze(As[0]), func_interp)
    #func_interp = np.einsum('ak, kj -> aj', np.squeeze(As[0]), func_interp) #uncomment this in case D = 1
    func_interp = np.transpose(func_interp, [1,2,0])
    #print("func interp shape = ", func_interp.shape)
    
    func_interp = func_interp.reshape(-1, D)
    #func_interp = func_interp.reshape(-1) #uncomment this in case D = 1
    return func_interp

# ...existing code...
def tensor_to_matrix(A):
    A_orig_shape = A.shape
    A_s = np.squeeze(A)
    s = A_s.shape
    if A_s.ndim == 1:
        mat = A_s.reshape(-1, 1)
    elif A_s.ndim == 2:
        mat = A_s
    elif A_s.ndim == 3:
        mat = A_s.reshape(s[0], s[1] * s[2])
    elif A_s.ndim == 4:
        mat = A_s.reshape(s[0] * s[1], s[2] * s[3])
    else:
        raise ValueError(f"Unsupported tensor ndim={A_s.ndim}, shape={A_orig_shape}")
    return mat, A_s.shape, A_orig_shape

def matrix_to_tensor(mat, squeezed_shape, orig_shape):
    # ricostruisce prima nella shape squeezed, poi riconduce a shape originale
    A_s = mat.reshape(squeezed_shape)
    return A_s.reshape(orig_shape)

def errors(As, func_vals_theo, interpolate=True):
    if interpolate:
        func_interp = interpolate_func(As)
    else:
        func_interp = As
    difference = func_vals_theo - func_interp
    err_max = np.max(np.abs(difference)) / np.max(np.abs(func_vals_theo))
    err_2 = np.linalg.norm(difference) / np.linalg.norm(func_vals_theo)
    return err_max, err_2

def error_vs_chi(As, sing_values_dict, func_vals_theo):
    """
    I expect chi_dict ={'n': [list of chi values to test]}
    """
    err_max_dict = {}
    err_2_dict = {}
    
    # compute full interpolation error
    err_max_full, err_2_full = errors(As, func_vals_theo)

    for n in sing_values_dict.keys():
        err_max_dict[n] = [err_max_full]  # start with full chi error
        err_2_dict[n] = [err_2_full]
        A_n = As[n]
        mat_n, sq_shape, orig_shape = tensor_to_matrix(A_n)
        U, S, Vh = np.linalg.svd(mat_n, full_matrices=False)
        for el in sing_values_dict[n][1:]:  # skip full chi since it's already computed
            mat_tr = U[:, :el] @ np.diag(S[:el]) @ Vh[:el, :]
            A_n_tr = matrix_to_tensor(mat_tr, sq_shape, orig_shape)
            As_tr = [a.copy() for a in As]
            As_tr[n] = A_n_tr
            err_max, err_2 = errors(As_tr, func_vals_theo)
            err_max_dict[n].append(err_max)
            err_2_dict[n].append(err_2)
    return err_max_dict, err_2_dict
