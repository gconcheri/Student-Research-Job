import numpy as np
from scipy.linalg.interpolative import interp_decomp

# Define interpolative compositions
# M ~= C @ Z, where C are columns of M, i.e. C[:, i] = M[:, j] for some j

def interpolative_decomposition(M, eps_or_k=1e-5, k_min=2):
    r = min(M.shape)
    if r <= k_min:
        k = r
        idx, proj = interp_decomp(M, eps_or_k=k)
    elif isinstance(eps_or_k, (int, np.int32, np.int64)):
        k = min(r, eps_or_k)
        idx, proj = interp_decomp(M, eps_or_k=k)
    else:
        k, idx,  proj = interp_decomp(M, eps_or_k=eps_or_k)
        if k <= k_min:
            k = min(r, k_min)
            idx, proj = interp_decomp(M, eps_or_k=k)
    C = M[:, idx[:k]]
    Z = np.concatenate([np.eye(k), proj], axis=1)[:, np.argsort(idx)]
    # check if NaNs or infs show up, fall back to leastsquares
    if not np.isfinite(proj).all():
        # identify indeces of polluted columns
        col_idxs = np.logical_not(np.isfinite(Z)).any(axis=0).nonzero()[0]
        # fill those with results from leastsquares
        Z[:, col_idxs] = np.linalg.lstsq(C, M[:, col_idxs], rcond=None)[0]
    return C, Z, k, idx[:k]

def find_ID_coeffs_via_iterated_leastsq(M, J, notJ=None):
    """
    Given a set of columns M[:, J], finds a matrix Z such that
    approximately M = M[:, J] @ Z by iteratively solving for
    each column M[:, J] @ Z[:, j] = M[:, j] with least squares.
    """
    # number of columns
    _, N = M.shape
    # fill empty matrix with solutions
    Z = np.zeros((len(J), N), dtype=M.dtype)
    # set columns J to identity in Z
    Z[np.arange(len(J)), J] = 1.
    # for the rest solve M[:, J] @ x = M[:, j] for x, then Z[:, j] = x
    if notJ is None:
        notJ = list(set(range(N)) - set(J))
    x = np.linalg.lstsq(M[:, J], M[:, notJ], rcond=None)[0] # x.shape = (len(J), N)
    Z[:, notJ] = x
    return Z

# define a wrapper to cache function evaluations
class function_cache_wrapper:
    
    def __init__(self, tensor):
        self.tensor = tensor
        self.num_calls = 0
        self.cache = {}
    
    def __call__(self, *args):
        self.num_calls += 1
        if args in self.cache:
            return self.cache[args]
        else:
            result = self.tensor(*args)
            self.cache[args] = result
            return result

def get_kwarg(arg, kwargs, default):
    if arg in kwargs:
        return kwargs[arg]
    else:
        return default

def tensor_cross_interpolation(tensor, L, d=2, mode='reset', cache=True, init_idxs=None, **kwargs):
    # check if caching of function calls is enabled
    if cache:
        tensor = function_cache_wrapper(tensor)
    # random initial choice
    if init_idxs is not None and isinstance(init_idxs[0], list):
        J = []
        I = []

        for j in range(1,L+1):
            J_l = []
            for id in init_idxs:
                J_id = id[j:]
                if not any(np.array_equal(J_id, x) for x in J_l):
                    J_l.append(J_id)
            J_l = np.array(J_l)
            J.append(J_l)

        for j in range(L):
            I_l = []
            for id in init_idxs:
                I_id = id[:j]
                if not any(np.array_equal(I_id, x) for x in I_l):
                    I_l.append(I_id)
            I_l = np.array(I_l)
            I.append(I_l)
        
        dtype = tensor(*init_idxs[0]).dtype
        
    elif init_idxs is not None:
        idxs = np.array(init_idxs)
        J = [idxs[j:].reshape(1, -1) for j in range(1, L+1)]
        I = [idxs[:j].reshape(1, -1) for j in range(L)]
        dtype = tensor(*idxs).dtype

    else:
        idxs = np.random.choice(d, size=(L))
        J = [idxs[j:].reshape(1, -1) for j in range(1, L+1)]
        I = [idxs[:j].reshape(1, -1) for j in range(L)]
        dtype = tensor(*idxs).dtype
    # run in either reset or accumulative mode
    if mode == 'reset':
        eps_or_chi = get_kwarg('eps_or_chi', kwargs, 1e-6)
        num_sweeps = get_kwarg('num_sweeps', kwargs, 6)
        # sweep
        I = reset_left_to_right_sweep(tensor, I, J, L, d, eps_or_chi, dtype=dtype)
        for i in range(num_sweeps-1):
            J = reset_right_to_left_sweep(tensor, I, J, L, d, eps_or_chi, dtype=dtype)
            I = reset_left_to_right_sweep(tensor, I, J, L, d, eps_or_chi, dtype=dtype)
        As, J = reset_get_MPS_right_to_left_sweep(tensor, I, J, L, d, eps_or_chi, dtype=dtype)
    elif mode == 'accumulative':
        num_sweeps = get_kwarg('num_sweeps', kwargs, 32)
        # sweep
        I = acc_left_to_right_sweep(tensor, I, J, L, d, dtype=dtype)
        for i in range(num_sweeps-1):
            J = acc_right_to_left_sweep(tensor, I, J, L, d, dtype=dtype)
            I = acc_left_to_right_sweep(tensor, I, J, L, d, dtype=dtype)
        As, J = acc_get_MPS_right_to_left_sweep(tensor, I, J, L, d, dtype=dtype)
    else:
        raise ValueError("Invalid argument for 'mode'! Must be either 'reset' or 'accumulative'.")
    if cache:
        return As, I, J, tensor
    return As, I, J

# reset-mode tensor cross interpolation
def reset_left_to_right_sweep(tensor, I, J, L, d, eps_or_chi, dtype=np.float64):
    # sweep left to right
    for bond in range(L-1):
        # construct local two-site tensor
        chil, _ = I[bond].shape
        chir, _ = J[bond+1].shape
        Pi = np.zeros((chil, d, d, chir), dtype=dtype)
        for i, I1 in enumerate(I[bond]):
            for s1 in range(d):
                for s2 in range(d):
                    for j, J2 in enumerate(J[bond+1]):
                        Pi[i, s1, s2, j] = tensor(*I1, s1, s2, *J2)
        Pi = Pi.reshape(chil * d, d * chir)
        # decompose using interpolative decomposition:
        # Pi = Z^T @ C^T, with C^T = Pi[idx,:]
        CT, ZT, k, idx = interpolative_decomposition(Pi.T, eps_or_k=eps_or_chi)
        # update indices using idxs c I[bond] x {0, 1, ..., d-1}
        I[bond+1] = np.array([np.append(I[bond][i//d], [i%d]) for i in idx])
    return I

def reset_right_to_left_sweep(tensor, I, J, L, d, eps_or_chi, dtype=np.float64):
    # sweep right to left
    for bond in range(L-2,-1,-1):
        # construct local two-site tensor
        chil, _ = I[bond].shape
        chir, _ = J[bond+1].shape
        Pi = np.zeros((chil, d, d, chir), dtype=dtype)
        for i, I1 in enumerate(I[bond]):
            for s1 in range(d):
                for s2 in range(d):
                    for j, J2 in enumerate(J[bond+1]):
                        Pi[i, s1, s2, j] = tensor(*I1, s1, s2, *J2)
        Pi = Pi.reshape(chil * d, d * chir)
        # decompose using interpolative decomposition:
        # Pi = C @ Z, C = Pi[:,idx]
        C, Z, k, idx = interpolative_decomposition(Pi, eps_or_k=eps_or_chi)
        # update indices using idxs c {0, 1, ..., d-1} x J[bond+1]
        J[bond] = np.array([np.append([i//chir], J[bond+1][i%chir]) for i in idx])
    return J

def reset_get_MPS_right_to_left_sweep(tensor, I, J, L, d, eps_or_chi, dtype=np.float64):
    # create list of MPS tensors to be filled
    As = [None for _ in range(L)]
    # sweep right to left
    for bond in range(L-2,-1,-1):
        # construct local two-site tensor
        chil, _ = I[bond].shape
        chir, _ = J[bond+1].shape
        Pi = np.zeros((chil, d, d, chir), dtype=dtype)
        for i, I1 in enumerate(I[bond]):
            for s1 in range(d):
                for s2 in range(d):
                    for j, J2 in enumerate(J[bond+1]):
                        Pi[i, s1, s2, j] = tensor(*I1, s1, s2, *J2)
        Pi = Pi.reshape(chil * d, d * chir)
        # decompose using interpolative decomposition:
        # Pi = C @ Z, C = Pi[:,idx]
        C, Z, k, idx = interpolative_decomposition(Pi, eps_or_k=eps_or_chi)
        # update indices using idxs c {0, 1, ..., d-1} x J[bond+1]
        J[bond] = np.array([np.append([i//chir], J[bond+1][i%chir]) for i in idx])
        # update tensors
        As[bond] = C.reshape(chil, d, k)
        As[bond+1] = Z.reshape(k, d, chir)
    return As, J

# accumulative tensor cross interpolation

def acc_left_to_right_sweep(tensor, I, J, L, d, dtype=np.float64):
    # sweep left to right
    for bond in range(L-1):
        if len(I[bond]) * d == len(I[bond+1]): # check if number of rows is already saturated
            pass
        else:
            # construct local two-site tensor
            chil, _ = I[bond].shape
            chir, _ = J[bond+1].shape
            Pi = np.zeros((chil, d, d, chir), dtype=dtype)
            for i, I1 in enumerate(I[bond]):
                for s1 in range(d):
                    for s2 in range(d):
                        for j, J2 in enumerate(J[bond+1]):
                            Pi[i, s1, s2, j] = tensor(*I1, s1, s2, *J2)
            Pi = Pi.reshape(chil * d, d * chir)
            # all rows of Pi = I[bond] x {0, ..., d-1}
            all_idxs = np.concatenate(
                           [np.append(I[bond],
                                      s1 * np.ones((chil, 1), dtype=np.int32),
                                      axis=1)
                            for s1 in range(d)],
                           axis=0)
            # turn multi-indices I[bond+1] c I[bond] x {0, ..., d-1} into numbered indexes
            I2 = list(np.argmax((I[bond+1][:, None] == all_idxs[None]).all(axis=2),
                                axis=1))
            # update index set I2
            N = Pi.shape[0]
            notI2 = list(set(range(N)) - set(I2)) # get indices not in I2
            # get current iteration for Pi_tilde from ID with rows in I2
            Z = find_ID_coeffs_via_iterated_leastsq(Pi.T, J=I2, notJ=notI2)
            Pi_tilde = Z.T @ Pi[I2, :]
            # compute errors and update index set
            errs = np.linalg.norm(Pi[notI2, :] - Pi_tilde[notI2, :], axis=1, ord=1) # row errors
            ell = notI2[np.argmax(errs)]
            I[bond+1] = np.append(I[bond+1],
                                  all_idxs[None, ell],
                                  axis=0)
    return I

def acc_right_to_left_sweep(tensor, I, J, L, d, dtype=np.float64):
    # sweep right to left
    for bond in range(L-2,-1,-1):
        if len(J[bond]) == d * len(J[bond+1]): # check if number of columns is already saturated
            pass
        else:
            # construct local two-site tensor
            chil, _ = I[bond].shape
            chir, _ = J[bond+1].shape
            Pi = np.zeros((chil, d, d, chir), dtype=dtype)
            for i, I1 in enumerate(I[bond]):
                for s1 in range(d):
                    for s2 in range(d):
                        for j, J2 in enumerate(J[bond+1]):
                            Pi[i, s1, s2, j] = tensor(*I1, s1, s2, *J2)
            Pi = Pi.reshape(chil * d, d * chir)
            # all columns of Pi = {0, ..., d-1} x J[bond+1]
            all_idxs = np.concatenate(
                           [np.append(s2 * np.ones((chir, 1), dtype=np.int32),
                                      J[bond+1],
                                      axis=1)
                            for s2 in range(d)],
                           axis=0)
            # turn multi-indices J[bond] c {0, ..., d-1} x J[bond+1] into numbered indexes
            J1 = list(np.argmax((J[bond][:, None] == all_idxs[None]).all(axis=2),
                                axis=1))
            # update index set J1
            N = Pi.shape[1]
            notJ1 = list(set(range(N)) - set(J1)) # get indices not in J1
            # get current iteration for Pi_tilde from ID with columns in J1
            Z = find_ID_coeffs_via_iterated_leastsq(Pi, J=J1, notJ=notJ1)
            Pi_tilde = Pi[:, J1] @ Z
            # compute errors and update index set
            errs = np.linalg.norm(Pi[:, notJ1] - Pi_tilde[:, notJ1], axis=0, ord=1) # column errors
            ell = notJ1[np.argmax(errs)]
            J[bond] = np.append(J[bond],
                                all_idxs[None, ell],
                                axis=0)
    return J

def acc_get_MPS_right_to_left_sweep(tensor, I, J, L, d, dtype=np.float64):
    # create list of MPS tensors to be filled
    As = [None for _ in range(L)]
    # sweep right to left
    for bond in range(L-2,-1,-1):
        # construct local two-site tensor
        chil, _ = I[bond].shape
        chir, _ = J[bond+1].shape
        Pi = np.zeros((chil, d, d, chir), dtype=dtype)
        for i, I1 in enumerate(I[bond]):
            for s1 in range(d):
                for s2 in range(d):
                    for j, J2 in enumerate(J[bond+1]):
                        Pi[i, s1, s2, j] = tensor(*I1, s1, s2, *J2)
        Pi = Pi.reshape(chil * d, d * chir)
        # all columns of Pi = {0, ..., d-1} x J[bond+1]
        all_idxs = np.concatenate(
                       [np.append(s2 * np.ones((chir, 1), dtype=np.int32),
                                  J[bond+1],
                                  axis=1)
                        for s2 in range(d)],
                       axis=0)
        # turn multi-indices J[bond] c {0, ..., d-1} x J[bond+1] into numbered indexes
        J1 = list(np.argmax((J[bond][:, None] == all_idxs[None]).all(axis=2),
                            axis=1))
        # only update the index set if the number of columns is not saturated
        if len(J[bond]) == d * len(J[bond+1]):
            pass
        else:
            # update index set J1
            N = Pi.shape[1]
            notJ1 = list(set(range(N)) - set(J1)) # get indices not in J1
            # get current iteration for Pi_tilde from ID with columns in J1
            Z = find_ID_coeffs_via_iterated_leastsq(Pi, J=J1, notJ=notJ1)
            Pi_tilde = Pi[:, J1] @ Z
            # compute errors and update index set
            errs = np.linalg.norm(Pi[:, notJ1] - Pi_tilde[:, notJ1], axis=0, ord=1) # column errors
            ell = notJ1[np.argmax(errs)]
            J1.append(ell)
            J[bond] = np.append(J[bond],
                                all_idxs[None, ell],
                                axis=0)
        # construct MPS tensor from index set
        Z = find_ID_coeffs_via_iterated_leastsq(Pi, J=J1)
        As[bond+1] = Z.reshape(len(J1), d, chir)
    # last tensor is slice
    As[0] = Pi[:, J1].reshape(chil, d, len(J1))
    return As, J