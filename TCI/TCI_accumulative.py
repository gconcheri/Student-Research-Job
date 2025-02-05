# %%
import time
import numpy as np
import matplotlib.pyplot as plt

# %%
def find_ID_coeffs_via_iterated_leastsq(A, J, notJ=None):
    """
    Given a set of columns A[:, J], finds a matrix Z such that
    approximately A = A[:, J] @ Z by iteratively solving for
    each column A[:, J] @ Z[:, j] = A[:, j] with least squares.
    """
    # number of columns
    _, N = A.shape
    # fill empty matrix with solutions
    Z = np.zeros((len(J), N), dtype=A.dtype)
    # set columns J to identity in Z
    Z[np.arange(len(J)), J] = 1.
    # for the rest solve A[:, J] @ x = A[:, j] for x, then Z[:, j] = x
    if notJ is None:
        notJ = list(set(range(N)) - set(J))
    x = np.linalg.lstsq(A[:, J], A[:, notJ], rcond=None)[0] # x.shape = (len(J), N)
    Z[:, notJ] = x
    return Z

# %%
class function:  # certain function f(x) with x given as binary

    def __init__(self, f):
        self.cache = {}
        self.f = f #store function passed during instantiation
        self.numcacheused = 0
        self.numvals = 0
        self.unique = 0


    def __call__(self, *args, **kwds):
        self.numvals+=1
        if args in self.cache:
            self.numcacheused+=1
            return self.cache[*args]
        else:
            self.unique+=1
            val = self.f(*args)
            self.cache[*args] = val
            return val
    
    def cache_size(self): #size of cache = number of current evaluations
        return len(self.cache) #return the number of entries in the cache

        

# %% [markdown]
# ### VERSION with accumulative TCI, here we don't use scipy's built-in ID, but we use an args max algorithm that lets us update the bond lists I_l, J_l by adding one index per iteration

# %%
# this version of the TCI only keeps track of pivots and creates the actual MPS tensors only in the final sweep
def accumulative_tensor_cross_interpolation(tensor, func_vals, L, d=2, sweeps = 10):
    # initial choice is all zeros
    idxs = np.zeros((L,), dtype=np.int32)

    I = [idxs[:j].reshape(1, -1) for j in range(L)] # creates list of I_l arrays
    J = [idxs[j:].reshape(1, -1) for j in range(1, L+1)] # list of J_l
    
    dtype = tensor(*idxs).dtype

    # sweep
    I = left_to_right_sweep(tensor, I, J, L, d, dtype)
    for i in range(sweeps-1):
        J = right_to_left_sweep(tensor, I, J, L, d, dtype)
        I = left_to_right_sweep(tensor, I, J, L, d, dtype)
    As = get_MPS_right_to_left_sweep(tensor, I, J, L, d, dtype)

    print('final err_max: ', err_max[-1])
    print('final err_2: ', err_2[-1])
    print()
    print('repeated evaluations: ', tensor.numcacheused)
    print('unique evaluations', tensor.unique)
    print('unique + repeated: ', tensor.numcacheused + tensor.unique)
    print('total evaluations: ', tensor.numvals)


    return As, J, eval, err_2, err_max


def left_to_right_sweep(tensor, I, J, L, d, dtype):
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
            errs = np.linalg.norm(Pi[notI2, :] - Pi_tilde[notI2, :], axis=1) # row errors
            ell = notI2[np.argmax(errs)]
            I[bond+1] = np.append(I[bond+1],
                                  all_idxs[None, ell],
                                  axis=0)
    return I

def right_to_left_sweep(tensor, I, J, L, d, dtype):
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
                    # implement the tensor cross interpolation
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
            errs = np.linalg.norm(Pi[:, notJ1] - Pi_tilde[:, notJ1], axis=0) # column errors
            ell = notJ1[np.argmax(errs)]
            J[bond] = np.append(J[bond],
                                all_idxs[None, ell],
                                axis=0)
    return J

def get_MPS_right_to_left_sweep(tensor, I, J, L, d, dtype):
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
            errs = np.linalg.norm(Pi[:, notJ1] - Pi_tilde[:, notJ1], axis=0) # column errors
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
    return As