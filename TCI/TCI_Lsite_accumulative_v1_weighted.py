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

#%%
class function:  # certain function f(x) with x given as binary

    def __init__(self, f):
        self.cache = {}
        self.runtime = 0
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
            self.runtime += int(''.join(map(str, args)), 2) #explanation of this code below
            return val
    
    def cache_size(self): #size of cache = number of current evaluations
        return len(self.cache) #return the number of entries in the cache

"""Now val defined in this class will be an array of shape (D,) with D number of points
in which correlation function is evaluated in space"""

#%%
def accumulative_tensor_cross_interpolation(tensor, func_vals, D, L, d=2, iters=6, 
                                            euclidean = True, weight = True, model = 0):
    #tensor must be function s.t. f(*il,σj,σj+1,*jr).shape = (D,) with D number of points in space
    # initial choice is all zeros
    idxs = np.zeros((L,), dtype=np.int32)
    #idxs = np.random.choice(d, size=(L)) #array of L random numbers from 0 to d-1 - index sigma

    dtype = np.complex128

    As = [None] * L

    J = [idxs[j:].reshape(1, -1) for j in range(1, L+1)]
    I = [idxs[:j].reshape(1, -1) for j in range(L)]

    runtime = []
    err_2list = []

    # sweep
    for i in range(iters):
        #print(f'Sweep: {i+1:d}.')
        As, I = left_to_right_sweep(tensor, As, I, J, L, d, D, dtype, euclidean, weight, model)
        As, J = right_to_left_sweep(tensor, As, I, J, L, d, D, dtype, euclidean, weight, model)
        runtime.append(tensor.runtime)
        func_interp = interpolation(As, D)
        difference = func_vals-func_interp #should be difference between 2 matrices
        err_2 = np.linalg.norm(difference)/np.linalg.norm(func_vals)
        err_2list.append(err_2)
    #in theory, at the end of these sweeps the first tensor of As should be the only one with the additional leg of dim D

    err_max = np.max(np.abs(difference))/np.max(np.abs(func_vals))
    err_2 = np.linalg.norm(difference)/np.linalg.norm(func_vals)
    evals = tensor.cache_size()

    print('err_max: ', err_max)
    print('err_2: ', err_2)
    print()
    print('repeated evaluations: ', tensor.numcacheused)
    print('unique evaluations', tensor.unique)
    print('unique + repeated: ', tensor.numcacheused + tensor.unique)
    print('total evaluations: ', tensor.numvals)
    print()


    return As, J, evals, err_2list, runtime

def interpolation(As, D):
    func_interp = np.squeeze(As[-1]) #now has two legs: ab
    for A in As[-2:0:-1]:
        func_interp = np.einsum('idk, kj -> idj', A, func_interp)
        func_interp = func_interp.reshape(A.shape[0], -1)
    if As[0].shape[0] == D:
        #print(As[0].shape)
        func_interp = np.einsum('dak, kj -> daj', np.squeeze(As[0]), func_interp)
        #func_interp = np.einsum('ak, kj -> aj', np.squeeze(As[0]), func_interp) #uncomment this in case D = 1
        func_interp = np.transpose(func_interp, [1,2,0])
        #print("func interp shape = ", func_interp.shape)
        
        func_interp = func_interp.reshape(-1, D)
        #func_interp = func_interp.reshape(-1) #uncomment this in case D = 1
        return func_interp
    else: 
        return print("Error")

def left_to_right_sweep(tensor, As, I, J, L, d, D, dtype, euclidean, weight, model):
    # sweep left to right
    for bond in range(L-1):
        #print(bond)
        # construct local two-site tensor
        chil, _ = I[bond].shape #chil = number of rows in array I_l: number of combinations (σ1,..,σl)
        chir, _ = J[bond+1].shape #which corresponds to number of "points" on which I am evaluating function
        
        Pi = np.zeros((D, chil, d, d, chir), dtype=dtype)
        for il in range(chil):
            for s1 in range(d):
                for s2 in range(d):
                    for jr in range(chir):
                        val = tensor(*I[bond][il,:],s1,s2,*J[bond+1][jr,:])
                        Pi[:,il,s1,s2,jr] = val
                        #print(val.shape)
        #print("Pi.shape = ", Pi.shape)
        Pi = np.transpose(Pi, [1,2,3,4,0])
        Pi = Pi.reshape(chil * d, d * chir * D)

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
        notI2binary = all_idxs[notI2]
        
        if len(I[bond+1]) == N:
            pass # don't add a row if set of rows is already maximal
        else:
            Z = find_ID_coeffs_via_iterated_leastsq(Pi.T, J=I2, notJ=notI2)
            Pi_tilde = Z.T @ Pi[I2, :]
            # compute errors and update index set
            if euclidean == True:
                errs = np.linalg.norm(Pi[notI2, :] - Pi_tilde[notI2, :], axis=1) # row errors
            else:
                errs = np.sum(np.abs(Pi[notI2, :] - Pi_tilde[notI2, :]), axis=1)
            if weight == True and bond < L//2+1:
                a,b = notI2binary.shape
                #we append set of zeros to each row such that the binary number is of length L
                #in this way, we can apply int(...) which turns binary number into an integer 
                #by multiplying rightmost digit for 2^0, second for 2^1, and so on up to the leftmost digit
                #which is multiplied by 2^L-1 according to our conventional notation (look corr func TCI comment)
                notI2binary = np.append(notI2binary, np.zeros((a, (L-b)), dtype=np.int32), axis = 1)
                integers = np.array([int(''.join(map(str, row)), 2) for row in notI2binary])
                ell = notI2[np.argmax(errs*weight_func(integers, model, L))]
            else:
                ell = notI2[np.argmax(errs)]
            I2.append(ell)
            notI2.remove(ell)
            I[bond+1] = np.append(I[bond+1],
                                  all_idxs[None, ell],
                                  axis=0)
        # shift canonicality center by ID with updated index set
        #find Z s.t. Pi = Z^T @ Pi[I2,:]
        Z = find_ID_coeffs_via_iterated_leastsq(Pi.T, J=I2, notJ=notI2)
        chi = len(I2)
        # update tensors
        As[bond] = Z.T.reshape(chil, d, chi)
        As[bond+1] = Pi[I2, :].reshape(chi, d, chir, D)
    return As, I

def right_to_left_sweep(tensor, As, I, J, L, d, D, dtype, euclidean, weight, model):
    # sweep right to left
    for bond in range(L-2,-1,-1):
        # construct local two-site tensor
        chil, _ = I[bond].shape
        chir, _ = J[bond+1].shape

        Pi = np.zeros((D, chil, d, d, chir), dtype=dtype)
        for il in range(chil):
            for s1 in range(d):
                for s2 in range(d):
                    for jr in range(chir):
                        val = tensor(*I[bond][il,:],s1,s2,*J[bond+1][jr,:])
                        #print(val.shape)
                        Pi[:,il,s1,s2,jr] = val
        Pi = Pi.reshape(D * chil * d, d * chir)

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
        notJ1binary = all_idxs[notJ1]
        if len(J[bond]) == N:
            pass # don't add a column if set of columns is already maximal
        else:
            # get current iteration for Pi_tilde from ID with columns in J1
            Z = find_ID_coeffs_via_iterated_leastsq(Pi, J=J1, notJ=notJ1)
            Pi_tilde = Pi[:, J1] @ Z            
            # compute errors and update index set
            if euclidean == True:
                errs = np.linalg.norm(Pi[:, notJ1] - Pi_tilde[:, notJ1], axis=0) # column errors
            else:
                errs = np.sum(np.abs(Pi[:, notJ1] - Pi_tilde[:, notJ1]), axis=0)

            if weight == True and bond < L//2+1: 
                a,b = notJ1binary.shape
                notJ1binary = np.append(np.zeros((a, (L-b)), dtype=np.int32), notJ1binary, axis = 1)
                integers = np.array([int(''.join(map(str, row)), 2) for row in notJ1binary])
                ell = notJ1[np.argmax(errs*weight_func(integers, model, L))]
            else:
                ell = notJ1[np.argmax(errs)]

            J1.append(ell)
            notJ1.remove(ell)
            J[bond] = np.append(J[bond],
                                all_idxs[None, ell],
                                axis=0)
        # shift canonicality center by ID with updated index set
        Z = find_ID_coeffs_via_iterated_leastsq(Pi, J=J1, notJ=notJ1)
        chi = len(J1)
        # update tensors
        As[bond] = Pi[:, J1].reshape(D, chil, d, chi)
        As[bond+1] = Z.reshape(chi, d, chir)
    return As, J

# def total_weight_func(errs, array, model, L):
#     a,b = array.shape
#     array = np.append(array, np.zeros((a, (L-b)), dtype=np.int32), axis = 1)
#     integers = np.array([int(''.join(map(str, row)), 2) for row in array])
#     return errs*weight_func(integers, model, L)

def weight_func(integer, model, L):
    if model == 0:
        return 1/(integer+0.001)
    elif model == 1:
        return 1/np.sqrt(integer+0.001)
    elif model ==2:
        return np.exp(-integer) 
    else:
        return (2**L-1-integer)*(2**L-1)